# ai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time

# --------------------------------------------------------------------------------------
# SYSTEM PROMPT (base unchanged; addendum appended)
# --------------------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = os.environ.get(
    "AI_BASE_SYSTEM_PROMPT",
    """
You are an analyst producing neutral, data-faithful survey summaries. Never invent data.
Never compute new percentages or totals. Only use values provided in the user JSON payload.
If a value is absent or marked as missing, say so briefly rather than inferring.
Keep tone impartial and concise.
"""
).strip()

ADDENDUM = """
ADDENDUM — DOMAIN CONTEXT (REQUIRED)
• This is a Public Service Employee Survey (PSES) for the federal public service.
• Your audience is HR leaders, people managers, and executives. Write in plain language,
  focusing on what the reported percentages mean for workforce experience (e.g., engagement,
  inclusion, workload/stress, harassment/discrimination, career development, enabling supports).
• Stay neutral and data-grounded. Offer short, descriptive insights (what, where, direction),
  not policy prescriptions. Do not speculate about causes or attribute intent.
• Respect privacy/suppression principles: if data for a subgroup/year is missing or suppressed,
  acknowledge that briefly instead of inferring.

ADDENDUM — HOW TO USE THE USER PAYLOAD FIELDS (REQUIRED)
• The user message is ALWAYS a JSON object for either a per-question task or an overall synthesis task.

PER-QUESTION PAYLOAD KEYS
  - task: "per_question_summary"
  - question_code: string (e.g., "Q44a")
  - question_text: string (verbatim prompt)
  - polarity: "POS" | "NEG" | "NEU"
  - reporting_metric:  # used unless distribution_only=true (see exception)
      - column: the exact column to narrate (e.g., "Positive", "Negative", "AGREE", "Answer1")
      - label: human description of that metric (e.g., "% selecting Strongly agree / Agree",
               "% negative", "% selecting Answer 1: <label>")
      - meaning_labels: list of option labels that define the metric (e.g., ["Strongly agree","Agree"]);
                        may be empty for single-option cases or if not applicable.
      - reporting_field: optional diagnostic name (e.g., "POSITIVE","NEGATIVE","AGREE","ANSWER1")
  - data: table you must rely on. Each row has "Year" and the reporting column.
          If subgroup analysis is enabled (category_in_play=true), rows may also have "Demographic".
  - category_in_play: boolean; if false, do NOT discuss subgroup gaps.
  - distribution_only: boolean; when true, DO NOT use reporting_metric at all.
      • Instead, interpret the table as a categorical distribution and narrate the listed
        answer options (Answer1..Answer6) for the latest year, optionally noting change vs prior years.
      • This is intended for questions like D57_A / D57_B that do not have an aggregate measure.

OVERALL PAYLOAD KEYS
  - task: "overall_synthesis"
  - selected_questions: list of { question_code, question_text, metric_label, meaning_labels?, distribution_only? }
      • When distribution_only=true for a question, avoid quoting an aggregate % for that item
        and exclude it from cross-question % comparisons. You may reference it qualitatively if needed.
  - matrix: list of row dicts (one per question row from the Summary matrix) with year:value pairs.
  - instructions: { compare_across_questions, trend_over_all_years, no_new_calculations, append_parenthetical_after_percent }

STRICT RULES
1) Narrate ONLY values provided. Do not switch columns or invent aggregates.
2) Parenthetical after every percentage (MANDATORY):
   • Immediately after each % number, append the exact aggregated response options in parentheses.
     Examples: 54% (Strongly agree/Agree); 32% (Excellent/Very good); 41% (Answer 1: “Yes”).
   • Build the parenthetical from 'meaning_labels' (joined with "/"). If 'meaning_labels' is empty,
     derive options from 'reporting_metric.label' (e.g., parse “% selecting Strongly agree / Agree”).
   • If you cannot derive them reliably, OMIT the parenthetical (do not paraphrase question text).
3) Trend over years: when multiple years exist, characterize direction across ALL years
   (improving/declining/stable/no clear trend). Cite years and values you actually see.
4) Demographic gaps (ONLY if category_in_play=true):
   • Identify the latest year, list each subgroup’s value with its parenthetical, then quantify the largest gap
     in p.p., and describe how that gap changed vs prior years.
   • If subgroup data is missing for some years, say so briefly.
5) Distribution-only exception (e.g., D57_A / D57_B):
   • If 'distribution_only' = true, ignore 'reporting_metric'. Use the table’s categorical options (e.g., Answer1..Answer6).
   • In the latest year, list the option breakdown (option label and % for each option that exists in the table).
   • If prior years exist, briefly indicate directional change (e.g., notable increases/decreases).
   • Do NOT compute a custom aggregation (e.g., do not sum options).
6) All numbers you mention must be present in the payload (or be simple differences such as p.p. gaps).
7) Be concise, verifiable, and avoid speculative language.

OUTPUT FORMAT
Return a single JSON object with one key:
  { "narrative": "<final short paragraph(s)>" }
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# FINAL ADDENDUM — LABEL COMPRESSION & PLACEMENT (REQUIRED)
# Keep all prior text unchanged; this addendum only clarifies formatting.
FINAL_ADDENDUM = """
FINAL ADDENDUM — Label compression & placement
• Placement: Always place the parentheses with the scale labels IMMEDIATELY AFTER each percentage
  (e.g., "54% (To a small/moderate/large extent/very large extent)"), not at the end of the sentence.
• Compression allowed: You MAY use a compressed form that preserves the full meaning and order of the labels
  by eliding repeated prefixes/suffixes. Example:
    Full: "(To a small extent/To a moderate extent/To a large extent/To a very large extent)"
    OK:   "(To a small/moderate/large extent/very large extent)"
• Consistency: Apply the same rule in both per-question summaries and the overall synthesis; when referencing
  a percentage for question X, append the labels for X (compressed if appropriate, otherwise full) right after the %.
• Do not reorder, merge, or rename categories; do not invent alternative wording beyond this compression.
""".strip()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

AI_SYSTEM_PROMPT = (BASE_SYSTEM_PROMPT + "\n\n" + ADDENDUM + "\n\n" + FINAL_ADDENDUM).strip()

# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------

def _safe_round(v: Any) -> Any:
    try:
        if v is None:
            return None
        f = float(v)
        return round(f, 1)
    except Exception:
        return v

def _df_minified_for_model(df, year_col: str, metric_col: Optional[str], include_demo: bool, distribution_only: bool):
    """
    Compact rows for the model:
      • distribution_only=False:
          keep ["Year", metric_col, ("Demographic" if include_demo)]
      • distribution_only=True:
          keep ["Year", ("Demographic" if include_demo), and any columns that look like Answer1..Answer6]
    """
    try:
        import pandas as pd
        if df is None:
            return []
        if not isinstance(df, pd.DataFrame):
            return []
        cols = list(df.columns)
        work = df.copy()

        if distribution_only:
            # Keep Year, optional Demographic, and all Answer1..6 columns present
            keep = []
            if year_col in cols:
                keep.append(year_col)
            if include_demo and "Demographic" in cols:
                keep.append("Demographic")
            answer_cols = [c for c in cols if str(c).strip().replace(" ", "").lower() in
                           [f"answer{i}" for i in range(1, 7)]]
            # Also accept "Answer 1" form
            for i in range(1, 7):
                alt = f"Answer {i}"
                if alt in cols and alt not in answer_cols:
                    answer_cols.append(alt)
            # Deduplicate preserving order
            seen = set(); answers_kept = []
            for c in answer_cols:
                k = c.lower().replace(" ", "")
                if k not in seen:
                    answers_kept.append(c); seen.add(k)
            keep += answers_kept

            if not keep:
                return []
            work = work[keep].copy()
            if year_col in work.columns:
                work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
            # coerce each AnswerN to numeric
            for c in answers_kept:
                work[c] = pd.to_numeric(work[c], errors="coerce")

            work = work.dropna(subset=[year_col])
            out = []
            for _, r in work.iterrows():
                row = {"Year": int(float(r[year_col]))}
                if include_demo and "Demographic" in work.columns and r.get("Demographic") is not None:
                    row["Demographic"] = str(r["Demographic"])
                # Attach only present AnswerN values
                for c in answers_kept:
                    val = r.get(c, None)
                    if val is not None:
                        try:
                            row[c] = _safe_round(val)
                        except Exception:
                            pass
                out.append(row)
            out.sort(key=lambda x: (x["Year"], x.get("Demographic","")))
            return out

        # Regular (aggregate) mode
        keep = []
        if year_col in cols:
            keep.append(year_col)
        if include_demo and "Demographic" in cols:
            keep.append("Demographic")
        if metric_col and metric_col in cols:
            keep.append(metric_col)
        if not keep:
            return []
        work = work[keep].copy()
        if year_col in work.columns:
            work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
        if metric_col in work.columns:
            work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
        work = work.dropna(subset=[year_col])
        if metric_col in work.columns:
            work = work.dropna(subset=[metric_col])

        out = []
        for _, r in work.iterrows():
            row = {"Year": int(float(r[year_col]))}
            if include_demo and "Demographic" in work.columns and r.get("Demographic") is not None:
                row["Demographic"] = str(r["Demographic"])
            if metric_col in work.columns:
                row[metric_col] = _safe_round(r[metric_col])
            out.append(row)
        out.sort(key=lambda x: (x["Year"], x.get("Demographic","")))
        return out
    except Exception:
        return []

# --------------------------------------------------------------------------------------
# PROMPT BUILDERS
# --------------------------------------------------------------------------------------

def build_per_q_prompt(
    *,
    question_code: str,
    question_text: str,
    df_disp,                    # pandas.DataFrame or already-minified list of dicts
    metric_col: Optional[str],  # EXACT column chosen by the app (Positive/Negative/AGREE/Answer1) or None in distribution mode
    metric_label: str,          # human-friendly description (e.g., "% selecting Strongly agree / Agree")
    category_in_play: bool,     # whether subgroups are present and should be analyzed
    # Optional hints:
    polarity: Optional[str] = None,                 # POS | NEG | NEU
    reporting_field: Optional[str] = None,          # e.g., "POSITIVE","NEGATIVE","AGREE","ANSWER1"
    meaning_indices: Optional[List[int]] = None,    # e.g., [1,2]
    meaning_labels: Optional[List[str]] = None,     # e.g., ["Strongly agree","Agree"]
    year_col: str = "Year",
    distribution_only: bool = False                 # EXCEPTION: D57_A / D57_B style
) -> str:
    """
    Returns the user message (JSON string) for a per-question analysis.
    """
    if isinstance(df_disp, list):
        data_rows = df_disp
    else:
        data_rows = _df_minified_for_model(
            df_disp,
            year_col=year_col,
            metric_col=metric_col,
            include_demo=bool(category_in_play),
            distribution_only=bool(distribution_only),
        )

    payload: Dict[str, Any] = {
        "task": "per_question_summary",
        "question_code": str(question_code),
        "question_text": str(question_text),
        "polarity": (polarity or "").upper() if polarity else "",
        "reporting_metric": {
            "column": metric_col or "",
            "label": metric_label,
            "meaning_labels": meaning_labels or [],
            "reporting_field": (reporting_field or ""),
        },
        "data": data_rows,             # strictly Year (+ optional Demographic) + chosen fields
        "category_in_play": bool(category_in_play),
        "distribution_only": bool(distribution_only),
        "instructions": {
            "no_math": True,
            "use_only_reporting_metric": not bool(distribution_only),
            "trend_over_all_years": True,
            "analyze_demographic_gaps": bool(category_in_play),
            "append_parenthetical_after_percent": True,
        },
    }
    if meaning_indices:
        payload["reporting_metric"]["meaning_indices"] = meaning_indices

    return json.dumps(payload, ensure_ascii=False)

def build_overall_prompt(
    *,
    tab_labels: List[str],
    pivot_df,  # pandas.DataFrame
    q_to_metric: Dict[str, str],
    code_to_text: Dict[str, str],
    q_to_meaning_labels: Optional[Dict[str, List[str]]] = None,
    q_distribution_only: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Overall synthesis prompt — synthesize across ALL selected questions and years.
    Includes per-question meaning_labels so the model can append parentheses after % consistently.
    Also includes distribution_only flags for D57_A/B-style items.
    """
    import pandas as pd

    data = []
    if isinstance(pivot_df, pd.DataFrame):
        try:
            pivot = pivot_df.copy()
            if "Question" not in pivot.reset_index().columns:
                pivot.index.name = "Question"
            pivot = pivot.reset_index()
            for _, r in pivot.iterrows():
                if "Question" not in r:
                    continue
                row = {"question_code": str(r["Question"])}
                for c in pivot_df.columns:
                    try:
                        yr = int(float(c))
                        row[str(yr)] = _safe_round(r[c])
                    except Exception:
                        continue
                data.append(row)
        except Exception:
            data = []

    selected = []
    for q in tab_labels:
        item = {
            "question_code": q,
            "question_text": code_to_text.get(q, ""),
            "metric_label": q_to_metric.get(q, "% of respondents"),
        }
        if q_to_meaning_labels and q in q_to_meaning_labels:
            item["meaning_labels"] = q_to_meaning_labels[q]
        if q_distribution_only and q in q_distribution_only:
            item["distribution_only"] = bool(q_distribution_only[q])
        selected.append(item)

    payload = {
        "task": "overall_synthesis",
        "selected_questions": selected,
        "matrix": data,  # list of dicts: each question row with year:value pairs
        "instructions": {
            "compare_across_questions": True,
            "trend_over_all_years": True,
            "no_new_calculations": True,
            "append_parenthetical_after_percent": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False)

# --------------------------------------------------------------------------------------
# LLM CALLER — IMPLEMENTED (OpenAI v1)
# --------------------------------------------------------------------------------------

def call_openai_json(*, system: str, user: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls OpenAI with JSON response mode.
    Requires OPENAI_API_KEY. Optional: AI_MODEL (default 'gpt-4o-mini').
    Returns (content_json_text, debug_hint_or_none).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback = json.dumps({"narrative": "AI is not configured (missing OPENAI_API_KEY)."}, ensure_ascii=False)
        return fallback, "OPENAI_API_KEY missing"

    model = os.environ.get("AI_MODEL", "gpt-4o-mini").strip()

    try:
        from openai import OpenAI
    except Exception as e:
        fallback = json.dumps({"narrative": "AI client missing. Please install openai>=1.0.0."}, ensure_ascii=False)
        return fallback, f"OpenAI import error: {type(e).__name__}"

    client = OpenAI(api_key=api_key)

    last_err = None
    for attempt in range(2):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=700,
            )
            content = rsp.choices[0].message.content
            try:
                _ = json.loads(content or "{}")
            except Exception:
                content = json.dumps({"narrative": (content or "").strip()})
            return content, None
        except Exception as e:
            last_err = e
            time.sleep(0.4)

    fb = json.dumps({"narrative": "The AI service is temporarily unavailable."}, ensure_ascii=False)
    return fb, f"LLM error: {type(last_err).__name__}"
