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
• The user message is ALWAYS a JSON object with these keys for each question:
  - question_code: string (e.g., "Q44a")
  - question_text: string (verbatim prompt)
  - polarity: "POS" | "NEG" | "NEU"
  - reporting_metric:
      - column: the exact column to narrate (e.g., "Positive", "Negative", "AGREE", "Answer1")
      - label: human description of that metric (e.g., "% selecting Strongly agree / Agree",
               "% reporting Discrimination", "% selecting Answer 1: <label>")
      - meaning_labels: list of human labels that define the metric (e.g., ["Strongly agree","Agree"]);
                       may be empty for single-option cases or if not applicable.
      - reporting_field: optional diagnostic name (e.g., "POSITIVE","NEGATIVE","AGREE","ANSWER1")
  - data: table you must rely on. It contains "Year" and the reporting column.
          If subgroup analysis is enabled (category_in_play = true), each row may also include "Demographic".
  - category_in_play: boolean indicating whether demographic subgroups are present; if false,
                      do NOT discuss subgroup gaps.

STRICT RULES
1) Narrate ONLY values from 'reporting_metric.column' in 'data'. Do not switch columns or compute alternative aggregates.
2) Use 'reporting_metric.label' and 'meaning_labels' to explain what the percentage represents.
   • NEG: phrase as "reporting <meaning>" (e.g., “reported that X adversely affected …”).
   • POS: phrase as "selecting <meaning>" (e.g., “reported they have the tools …”).
   • NEU: base interpretation on scale labels in meaning_labels (e.g., Excellent/Very good/Good vs Fair/Poor).
3) **Parenthetical definition after every percentage (MANDATORY):**
   Immediately after each percentage you state, append the aggregated scale/definition in parentheses, using the
   metric and meaning labels from the payload. Examples:
     - 54% (Strongly agree/Agree)
     - 46% (reporting “Discrimination”)
     - 32% (Excellent/Very good)
   Apply consistently to single values and subgroup listings. For differences (e.g., “gap 5 p.p.”) you do not add a parenthesis to the delta itself.
4) Trend over years: when multiple years are present, characterize the direction across ALL years
   (improving/declining/stable/no clear trend). Cite years and values you actually see.
5) Demographic gaps (ONLY if category_in_play = true):
   • Identify the most recent year present (e.g., 2024) and the subgroups available for that year.
   • First list each subgroup’s latest-year value with its parenthetical definition (e.g., “2024: English 52% (Strongly agree/Agree), French 47% (Strongly agree/Agree)”).
   • Then quantify the largest absolute gap between any two subgroups for that year in percentage points.
     Example: “largest subgroup gap was 5 p.p. between English and French.”
   • Describe how that gap changed vs prior years (widened/narrowed/stable) using only provided values.
   • If subgroup data is incomplete for some years, acknowledge briefly (e.g., “no subgroup data for 2022”).
   • Do NOT discuss gaps if category_in_play is false.
6) All numbers you mention must be present in 'data' (or be simple differences of those numbers).
7) Be concise, verifiable, and avoid speculative language.

OUTPUT FORMAT
Return a single JSON object with one key:
  { "narrative": "<final short paragraph(s)>" }
"""

AI_SYSTEM_PROMPT = (BASE_SYSTEM_PROMPT + "\n\n" + ADDENDUM).strip()

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

def _df_minified_for_model(df, year_col: str, metric_col: str, include_demo: bool):
    """
    Compact rows for the model:
      • If include_demo=True and 'Demographic' exists: keep ["Year","Demographic", metric_col]
      • Else: keep ["Year", metric_col]
    """
    try:
        import pandas as pd
        if df is None:
            return []
        if not isinstance(df, pd.DataFrame):
            return []
        cols = list(df.columns)
        keep = []
        if year_col in cols:
            keep.append(year_col)
        if include_demo and "Demographic" in cols:
            keep.append("Demographic")
        if metric_col in cols:
            keep.append(metric_col)
        if not keep:
            return []
        work = df[keep].copy()
        # Coerce numerics for the metric and year
        if year_col in work.columns:
            work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
        work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
        work = work.dropna(subset=[year_col, metric_col])
        # Round metric; cast Year to int
        out = []
        for _, r in work.iterrows():
            row = {"Year": int(float(r[year_col]))}
            if include_demo and "Demographic" in work.columns and r.get("Demographic") is not None:
                row["Demographic"] = str(r["Demographic"])
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
    metric_col: str,            # EXACT column chosen by the app (Positive/Negative/AGREE/Answer1)
    metric_label: str,          # human-friendly description (e.g., "% selecting Strongly agree / Agree")
    category_in_play: bool,     # whether subgroups are present and should be analyzed
    # Optional hints (backward-compatible):
    polarity: Optional[str] = None,                 # POS | NEG | NEU
    reporting_field: Optional[str] = None,          # e.g., "POSITIVE","NEGATIVE","AGREE","ANSWER1"
    meaning_indices: Optional[List[int]] = None,    # e.g., [1,2]
    meaning_labels: Optional[List[str]] = None,     # e.g., ["Strongly agree","Agree"]
    year_col: str = "Year"
) -> str:
    """
    Returns the user message (JSON string) for a per-question analysis.
    Base system prompt is not changed; we include the extra fields the model needs.
    """
    if isinstance(df_disp, list):
        data_rows = df_disp
    else:
        data_rows = _df_minified_for_model(
            df_disp,
            year_col=year_col,
            metric_col=metric_col,
            include_demo=bool(category_in_play)
        )

    payload: Dict[str, Any] = {
        "task": "per_question_summary",
        "question_code": str(question_code),
        "question_text": str(question_text),
        "polarity": (polarity or "").upper() if polarity else "",
        "reporting_metric": {
            "column": metric_col,
            "label": metric_label,
            "meaning_labels": meaning_labels or [],
            "reporting_field": (reporting_field or ""),
        },
        "data": data_rows,             # strictly Year (+ optional Demographic) + chosen metric
        "category_in_play": bool(category_in_play),
        "instructions": {
            "no_math": True,
            "use_only_reporting_metric": True,
            "trend_over_all_years": True,
            "analyze_demographic_gaps": bool(category_in_play),
            "append_parenthetical_after_percent": True,  # explicit hint for new rule
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
) -> str:
    """
    Overall synthesis prompt — synthesize across ALL selected questions and years.
    """
    import pandas as pd

    data = []
    if isinstance(pivot_df, pd.DataFrame):
        try:
            pivot = pivot_df.copy()
            if pivot.index.name is None or str(pivot.index.name).lower() != "question":
                pivot.index.name = "Question"
            pivot = pivot.reset_index()
            for _, r in pivot.iterrows():
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

    payload = {
        "task": "overall_synthesis",
        "selected_questions": [
            {
                "question_code": q,
                "question_text": code_to_text.get(q, ""),
                "metric_label": q_to_metric.get(q, "% of respondents"),
            }
            for q in tab_labels
        ],
        "matrix": data,  # list of dicts: each question row with year:value pairs
        "instructions": {
            "compare_across_questions": True,
            "trend_over_all_years": True,
            "no_new_calculations": True,
            "append_parenthetical_after_percent": True,  # require parentheses in overall too
        },
    }
    return json.dumps(payload, ensure_ascii=False)

# --------------------------------------------------------------------------------------
# LLM CALLER — IMPLEMENTED
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
