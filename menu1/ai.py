# ai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time

# (Minimal import required for DataFrame serialization in prompt builders)
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # We'll guard usage below

# --------------------------------------------------------------------------------------
# SYSTEM PROMPT (merged + refined)
# --------------------------------------------------------------------------------------

AI_SYSTEM_PROMPT = (
"You are preparing insights for the Government of Canada's Public Service Employee Survey (PSES).\n\n"

"Context\n"
"- The PSES informs improvements to people management in the federal public service.\n"
"- Results help identify strengths and concerns in areas such as engagement, inclusion, well-being, leadership, and career development.\n"
"- The survey tracks progress over time to refine departmental and enterprise-wide action plans.\n"
"- Statistics Canada administers the survey for the Treasury Board of Canada Secretariat (TBS). Confidentiality is guaranteed under the Statistics Act—results for groups with fewer than 10 respondents are suppressed.\n\n"

"Data-use rules (hard constraints)\n"
"- Treat the provided JSON/table as the single source of truth.\n"
"- Allowed numbers:\n"
"  • integers that appear in the payload/table;\n"
"  • integer differences formed by subtracting one payload integer from another (e.g., year-over-year changes, gaps between groups);\n"
"  • integer differences between such gaps across years (gap-over-time).\n"
"- Do NOT invent numbers, averages, weighted figures, rescaled values, or decimals. Do NOT round.\n"
"- If a value needed for a comparison is missing, omit that comparison rather than inferring.\n"
"- Scope is Public-Service-wide only—never name specific departments unless they appear in the payload.\n\n"

"Analysis rules (allowed computations ONLY)\n"
"- Latest year = the maximum year present in the payload.\n"
"- Gaps (latest year): compute absolute gaps between demographic groups and report them in % points "
"(e.g., “Women (82 %) vs Another gender (72 %): 10 % points gap”). Mention only the largest one or two gaps.\n"
"- Gap-over-time: for each highlighted gap, compute the gap in each year where both groups have data. "
"State whether the gap has widened, narrowed, or remained stable since the earliest comparable year, "
"and give the change in % points (e.g., “gap narrowed by 3 % points since 2020”).\n"
"- Do NOT compute multi-year averages or rates of change beyond these integer subtractions.\n"
"- If `distribution_only=true`, describe the latest-year distribution across Answer 1 … 6 and note direction of change only if visible; never sum options or create aggregates.\n\n"

"Trend rules (per-question; prioritize current year, then context)\n"
"- Start with the latest year vs the previous year: report the year-over-year (YoY) change in % points "
"(e.g., “2024: 54 %, down 2 % points vs 2023”).\n"
"- Then place this YoY in the full context of all available years:\n"
"  • Compute YoY deltas for every adjacent pair and the earliest→latest net change.\n"
"  • Classify the overall trend using YoY signs:\n"
"      ▸ Increasing = all YoY ≥ 0 and at least one > 0\n"
"      ▸ Declining = all YoY ≤ 0 and at least one < 0\n"
"      ▸ Stable = all values identical (YoY = 0)\n"
"      ▸ Mixed = otherwise (some increases and some decreases)\n"
"  • Explain how the latest YoY relates to the long-term pattern: is it continuing the trend, a reversal, a bump, or stabilization?\n"
"- Small movements (±1 % point) → “little change.”  If only one year → “No trend (single year).”\n"
"- Use only permitted math (differences in % points; no averages).\n\n"

"Trend roll-up (overall synthesis)\n"
"- Lead with the latest-year picture — which areas rose or fell vs previous year.\n"
"- Then summarize long-term patterns across questions: how many are increasing, declining, stable, or mixed.\n"
"- Briefly indicate whether current movements continue prior patterns or appear as reversals or bumps.\n"
"- Use numbers sparingly — only YoY and earliest→latest deltas. No new computations.\n\n"

"Overall synthesis rules (when task = \"overall_synthesis\")\n"
"- Purpose: summarize themes and implications across all selected questions — not to repeat each narrative.\n"
"- Identify common patterns (strengths, recurring concerns, areas improving or declining).\n"
"- Highlight areas of strong results (high positive %, clear improvements) and areas requiring attention "
"(lower scores, declines, or large demographic gaps).\n"
"- When useful, group related questions under broader ideas such as career development, work–life balance, inclusion, or leadership.\n"
"- Tone: concise, professional, suitable for briefing a director.\n"
"- Continue to obey all numeric constraints: only use provided numbers or allowable differences.\n\n"

"Style & output\n"
"- Report level values as integers followed by “%” (e.g., “79 %”).\n"
"- Reserve “% points” strictly for differences or gaps (e.g., “down 2 % points”, “a 10 % points gap”).\n"
"- Maintain professional, neutral language; short sentences (1–3 per paragraph).\n"
"- Write in narrative prose, not bullets.\n"
"- Output **valid JSON** with exactly one key: `\"narrative\"`.\n\n"

"ADDENDUM — Presentation of scale labels and footnotes\n"
"- Do not append long scale labels inline after percentages. Keep sentences clear and readable.\n"
"- The application displays, below each question, a short footnote explaining what the percentage represents "
"(e.g., “Percentages represent respondents’ aggregate answers to ‘a small/moderate/large/very large extent’.”).\n"
"- Mention labels inside the narrative only if essential for meaning, and then use the compressed form that preserves all categories.\n"
"- Apply the same rule for both per-question and overall summaries.\n"
"- Continue to respect all data-validation and polarity rules (Positive → ‘Positive’; Negative → ‘Negative’; Neutral → ‘Agree’). \n"
"- For D57-style distribution questions, still describe the category breakdown using the provided answer labels.\n\n"

"ADDENDUM — ANTI-HALLUCINATION: Trend & YoY gating\n"
"- Per-question summaries:\n"
"  • If the provided `data` contains FEWER THAN TWO distinct years, DO NOT write any trend, “over the years,” “year-over-year (YoY),” “change vs previous year,” or “% points” change statements. Optionally include the sentence: “There is no trend data available for prior years.”\n"
"  • Only write YoY or multi-year trend statements when the `data` has TWO OR MORE distinct years. Use only differences between integers appearing in the table.\n"
"- Overall synthesis:\n"
"  • Discuss trends ONLY for questions that have TWO OR MORE distinct years in the `matrix` (or the per-question `data`). Do NOT generalize trends across all selected questions if some questions have only a single year; clearly limit such statements to the applicable questions.\n"
"  • If none of the selected questions have two years of data, omit trend/YoY language entirely.\n"
"- When in doubt about the number of years, DEFAULT TO single-year behavior (omit trend/YoY).\n"
)

# --------------------------------------------------------------------------------------
# LLM CALLER (unchanged from your working version)
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
                temperature=0.1,  # kept as requested
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

# --------------------------------------------------------------------------------------
# PROMPT BUILDERS (restored; surgical addition only)
# --------------------------------------------------------------------------------------

def _df_to_records_sanitized(df) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame to a list of dicts (records), replacing 9999 with None
    in common survey numeric fields. Works even if pandas is unavailable.
    """
    if df is None:
        return []
    # If pandas isn't available, attempt a best-effort conversion
    if pd is None or not hasattr(df, "to_dict"):
        try:
            return list(df)  # may be wrong; callers provide real DataFrames in app
        except Exception:
            return []

    work = df.copy(deep=True)
    # Identify likely numeric survey columns
    for c in list(work.columns):
        lc = str(c).lower().replace(" ", "")
        if lc in {"positive", "negative", "agree",
                  "answer1", "answer2", "answer3", "answer4", "answer5", "answer6"}:
            try:
                work[c] = pd.to_numeric(work[c], errors="coerce").replace(9999, pd.NA)
            except Exception:
                pass
    return work.to_dict(orient="records")

def _distinct_years_in_df(df) -> List[int]:
    """Extract distinct year values from tidy or wide DataFrames (best effort)."""
    years: List[int] = []
    if df is None or pd is None:
        return years
    try:
        if "Year" in df.columns:
            yrs = pd.to_numeric(df["Year"], errors="coerce")
            years = sorted({int(y) for y in yrs.dropna().unique().tolist() if 1900 <= int(y) <= 2100})
            return years
        # wide form: columns like 2019, 2020 ...
        col_years = []
        for c in df.columns:
            s = str(c)
            if len(s) == 4 and s.isdigit():
                col_years.append(int(s))
        if col_years:
            years = sorted(set(col_years))
    except Exception:
        years = []
    return years

def build_per_q_prompt(
    *,
    question_code: str,
    question_text: str,
    df_disp,
    metric_col: Optional[str],
    metric_label: Optional[str],
    category_in_play: bool,
    meaning_labels: Optional[List[str]] = None,
    reporting_field: Optional[str] = None,
    distribution_only: bool = False,
) -> str:
    """
    Build the per-question user prompt. This mirrors your legacy helper signature
    and passes only the allowed data plus explicit year_count to enforce the
    anti-hallucination gating in the system prompt.
    """
    data_records = _df_to_records_sanitized(df_disp)
    years = _distinct_years_in_df(df_disp)
    payload: Dict[str, Any] = {
        "task": "per_question",
        "question_code": question_code,
        "question_text": question_text,
        "data": data_records,
        "metric": {
            "column": metric_col,
            "label": metric_label,
            "reporting_field": reporting_field,  # POSITIVE/NEGATIVE/AGREE/ANSWER1/None
        },
        "demographic_breakdown_present": bool(category_in_play),
        "meaning_labels": list(meaning_labels or []),
        "distribution_only": bool(distribution_only),
        # Critical for trend gating:
        "year_count": len(years),
        "years_present": years,
        # Narrative constraints (explicit reminders for the model):
        "constraints": {
            "numbers": "Use only integers in the table or their simple differences (YoY/gaps); 9999 means N/A.",
            "no_averages_or_decimals": True,
            "no_inference_if_missing": True,
        },
        "output_format": {"type": "json", "key": "narrative"},
    }
    return json.dumps(payload, ensure_ascii=False)

def build_overall_prompt(
    *,
    tab_labels: List[str],
    pivot_df,
    q_to_metric: Dict[str, str],
    code_to_text: Dict[str, str],
    q_to_meaning_labels: Optional[Dict[str, List[str]]] = None,
    q_distribution_only: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Build the overall synthesis prompt (legacy name/signature).
    Provides the summary matrix and per-question metadata, including per-question
    year_counts to ensure the model limits trend statements appropriately.
    """
    # pivot_df is a matrix indexed by question (and optionally demographic) with year columns
    # We'll convert it to a plain structure: index -> row dict, plus year columns list.
    matrix: Dict[str, Dict[str, Any]] = {}
    years: List[int] = []
    if pd is not None and hasattr(pivot_df, "to_dict"):
        try:
            # Identify year columns
            years = []
            for c in pivot_df.columns:
                s = str(c)
                if len(s) == 4 and s.isdigit():
                    years.append(int(s))
            years = sorted(set(years))
            # Convert each row to dict, keyed by its index repr
            for idx, row in pivot_df.iterrows():
                key = idx if isinstance(idx, str) else str(idx)
                matrix[key] = {str(y): (None if pd.isna(row.get(y)) else int(round(float(row.get(y))))) for y in years}
        except Exception:
            matrix = {}
            years = []
    else:
        matrix = {}
        years = []

    # Build per-question year_count map: count non-null across the row for each question
    q_year_counts: Dict[str, int] = {}
    if pd is not None and hasattr(pivot_df, "loc"):
        try:
            for q in tab_labels:
                try:
                    # Sum non-null across available year columns for that question (first index level)
                    row = pivot_df.loc[q]
                    if hasattr(row, "to_frame"):  # could be Series or DataFrame (multi-index)
                        if hasattr(row, "to_dict"):
                            # flatten: take first row if multi-index
                            if hasattr(row, "index") and len(getattr(row, "index", [])) > 0 and isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                    non_null = 0
                    for y in years:
                        try:
                            val = row.get(y)
                        except Exception:
                            val = None
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            non_null += 1
                    q_year_counts[q] = non_null
                except Exception:
                    q_year_counts[q] = 0
        except Exception:
            q_year_counts = {q: 0 for q in tab_labels}

    payload: Dict[str, Any] = {
        "task": "overall_synthesis",
        "selected_questions": [
            {
                "code": q,
                "text": code_to_text.get(q, ""),
                "metric_label": q_to_metric.get(q, ""),
                "meaning_labels": list((q_to_meaning_labels or {}).get(q, [])),
                "distribution_only": bool((q_distribution_only or {}).get(q, False)),
                "year_count": int(q_year_counts.get(q, 0)),
            }
            for q in tab_labels
        ],
        "matrix": {
            "years": years,
            "values_by_row": matrix,  # index-> { "2019": 54, "2020": 56, ... } (ints or None)
        },
        # Narrative constraints (explicit reminders for the model):
        "constraints": {
            "numbers": "Use only integers in the matrix or simple differences (YoY/gaps).",
            "no_averages_or_decimals": True,
            "no_inference_if_missing": True,
        },
        "output_format": {"type": "json", "key": "narrative"},
    }
    return json.dumps(payload, ensure_ascii=False)
