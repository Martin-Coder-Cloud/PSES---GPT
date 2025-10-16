# menu1/ai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time

# Optional pandas import
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

__all__ = [
    "AI_SYSTEM_PROMPT",
    "call_openai_json",
    "build_per_q_prompt",
    "build_overall_prompt",
]

# --------------------------------------------------------------------------------------
# SYSTEM PROMPT (same as approved; added single-year style guard)
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
"- If `distribution_only=true`, describe only the latest-year distribution; never create aggregates.\n\n"

"Trend rules (per-question; prioritize current year, then context)\n"
"- Start with the latest year vs the previous year: report the year-over-year (YoY) change in % points "
"(e.g., “2024: 54 %, down 2 % points vs 2023”).\n"
"- Then place this YoY in context of all available years:\n"
"  • Compute YoY deltas and the earliest→latest net change.\n"
"  • Classify as increasing, declining, stable, or mixed.\n"
"  • Explain how the latest YoY relates to the long-term pattern.\n"
"- Small movements (±1 % point) → “little change.”  If only one year → “No trend (single year).”\n\n"

"Overall synthesis rules (when task = \"overall_synthesis\")\n"
"- Summarize themes and implications across questions — not to repeat each narrative.\n"
"- Identify common strengths and areas requiring attention.\n"
"- Highlight areas of improvement or decline only when supported by data.\n"
"- Keep tone concise, professional, suitable for a director briefing.\n\n"

"Style & output\n"
"- Report level values as integers followed by “%”. Reserve “% points” only for differences or gaps.\n"
"- Write in short neutral sentences (1–3 per paragraph).\n"
"- Output **valid JSON** with exactly one key: `\"narrative\"`.\n\n"

"ADDENDUM — Presentation of scale labels and footnotes\n"
"- Do not append long scale labels inline after percentages. Keep sentences clear and readable.\n"
"- The application displays, below each question, a short footnote explaining what the percentage represents "
"(e.g., “Percentages represent respondents’ aggregate answers to ‘a small/moderate/large/very large extent’.”).\n"
"- Mention labels inside the narrative only if essential for meaning, and then use the compressed form.\n\n"

"ADDENDUM — Anti-hallucination: strict trend gating\n"
"- You must obey the `allow_trend` flag in the payload:\n"
"  • If `allow_trend=false`, you must NOT write any sentences about change, YoY, “since <year>”, or “trend”. "
"    You may instead state: “There is no trend data available for prior years.”\n"
"  • If `allow_trend=true`, trend/YoY statements are allowed but must use only years appearing in `years_present`.\n"
"- Never mention or invent years not listed in `years_present`.\n"
"- In overall synthesis, discuss trends only for questions with `allow_trend=true`. "
"Do not generalize trends across all questions if others have single-year data.\n"
"- When uncertain, default to single-year phrasing.\n\n"

"ADDENDUM — Single-year style guard (readability & consistency)\n"
"- When `allow_trend=false`, open with a full sentence — do NOT use telegraphic formats like “YYYY: 54 %* …”. "
"  Start with: “In <LATEST_YEAR>, <VALUE>%* …”.\n"
"- <LATEST_YEAR> must be the maximum of `years_present`. <VALUE> must be the reported metric for that year as given in the table.\n"
"- After this sentence, you may add: “There is no trend data available for prior years.”\n"
)

# --------------------------------------------------------------------------------------
# OpenAI call (unchanged)
# --------------------------------------------------------------------------------------

def call_openai_json(*, system: str, user: str) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback = json.dumps({"narrative": "AI is not configured (missing OPENAI_API_KEY)."}, ensure_ascii=False)
        return fallback, "OPENAI_API_KEY missing"

    model = os.environ.get("AI_MODEL", "gpt-4o-mini").strip()
    try:
        from openai import OpenAI
    except Exception as e:
        fb = json.dumps({"narrative": "AI client missing. Please install openai>=1.0.0."}, ensure_ascii=False)
        return fb, f"OpenAI import error: {type(e).__name__}"

    client = OpenAI(api_key=api_key)
    last_err = None
    for attempt in range(2):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
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

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _df_to_records_sanitized(df) -> List[Dict[str, Any]]:
    if df is None or pd is None:
        return []
    work = df.copy(deep=True)
    for c in work.columns:
        lc = str(c).lower().replace(" ", "")
        if lc in {"positive","negative","agree",
                  "answer1","answer2","answer3","answer4","answer5","answer6"}:
            try:
                work[c] = pd.to_numeric(work[c], errors="coerce").replace(9999, pd.NA)
            except Exception:
                pass
    return work.to_dict(orient="records")

def _distinct_valid_years(df, metric_col: Optional[str]) -> List[int]:
    """Return list of years with non-null metric values."""
    if df is None or pd is None or metric_col is None or metric_col not in df.columns:
        return []
    try:
        work = df[[metric_col]].copy()
        years: List[int] = []
        if "Year" in df.columns:
            valid = df.loc[df[metric_col].notna(), "Year"]
            years = sorted({int(y) for y in pd.to_numeric(valid, errors="coerce").dropna().tolist() if 1900 <= int(y) <= 2100})
        else:
            for c in df.columns:
                if len(str(c)) == 4 and str(c).isdigit():
                    if df[c].notna().any():
                        years.append(int(c))
        return sorted(set(years))
    except Exception:
        return []

# --------------------------------------------------------------------------------------
# Prompt builders (unchanged except for allow_trend/years_present we added earlier)
# --------------------------------------------------------------------------------------

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
    data_records = _df_to_records_sanitized(df_disp)
    years_present = _distinct_valid_years(df_disp, metric_col)
    allow_trend = len(years_present) >= 2
    payload = {
        "task": "per_question",
        "question_code": question_code,
        "question_text": question_text,
        "data": data_records,
        "metric": {
            "column": metric_col,
            "label": metric_label,
            "reporting_field": reporting_field,
        },
        "demographic_breakdown_present": bool(category_in_play),
        "meaning_labels": list(meaning_labels or []),
        "distribution_only": bool(distribution_only),
        "allow_trend": allow_trend,
        "years_present": years_present,
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
    matrix: Dict[str, Dict[str, Any]] = {}
    years: List[int] = []
    if pd is not None and hasattr(pivot_df, "columns"):
        try:
            for c in pivot_df.columns:
                if len(str(c)) == 4 and str(c).isdigit():
                    years.append(int(c))
            years = sorted(set(years))
            for idx, row in pivot_df.iterrows():
                key = idx if isinstance(idx, str) else str(idx)
                matrix[key] = {
                    str(y): (None if pd.isna(row.get(y)) else int(round(float(row.get(y))))) for y in years
                }
        except Exception:
            pass

    selected_questions = []
    for q in tab_labels:
        try:
            allow_trend = False
            years_present = []
            if q in pivot_df.index:
                row = pivot_df.loc[q]
                non_null_years = []
                for y in years:
                    try:
                        val = row.get(y)
                    except Exception:
                        val = None
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        non_null_years.append(y)
                years_present = sorted(non_null_years)
                allow_trend = len(non_null_years) >= 2
            selected_questions.append({
                "code": q,
                "text": code_to_text.get(q, ""),
                "metric_label": q_to_metric.get(q, ""),
                "meaning_labels": list((q_to_meaning_labels or {}).get(q, [])),
                "distribution_only": bool((q_distribution_only or {}).get(q, False)),
                "allow_trend": allow_trend,
                "years_present": years_present,
            })
        except Exception:
            selected_questions.append({
                "code": q, "text": code_to_text.get(q, ""),
                "metric_label": q_to_metric.get(q, ""),
                "meaning_labels": [], "distribution_only": False,
                "allow_trend": False, "years_present": []
            })

    payload = {
        "task": "overall_synthesis",
        "selected_questions": selected_questions,
        "matrix": {"years": years, "values_by_row": matrix},
        "output_format": {"type": "json", "key": "narrative"},
    }
    return json.dumps(payload, ensure_ascii=False)
