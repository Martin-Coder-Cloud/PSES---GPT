# menu1/ai.py
"""
AI prompt and calling utilities for Menu 1.

Exports:
- AI_SYSTEM_PROMPT: strict system instruction for the model (exact October 6 version).
- build_per_q_prompt(...): NA-safe JSON "user" payload for a single question.
- build_overall_prompt(...): NA-safe JSON "user" payload for a multi-question summary (requires code_to_text).
- call_openai_json(...): robust caller that returns (json_text, error_hint).
- extract_narrative(...): helper to pull 'narrative' string out of model JSON.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import os
import time
import pandas as pd

# -----------------------------
# Exact AI system prompt (Option 1 with gap-over-time, % everywhere)
# -----------------------------
AI_SYSTEM_PROMPT = (
    "You are preparing insights for the Government of Canada's Public Service Employee Survey (PSES).\n\n"
    "Context\n"
    "- The PSES informs improvements to people management in the federal public service.\n"
    "- Results help identify strengths and concerns in areas such as employee engagement, equity and inclusion, and workplace well-being.\n"
    "- The survey tracks progress over time to refine action plans. Statistics Canada administers the survey with the Treasury Board of Canada Secretariat. Confidentiality is guaranteed under the Statistics Act (grouped reporting; results for groups <10 are suppressed).\n\n"
    "Data-use rules (hard constraints)\n"
    "- Treat the provided JSON/table as the single source of truth.\n"
    "- Allowed numbers: integers that appear in the payload/table; integer differences formed by subtracting one payload integer from another (e.g., year-over-year changes, gaps between groups); and integer differences between such gaps across years (gap-over-time).\n"
    "- Do NOT invent numbers, averages, weighted figures, percentages, rescaled values, or decimals. Do NOT round.\n"
    "- If a value needed for a comparison is missing, omit that comparison rather than inferring.\n"
    "- Public Service–wide scope ONLY; do not reference specific departments unless present in the payload.\n\n"
    "Analysis rules (allowed computations ONLY)\n"
    "- Latest year = the maximum year present in the payload.\n"
    "- Trend (overall): If a previous year exists, compute the signed change (latest - previous) as an integer and report it as a change in % points (e.g., \"down 2% points\"). If not, skip.\n"
    "- Gaps (latest year): Compute absolute gaps between demographic groups (integer subtraction) and report them in % points (e.g., \"Women (82%) vs Another gender (72%): 10% points gap\"). Mention only the largest 1–2 gaps.\n"
    "- Gap-over-time: For each highlighted gap, compute the gap for each year where BOTH groups have values. State whether the gap has widened, narrowed, or remained stable since the earliest year with both groups (or vs the previous if only two years exist), and report the change in % points (e.g., \"gap narrowed by 3% points since 2020\"). If fewer than two such years exist, omit this sentence.\n"
    "- Do NOT compute multi-year averages, rates of change, or anything beyond the integer subtractions described above.\n\n"
    "Style & output\n"
    "- Report level values as integers followed by a percent sign (e.g., \"79%\", \"84%\").\n"
    "- Reserve \"percent points\" strictly for differences or gaps: write them as integers followed by \"% points\" (e.g., \"down 2% points\", \"a 10% points gap\"). Never describe a level as \"points\".\n"
    "- Professional, concise, neutral. Narrative style (1–3 short sentences, no lists).\n"
    "- Output VALID JSON with exactly one key: \"narrative\".\n"
)

# -----------------------------
# Sentinels (fail-safes)
# -----------------------------
_NO_DATA_PER_Q_PREFIX = "__NO_DATA_PER_Q__:"
_NO_DATA_OVERALL = "__NO_DATA_OVERALL__"

# -----------------------------
# Internal helpers (NA-safe)
# -----------------------------
def _to_py_int(x) -> Optional[int]:
    """Return int(x) or None if missing/NA."""
    try:
        if x is None:
            return None
        if isinstance(x, float):
            if pd.isna(x):
                return None
            return int(round(x))
        if isinstance(x, int):
            return int(x)
        val = pd.to_numeric(x, errors="coerce")
        if pd.isna(val):
            return None
        return int(round(val))
    except Exception:
        return None

def _detect_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Year", "year", "SURVEYR", "survey_year"):
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c)
        if len(s) == 4 and s.isdigit() and 1900 <= int(s) <= 2100:
            return "Year"
    return None

def _detect_demo_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Demographic", "group", "Group", "DEMOLBL", "demographic"):
        if c in df.columns:
            return c
    return None

def _coerce_metric_series(s: pd.Series) -> pd.Series:
    """9999 and NA -> NA; otherwise safe-int (nullable)."""
    if s is None or s.empty:
        return s
    s2 = pd.to_numeric(s.replace({9999: None}), errors="coerce").astype("Int64")
    return s2

# -----------------------------
# Per-question prompt builder
# -----------------------------
def build_per_q_prompt(
    *,
    question_code: str,
    question_text: str,
    df_disp: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    category_in_play: bool
) -> str:
    """
    Build the per-question JSON payload for the model (as a string).
    NA-safe: never forces int on NaN; missing values become nulls in JSON.
    If no usable rows exist, returns a sentinel that the caller can handle.
    """
    if df_disp is None or df_disp.empty or metric_col not in df_disp.columns:
        return f"{_NO_DATA_PER_Q_PREFIX}{question_code}"

    df = df_disp.copy()

    ycol = _detect_year_col(df)
    if not ycol:
        return f"{_NO_DATA_PER_Q_PREFIX}{question_code}"

    gcol = _detect_demo_col(df)
    if not gcol:
        gcol = "__Demographic__"
        df[gcol] = "All respondents"

    df["_VAL_"] = _coerce_metric_series(df[metric_col])

    if ycol != "Year":
        df["Year"] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")
    else:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    rows = []
    for _, r in df.iterrows():
        y = _to_py_int(r.get("Year"))
        v = _to_py_int(r.get("_VAL_"))
        if y is None:
            continue
        rows.append({
            "year": y,
            "group": str(r.get(gcol, "All respondents")),
            "value": v
        })

    if not rows:
        return f"{_NO_DATA_PER_Q_PREFIX}{question_code}"

    payload = {
        "question": {
            "code": question_code,
            "text": question_text or question_code,
            "metric": metric_label
        },
        "category_in_play": bool(category_in_play),
        "rows": rows
    }
    return json.dumps(payload, ensure_ascii=False)

# -----------------------------
# Overall (multi-question) prompt builder  — REQUIRES code_to_text
# -----------------------------
def build_overall_prompt(
    *,
    tab_labels: List[str],
    pivot_df: pd.DataFrame,
    q_to_metric: Dict[str, str],
    code_to_text: Dict[str, str],
) -> str:
    """
    Build an overall JSON payload across multiple questions (as a string).
    REQUIREMENT: code_to_text is mandatory and used to label each question with its meaning.
    NA-safe melt of the pivot. If nothing usable, return sentinel.
    Assumes pivot rows are keyed by question code (in index or a column).
    """
    if pivot_df is None or pivot_df.empty or not tab_labels:
        return _NO_DATA_OVERALL

    pv = pivot_df.copy()
    if pv.index.name or pv.index.names:
        pv = pv.reset_index()

    # Detect year-like columns (ints or 'YYYY')
    year_cols = []
    for c in pv.columns:
        try:
            if isinstance(c, int) and 1900 <= c <= 2100:
                year_cols.append(c)
            else:
                s = str(c)
                if len(s) == 4 and s.isdigit():
                    y = int(s)
                    if 1900 <= y <= 2100:
                        year_cols.append(c)
        except Exception:
            continue
    id_candidates = [c for c in pv.columns if c not in year_cols]
    preferred_names = {"question_code", "question", "code"}
    qcol = None
    for c in id_candidates:
        if str(c).lower() in preferred_names:
            qcol = c
            break
    if qcol is None:
        if not id_candidates:
            return _NO_DATA_OVERALL
        qcol = id_candidates[0]

    if not year_cols:
        year_cols = [c for c in pv.columns if c != qcol]

    long = pv.melt(id_vars=[qcol], value_vars=year_cols, var_name="year", value_name="value")
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["value"] = _coerce_metric_series(long["value"])
    long = long[long[qcol].astype(str).isin([str(q) for q in tab_labels])]

    rows = []
    for _, r in long.iterrows():
        y = _to_py_int(r.get("year"))
        v = _to_py_int(r.get("value"))
        if y is None or v is None:
            continue
        qcode = str(r.get(qcol))
        rows.append({
            "question_code": qcode,
            "question_text": code_to_text.get(qcode, qcode),
            "year": y,
            "value": v
        })

    if not rows:
        return _NO_DATA_OVERALL

    latest_year = max(r["year"] for r in rows)

    questions = [
        {"code": code, "text": code_to_text.get(code, code), "metric": q_to_metric.get(code, "% positive")}
        for code in tab_labels
    ]

    payload = {
        "latest_year": int(latest_year),
        "questions": questions,
        "metric_by_question": q_to_metric,
        "rows": rows
    }

    # Small natural-language preface to anchor behavior; the model still receives strict JSON
    user_msg = (
        "Synthesize cross-question patterns using the question texts (not codes). "
        "Use ONLY numbers present in the rows; describe differences in % points.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return user_msg

# -----------------------------
# Caller: OpenAI (json)
# -----------------------------
def call_openai_json(system: str, user: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Make a JSON-mode chat completion call with retries.
    Returns (json_text, error_hint).
    """
    if user.startswith(_NO_DATA_PER_Q_PREFIX):
        qcode = user.split(_NO_DATA_PER_Q_PREFIX, 1)[-1] or "the selected question"
        return json.dumps({"narrative": f"No AI summary: no data available for {qcode} under the current filters."}), None
    if user == _NO_DATA_OVERALL:
        return json.dumps({"narrative": "No overall AI summary: no data available under the current filters."}), None

    api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_APIKEY", "")
    model = os.environ.get("OPENAI_MODEL", "") or "gpt-4o-mini"

    if not api_key:
        return json.dumps({"narrative": "AI is not configured (no API key)."}), "missing_api_key"

    try:
        from openai import OpenAI  # modern SDK
        client = OpenAI(api_key=api_key)
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = (resp.choices[0].message.content or "").strip()
                return content, None
            except Exception as e:
                if attempt == 0:
                    time.sleep(0.8)
                else:
                    return json.dumps({"narrative": "AI request failed."}), f"openai_error:{type(e).__name__}"
    except Exception:
        # Legacy fallback
        try:
            import openai  # type: ignore
            openai.api_key = api_key
            for attempt in range(2):
                try:
                    resp = openai.ChatCompletion.create(
                        model=model,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    content = (resp.choices[0].message["content"] or "").strip()
                    return content, None
                except Exception as e:
                    if attempt == 0:
                        time.sleep(0.8)
                    else:
                        return json.dumps({"narrative": "AI request failed."}), f"openai_error:{type(e).__name__}"
        except Exception:
            return json.dumps({"narrative": "AI is unavailable in this environment."}), "no_openai_sdk"

# -----------------------------
# Extractor
# -----------------------------
def extract_narrative(json_text: str) -> Optional[str]:
    """Parse the model's JSON and return the 'narrative' string if present."""
    if not json_text:
        return None
    try:
        obj = json.loads(json_text)
        val = obj.get("narrative") if isinstance(obj, dict) else None
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        return None
    return None
