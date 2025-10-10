# menu1/ai.py
"""
AI prompt and calling utilities for Menu 1.

Exports:
- AI_SYSTEM_PROMPT: strict system instruction for the model (your base text lives here).
- build_per_q_prompt(...): NA-safe JSON "user" payload for a single question.
- build_overall_prompt(...): NA-safe JSON "user" payload for a multi-question summary (requires code_to_text).
- call_openai_json(...): robust caller that returns (json_text, error_hint).
- extract_narrative(...): helper to pull 'narrative' string out of model JSON.

Notes (2025-10-10, surgical updates):
- We DO NOT overwrite your base prompt logic; we only APPEND a small addendum at call-time.
- Per/overall payload builders accept OPTIONAL fields:
    polarity_hint, reporting_field, meaning_indices, meaning_labels
  These help the model phrase results using the correct aggregate and labels, with ZERO math
  except explicitly allowed gaps — and now all-years trend classification.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import time
import pandas as pd

# -----------------------------
# Your base system prompt (kept intact)
# If you already define this elsewhere, you may ignore/override this constant by
# passing a 'system' string into call_openai_json(...).
# -----------------------------
AI_SYSTEM_PROMPT = (
    "You are preparing insights for the Government of Canada's Public Service Employee Survey (PSES).\n\n"
    "Context\n"
    "- The PSES informs improvements to people management in the the federal public service.\n"
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
# Addendum we APPEND at call-time (does not alter base prompt above)
# Explicitly enforces using provided aggregates + labels, allows gap math only,
# and adds all-years trend classification (consistent increase/decline, steady, mixed).
# -----------------------------
AI_REPORTING_ADDENDUM = """
AI Summary — Reporting Addendum (use provided aggregates; allowed math limited to gaps and trend classification)

1) Source of truth
   • Use the numeric values exactly as provided in the payload.
   • Do not compute complements (e.g., 100−POSITIVE), sum answers, average, rescale, or infer values.

2) Which number to report
   • For each question, a single reporting field is already selected by the app:
     POSITIVE, NEGATIVE, AGREE, or a specific ANSWERi (e.g., ANSWER1).
   • Narrate only that field’s values for the rows provided.

3) Meaning & labels
   • The payload may include meaning indices (e.g., 1,2) and meaning labels (e.g., "Excellent/Very good").
   • Use these labels verbatim to describe what the reported % represents.

4) Polarity (phrasing only)
   • A polarity_hint may be provided: POS → favourable phrasing; NEG → problem phrasing; NEU → phrase directly with labels.
   • Never transform numbers because of polarity.

5) Years & demographics
   • Focus on the years provided for each question. With demographics, refer only to the groups present in the payload.

6) Structure & tone
   • For each question: 1–2 concise sentences using the chosen field and its labels.
   • If multiple questions are present, add a brief Overall line comparing reported values at face value.
   • Do not mention validation in the narrative.

7) Gap analysis (the ONLY subgroup/temporal math)
   • You may compute percentage-point differences using numbers explicitly present in the payload:
     – Trend (two-year): latest vs previous year for the same group/measure.
     – Gaps (latest): differences between groups in the same year/measure.
     – Gap-over-time: change in a subgroup gap across two years, when both groups exist in both years.
   • No other calculations are allowed.

8) Trend analysis across ALL years (classification rules; integers only)
   • Apply these rules to the ordered sequence of yearly values included in the payload (per question/group).
   • If fewer than 2 years exist for a measure, omit the trend statement.

   Allowed computations:
     – Year-over-year integer subtraction: Δ_i = value_{year_{i+1}} − value_{year_i}
     – Net change across the window: Net = Last − First (integer)
     – Integer comparisons only

   Default thresholds:
     – Tiny fluctuation tolerance per step: ±1 % point
     – Minimum absolute net change to call a clear trend: ≥ 3 % points

   Classification (choose one):
     1) Flat / No trend: all year-over-year changes are exactly 0.
     2) Steady: every year-over-year change is within ±1 pp AND |Net| ≤ 1 pp.
     3) Consistent increase: all Δ_i ≥ 0, at least one Δ_i ≥ +1 pp, AND Net ≥ +3 pp.
     4) Consistent decline: all Δ_i ≤ 0, at least one Δ_i ≤ −1 pp, AND Net ≤ −3 pp.
     5) Mixed / Variable: anything else (ups and downs or small inconsistent moves).

   Wording examples:
     – "Across 2019–2024, results show a consistent increase (up +5% points overall)."
     – "Across 2019–2024, the pattern is steady (changes within ±1% point; no material net change)."
     – "Across 2019–2024, results are mixed/variable (ups and downs with no clear direction)."
""".strip()

def _compose_system_prompt(system: Optional[str]) -> str:
    """Return base system prompt + addendum without altering the original constant."""
    base = (system or AI_SYSTEM_PROMPT).rstrip()
    return f"{base}\n\n{AI_REPORTING_ADDENDUM}"

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
# Per-question prompt builder (accepts optional added fields)
# -----------------------------
def build_per_q_prompt(
    *,
    question_code: str,
    question_text: str,
    df_disp: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    category_in_play: bool,
    # OPTIONAL metadata for explicit phrasing (backward-compatible)
    polarity_hint: Optional[str] = None,           # "POS" | "NEG" | "NEU"
    reporting_field: Optional[str] = None,         # "POSITIVE" | "NEGATIVE" | "AGREE" | "ANSWER1".. "ANSWER7"
    meaning_indices: Optional[List[int]] = None,   # e.g., [4,5]
    meaning_labels: Optional[List[str]] = None,    # e.g., ["Often","Always"]
) -> str:
    """
    Build the per-question JSON payload for the model (as a string).
    NA-safe: never forces int on NaN; missing values become nulls in JSON.
    If no usable rows exist, returns a sentinel that the caller can handle.

    IMPORTANT: We send the rows exactly as they appear in df_disp for the chosen metric_col.
    Include ALL years you want the model to consider for the trend classification.
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

    rows: List[Dict[str, Any]] = []
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

    payload: Dict[str, Any] = {
        "question": {
            "code": question_code,
            "text": question_text or question_code,
            "metric": metric_label  # kept for backward compatibility
        },
        "category_in_play": bool(category_in_play),
        "rows": rows
    }

    # Include optional explicit metadata if provided
    if polarity_hint:
        payload["question"]["polarity_hint"] = str(polarity_hint)
    if reporting_field:
        payload["question"]["reporting_field"] = str(reporting_field)
    if meaning_indices is not None:
        payload["question"]["meaning_indices"] = [int(i) for i in meaning_indices if isinstance(i, (int, float, str))]
    if meaning_labels is not None:
        payload["question"]["meaning_labels"] = [str(s) for s in meaning_labels]

    return json.dumps(payload, ensure_ascii=False)

# -----------------------------
# Overall (multi-question) prompt builder — REQUIRES code_to_text
# Optional dicts for explicit metadata per question.
# -----------------------------
def build_overall_prompt(
    *,
    tab_labels: List[str],
    pivot_df: pd.DataFrame,
    q_to_metric: Dict[str, str],
    code_to_text: Dict[str, str],
    # OPTIONAL metadata maps (backward-compatible)
    code_to_polarity_hint: Optional[Dict[str, str]] = None,
    code_to_reporting_field: Optional[Dict[str, str]] = None,
    code_to_meaning_indices: Optional[Dict[str, List[int]]] = None,
    code_to_meaning_labels: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Build an overall JSON payload across multiple questions (as a string).
    REQUIREMENT: code_to_text is mandatory and used to label each question with its meaning.
    NA-safe melt of the pivot. If nothing usable, return sentinel.

    IMPORTANT: Include the years you want the model to consider; this enables the
    all-years trend classification in the addendum.
    """
    if pivot_df is None or pivot_df.empty or not tab_labels:
        return _NO_DATA_OVERALL

    pv = pivot_df.copy()
    if pv.index.name or pv.index.names:
        pv = pv.reset_index()

    # Detect year-like columns (ints or 'YYYY')
    year_cols: List[Any] = []
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

    rows: List[Dict[str, Any]] = []
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

    questions: List[Dict[str, Any]] = []
    for code in tab_labels:
        entry: Dict[str, Any] = {
            "code": code,
            "text": code_to_text.get(code, code),
            "metric": q_to_metric.get(code, "% of respondents")  # kept for backward compatibility
        }
        # Optional explicit metadata per question
        if code_to_polarity_hint and code in code_to_polarity_hint:
            entry["polarity_hint"] = code_to_polarity_hint[code]
        if code_to_reporting_field and code in code_to_reporting_field:
            entry["reporting_field"] = code_to_reporting_field[code]
        if code_to_meaning_indices and code in code_to_meaning_indices:
            entry["meaning_indices"] = [int(i) for i in code_to_meaning_indices[code]]
        if code_to_meaning_labels and code in code_to_meaning_labels:
            entry["meaning_labels"] = [str(s) for s in code_to_meaning_labels[code]]
        questions.append(entry)

    payload = {
        "latest_year": int(latest_year),
        "questions": questions,
        "metric_by_question": q_to_metric,
        "rows": rows
    }

    # Small natural-language preface to anchor behavior; the model still receives strict JSON
    user_msg = (
        "Synthesize cross-question patterns using the question texts (not codes). "
        "Use ONLY numbers present in the rows; describe differences in % points. "
        "Consider all years provided for each question when discussing trend classification.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return user_msg

# -----------------------------
# Caller: OpenAI (json)
# We append the addendum to whatever 'system' is passed, without editing the base text.
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

    # Compose final system prompt (base + addendum) without touching the original constant
    system_final = _compose_system_prompt(system)

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
                        {"role": "system", "content": system_final},
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
                            {"role": "system", "content": system_final},
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
