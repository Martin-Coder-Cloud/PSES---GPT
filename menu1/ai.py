# menu1/ai.py
"""
AI prompt and calling utilities for Menu 1.

Exports:
- AI_SYSTEM_PROMPT: strict system instruction for the model.
- build_per_q_prompt(...): NA-safe JSON "user" payload for a single question.
- build_overall_prompt(...): NA-safe JSON "user" payload for a multi-question summary.
- call_openai_json(...): robust caller that returns (json_text, error_hint).
- extract_narrative(...): helper to pull 'narrative' string out of model JSON.

Design goals:
- Never invent numbers: only serialize integers from the provided tables.
- Treat missing values as None (JSON null) and short-circuit to a friendly message if there's no usable data.
- Percent everywhere; changes are "percentage points".
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import os
import time

import pandas as pd

# -----------------------------
# System prompt (strict; % + % points)
# -----------------------------
AI_SYSTEM_PROMPT = (
    "You are preparing insights for the Government of Canada’s Public Service Employee Survey (PSES).\n\n"
    "Context\n"
    "- The PSES informs improvements to people management across the federal public service.\n"
    "- Results help identify strengths and concerns (engagement, equity, inclusion, well-being).\n"
    "- Trends are tracked across survey cycles.\n"
    "- Confidentiality is guaranteed; groups <10 are suppressed.\n\n"
    "Hard constraints\n"
    "- Use ONLY the provided JSON/table. Do NOT invent, assume, extrapolate, or impute values.\n"
    "- Scope is Public Service–wide unless the payload explicitly includes groups.\n"
    "- Treat all numeric values as integer PERCENTAGES; present them with the % sign (e.g., 79%).\n"
    "- Describe changes as percentage points (e.g., “down 2 percentage points”).\n\n"
    "Analysis rules\n"
    "- Start with the latest-year result for the selected question.\n"
    "- Describe the trend from the earliest year in the payload to the latest year.\n"
    "- If groups are present for the latest year, report the largest gap (in percentage points) and also state whether\n"
    "  that gap has widened, narrowed, or remained similar compared with an earlier year (only if both groups have data).\n"
    "- Only discuss values present in the payload. If something is missing, do not mention it.\n\n"
    "Style & output\n"
    "- Professional, concise, neutral; 1–3 short paragraphs.\n"
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
        if pd.isna(x):  # pandas NA/NaN
            return None
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
    return None


def _detect_demo_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Demographic", "demographic", "group_label", "group"):
        if c in df.columns:
            return c
    return None


def _coerce_metric_series(s: pd.Series) -> pd.Series:
    """Numeric with NA; map 9999 -> NA; leave as float/nullable for safety."""
    out = pd.to_numeric(s, errors="coerce")
    try:
        out = out.mask(out == 9999, other=pd.NA)
    except Exception:
        pass
    return out


def _is_year_like(col) -> bool:
    """True if column label looks like a survey year (int or 4-digit string in [1900,2100])."""
    try:
        if isinstance(col, int):
            return 1900 <= col <= 2100
        s = str(col)
        if len(s) == 4 and s.isdigit():
            y = int(s)
            return 1900 <= y <= 2100
        return False
    except Exception:
        return False


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

    # Identify columns
    ycol = _detect_year_col(df)
    if not ycol:
        return f"{_NO_DATA_PER_Q_PREFIX}{question_code}"

    gcol = _detect_demo_col(df)
    if not gcol:
        # If no demographic label present in the display table, synthesize an "All respondents" view
        gcol = "__Demographic__"
        df[gcol] = "All respondents"

    # Metric coercion (9999 -> NA)
    df[metric_col] = _coerce_metric_series(df[metric_col])
    # Year coercion
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")

    # Serialize only rows with both a Year and a metric value
    rows: List[Dict[str, Optional[int]]] = []
    for _, r in df.iterrows():
        y = _to_py_int(r.get(ycol))
        v = _to_py_int(r.get(metric_col))
        if y is None or v is None:
            continue
        rows.append({
            "year": y,
            "group": str(r.get(gcol)) if pd.notna(r.get(gcol)) else "All respondents",
            "value": v
        })

    if not rows:
        return f"{_NO_DATA_PER_Q_PREFIX}{question_code}"

    latest_year = max(r["year"] for r in rows)

    payload = {
        "question": {
            "code": str(question_code),
            "text": str(question_text or ""),
        },
        "metric_label": str(metric_label or "% positive"),
        "units": "percent",
        "category_in_play": bool(category_in_play),
        "latest_year": int(latest_year),
        "rows": rows
    }

    user_msg = (
        "Analyze the PSES results below without inventing any numbers. "
        "All values are integer percentages.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return user_msg


# -----------------------------
# Overall (multi-question) prompt builder
# -----------------------------
def build_overall_prompt(
    *,
    tab_labels: List[str],
    pivot_df: pd.DataFrame,
    q_to_metric: Dict[str, str],
) -> str:
    """
    Build an overall JSON payload across multiple questions (as a string).
    NA-safe melt of the pivot. If nothing usable, return sentinel.
    Assumes pivot rows are keyed by question code (either in index or a column).
    """
    if pivot_df is None or pivot_df.empty or not tab_labels:
        return _NO_DATA_OVERALL

    pv = pivot_df.copy()
    if pv.index.name or pv.index.names:
        pv = pv.reset_index()

    # Detect year columns by label and pick question id column from non-year candidates
    year_cols = [c for c in pv.columns if _is_year_like(c)]
    id_candidates = [c for c in pv.columns if c not in year_cols]

    # Prefer explicit question-like labels among id candidates; else first non-year column
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

    # If somehow no year columns detected, treat everything except qcol as year values
    if not year_cols:
        year_cols = [c for c in pv.columns if c != qcol]

    # Melt wide (years as columns) into long rows
    long = pv.melt(id_vars=[qcol], value_vars=year_cols, var_name="year", value_name="value")

    # Coerce types
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["value"] = _coerce_metric_series(long["value"])

    # Keep only requested questions
    long = long[long[qcol].astype(str).isin([str(q) for q in tab_labels])]

    # Build rows (skip NA)
    rows: List[Dict[str, Optional[int]]] = []
    for _, r in long.iterrows():
        y = _to_py_int(r.get("year"))
        v = _to_py_int(r.get("value"))
        if y is None or v is None:
            continue
        rows.append({
            "question_code": str(r.get(qcol)),
            "year": y,
            "value": v
        })

    if not rows:
        return _NO_DATA_OVERALL

    latest_year = max(r["year"] for r in rows)

    payload = {
        "latest_year": int(latest_year),
        "metric_by_question": q_to_metric,  # e.g., {"Q01": "% positive", ...}
        "rows": rows
    }

    user_msg = (
        "Synthesize cross-question patterns without inventing any numbers. "
        "All values are integer percentages.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return user_msg


# -----------------------------
# OpenAI caller (tuple return)
# -----------------------------
def call_openai_json(*, system: str, user: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Invoke the OpenAI chat API and return (json_text, error_hint).
    Fail-safe:
      - If `user` is a sentinel indicating no data, return a clear JSON message immediately.
    """
    # Fail-safe short-circuits
    if user.startswith(_NO_DATA_PER_Q_PREFIX):
        qcode = user.split(_NO_DATA_PER_Q_PREFIX, 1)[-1] or "the selected question"
        return json.dumps({"narrative": f"No AI summary: no data available for {qcode} under the current filters."}), None
    if user == _NO_DATA_OVERALL:
        return json.dumps({"narrative": "No overall AI summary: no data available under the current filters."}), None

    api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_APIKEY", "")
    model = os.environ.get("OPENAI_MODEL", "") or "gpt-4o-mini"

    if not api_key:
        return json.dumps({"narrative": "AI is not configured (no API key)."}), "missing_api_key"

    # Prefer the modern SDK if available
    try:
        from openai import OpenAI  # type: ignore
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
        # Fallback to legacy openai package
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
                    content = (resp["choices"][0]["message"]["content"] or "").strip()
                    # Ensure JSON envelope
                    try:
                        json.loads(content)
                        ok = True
                    except Exception:
                        ok = False
                    if not ok:
                        content = json.dumps({"narrative": content})
                    return content, None
                except Exception as e:
                    if attempt == 0:
                        time.sleep(0.8)
                    else:
                        return json.dumps({"narrative": "AI request failed."}), f"openai_error:{type(e).__name__}"
        except Exception:
            return json.dumps({"narrative": "AI is unavailable in this environment."}), "no_sdk"

    # Should not reach
    return json.dumps({"narrative": "AI request failed."}), "unknown"


# -----------------------------
# Narrative extractor
# -----------------------------
def extract_narrative(json_text: str) -> Optional[str]:
    """
    Parse the model's JSON and return the 'narrative' string if present.
    """
    if not json_text:
        return None
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            val = obj.get("narrative")
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        return None
    return None
