# app/menu1/ai.py
"""
AI prompt and calling utilities for Menu 1.

PUBLIC API:
- AI_SYSTEM_PROMPT: system instruction used for the model
- build_per_q_prompt(...): builds the JSON "user" payload for a single question
- build_overall_prompt(...): builds the JSON "user" payload for the multi-question overview
- call_openai_json(...): calls the model in JSON mode and returns (json_text, error_hint)
- extract_narrative(...): parses the model JSON and returns the "narrative" text
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import os
import re

import pandas as pd

try:
    # Expected to exist in your app; if not, set OPENAI_MODEL env var.
    from .constants import DEFAULT_OPENAI_MODEL  # type: ignore
except Exception:
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # safe fallback; can be overridden by env OPENAI_MODEL

# -----------------------------
# Exact AI system prompt (Option 1 with gap-over-time)
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
    "- Begin with the latest year's result for the selected question (metric_label) if present.\n"
    "- Trend (overall): If a previous year exists, compute the signed change in points (latest - previous) as an integer and report it (e.g., \"down 2 points\"). If not, skip.\n"
    "- Gaps (latest year): Compute absolute gaps in points between demographic groups (integer subtraction). Mention only the largest 1–2 gaps and state which group is higher/lower (e.g., \"Women (82) vs Another gender (72): 10-point gap\").\n"
    "- Gap-over-time: For each highlighted gap, compute the gap for each year where BOTH groups have values. State whether the gap has widened, narrowed, or remained stable since the earliest year with both groups (or, if only two adjacent years exist, vs the previous), and report the change in points with the reference year (e.g., \"gap narrowed by 3 points since 2020\"). If fewer than two such years exist, omit this sentence.\n"
    "- Do NOT compute multi-year averages, rates of change, or anything beyond the integer subtractions described above.\n\n"
    "Style & output\n"
    "- Professional, concise, neutral. Narrative style (1–3 short sentences, no lists).\n"
    "- When citing values, keep them as plain integers (optionally append \"%\" if the UI uses percent symbols) and use \"points\" for changes/gaps.\n"
    "- Output VALID JSON with exactly one key: \"narrative\".\n"
)

# =====================================================================================
# Payload builders
# =====================================================================================

def _is_all_label(x: object) -> bool:
    s = str(x or "").strip().lower()
    return s in {"all", "all respondents", "all in category"}

def _to_int_strict(v: object) -> Optional[int]:
    try:
        return int(v)  # already integer-like
    except Exception:
        try:
            f = float(v)
            return int(round(f))
        except Exception:
            return None

def _series_json(df_disp: pd.DataFrame, metric_col: str) -> List[Dict[str, int]]:
    """
    Build a [{year, value}] series from df_disp.
    - If a Demographic column exists, use ONLY rows where Demographic indicates the PS-wide baseline (All respondents).
    - No averaging or derived calculations are performed.
    """
    s = df_disp.copy()
    if "Year" not in s.columns or metric_col not in s.columns:
        return []
    if "Demographic" in s.columns:
        s = s[s["Demographic"].map(_is_all_label, na_action="ignore")]
    s = s.dropna(subset=["Year"])
    s = s.sort_values("Year")
    out: List[Dict[str, int]] = []
    for _, r in s.iterrows():
        y = _to_int_strict(r.get("Year"))
        v = _to_int_strict(r.get(metric_col))
        if y is None or v is None:
            continue
        out.append({"year": y, "value": v})
    return out

def _groups_json_for_year(df_disp: pd.DataFrame, metric_col: str, year: int) -> List[Dict[str, object]]:
    """
    Returns [{label, value}] for a single year if a Demographic column exists in df_disp.
    Uses values as-is (integer cast only). No averaging.
    """
    if "Demographic" not in df_disp.columns:
        return []
    s = df_disp.copy()
    try:
        s = s[pd.to_numeric(s["Year"], errors="coerce").astype("Int64") == int(year)]
    except Exception:
        return []
    s = s.dropna(subset=["Demographic"])
    out: List[Dict[str, object]] = []
    for _, r in s.iterrows():
        label = str(r.get("Demographic"))
        v = _to_int_strict(r.get(metric_col))
        if v is None:
            continue
        out.append({"label": label, "value": v})
    return out

def build_per_q_prompt(
    question_code: str,
    question_text: str,
    df_disp: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    category_in_play: bool,
) -> str:
    """
    Build the per-question JSON payload expected by the model.
    Returns a JSON string (user message content).
    """
    latest_year_val = pd.to_numeric(df_disp.get("Year"), errors="coerce").max()
    latest_year = int(latest_year_val) if pd.notna(latest_year_val) else None

    group_snapshot: List[Dict[str, object]] = []
    if category_in_play and latest_year is not None:
        group_snapshot = _groups_json_for_year(df_disp, metric_col, latest_year)

    series_rows = _series_json(df_disp, metric_col)

    payload = {
        "question_code": str(question_code),
        "question_text": str(question_text),
        "metric_label": str(metric_label),
        "series": series_rows,                 # [{year, value}] integers (baseline only if available)
        "latest_year": latest_year,            # int or None
        "groups_latest_year": group_snapshot,  # [{label, value}] integers
    }
    return json.dumps(payload, ensure_ascii=False)

def build_overall_prompt(
    tiles: List[Dict[str, object]],  # [{"question_code","question_text","latest_year","latest_value_int"}]
) -> str:
    """
    Build a compact payload for the multi-question overview.
    """
    clean: List[Dict[str, object]] = []
    for t in tiles:
        try:
            code = str(t.get("question_code"))
            text = str(t.get("question_text"))
            yr = int(t.get("latest_year"))
            val = int(t.get("latest_value_int"))
            clean.append({"question_code": code, "question_text": text, "latest_year": yr, "latest_value": val})
        except Exception:
            continue
    payload = {"overview": clean}
    return json.dumps(payload, ensure_ascii=False)

# =====================================================================================
# Model caller (JSON mode) and helpers
# =====================================================================================

_JSON_TAIL_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)

def call_openai_json(
    user_payload_json: Optional[str] = None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    *,
    # Back-compat keyword aliases (accepted but optional)
    system: Optional[str] = None,
    model: Optional[str] = None,
    user: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    **kwargs,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls the OpenAI API with a JSON-output instruction.
    Returns (json_text, error_hint). On error, (None, hint).

    Back-compat:
      - accepts legacy keyword aliases: system -> system_prompt, model -> model_name, user -> user_payload_json
      - if payload is missing/empty, returns a clear error hint instead of raising
    """
    payload = user_payload_json or user
    if not payload:
        return None, "empty or missing user_payload_json"

    try:
        import openai  # type: ignore
    except Exception:
        return None, "openai package not available"

    resolved_model = model_name or model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    sys_prompt = system_prompt or system or AI_SYSTEM_PROMPT

    # Try the new-style client first; fall back to legacy if needed
    try:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": payload},
            ],
        )
        return (resp.choices[0].message.content, None)
    except Exception:
        try:
            out = openai.ChatCompletion.create(
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": payload},
                ],
            )
            content = out["choices"][0]["message"]["content"]
            return (content, None)
        except Exception as e2:
            return None, f"openai call failed: {e2}"

def extract_narrative(json_text: Optional[str]) -> Optional[str]:
    """
    Extracts the "narrative" field from a JSON string.
    Robust to leading/trailing noise by trimming to the last JSON object.
    """
    if not json_text:
        return None
    txt = str(json_text).strip()
    m = _JSON_TAIL_RE.search(txt)
    if m:
        txt = m.group(0)
    try:
        obj = json.loads(txt)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    val = obj.get("narrative")
    if val is None:
        return None
    return str(val).strip() or None
