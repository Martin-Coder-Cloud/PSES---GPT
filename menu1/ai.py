# app/menu1/ai.py
"""
AI prompt and calling utilities for Menu 1.

PUBLIC API (unchanged names):
- AI_SYSTEM_PROMPT
- build_per_q_prompt(...): builds the JSON "user" payload for a single question
- build_overall_prompt(...): builds the JSON "user" payload for the multi-question summary
- call_openai_json(...): robust caller that returns (json_text, error_hint)
- extract_narrative(...): safe JSON parse helper returning the text narrative or None
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import os
import re
import pandas as pd

from .constants import DEFAULT_OPENAI_MODEL

# =====================================================================================
# System prompt (Option 1 — approved)
# =====================================================================================
AI_SYSTEM_PROMPT: str = (
    "You are preparing insights for the Government of Canada's Public Service Employee Survey (PSES).\n\n"
    "Context\n"
    "- The PSES informs improvements to people management in the federal public service.\n"
    "- Results help identify strengths and concerns in areas such as engagement, inclusion, and well-being.\n"
    "- Statistics Canada administers the survey with TBS. Confidentiality is guaranteed under the Statistics Act (grouped reporting; <10 suppressed).\n\n"
    "Data-use rules (hard constraints)\n"
    "- Treat the provided JSON/table as the single source of truth.\n"
    "- Use ONLY integers that appear in the payload/table OR integers that are EXACT point differences between two payload integers.\n"
    "- Do NOT invent numbers, averages, weighted figures, percentages, rescaled values, or decimals. Do NOT round.\n"
    "- If a value needed for a comparison is missing, omit that comparison rather than inferring.\n"
    "- Refer only to years, groups, and labels present in the payload.\n\n"
    "Analysis rules (allowed computations ONLY)\n"
    "- Latest year = the maximum year present in the payload.\n"
    "- Trend: If a previous year exists, compute the signed change in points (latest - previous) as an integer and report it (e.g., \"down 2 points\"). If not, skip.\n"
    "- Gaps: In the latest year, compute absolute gaps in points between demographic groups (integer subtraction). Mention only the largest 1–2 gaps.\n"
    "- Direction: For gaps, state which group is higher/lower (e.g., \"Women (82) vs Another gender (72): 10-point gap\").\n"
    "- Do NOT compute multi-year averages, rates of change, or anything beyond simple integer subtraction.\n\n"
    "Style & output\n"
    "- Professional, concise, neutral. 1–3 short sentences.\n"
    "- When citing values, keep them as plain integers (optionally append \"%\" if the UI uses percent symbols) and use \"points\" for changes/gaps.\n"
    "- Output VALID JSON with exactly one key: \"narrative\".\n"
)

# =====================================================================================
# Helpers to build model payloads (public names preserved)
# =====================================================================================

def _series_json(df_disp: pd.DataFrame, metric_col: str) -> List[Dict[str, int]]:
    """
    Build a [{year, value}] series from a display dataframe.
    NOTE: Leaves any intra-year shaping exactly as df_disp provides it; converts to ints.
    """
    rows: List[Dict[str, int]] = []
    s = df_disp.copy()
    s = s.dropna(subset=["Year"])
    s = s.sort_values("Year")
    for _, r in s.iterrows():
        try:
            y = int(r["Year"])
        except Exception:
            continue
        v_raw = r.get(metric_col)
        try:
            v = int(v_raw)
        except Exception:
            try:
                v = int(round(float(v_raw)))
            except Exception:
                continue
        rows.append({"year": y, "value": v})
    return rows

def _groups_json_for_year(df_disp: pd.DataFrame, metric_col: str, year: int) -> List[Dict[str, object]]:
    """
    Returns [{label, value}] for a single year if a Demographic column exists in df_disp.
    """
    if "Demographic" not in df_disp.columns:
        return []
    s = df_disp[df_disp["Year"].astype("Int64") == int(year)].copy()
    s = s.dropna(subset=["Demographic"])
    out: List[Dict[str, object]] = []
    for _, r in s.iterrows():
        label = str(r["Demographic"])
        v_raw = r.get(metric_col)
        try:
            v = int(v_raw)
        except Exception:
            try:
                v = int(round(float(v_raw)))
            except Exception:
                continue
        out.append({"label": label, "value": v})
    return out

def build_per_q_prompt(
    question_code: str,
    question_text: str,
    df_disp: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    category_in_play: bool
) -> str:
    """
    Build the per-question JSON payload expected by the model.
    Returns a JSON string (user message content).
    """
    latest_year = pd.to_numeric(df_disp["Year"], errors="coerce").max()

    # Snapshot of groups at the latest year (if demographics in play)
    group_snapshot: List[Dict[str, int]] = []
    if category_in_play and "Demographic" in df_disp.columns and pd.notna(latest_year):
        try:
            latest_int = int(latest_year)
        except Exception:
            latest_int = None
        if latest_int is not None:
            group_snapshot = _groups_json_for_year(df_disp, metric_col, latest_int)

    series_rows = _series_json(df_disp, metric_col)

    payload = {
        "question_code": str(question_code),
        "question_text": str(question_text),
        "metric_label": str(metric_label),
        "series": series_rows,                  # [{year, value}] integers
        "latest_year": int(latest_year) if pd.notna(latest_year) else None,
        "groups_latest_year": group_snapshot,   # [{label, value}] integers
    }
    return json.dumps(payload, ensure_ascii=False)

def build_overall_prompt(
    tiles: List[Dict[str, object]],  # list of {"question_code","question_text","latest_year","latest_value_int"}
) -> str:
    """
    Build a compact payload for the multi-question overview.
    Returns a JSON string (user message content).
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
# Model caller (JSON mode) and helpers — public name preserved
# =====================================================================================

_JSON_TAIL_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)

def call_openai_json(
    user_payload_json: Optional[str] = None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    *,
    # Back-compat aliases accepted (safe no-ops if unused)
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    **kwargs,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls the OpenAI API with a JSON-output instruction.
    Returns (json_text, error_hint). On error, (None, hint).

    Back-compat:
      - accepts legacy keyword aliases: system -> system_prompt, model -> model_name
      - if payload is missing/empty, returns a clear error hint instead of raising
    """
    if not user_payload_json:
        return None, "empty or missing user_payload_json"

    try:
        import openai  # type: ignore
    except Exception:
        return None, "openai package not available"

    resolved_model = model_name or model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    sys_prompt = system_prompt or system or AI_SYSTEM_PROMPT

    try:
        # New-style client (if available)
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_payload_json},
            ],
        )
        return (resp.choices[0].message.content, None)
    except Exception:
        # Fallback to legacy API if the above path fails
        try:
            out = openai.ChatCompletion.create(
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_payload_json},
                ],
            )
            content = out["choices"][0]["message"]["content"]
            return (content, None)
        except Exception as e2:
            return None, f"openai call failed: {e2}"

def extract_narrative(json_text: Optional[str]) -> Optional[str]:
    """
    Extracts the "narrative" field from a JSON string.
    Robust to trailing tokens before the final JSON object.
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
