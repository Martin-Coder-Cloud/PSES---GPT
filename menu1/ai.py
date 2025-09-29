# app/menu1/ai.py
"""
AI prompt and calling utilities for Menu 1.

- AI_SYSTEM_PROMPT: a concise but explicit system instruction
- build_per_q_prompt(...): builds the JSON "user" payload for a single question
- build_overall_prompt(...): builds the JSON "user" payload for the multi-question summary
- call_openai_json(...): robust caller that returns (json_text, error_hint)
- extract_narrative(...): safe JSON parse helper returning the text narrative or None
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import time
import os

import pandas as pd
import streamlit as st

from .constants import DEFAULT_OPENAI_MODEL

# =====================================================================================
# System prompt (refined)
# =====================================================================================
AI_SYSTEM_PROMPT: str = (
    "You draft insights for the Government of Canada’s Public Service Employee Survey (PSES).\n\n"
    "Scope & data boundaries\n"
    "- Use ONLY the numbers in the payload provided by the user message. Do not infer missing values.\n"
    "- Focus on Public Service–wide results. Ignore departments unless explicitly present.\n"
    "- If trend or gap comparisons aren’t possible (e.g., a single year or single group), say so briefly.\n\n"
    "How to analyze\n"
    "- Start from the LATEST year present (typically 2024) and state its value with the correct metric label.\n"
    "- Trends: compare the latest year with the EARLIEST available year. Classify the change as:\n"
    "  • stable ≤ 1 point; • slight > 1–2 points; • notable > 2 points.\n"
    "- Demographic gaps (latest year): highlight the biggest gap(s). Classify gap size as:\n"
    "  • minimal ≤ 2 points; • notable > 2–5 points; • important > 5 points.\n"
    "- If multiple groups exist across years, mention whether gaps look wider, narrower, or similar versus earlier.\n"
    "- When data are sparse, use cautious wording like “based on available years”.\n\n"
    "Style & output\n"
    "- Professional, concise, neutral; plain language suitable for a public service audience.\n"
    "- Per-question: ONE short paragraph (no lists).\n"
    "- Overall (multi-question): 1–2 short paragraphs at most.\n"
    "- Output VALID JSON with exactly one key: \"narrative\". No extra keys, no markdown.\n"
)

# =====================================================================================
# Prompt builders
# =====================================================================================

def _series_json(df_disp: pd.DataFrame, metric_col: str) -> List[Dict[str, float]]:
    """
    Build a [{year, value}] series from a display dataframe.
    If a Demographic column exists, averages by year (public-service level view).
    """
    rows: List[Dict[str, float]] = []
    s = df_disp.copy()
    if "Demographic" in s.columns:
        s = s.groupby("Year", as_index=False)[metric_col].mean(numeric_only=True)
        s = s.rename(columns={metric_col: "Metric"})
    else:
        s = s[["Year", metric_col]].rename(columns={metric_col: "Metric"})
    s = s.dropna(subset=["Year"]).sort_values("Year")
    for _, r in s.iterrows():
        try:
            y = int(r["Year"])
        except Exception:
            y = r["Year"]
        val = r["Metric"]
        rows.append({"year": int(y), "value": float(val) if pd.notna(val) else None})
    return rows


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
    Build the per-question JSON payload expected by the model.
    Returns a JSON string.
    """
    latest_year = pd.to_numeric(df_disp["Year"], errors="coerce").max()

    # Snapshot of groups at the latest year (if demographics in play)
    group_snapshot: List[Dict[str, float]] = []
    if category_in_play and "Demographic" in df_disp.columns and pd.notna(latest_year):
        g = df_disp[pd.to_numeric(df_disp["Year"], errors="coerce") == latest_year][["Demographic", metric_col]].dropna()
        g = g.sort_values(metric_col, ascending=False)
        if not g.empty:
            # include top and bottom groups to frame the gap
            top = g.iloc[0].to_dict()
            bot = g.iloc[-1].to_dict()
            group_snapshot = [
                {"demographic": str(top["Demographic"]), "value": float(top[metric_col])},
                {"demographic": str(bot["Demographic"]), "value": float(bot[metric_col])},
            ]

    payload = {
        "question_code": question_code,
        "question_text": question_text,
        "metric_label": metric_label,
        "series_by_year": _series_json(df_disp, metric_col),
        "latest_year_group_snapshot": group_snapshot,
        # Soft guidance to keep per-question output to ONE short paragraph:
        "output_style_hint": "Return exactly one short paragraph in 'narrative'.",
    }
    return json.dumps(payload, ensure_ascii=False)


def build_overall_prompt(
    *,
    tab_labels: List[str],                     # list of question codes (e.g., ["Q01","Q12"])
    pivot_df: pd.DataFrame,                    # index=QuestionLabel (code only or code×demo), columns=years
    q_to_metric: Dict[str, str]                # mapping: question_code -> metric_label (e.g., "% positive")
) -> str:
    """
    Build the overall summary payload for multiple questions.
    Returns a JSON string.
    """
    items: List[Dict] = []

    # Ensure we only iterate questions that actually exist in the pivot
    pivot_index = pivot_df.index.tolist()
    # pivot index might be tuples (Question, Demographic) if that mode is used; we only expect question codes here
    for q in tab_labels:
        # try exact match; if pivot index is tuples, prefer rows whose first element equals q
        rows = []
        if all(isinstance(ix, tuple) for ix in pivot_index):
            # aggregate across demographics for the overall story
            sub = pivot_df.loc[[ix for ix in pivot_index if ix and ix[0] == q]]
            if not sub.empty:
                # mean across demo rows per year
                vals = sub.groupby(level=0).mean(numeric_only=True)
                series_row = vals
            else:
                series_row = None
        else:
            if q in pivot_df.index:
                series_row = pivot_df.loc[[q]]
            else:
                series_row = None

        values_by_year: Dict[int, float] = {}
        if series_row is not None and not series_row.empty:
            row = series_row.mean(numeric_only=True) if len(series_row) > 1 else series_row.iloc[0]
            for y in series_row.columns:
                v = row.get(y)
                if pd.notna(v):
                    try:
                        values_by_year[int(y)] = float(v)
                    except Exception:
                        continue

        items.append({
            "question_code": q,
            "metric_label": q_to_metric.get(q, "% positive"),
            "values_by_year": values_by_year
        })

    payload = {
        "questions": items,
        "notes": "Synthesize overall patterns across questions using each question's metric_label. Keep it brief (1–2 short paragraphs)."
    }
    return json.dumps(payload, ensure_ascii=False)

# =====================================================================================
# Caller + helpers
# =====================================================================================

def _resolve_model(default_model: Optional[str] = None) -> str:
    """
    Decide which model to use. Prefers st.secrets, then env var, then constant.
    """
    return (
        st.secrets.get("OPENAI_MODEL", "") or
        os.environ.get("OPENAI_MODEL", "") or
        (default_model or DEFAULT_OPENAI_MODEL)
    )


def call_openai_json(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 2
) -> Tuple[str, Optional[str]]:
    """
    Call OpenAI chat with a JSON response expectation.
    Returns: (json_text, error_hint). Never raises.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "", "no_api_key"

    model = model or _resolve_model()

    # Prefer new SDK if available
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        hint = "unknown_error"
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                content = resp.choices[0].message.content or ""
                return content, None
            except Exception as e:
                hint = f"openai_err_{attempt+1}: {type(e).__name__}"
                time.sleep(0.8 * (attempt + 1))
        return "", hint
    except Exception:
        # Legacy package fallback
        try:
            import openai  # type: ignore
            openai.api_key = api_key
            hint = "unknown_error"
            for attempt in range(max_retries + 1):
                try:
                    resp = openai.ChatCompletion.create(
                        model=model,
                        temperature=temperature,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    )
                    content = resp["choices"][0]["message"]["content"] or ""
                    return content, None
                except Exception as e:
                    hint = f"openai_legacy_err_{attempt+1}: {type(e).__name__}"
                    time.sleep(0.8 * (attempt + 1))
            return "", hint
        except Exception:
            return "", "no_openai_sdk"


def extract_narrative(json_text: str) -> Optional[str]:
    """
    Parse the model's JSON and return the 'narrative' string if present.
    """
    if not json_text:
        return None
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict) and isinstance(obj.get("narrative"), str) and obj["narrative"].strip():
            return obj["narrative"].strip()
    except Exception:
        return None
    return None
