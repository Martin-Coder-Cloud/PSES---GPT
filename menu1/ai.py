# menu1/ai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import math

import pandas as pd
import streamlit as st


# ------------------------ System prompt (preserve if provided) ------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are an analyst specializing in the Public Service Employee Survey (PSES). "
    "Write concise, factual, HR-relevant summaries strictly grounded in the structured data I provide. "
    "Interpret the percentages that are already computed (no fresh calculations other than stating year-over-year or "
    "across-years direction of change and subgroup gaps, which are described in the data). "
    "When multiple years are available, evaluate the pattern across all years: rising, declining, steady, mixed, or no trend. "
    "For subgroup breakdowns (e.g., official language, gender, tenure), report the latest-year gaps and briefly note how "
    "those gaps changed across prior years if the tables allow it. "
    "Avoid policy advice; keep an HR management lens appropriate to the federal public service context."
)

# If the app preloads a system prompt, use it as-is (no changes).
AI_SYSTEM_PROMPT: str = st.session_state.get("AI_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)


# ------------------------ Utilities ------------------------

def _is_number(x: Any) -> bool:
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False

def _clean_value(v: Any) -> Any:
    # Normalize 9999 -> None and round floats reasonably
    try:
        if v == 9999:
            return None
        if isinstance(v, float):
            if math.isnan(v):
                return None
            # keep one decimal for % values; leave integers alone
            if abs(v - round(v)) < 1e-9:
                return int(round(v))
            return round(v, 1)
        return v
    except Exception:
        return v

def _df_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    # Ensure JSON-safe rows, with 9999 stripped
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for c, v in row.items():
            rec[str(c)] = _clean_value(v)
        out.append(rec)
    return out


# ------------------------ Prompt Builders (surgical additions only) ------------------------

def build_per_q_prompt(
    *,
    question_code: str,
    question_text: str,
    df_disp: pd.DataFrame,
    metric_col: Optional[str],
    metric_label: str,
    category_in_play: bool,
    meaning_labels: Optional[List[str]] = None,   # <-- ADDED
    reporting_field: Optional[str] = None,        # unchanged (may be None)
    distribution_only: bool = False               # unchanged (D57_* support)
) -> str:
    """
    Build the per-question prompt payload. The only change is that we now include
    `meaning_labels` and add one tiny rule about appending them as a parenthetical.
    """
    payload = {
        "question_code": question_code,
        "question_text": question_text,
        "metric_column": metric_col,
        "metric_label": metric_label,               # e.g., "% negative", "% positive", "% agree", etc.
        "reporting_field": reporting_field,         # e.g., "NEGATIVE", "POSITIVE", "AGREE", or None for D57_*
        "distribution_only": bool(distribution_only),
        "category_in_play": bool(category_in_play), # whether demographic breakdowns exist in df_disp
        "meaning_labels": list(meaning_labels or []),  # <-- ADDED (labels for the aggregated metric)
        "data": _df_to_records(df_disp),            # the table you see in the UI, already cleaned
    }

    # --- Minimal additive instruction (do not remove or alter existing behaviour) ---
    extra_rule = (
        "STYLE & PARENTHETICAL: After each percentage you report, append a parenthetical using the provided "
        "`meaning_labels`, joined with '/', e.g. (To a small extent/To a moderate extent/To a large extent/"
        "To a very large extent). If `meaning_labels` is empty or not provided, omit the parenthetical. "
        "Do not substitute the metric name (e.g., 'Negative') as the parenthetical—use only the labels. "
        "When `distribution_only` is true (e.g., D57_*), do not aggregate; describe the distribution across the "
        "visible response options (ignore any null/9999 rows). "
        "When a demographic category is present, in the latest year: identify the largest subgroup gap and state "
        "how that gap changed across prior years if apparent. "
        "For years, comment on the trend across all available years (rising, declining, steady, mixed, or insufficient data). "
        "Keep the federal public service HR context in mind."
    )

    # Your app expects the 'user' content to be a JSON object with instructions.
    # We return a single string that contains a JSON blob followed by a short instruction block.
    return json.dumps({"payload": payload, "instructions": extra_rule}, ensure_ascii=False)


def build_overall_prompt(
    *,
    tab_labels: List[str],
    pivot_df: pd.DataFrame,
    q_to_metric: Dict[str, str],
    code_to_text: Dict[str, str],
    q_to_meaning_labels: Optional[Dict[str, List[str]]] = None,  # <-- ADDED
    q_distribution_only: Optional[Dict[str, bool]] = None        # unchanged
) -> str:
    """
    Build the overall synthesis prompt. Added `q_to_meaning_labels` so overall can
    echo the same parenthetical as per-question summaries.
    """
    payload = {
        "selected_questions": tab_labels,
        "pivot": _df_to_records(pivot_df),        # summary pivot (already polarity-aware)
        "metric_by_question": q_to_metric,        # e.g., {"Q44a": "% negative", ...}
        "question_texts": code_to_text,           # {"Q44a": "...", ...}
        "meaning_labels_by_question": q_to_meaning_labels or {},  # <-- ADDED
        "distribution_only_flags": q_distribution_only or {},
    }

    extra_rule = (
        "OVERALL TASK: Synthesize cross-question patterns using only the provided pivot and metadata. "
        "If you cite any per-question percentage, append a parenthetical using "
        "meaning_labels_by_question[question_code] joined with '/'. "
        "If none provided for that question, omit the parenthetical. "
        "Highlight which items are highest/lowest in the latest year, note any coherent trends across ALL years, "
        "and summarize subgroup disparities in the latest year (and how those disparities changed across years) "
        "when subgroup rows are present in the pivot. Keep it concise and relevant to HR management in the federal "
        "public service context. Avoid inventing numbers."
    )

    return json.dumps({"payload": payload, "instructions": extra_rule}, ensure_ascii=False)


# ------------------------ LLM Call Wrapper (kept safe & flexible) ------------------------

def call_openai_json(
    *,
    system: str,
    user: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Flexible wrapper used by Menu 1.
    1) If the host app injects a callable in session_state["call_openai_json"], delegate to it.
    2) Else, attempt a direct OpenAI call (response_format=json). Model can be provided via
       session_state["openai_model"]; default to 'gpt-4o-mini' if unset.
    3) On any error or if OpenAI SDK is unavailable, return ("{}", None) gracefully.
    """
    # Delegate to an app-provided function if present (preserves your existing wiring).
    f = st.session_state.get("call_openai_json")
    if callable(f):
        try:
            return f(system=system, user=user)  # expected to return (content, hint)
        except Exception:
            pass

    # Try a direct OpenAI call if the SDK and key are available.
    try:
        import openai  # type: ignore
        model = st.session_state.get("openai_model", "gpt-4o-mini")
        client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai

        # Support either new client or legacy
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or AI_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = resp.choices[0].message.content if resp and resp.choices else "{}"
        else:
            # Very old sdk fallback
            resp = client.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or AI_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else "{}"

        # Try to ensure valid JSON string
        try:
            json.loads(content or "{}")
        except Exception:
            content = "{}"

        return content, None
    except Exception:
        return "{}", None
