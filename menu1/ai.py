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
Use plain, professional language suited to HR reporting in the Government of Canada.
Write concise sentences and avoid speculation, advocacy, or policy advice.

Core rules (must follow):
• Do not paraphrase or compress the meaning of scale labels unless explicitly allowed in the user or system instructions.
• Use the provided meaning labels exactly as supplied by metadata for each question when reporting percentages.
• For D57_a and D57_b, report the distribution across Answer1..Answer6 using only the values provided (exclude any 9999).
• For polarity:
  - POS → narrate the 'Positive' metric; if missing/9999, fall back: Positive → Agree → Answer1 → Negative
  - NEG → narrate the 'Negative' metric; if missing/9999, fall back: Negative → Agree → Answer1 → Positive
  - NEU → narrate the 'Agree' metric;   if missing/9999, fall back: Agree → Answer1 → Positive → Negative
• Do not treat 9999 as data; it means “not applicable” and must be excluded from reporting and validation.
• Trend analysis: determine whether each question’s series across ALL available years is rising, declining, steady, mixed, or no trend.
• Demographic gaps (latest year): when subgroup tables exist, identify the largest latest-year gap and, where possible from tables, how that gap changed relative to prior years.
• Overall synthesis: summarize across the already-produced per-question results; do not recompute or invent any numbers. Reuse the same metrics/labels used in the per-question outputs.
• Validation: every number in the narrative must match a value shown in the tables (or a simple difference/change derived from those numbers).

Tone & format:
• Be succinct, neutral, and data-first. Avoid tautologies and repetition.
• Write in full sentences suitable for an executive readout.
• If data is missing/suppressed, acknowledge briefly without inferring.
• Use en dashes (–) in question headers as provided by the payload; do not alter the question text.

ADDENDUM — HOW TO USE THE USER PAYLOAD FIELDS (REQUIRED)
• The user message is ALWAYS a JSON object for either a per-question task or an overall synthesis task.

PER-QUESTION PAYLOAD KEYS
  - task: "per_question_summary"
  - question_code: string (e.g., "Q44a")
  - question_text: string (verbatim prompt)
  - polarity: "POS" | "NEG" | "NEU"
  - reporting_metric:  # used unless distribution_only=true (see exception)
      - column: the exact column to narrate (e.g., "Positive", "Negative", "AGREE", "Answer1")
      - labels_full: a single string with exact labels in order, e.g. "(Strongly agree/Agree)"
      - labels_compressed: a single string you may use if permitted, preserving meaning/order, e.g. "(Strongly/Somewhat agree)"
  - years: list of {year:int, value:float} used for trend
  - latest_year: int
  - latest_value: float (for the chosen reporting metric)
  - demographics: optional { "category": str, "subgroups": [ { "name": str, "year": int, "value": float }, ... ] }
  - distribution_only: bool (true for D57_a/b)
  - notes: optional extra instructions

OVERALL PAYLOAD KEYS
  - task: "overall_synthesis"
  - items: list of objects, each with:
      - question_code, question_text, polarity
      - reporting_metric.column
      - labels_full, labels_compressed
      - latest_year, latest_value
      - years: list for trend
      - demographics: optional latest-year subgroup values and prior-year context if available

Failure handling:
• If the payload lacks necessary numbers, state that the specific item cannot be summarized due to missing data. Do not invent.
• If distribution_only is true, do NOT refer to Positive/Negative/Agree; only narrate the provided Answer1..Answer6 distribution.

------------------------------------------------------------------------------
ADDENDUM — FORMATTING RULES FOR SCALE LABELS (REQUIRED)
• When you write a percentage, place the parentheses containing the scale labels
  IMMEDIATELY AFTER the percentage (e.g., "54% (To a small/moderate/large extent/very large extent)").
• You MAY use the provided labels_compressed when it fully preserves the meaning and order
  of the labels; otherwise, use labels_full exactly as provided. Do not invent new wording.
• Apply the SAME rule in both per-question summaries and the overall synthesis:
  for any percentage tied to a specific question, append that question’s labels (compressed if provided, else full)
  right after the %, not at the end of the sentence.
• Do not shorten/omit categories beyond the provided compressed form. Do not swap, merge, or reorder labels.
------------------------------------------------------------------------------
""",
)

# Backward-compat alias for callers that import AI_SYSTEM_PROMPT
AI_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

# --------------------------------------------------------------------------------------
# Public API: build payloads and call the LLM (unchanged interface)
# --------------------------------------------------------------------------------------

def _coalesce_label_strings(meaning_labels: List[str]) -> Tuple[str, str]:
    """
    Return (labels_full, labels_compressed) from a list of labels, preserving order.
    Compression rule: detect a common prefix and suffix; keep full text on the first
    and last item; elide prefix/suffix in middle items when safe. Fall back to full.
    Examples:
      ["To a small extent","To a moderate extent","To a large extent","To a very large extent"]
        -> full="(To a small extent/To a moderate extent/To a large extent/To a very large extent)"
           compressed="(To a small/moderate/large extent/very large extent)"
    """
    # full form
    full = "(" + "/".join(meaning_labels) + ")"
    # attempt compression (very conservative)
    try:
        # Find longest common prefix and suffix across all labels
        from os.path import commonprefix
        prefix = commonprefix(meaning_labels)
        # common suffix: operate on reversed strings
        rev = [s[::-1] for s in meaning_labels]
        suffix_rev = commonprefix(rev)
        suffix = suffix_rev[::-1]
        # Build compressed parts
        parts: List[str] = []
        for i, lab in enumerate(meaning_labels):
            core = lab
            if prefix and lab.startswith(prefix):
                core = core[len(prefix):]
            if suffix and core.endswith(suffix):
                core = core[: -len(suffix)]
            # Keep first and last with prefix/suffix if core would be empty
            if not core.strip():
                core = lab
            # Re-attach suffix for readability on every item
            # Keep prefix only on the first item if it adds clarity
            if i == 0 and lab.startswith(prefix):
                core = prefix + core
            if suffix and lab.endswith(suffix) and not core.endswith(suffix):
                core = core + suffix
            parts.append(core)
        compressed = "(" + "/".join(parts) + ")"
        # sanity: ensure no item is empty and order preserved
        if all(p.strip() for p in parts) and len(parts) == len(meaning_labels):
            return full, compressed
        return full, full
    except Exception:
        return full, full


def build_per_question_user_payload(
    *,
    question_code: str,
    question_text: str,
    polarity: str,
    reporting_column: str,
    meaning_labels: List[str],
    years: List[Dict[str, Any]],
    latest_year: int,
    latest_value: Optional[float],
    demographics: Optional[Dict[str, Any]],
    distribution_only: bool,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    labels_full, labels_compressed = _coalesce_label_strings(meaning_labels or [])
    payload: Dict[str, Any] = {
        "task": "per_question_summary",
        "question_code": question_code,
        "question_text": question_text,
        "polarity": polarity,
        "reporting_metric": {
            "column": reporting_column,
            "labels_full": labels_full,
            "labels_compressed": labels_compressed,
        },
        "years": years,
        "latest_year": latest_year,
        "latest_value": latest_value,
        "demographics": demographics,
        "distribution_only": bool(distribution_only),
    }
    if notes:
        payload["notes"] = notes
    return payload


def build_overall_user_payload(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Items are expected to already include labels_full/labels_compressed from per-question phase
    return {"task": "overall_synthesis", "items": items}


# --------------------------------------------------------------------------------------
# LLM call wrapper (unchanged behavior)
# --------------------------------------------------------------------------------------

def call_llm_openai_chat(
    client,
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 900,
    retries: int = 2,
    timeout_s: float = 20.0,
) -> Tuple[str, Optional[str]]:
    """
    Returns (content_str, error_str). On success, content_str is the assistant message
    content (string). If the assistant returns JSON, we do NOT parse it here—callers
    may parse upstream. On failure, content_str contains a fallback JSON with a message
    and error_str contains a short reason.
    """
    last_err: Optional[BaseException] = None
    for _ in range(retries + 1):
        try:
            rsp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": BASE_SYSTEM_PROMPT if system_prompt is None else system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                max_tokens=max_tokens,
            )
            content = rsp.choices[0].message.content
            # If it's not valid JSON, leave it as text; callers decide how to render/validate.
            try:
                _ = json.loads(content or "{}")
            except Exception:
                content = json.dumps({"narrative": (content or "").strip()}, ensure_ascii=False)
            return content, None
        except Exception as e:
            last_err = e
            time.sleep(0.4)

    fb = json.dumps({"narrative": "The AI service is temporarily unavailable."}, ensure_ascii=False)
    return fb, f"LLM error: {type(last_err).__name__}"
