# menu1/ai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time
import re

# Optional pandas import
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

__all__ = [
    "AI_SYSTEM_PROMPT",
    "build_ai_payload_for_question",
    "build_ai_payload_for_overall",
    "call_openai_chat",
    "produce_ai_summary_for_question",
    "produce_ai_overall_synthesis",
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

AI_SYSTEM_PROMPT = (
    "You are an AI survey analyst for the Government of Canada.\n"
    "Your role is to interpret Public Service Employee Survey (PSES) results and\n"
    "produce clear, neutral, and accurate written summaries based solely on the\n"
    "tabular data and metadata provided in the payload.\n\n"
    "Context\n"
    "- The data represent responses from public service employees who participated\n"
    "  in the PSES across the federal public service.\n"
    "- All percentages represent the share of respondents who selected each\n"
    "  response option (e.g., Strongly agree, Somewhat agree, etc.) out of all\n"
    "  valid responses for that subgroup and year.\n"
    "- Unless stated otherwise, “Positive” refers to the combined percentage of\n"
    "  respondents selecting favourable options (e.g., Strongly agree/Somewhat\n"
    "  agree), and “Negative” refers to combined unfavourable options.\n"
    "- \"n\" is the number of respondents in that subgroup for that question/year.\n"
    "- Scope is Public-Service-wide only—never name specific departments unless\n"
    "  they appear in the payload.\n\n"
    "Analysis rules (allowed computations ONLY)\n"
    "- Latest year = the maximum year present in the payload.\n"
    "- Gaps (latest year): compute absolute gaps between demographic groups and\n"
    "  report them in % points (e.g., “Women (82 %) vs Another gender (72 %):\n"
    "  10 % points gap”). Mention only the largest one or two gaps.\n"
    "- Gap-over-time policy:\n"
    "  • When `gap_over_time_policy` is \"earliest_to_latest\", compare the\n"
    "    earliest available year with the latest available year for each group\n"
    "    and, when relevant, the gap between them.\n"
    "  • When `gap_over_time_policy` is \"latest_vs_previous\", compare the\n"
    "    latest year only to the immediately previous year in the data.\n"
    "  • When `gap_over_time_policy` is \"none\", do not compute gap trends.\n"
    "- Trend (per group or overall): when at least two years exist, you may\n"
    "  compute simple differences in percentage points between:\n"
    "  • earliest vs latest year; and/or\n"
    "  • latest vs previous year, if explicitly requested in the payload.\n"
    "- You may say that a result “increased”, “decreased”, or “remained stable”\n"
    "  when those differences are non-zero, small, or zero, but avoid making\n"
    "  claims about statistical significance.\n"
    "- Do not compute multi-year averages or rates of change beyond these integer\n"
    "  subtractions.\n"
    "- If `distribution_only=true`, describe only the latest-year distribution;\n"
    "  never create aggregates.\n\n"
    "Trend rules (per-question; prioritize current year, then context)\n"
    "- When referencing differences between years (e.g. 2024 vs 2022), be precise\n"
    "  about which years are being compared. Use language such as “Since 2019”\n"
    "  only when you are explicitly comparing earliest vs latest years.\n"
    "- Do not describe a change from the peak year as “since 2019” unless 2019 is\n"
    "  actually the earliest comparable year. If the peak occurs in 2020 or 2022,\n"
    "  describe it relative to that peak (e.g., “down from a peak of 78 % in\n"
    "  2020”), rather than making an incorrect “since 2019” statement.\n"
    "- If multiple years are present, you may:\n"
    "  • identify the highest and lowest year values; and\n"
    "  • describe whether the overall pattern appears generally increasing,\n"
    "    decreasing, or mixed.\n"
    "- Keep the description simple and neutral: do not over-interpret small\n"
    "  fluctuations or treat them as meaningful trends.\n"
    "- When describing trends, always anchor them to the exact years involved\n"
    "  (e.g., “between 2019 and 2024” or “between 2022 and 2024”).\n"
    "- If the data show a decline from 2019 to 2020 and then stability, describe\n"
    "  that pattern accurately (e.g., “Results declined between 2019 and 2020\n"
    "  and then remained stable through 2024”).\n"
    "- If multiple years are given but there is no clear pattern, you may say the\n"
    "  pattern is “mixed” or “shows no consistent trend”.\n"
    "- Never invent or assume survey years that are not in the payload.\n"
    "- Do not claim that something is “since the earliest year” unless you are\n"
    "  explicitly comparing earliest and latest values.\n"
    "- If the payload includes `trend_reference_year`, treat it as the baseline\n"
    "  year for describing change when requested.\n\n"
    "Overall synthesis rules (when task = \"overall_synthesis\")\n"
    "- Purpose: provide a high-level synthesis across all selected questions, not a recap of individual question summaries.\n"
    "- Do not mention question codes (e.g., \"Q08\") or restate detailed results for each question; those belong only in the question-level summaries.\n"
    "- Do not repeat the same statistics already provided above. Avoid sentences like \"In 2024, 72 %...\" or \"For this question, 77 %...\".\n"
    "- You may use at most one or two percentages in total in the overall synthesis, and only when they support a cross-question conclusion (e.g., \"both indicators remain above 70 %\").\n"
    "- Focus on what the selected questions have in common:\n"
    "  • whether they are generally increasing, declining, stable, or mixed over time;\n"
    "  • whether they point to similar strengths or recurring concerns;\n"
    "  • whether demographic differences are generally minimal, modest, or notable across the selected questions.\n"
    "- Describe demographic patterns only in broad terms (e.g., \"results are broadly similar between English and French respondents\") unless a clear, recurring gap is visible across several questions. Do not list numeric gaps for each question.\n"
    "- Keep the narrative anchored to the population: at least once, refer explicitly to \"respondents across the public service\" or an equivalent formulation.\n"
    "- If trends differ across questions (some improving, others declining), describe this explicitly as a mixed pattern rather than forcing a single story.\n"
    "- If trend data are limited or inconsistent for many questions, acknowledge the limitation and focus on the most reliable signals.\n"
    "- Tone: concise, executive-level. Aim for 2–3 short paragraphs that could stand alone in a briefing note.\n"
    "- Continue to obey all numeric constraints: only use provided numbers or allowed differences; do not introduce new calculations or statistics.\n"
    "- Your overall synthesis should typically follow this structure:\n"
    "  1) First paragraph: describe the overall direction and level of results across the selected questions (e.g., broadly positive but softening over time, stable, or mixed), framed at the public service level and without citing individual question codes or detailed statistics.\n"
    "  2) Second paragraph: describe broad demographic patterns across the questions (e.g., results are generally similar between groups, or certain groups tend to report lower positive responses), again without walking through each question one by one.\n"
    "  3) Third paragraph (optional): provide one high-level takeaway about what the combined pattern suggests for employee experience across the public service (e.g., areas to watch, recurring concerns), without making policy prescriptions.\n\n"
    "Style & output\n"
    "- Report level values as integers followed by “%” (e.g., “79 %”).\n"
    "- Respect the language and labels from the metadata when naming\n"
    "  demographic groups or questions.\n"
    "- Use neutral wording; avoid strong or emotional language.\n"
    "- Do not make policy recommendations. You may highlight areas that\n"
    "  “may warrant attention” but not prescribe interventions.\n"
    "- Always respect any additional constraints provided in the `controls`\n"
    "  object of the payload.\n\n"
    "Your output must be a single JSON object with these keys:\n"
    "- \"summary\": the main narrative as a plain-text string.\n"
    "- \"notes\": optional clarifications or caveats (or an empty string).\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: SAFE LOOKUP / NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and (x != x)):  # NaN check
            return None
        return float(x)
    except Exception:
        return None


def _coerce_dataframe(df_like: Any) -> Optional["pd.DataFrame"]:
    """Best-effort coercion of a dict/list into a pandas DataFrame."""
    if pd is None:
        return None
    if isinstance(df_like, pd.DataFrame):
        return df_like
    try:
        return pd.DataFrame(df_like)
    except Exception:
        return None


def _normalize_year_col(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rename/ensure a 'Year' column for the AI payload (string)."""
    out = df.copy()
    # Prefer existing "Year" if present
    if "Year" not in out.columns:
        # Try common variants
        for c in list(out.columns):
            lc = str(c).strip().lower()
            if lc in ("year", "surveyr"):
                out = out.rename(columns={c: "Year"})
                break
    if "Year" not in out.columns:
        # Fallback: create from year or SURVEYR
        for c in list(out.columns):
            lc = str(c).strip().lower()
            if lc in ("year", "surveyr"):
                out["Year"] = out[c]
                break
    # Make sure Year is string for JSON
    if "Year" in out.columns:
        out["Year"] = out["Year"].apply(lambda v: str(v) if v is not None else "")
    return out


def _normalize_demo_col(df: "pd.DataFrame", demo_col: str = "Demographic") -> "pd.DataFrame":
    """Ensure a Demographic column if possible."""
    out = df.copy()
    if demo_col in out.columns:
        return out
    # If there is a group label column with another name, adopt it.
    for c in out.columns:
        lc = str(c).strip().lower()
        if "demographic" in lc or lc in ("group", "group_label", "subgroup"):
            out = out.rename(columns={c: demo_col})
            return out
    # Otherwise, if there is a column like "group_value", use that
    for c in out.columns:
        lc = str(c).strip().lower()
        if lc == "group_value":
            out = out.rename(columns={c: demo_col})
            return out
    # If nothing, leave as is; the calling code may treat this as overall.
    return out


def _extract_years_from_df(df: Optional["pd.DataFrame"]) -> List[int]:
    years: List[int] = []
    if df is None or df.empty:
        return years
    # Look for "Year" or anything that looks like survey year
    cols = [c for c in df.columns]
    year_col = None
    for c in cols:
        lc = str(c).strip().lower()
        if lc in ("year", "surveyr"):
            year_col = c
            break
    if year_col is None and "Year" in df.columns:
        year_col = "Year"
    if year_col is None:
        return years
    for v in df[year_col].unique():
        iv = _safe_int(v)
        if iv is not None:
            years.append(iv)
    years = sorted(set(years))
    return years


# ─────────────────────────────────────────────────────────────────────────────
# PAYLOAD BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_ai_payload_for_question(
    question_code: str,
    question_text: str,
    df_latest: Any,
    df_trend: Any,
    demographic_label: Optional[str],
    subgroup_label: Optional[str],
    controls: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the JSON payload for a single-question narrative task.

    Parameters
    ----------
    question_code:
        e.g. "Q08"
    question_text:
        Full English text of the survey question.
    df_latest:
        DataFrame (or coercible) of latest-year results by group/demographic.
        Expected columns (flexible names allowed): Year, Demographic, Positive, Neutral, Negative, n
    df_trend:
        DataFrame (or coercible) of multi-year results for the same question and demo selection.
    demographic_label:
        Name of the demographic category (e.g., "Official language", "Gender") or None for overall.
    subgroup_label:
        Specific subgroup label (e.g., "Women", "French") if focus is on a single subgroup, else None.
    controls:
        Optional dictionary with keys like:
          - "gap_over_time_policy": "earliest_to_latest" | "latest_vs_previous" | "none"
          - "distribution_only": bool
          - "max_years_in_trend": int
    """
    if controls is None:
        controls = {}

    df_latest_co = _coerce_dataframe(df_latest)
    df_trend_co = _coerce_dataframe(df_trend)

    if df_latest_co is not None:
        df_latest_co = _normalize_year_col(_normalize_demo_col(df_latest_co, demo_col="Demographic"))
    if df_trend_co is not None:
        df_trend_co = _normalize_year_col(_normalize_demo_col(df_trend_co, demo_col="Demographic"))

    years_latest = _extract_years_from_df(df_latest_co)
    years_trend = _extract_years_from_df(df_trend_co)

    payload: Dict[str, Any] = {
        "task": "question_summary",
        "question": {
            "code": question_code,
            "text": question_text,
        },
        "demographic_context": {
            "category_label": demographic_label or "All respondents",
            "subgroup_label": subgroup_label,
        },
        "tables": {
            "latest": df_latest_co.to_dict(orient="records") if df_latest_co is not None else [],
            "trend": df_trend_co.to_dict(orient="records") if df_trend_co is not None else [],
        },
        "years": {
            "latest_table_years": years_latest,
            "trend_table_years": years_trend,
        },
        "controls": {
            "gap_over_time_policy": controls.get("gap_over_time_policy", "earliest_to_latest"),
            "distribution_only": bool(controls.get("distribution_only", False)),
            "max_years_in_trend": int(controls.get("max_years_in_trend", 6)),
        },
    }
    return payload


def build_ai_payload_for_overall(
    question_payloads: List[Dict[str, Any]],
    overall_controls: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON payload for an overall synthesis across multiple questions.

    Parameters
    ----------
    question_payloads:
        A list of payloads previously built by `build_ai_payload_for_question`,
        each containing the question meta, tables, and controls used.
    overall_controls:
        Optional dict that can include:
          - "no_repetition": bool — if True, do not restate each question narrative;
          - "cross_question_only": bool — focus strictly on themes across questions;
          - "hr_insights_allowed": bool — allow brief HR lens (e.g., areas to watch);
          - "ban_external_context": bool — do not use any external benchmarks.
    """
    if overall_controls is None:
        overall_controls = {}

    # Light normalization: strip out heavy tables but keep summary-level info
    compact_questions: List[Dict[str, Any]] = []
    for qp in question_payloads:
        qmeta = qp.get("question", {})
        demo_ctx = qp.get("demographic_context", {})
        years = qp.get("years", {})
        controls = qp.get("controls", {})
        latest_tbl = qp.get("tables", {}).get("latest", [])
        trend_tbl = qp.get("tables", {}).get("trend", [])

        # We can include the reduced tables; the system prompt restricts how they are used.
        compact_questions.append(
            {
                "question": {
                    "code": qmeta.get("code"),
                    "text": qmeta.get("text"),
                },
                "demographic_context": demo_ctx,
                "years": years,
                "controls": controls,
                "tables": {
                    "latest": latest_tbl,
                    "trend": trend_tbl,
                },
            }
        )

    payload: Dict[str, Any] = {
        "task": "overall_synthesis",
        "questions": compact_questions,
        "overall_controls": {
            "no_repetition": bool(overall_controls.get("no_repetition", True)),
            "cross_question_only": bool(overall_controls.get("cross_question_only", True)),
            "hr_insights_allowed": bool(overall_controls.get("hr_insights_allowed", True)),
            "ban_external_context": bool(overall_controls.get("ban_external_context", True)),
        },
    }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI CHAT CALL (GENERIC)
# ─────────────────────────────────────────────────────────────────────────────

def call_openai_chat(
    client,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
    temperature: float = 0.3,
    max_tokens: int = 900,
    timeout: Optional[float] = 30.0,
) -> Dict[str, Any]:
    """
    Generic wrapper around the Chat Completions API.

    This function does NOT know anything about PSES specifically; it just sends
    the system prompt and the payload as user content.

    Parameters
    ----------
    client:
        An OpenAI client instance.
    model:
        The chat model ID (e.g., 'gpt-4o-mini').
    system_prompt:
        System-level instructions string.
    payload:
        JSON-serializable dict with task, tables, and controls.
    temperature:
        Sampling temperature for the model.
    max_tokens:
        Maximum tokens for the completion.
    timeout:
        Optional timeout in seconds (unused if client does not support it).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    extra_args: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # If the client supports a timeout parameter, we can pass it;
    # otherwise, this will be ignored.
    try:
        response = client.chat.completions.create(**extra_args)
    except TypeError:
        # Fallback without unknown kwargs
        extra_args.pop("timeout", None)
        response = client.chat.completions.create(**extra_args)

    if not response or not getattr(response, "choices", None):
        raise RuntimeError("No response choices returned by the OpenAI API.")

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response content from the OpenAI API.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: wrap raw text if not valid JSON
        parsed = {
            "summary": content,
            "notes": "",
        }
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPERS FOR THIS APP
# ─────────────────────────────────────────────────────────────────────────────

def produce_ai_summary_for_question(
    client,
    model: str,
    question_code: str,
    question_text: str,
    df_latest: Any,
    df_trend: Any,
    demographic_label: Optional[str],
    subgroup_label: Optional[str],
    controls: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level helper: build payload and call the model for a single question.

    Returns
    -------
    dict with keys:
      - "summary": main narrative
      - "notes": clarifications or caveats (may be empty)
      - "raw_payload": the payload that was sent
    """
    payload = build_ai_payload_for_question(
        question_code=question_code,
        question_text=question_text,
        df_latest=df_latest,
        df_trend=df_trend,
        demographic_label=demographic_label,
        subgroup_label=subgroup_label,
        controls=controls,
    )
    result = call_openai_chat(
        client=client,
        model=model,
        system_prompt=AI_SYSTEM_PROMPT,
        payload=payload,
    )
    return {
        "summary": result.get("summary", ""),
        "notes": result.get("notes", ""),
        "raw_payload": payload,
    }


def produce_ai_overall_synthesis(
    client,
    model: str,
    question_results: List[Dict[str, Any]],
    overall_controls: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level helper: build overall synthesis payload from many question payloads
    (or from their associated metadata) and call the model.

    Parameters
    ----------
    question_results:
        List of dicts, each expected to contain (at least):
          - "raw_payload": the payload originally passed for that question, OR
          - an equivalent structure built by the caller.
    overall_controls:
        Optional overall_controls dict for build_ai_payload_for_overall().

    Returns
    -------
    dict with keys:
      - "summary": overall narrative
      - "notes": clarifications or caveats (may be empty)
      - "raw_payload": the payload that was sent
    """
    # Extract per-question payloads
    question_payloads: List[Dict[str, Any]] = []
    for qr in question_results:
        rp = qr.get("raw_payload") or qr.get("payload")
        if isinstance(rp, dict):
            question_payloads.append(rp)

    payload = build_ai_payload_for_overall(
        question_payloads=question_payloads,
        overall_controls=overall_controls,
    )
    result = call_openai_chat(
        client=client,
        model=model,
        system_prompt=AI_SYSTEM_PROMPT,
        payload=payload,
    )
    return {
        "summary": result.get("summary", ""),
        "notes": result.get("notes", ""),
        "raw_payload": payload,
    }
