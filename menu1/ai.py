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
    "call_openai_json",
    "build_per_q_prompt",
    "build_overall_prompt",
]

# --------------------------------------------------------------------------------------
# SYSTEM PROMPT (updated: zero-gap guard added; all prior approved rules preserved)
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
"- Gap-over-time policy:\n"
"  • When `gap_over_time_policy` is \"earliest_to_latest\", compute the gap in the earliest comparable year and the latest year; report whether the gap widened, narrowed, or remained stable, with the absolute change in % points.\n"
"  • Only fall back to latest vs previous year when there are exactly two comparable years.\n"
"  • You may describe direction using signs of adjacent gap deltas (e.g., mostly widening) without creating new numbers beyond allowed differences.\n"
"- Do NOT compute multi-year averages or rates of change beyond these integer subtractions.\n"
"- If `distribution_only=true`, describe only the latest-year distribution; never create aggregates.\n\n"
"Trend rules (per-question; prioritize current year, then context)\n"
"- Start with the latest year vs the previous year only when at least two years are available: report the YoY change in % points "
"(e.g., “2024: 54 %, down 2 % points vs 2023”).\n"
"- Then place this YoY in the context of all available years:\n"
"  • Compute YoY deltas for each adjacent pair and the earliest→latest net change.\n"
"  • Classify the overall pattern using YoY signs: increasing, declining, stable, or mixed.\n"
"  • Explain how the latest YoY relates to the long-term pattern (continuation, reversal, bump, or stabilization).\n"
"- Small movements (±1 % point) → “little change.”  If only one year → “No trend (single year).”\n"
"- Use only permitted math (differences in % points; no averages).\n\n"
"Trend roll-up (overall synthesis)\n"
"- Lead with the latest-year picture — which areas rose or fell vs previous year (only where multi-year data exist).\n"
"- Summarize long-term patterns across questions: how many are increasing, declining, stable, or mixed.\n"
"- Briefly indicate whether current movements continue prior patterns or appear as reversals or bumps.\n"
"- Use numbers sparingly — only YoY and earliest→latest deltas. No new computations.\n\n"
"Overall synthesis rules (when task = \"overall_synthesis\")\n"
"- Purpose: summarize themes and implications across all selected questions — not to repeat each narrative.\n"
"- Respect `overall_controls`:\n"
"  • If `no_repetition=true`, do NOT restate per-question values or mini-summaries. Synthesize only what is common across the selected questions.\n"
"  • If `cross_question_only=true`, keep the synthesis strictly inside the selected questions (and their demographic/metric scope). Do NOT introduce topics, entities, or measures that are not part of the selection.\n"
"  • If `hr_insights_allowed=true`, you may include brief HR-oriented implications **only** when they follow directly from permitted computations (YoY deltas, earliest→latest deltas, gaps and gap-over-time). Use cautious, evidence-linked language such as “remains a relative strength”, “suggests an area to watch”, or “results are broadly similar across groups”.\n"
"  • If `ban_external_context=true`, do NOT use external benchmarks, causes, policies, or departments unless they are explicitly part of the payload.\n"
"- Identify common patterns (strengths, recurring concerns, areas improving or declining) using only allowed computations.\n"
"- Highlight areas of strong results (high positive %, clear improvements) and areas requiring attention (lower scores, declines, or large demographic gaps) **only when such information is derivable from the provided tables**.\n"
"- When useful, group related questions under broader ideas such as career development, work–life balance, inclusion, or leadership — but only if those ideas are evident from the selected questions.\n"
"- If signals conflict across questions (some up, some down), describe it as a mixed pattern rather than over-generalizing.\n"
"- If trend data are insufficient for many items, acknowledge the limitation and avoid trend claims.\n"
"- Tone: concise, professional, suitable for briefing a director.\n"
"- Continue to obey all numeric constraints: only use provided numbers or allowable differences.\n\n"
"Style & output\n"
"- Report level values as integers followed by “%” (e.g., “79 %”).\n"
"- Reserve “% points” strictly for differences or gaps (e.g., “down 2 % points”, “a 10 % points gap”).\n"
"- Maintain professional, neutral language; short sentences (1–3 per paragraph).\n"
"- Write in narrative prose, not bullets.\n"
"- Output **valid JSON** with exactly one key: `\"narrative\"`.\n\n"
"ADDENDUM — Presentation of scale labels and footnotes\n"
"- Do not append long scale labels inline after percentages. Keep sentences clear and readable.\n"
"- The application displays, below each question, a short footnote explaining what the percentage represents "
"(e.g., “Percentages represent respondents’ aggregate answers to ‘a small/moderate/large/very large extent’.”).\n"
"- Mention labels inside the narrative only if essential for meaning, and then use the compressed form that preserves all categories.\n"
"- Apply the same rule for both per-question and overall summaries.\n"
"- Continue to respect all data-validation and polarity rules (Positive → ‘Positive’; Negative → ‘Negative’; Neutral → ‘Agree’). \n"
"- For D57-style distribution questions, still describe the category breakdown using the provided answer labels.\n\n"
"ADDENDUM — Anti-hallucination: strict trend gating\n"
"- You must obey the `allow_trend` flag in the payload:\n"
"  • If `allow_trend=false`, you must NOT write any sentences about change, YoY, “since <year>”, or “trend”. "
"    You may instead state: “There is no trend data available for prior years.”\n"
"- Never mention or invent years not listed in `years_present`.\n"
"- In overall synthesis, discuss trends only for questions with `allow_trend=true`. "
"Do not generalize trends across all questions if others have single-year data.\n"
"- When uncertain, default to single-year phrasing.\n\n"
"ADDENDUM — Single-year style guard (readability & consistency)\n"
"- When `allow_trend=false`, open with a full sentence — do NOT use telegraphic formats like “YYYY: 54 %* …”. "
"  Start with: “In <LATEST_YEAR>, <VALUE>%* …”.\n"
"- <LATEST_YEAR> must be the maximum of `years_present`. <VALUE> must be the reported metric for that year as given in the table.\n"
"- After this sentence, you may add: “There is no trend data available for prior years.”\n\n"
"ADDENDUM — Demographic gaps (latest year) and change-over-time\n"
"- When the payload indicates `demographic_breakdown_present=true`, you MUST:\n"
"  • Identify the latest year present in `years_present` and report the largest gap between demographic groups for the reported metric in that year, as an absolute difference in % points. Prefer integrated phrasing within the paragraph.\n"
"  • Compute gap-over-time from the earliest comparable year to the latest year (or latest vs previous if only two years exist), and state whether it widened, narrowed, or remained stable, with the absolute change in % points.\n"
"- Use only numbers visible in the table for the relevant groups and years. Do not infer values for missing years or groups.\n"
"- If fewer than two groups have values in the latest year, omit gap reporting.\n"
"- In overall synthesis, summarize notable gaps across the selected questions strictly from the provided tables; do not recompute or average.\n"
"- **Zero-gap guard (strict phrasing rules):**\n"
"  • If the absolute gap between groups in the latest year is ≤ 0.4 % points (effectively zero), do **not** describe one group as higher or lower. Instead state that results are identical or indistinguishable (e.g., “72 % each” or “virtually identical”).\n"
"  • If the absolute gap is between 0.5 and 2.0 % points, use “minimal” or “negligible difference” and avoid “slightly higher/lower”.\n"
"  • Retain existing magnitude wording for larger gaps (e.g., modest/notable) as already specified below.\n\n"
"ADDENDUM — Narrative and readability\n"
"- Express insights in a natural, narrative tone suitable for executive briefing notes.\n"
"- Vary sentence structure to avoid repetitive phrasing such as “The largest gap is…” or “The difference is…”.\n"
"- Integrate gap observations smoothly within the paragraph. For example: "
"“Results show a modest 3 % points difference between English (54 %) and French (51 %) respondents, indicating comparable experiences.”\n"
"- Keep concise yet fluid phrasing — avoid telegraphic, list-like statements.\n"
"- Maintain neutrality and factuality. Do not speculate or infer causes.\n\n"
"ADDENDUM — Stylistic flexibility for demographic comparisons\n"
"- You may vary how you describe demographic differences as long as values and gaps remain correct.\n"
"- Acceptable variations include:\n"
"  • “Results were broadly similar across groups, with only a 1 % point difference between English (54 %) and French (53 %).”\n"
"  • “English respondents reported slightly higher negative impacts (37 %) than French respondents (27 %), a 10 % points gap.”\n"
"  • “A modest 3 % points difference separates English (18 %) and French (21 %) respondents.”\n"
"- Adjust tone to the size of the difference:\n"
"  • ≤ 2 % points → “minimal”, “negligible”, “results are similar”\n"
"  • 3–6 % points → “modest”, “slightly higher/lower”\n"
"  • ≥ 7 % points → “notable”, “considerable”\n"
"- Integrate demographic commentary smoothly into the narrative rather than as a rigid template.\n"
"- Avoid repeating identical sentence templates across questions.\n\n"
"ADDENDUM — Global narrative flexibility (applies to all summaries)\n"
"- Apply natural, fluent phrasing to all narrative outputs: per-question summaries, demographic comparisons, and the overall synthesis.\n"
"- Vary sentence openings and structure; avoid repetitive templates like “The largest gap is…”, “There is…”, “Among groups…”, or “In <year>, <value>%…”.\n"
"- Prefer integrated narrative over list-like statements. Combine closely related facts into one smooth sentence where appropriate.\n"
"- Use qualitative cues matched to magnitude (without adding new numbers):\n"
"  • ≤ 2 % points → “minimal”, “negligible”, “results are similar”\n"
"  • 3–6 % points → “modest”, “slightly higher/lower”\n"
"  • ≥ 7 % points → “notable”, “considerable”\n"
"- Examples (keep values exactly as provided):\n"
"  • Single-year (no trend): “In 2024, 54 %* reported negative impacts. Results were nearly identical across groups (English 54 %, French 53 %). There is no trend data available for prior years.”\n"
"  • With demographic difference: “English respondents reported slightly higher negative impacts (37 %) than French respondents (27 %), a modest 10 % points gap.”\n"
"  • Overall synthesis: “Work–life conflict stands out at 54 %, while language training (34 %) and accommodation issues (31 %) also affect many. Discrimination (25 %) and accessibility (19 %) are lower but still consequential.”\n"
"- Maintain all anti-hallucination rules:\n"
"  • Do not invent years or values. Use only numbers present in the payload/tables and allowable differences.\n"
"  • If `allow_trend=false`, include the exact sentence: “There is no trend data available for prior years.”\n"
"- Keep tone concise and neutral, suitable for executive briefings.\n"
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
# Helpers (existing + normalization helper)
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
    """Return list of years with non-null metric values for the chosen metric."""
    if df is None or pd is None or metric_col is None or metric_col not in df.columns:
        return []
    try:
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

def _normalize_meaning_labels(
    meaning_labels: Optional[List[str]],
    reporting_field: Optional[str],
    df_disp=None
) -> List[str]:
    """
    Minimal, defensive normalization so the footnote always matches the chosen metric.
    - Deduplicates and trims labels
    - Ensures they are separated with '/'
    - Guards against obvious polarity mismatches (e.g., AGREE with 'disagree' labels)
    NOTE: This does not recompute indices; it only sanitizes what the caller provides.
    """
    labels: List[str] = []
    if meaning_labels is None:
        labels = []
    else:
        flat: List[str] = []
        for item in meaning_labels:
            if item is None:
                continue
            s = str(item).strip()
            if not s:
                continue
            parts = re.split(r"[\/\|,;]+", s)
            if len(parts) == 1:
                flat.append(s)
            else:
                flat.extend(p.strip() for p in parts if p.strip())
        labels = flat or []

    seen = set()
    clean: List[str] = []
    for s in labels:
        ss = re.sub(r"\s+", " ", s).strip()
        if ss and ss.lower() not in seen:
            seen.add(ss.lower())
            clean.append(ss)

    rf = (reporting_field or "").upper().strip()
    if rf == "AGREE":
        looks_disagree = all(any(tok in s.lower() for tok in ("disagree", "no")) for s in clean) if clean else False
        if looks_disagree:
            agreeish = []
            if df_disp is not None:
                for col in df_disp.columns:
                    lc = str(col).lower()
                    if any(key in lc for key in ("agree", "yes")) and not any(bad in lc for bad in ("disagree", "no")):
                        agreeish.append(str(col).strip())
            clean = agreeish or ["Yes"]
    elif rf == "POSITIVE":
        if clean and all("disagree" in s.lower() for s in clean):
            posish = []
            if df_disp is not None:
                for col in df_disp.columns:
                    lc = str(col).lower()
                    if any(key in lc for key in ("agree", "satisfied", "positive", "good", "always", "often", "to a very large extent", "to a large extent")) \
                       and "disagree" not in lc:
                        posish.append(str(col).strip())
            clean = posish or clean
    elif rf == "NEGATIVE":
        if clean and all(any(key in s.lower() for key in ("agree", "satisfied", "positive")) for s in clean):
            negish = []
            if df_disp is not None:
                for col in df_disp.columns:
                    lc = str(col).lower()
                    if any(key in lc for key in ("disagree", "dissatisfied", "negative", "rarely", "never", "to a small extent", "to a very small extent")):
                        negish.append(str(col).strip())
            clean = negish or clean

    if not clean:
        clean = labels or ["Yes"] if rf == "AGREE" else labels or []

    final_labels = []
    seen2 = set()
    for s in clean:
        ss = re.sub(r"\s+", " ", s).strip()
        if ss and ss.lower() not in seen2:
            seen2.add(ss.lower())
            final_labels.append(ss)

    return final_labels

# --------------------------------------------------------------------------------------
# Prompt builders (same signatures; added zero_gap_guard flag)
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
    # Normalize the meaning labels in a polarity-aware way (non-invasive)
    norm_labels = _normalize_meaning_labels(meaning_labels, reporting_field, df_disp=df_disp)

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
        "meaning_labels": list(norm_labels),
        "distribution_only": bool(distribution_only),
        "allow_trend": allow_trend,
        "years_present": years_present,
        "gap_over_time_policy": "earliest_to_latest",
        "zero_gap_guard": True,
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

            raw_labels = list((q_to_meaning_labels or {}).get(q, []))
            safe_labels = _normalize_meaning_labels(raw_labels, reporting_field=None, df_disp=None)

            selected_questions.append({
                "code": q,
                "text": code_to_text.get(q, ""),
                "metric_label": q_to_metric.get(q, ""),
                "meaning_labels": list(safe_labels),
                "distribution_only": bool((q_distribution_only or {}).get(q, False)),
                "allow_trend": allow_trend,
                "years_present": years_present,
            })
        except Exception:
            selected_questions.append({
                "code": q, "text": code_to_text.get(q, ""),
                "metric_label": q_to_metric.get(q, ""),
                "meaning_labels": [],
                "distribution_only": False,
                "allow_trend": False,
                "years_present": []
            })

    payload = {
        "task": "overall_synthesis",
        "selected_questions": selected_questions,
        "matrix": {"years": years, "values_by_row": matrix},
        "overall_controls": {
            "no_repetition": True,
            "cross_question_only": True,
            "hr_insights_allowed": True,
            "ban_external_context": True
        },
        "output_format": {"type": "json", "key": "narrative"},
    }
    return json.dumps(payload, ensure_ascii=False)
