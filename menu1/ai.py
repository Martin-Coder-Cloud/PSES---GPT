# ai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time

# --------------------------------------------------------------------------------------
# SYSTEM PROMPT (merged + refined)
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
"- Gap-over-time: for each highlighted gap, compute the gap in each year where both groups have data. "
"State whether the gap has widened, narrowed, or remained stable since the earliest comparable year, "
"and give the change in % points (e.g., “gap narrowed by 3 % points since 2020”).\n"
"- Do NOT compute multi-year averages or rates of change beyond these integer subtractions.\n"
"- If `distribution_only=true`, describe the latest-year distribution across Answer 1 … 6 and note direction of change only if visible; never sum options or create aggregates.\n\n"

"Trend rules (per-question; prioritize current year, then context)\n"
"- Start with the latest year vs the previous year: report the year-over-year (YoY) change in % points "
"(e.g., “2024: 54 %, down 2 % points vs 2023”).\n"
"- Then place this YoY in the full context of all available years:\n"
"  • Compute YoY deltas for every adjacent pair and the earliest→latest net change.\n"
"  • Classify the overall trend using YoY signs:\n"
"      ▸ Increasing = all YoY ≥ 0 and at least one > 0\n"
"      ▸ Declining = all YoY ≤ 0 and at least one < 0\n"
"      ▸ Stable = all values identical (YoY = 0)\n"
"      ▸ Mixed = otherwise (some increases and some decreases)\n"
"  • Explain how the latest YoY relates to the long-term pattern: is it continuing the trend, a reversal, a bump, or stabilization?\n"
"- Small movements (±1 % point) → “little change.”  If only one year → “No trend (single year).”\n"
"- Use only permitted math (differences in % points; no averages).\n\n"

"Trend roll-up (overall synthesis)\n"
"- Lead with the latest-year picture — which areas rose or fell vs previous year.\n"
"- Then summarize long-term patterns across questions: how many are increasing, declining, stable, or mixed.\n"
"- Briefly indicate whether current movements continue prior patterns or appear as reversals or bumps.\n"
"- Use numbers sparingly — only YoY and earliest→latest deltas. No new computations.\n\n"

"Overall synthesis rules (when task = \"overall_synthesis\")\n"
"- Purpose: summarize themes and implications across all selected questions — not to repeat each narrative.\n"
"- Identify common patterns (strengths, recurring concerns, areas improving or declining).\n"
"- Highlight areas of strong results (high positive %, clear improvements) and areas requiring attention "
"(lower scores, declines, or large demographic gaps).\n"
"- When useful, group related questions under broader ideas such as career development, work–life balance, inclusion, or leadership.\n"
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
"- For D57-style distribution questions, still describe the category breakdown using the provided answer labels."
)

# --------------------------------------------------------------------------------------
# LLM CALLER (unchanged from your working version)
# --------------------------------------------------------------------------------------

def call_openai_json(*, system: str, user: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls OpenAI with JSON response mode.
    Requires OPENAI_API_KEY. Optional: AI_MODEL (default 'gpt-4o-mini').
    Returns (content_json_text, debug_hint_or_none).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback = json.dumps({"narrative": "AI is not configured (missing OPENAI_API_KEY)."}, ensure_ascii=False)
        return fallback, "OPENAI_API_KEY missing"

    model = os.environ.get("AI_MODEL", "gpt-4o-mini").strip()

    try:
        from openai import OpenAI
    except Exception as e:
        fallback = json.dumps({"narrative": "AI client missing. Please install openai>=1.0.0."}, ensure_ascii=False)
        return fallback, f"OpenAI import error: {type(e).__name__}"

    client = OpenAI(api_key=api_key)

    last_err = None
    for attempt in range(2):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
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
# LEGACY HELPERS (added for backward compatibility; serialization only)
# --------------------------------------------------------------------------------------

def _minify_df_for_model(
    df, *,
    year_col: str = "Year",
    metric_col: Optional[str] = None,
    include_demo: bool = True,
    distribution_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert a per-question display DataFrame into compact rows for the model.
    - distribution_only=False: keep Year, optional Demographic, and the chosen metric column.
    - distribution_only=True: keep Year, optional Demographic, and any Answer1..Answer6 columns.
    Values of 9999 are excluded (treated as missing).
    """
    try:
        import pandas as pd  # local import to avoid hard dependency unless needed
        if df is None:
            return []
        if isinstance(df, list):
            # already minified
            return df  # type: ignore[return-value]
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []

        work = df.copy()
        cols = list(work.columns)

        def _to_num(s):
            try:
                f = float(s)
                return None if f == 9999 else f
            except Exception:
                return None

        if distribution_only:
            keep = []
            if year_col in cols: keep.append(year_col)
            if include_demo and "Demographic" in cols: keep.append("Demographic")
            # collect Answer1..Answer6 (with or without space)
            ans = []
            for i in range(1, 7):
                for name in (f"Answer{i}", f"Answer {i}"):
                    if name in cols and name not in ans:
                        ans.append(name)
            keep += ans
            if not keep:
                return []
            work = work[keep].copy()
            # coerce numerics and drop missing year
            if year_col in work.columns:
                work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
            for c in ans:
                work[c] = work[c].apply(_to_num)
            work = work.dropna(subset=[year_col])
            out: List[Dict[str, Any]] = []
            for _, r in work.iterrows():
                row: Dict[str, Any] = {"Year": int(float(r[year_col]))}
                if include_demo and "Demographic" in work.columns and pd.notna(r.get("Demographic", None)):
                    row["Demographic"] = str(r["Demographic"])
                for c in ans:
                    v = r.get(c, None)
                    if v is not None:
                        row[c] = v
                out.append(row)
            out.sort(key=lambda x: (x["Year"], x.get("Demographic","")))
            return out

        # aggregate metric mode
        keep = []
        if year_col in cols: keep.append(year_col)
        if include_demo and "Demographic" in cols: keep.append("Demographic")
        if metric_col and metric_col in cols: keep.append(metric_col)
        if not keep:
            return []
        work = work[keep].copy()
        if year_col in work.columns:
            work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
        if metric_col in work.columns:
            work[metric_col] = work[metric_col].apply(_to_num)
        work = work.dropna(subset=[year_col])
        if metric_col in work.columns:
            work = work.dropna(subset=[metric_col])
        out: List[Dict[str, Any]] = []
        for _, r in work.iterrows():
            row: Dict[str, Any] = {"Year": int(float(r[year_col]))}
            if include_demo and "Demographic" in work.columns and r.get("Demographic") is not None:
                row["Demographic"] = str(r["Demographic"])
            if metric_col in work.columns and r.get(metric_col, None) is not None:
                row[metric_col] = r[metric_col]
            out.append(row)
        out.sort(key=lambda x: (x["Year"], x.get("Demographic","")))
        return out
    except Exception:
        return []

def build_per_q_prompt(
    *,
    question_code: str,
    question_text: str,
    df_disp,                    # pandas.DataFrame or already-minified list of dicts
    metric_col: Optional[str],  # EXACT column chosen by the app (Positive/Negative/Agree/Answer1) or None in distribution mode
    metric_label: str,          # human-friendly description (e.g., "% selecting Strongly agree / Agree")
    category_in_play: bool,     # whether subgroups are present and should be analyzed
    meaning_labels: Optional[List[str]] = None,     # e.g., ["Strongly agree","Agree"]
    reporting_field: Optional[str] = None,          # e.g., "POSITIVE","NEGATIVE","AGREE","ANSWER1"
    year_col: str = "Year",
    distribution_only: bool = False                 # EXCEPTION: D57_A / D57_B style
) -> str:
    """
    Legacy helper: returns the USER message JSON string for a per-question analysis.
    Serialization only; no extra computations.
    """
    data_rows = _minify_df_for_model(
        df_disp,
        year_col=year_col,
        metric_col=metric_col,
        include_demo=bool(category_in_play),
        distribution_only=bool(distribution_only),
    )
    payload: Dict[str, Any] = {
        "task": "per_question_summary",
        "question_code": str(question_code),
        "question_text": str(question_text),
        "reporting_metric": {
            "column": metric_col or "",
            "label": metric_label or "",
            "meaning_labels": meaning_labels or [],
            "reporting_field": reporting_field or "",
        },
        "data": data_rows,
        "category_in_play": bool(category_in_play),
        "distribution_only": bool(distribution_only),
    }
    return json.dumps(payload, ensure_ascii=False)

def build_overall_prompt(
    *,
    tab_labels: List[str],
    pivot_df,  # pandas.DataFrame
    q_to_metric: Dict[str, str],
    code_to_text: Dict[str, str],
    q_to_meaning_labels: Optional[Dict[str, List[str]]] = None,
    q_distribution_only: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Legacy helper: returns the USER message JSON string for overall synthesis.
    Uses the summary pivot and per-question metadata the renderer already resolved.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None  # type: ignore

    matrix: List[Dict[str, Any]] = []
    if pd is not None and isinstance(pivot_df, pd.DataFrame) and not pivot_df.empty:
        piv = pivot_df.copy()
        if "Question" not in piv.reset_index().columns:
            piv.index.name = "Question"
        piv = piv.reset_index()
        for _, r in piv.iterrows():
            if "Question" not in r:
                continue
            row = {"question_code": str(r["Question"])}
            for c in pivot_df.columns:
                try:
                    yr = int(float(c))
                    val = r[c]
                    try:
                        f = float(val)
                        if f == 9999:
                            continue
                        row[str(yr)] = f
                    except Exception:
                        continue
                except Exception:
                    continue
            matrix.append(row)

    selected = []
    for q in tab_labels:
        item = {
            "question_code": q,
            "question_text": code_to_text.get(q, ""),
            "metric_label": q_to_metric.get(q, "% value"),
        }
        if q_to_meaning_labels and q in q_to_meaning_labels:
            item["meaning_labels"] = q_to_meaning_labels[q] or []
        if q_distribution_only and q in q_distribution_only:
            item["distribution_only"] = bool(q_distribution_only[q])
        selected.append(item)

    payload = {
        "task": "overall_synthesis",
        "selected_questions": selected,
        "matrix": matrix,
        "instructions": {
            "compare_across_questions": True,
            "trend_over_all_years": True,
            "no_new_calculations": True
        },
    }
    return json.dumps(payload, ensure_ascii=False)
