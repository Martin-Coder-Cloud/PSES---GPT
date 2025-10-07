"""
Menu 1: Results rendering
- Summary tab (code-only rows; years as columns)
- Per-question tabs with distribution tables
- Source link shown directly under each tabulation (as a clickable title, no raw URL)
- AI summaries:
    • One short paragraph per question
    • Overall summary only when multiple questions are selected
- Excel export (Summary + each question + AI Summary)
- Start a new search button shown in a persistent footer (visible on any tab)
- AI cache to prevent re-calling on reruns/navigations with unchanged results
"""

from __future__ import annotations
from typing import Dict, Callable, Any, Tuple, List, Set, Optional
import io
import json
import hashlib
import re

import pandas as pd
import streamlit as st

from ..ai import AI_SYSTEM_PROMPT  # use the exact system prompt your ai.py defines

# ----- small helpers ----------------------------------------------------------

def _hash_key(obj: Any) -> str:
    """Stable hash for cache keys (works with dicts/lists/pandas) without mutating dtypes."""
    try:
        if isinstance(obj, pd.DataFrame):
            payload = obj.to_csv(index=True, na_rep="")
        else:
            payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _ai_cache_get(key: str):
    cache = st.session_state.get("menu1_ai_cache", {})
    return cache.get(key)

def _ai_cache_put(key: str, value: dict):
    cache = st.session_state.get("menu1_ai_cache", {})
    cache[key] = value
    st.session_state["menu1_ai_cache"] = cache

def _source_link_line(source_title: str, source_url: str) -> None:
    # Requirements: show the title as a clickable link (no raw URL), placed directly under the table.
    st.markdown(
        f"<div style='margin-top:6px; font-size:0.9rem;'>Source: "
        f"<a href='{source_url}' target='_blank'>{source_title}</a></div>",
        unsafe_allow_html=True
    )

# ====================== FACT-CHECK VALIDATOR HELPERS (kept but NOT used while disabled) ======================

_INT_RE = re.compile(r"-?\d+")

def _is_year_like(n: int) -> bool:
    # treat any 4-digit year-ish as a "year token" we ignore in validation
    return 1900 <= n <= 2100

def _is_year_label(col) -> bool:
    """
    True if a column label looks like a survey year (int 1900–2100 or 4-digit string in that range).
    Used only by the validator to detect wide tables (years as columns).
    """
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

def _to_int(v: Any) -> Optional[int]:
    """
    Safe integer caster for fact-check logic:
    - returns None for None/NaN/pd.NA/non-numeric
    - returns int(...) for valid integers/floats
    """
    try:
        if v is None:
            return None
        # pandas NA support (without hard dependency)
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return None
        except Exception:
            pass
        if isinstance(v, int):
            return int(v)
        f = float(v)
        try:
            import math
            if math.isnan(f):
                return None
        except Exception:
            pass
        return int(round(f))
    except Exception:
        return None

def _pick_display_metric(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    """
    Choose which integer column the table is displaying.
    Preference:
      1) explicit `prefer` if provided and present,
      2) 'value_display',
      3) 'AGREE',
      4) 'SCORE100'
    Returns the column name or None if none found.
    """
    if prefer and prefer in df.columns:
        return prefer
    for c in ("value_display", "AGREE", "SCORE100"):
        if c in df.columns:
            return c
    return None

def _allowed_numbers_from_disp(df: pd.DataFrame, metric_col: str) -> Tuple[Set[int], Set[int]]:
    """
    Build the allowed set of integers from a per-question display dataframe.
    Returns (allowed_numbers, visible_years).

    NOTE: This helper is retained for later re-enable of the validator,
    but it is NOT called while the validator is disabled.
    """
    if df is None or df.empty:
        return set(), set()

    # ---- Detect shape; if wide, melt to long with 'Year' + '__MVAL__' ----
    metric_col_work = metric_col
    if "Year" not in df.columns or metric_col not in df.columns:
        # Identify year-like columns
        ycols = [c for c in df.columns if _is_year_label(c)]
        if ycols:
            id_cols = [c for c in df.columns if c not in ycols]
            melted = df.melt(id_vars=id_cols, value_vars=ycols, var_name="Year", value_name="__MVAL__")
            work = melted.copy()
            metric_col_work = "__MVAL__"
        else:
            # No Year column and no obvious year-like headers → nothing to validate
            return set(), set()
    else:
        work = df.copy()

    # Normalize types
    work["__Year__"] = pd.to_numeric(work.get("Year"), errors="coerce").astype("Int64")
    if "Demographic" not in work.columns:
        work["Demographic"] = "All respondents"

    # Integerize metric values where possible (NA-safe)
    if metric_col_work not in work.columns:
        return set(), set()
    work["__Val__"] = work[metric_col_work].apply(_to_int)

    # Visible years
    years: Set[int] = set(int(y) for y in work["__Year__"].dropna().unique())

    # Base values
    allowed: Set[int] = set(int(v) for v in work["__Val__"].dropna().unique())

    # YoY diffs (within each group label)
    gdf_work = work.dropna(subset=["__Val__", "__Year__"])
    for _, gdf in gdf_work.groupby("Demographic", dropna=False):
        vals = [int(v) for v in gdf.sort_values("__Year__")["__Val__"].tolist() if v is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                allowed.add(abs(vals[j] - vals[i]))

    # Latest-year gaps
    if years:
        latest = max(years)
        ydf = work[(work["__Year__"] == latest) & work["__Val__"].notna()]
        vals = list(ydf[["Demographic", "__Val__"]].itertuples(index=False, name=None))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                vi = int(vals[i][1]); vj = int(vals[j][1])
                allowed.add(abs(vi - vj))

        # Gap-over-time: for each pair of groups, compare gaps across years
        groups = sorted(ydf["Demographic"].astype(str).unique().tolist())
        # Build year->val maps per group
        maps: Dict[str, Dict[int, int]] = {}
        for g in work["Demographic"].astype(str).unique().tolist():
            gmap: Dict[int, int] = {}
            # ---- FIXED: .astype(str) parentheses and bracket placement ----
            for _, r in work[work["Demographic"].astype(str) == g].iterrows():
                y = r["__Year__"]; v = r["__Val__"]
                if pd.notna(y) and v is not None:
                    gmap[int(y)] = int(v)
            if gmap:
                maps[g] = gmap
        # For each pair, compute latest gap and delta vs earlier years with both groups present
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                m1 = maps.get(g1, {}); m2 = maps.get(g2, {})
                if latest not in m1 or latest not in m2:
                    continue
                gap_latest = abs(m1[latest] - m2[latest])
                # against all earlier years where both exist
                for y in sorted(years):
                    if y >= latest:
                        continue
                    if y in m1 and y in m2:
                        gap_prev = abs(m1[y] - m2[y])
                        allowed.add(abs(gap_latest - gap_prev))

    return allowed, years

def _extract_datapoint_integers_with_sentences(text: str) -> List[Tuple[int, str]]:
    """
    Extract only 'data-point' integers from text, keeping the sentence they appear in.
    NOTE: Validator is disabled; helper kept for later.
    """
    if not text:
        return []

    sentences = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    found: List[Tuple[int, str]] = []

    patterns = [
        re.compile(r"\((\d+)\)"),
        re.compile(r"\b(\d+)\s*points?\b", re.IGNORECASE),
        re.compile(r"\b(\d+)\s*%\b"),
        re.compile(r"\b(?:is|was|were|at|reached|stood(?:\s+at)?)\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bfrom\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bto\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bvs\.?\s+(\d+)\b", re.IGNORECASE),
    ]

    for s in sentences:
        nums: Set[int] = set()
        for pat in patterns:
            for m in pat.finditer(s):
                try:
                    n = int(m.group(1))
                    nums.add(n)
                except Exception:
                    continue
        for n in sorted(nums):
            found.append((n, s))
    return found

def _validate_narrative(narrative: str, allowed: Set[int], years: Set[int]) -> dict:
    """
    Validate narrative integers against the allowed set; ignore year numbers.
    NOTE: Validator is disabled; helper kept for later.
    """
    if not narrative:
        return {"ok": True, "bad_numbers": set(), "problems": []}
    return {"ok": True, "bad_numbers": set(), "problems": []}

# ==================== AI narrative computation ====================

def _compute_ai_narratives(
    *,
    tab_labels: List[str],
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    per_q_metric_label: Dict[str, str],
    code_to_text: Dict[str, str],
    demo_selection: Optional[str],
    pivot: pd.DataFrame,
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
) -> Tuple[Dict[str, str], Optional[str]]:
    """Build per-question and overall narratives (writes to cache, returns dicts)."""
    per_q_narratives: Dict[str, str] = {}
    overall_narrative: Optional[str] = None

    # Per-question
    for q in tab_labels:
        df_disp = per_q_disp[q]
        metric_col = per_q_metric_col[q]
        metric_label = per_q_metric_label[q]
        qtext = code_to_text.get(q, "")
        with st.spinner(f"AI — analyzing {q}…"):
            content, _hint = call_openai_json(
                system=AI_SYSTEM_PROMPT,
                user=build_per_q_prompt(
                    question_code=q,
                    question_text=qtext,
                    df_disp=df_disp,
                    metric_col=metric_col,
                    metric_label=metric_label,
                    category_in_play=(demo_selection != "All respondents")
                )
            )
        try:
            j = json.loads(content) if content else {}
            per_q_narratives[q] = (j.get("narrative") or "").strip()
        except Exception:
            per_q_narratives[q] = ""

    # Overall (only if multiple questions)
    if len(tab_labels) > 1:
        with st.spinner("AI — synthesizing overall pattern…"):
            q_to_metric = {q: per_q_metric_label[q] for q in tab_labels}
            content, _hint = call_openai_json(
                system=AI_SYSTEM_PROMPT,
                user=build_overall_prompt(
                    tab_labels=tab_labels,
                    pivot_df=pivot,
                    q_to_metric=q_to_metric
                )
            )
        try:
            j = json.loads(content) if content else {}
            overall_narrative = (j.get("narrative") or "").strip()
        except Exception:
            overall_narrative = None

    return per_q_narratives, overall_narrative

# ----- main renderer ----------------------------------------------------------

def tabs_summary_and_per_q(
    *,
    payload: Dict[str, Any],
    ai_on: bool,
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
    source_url: str,
    source_title: str,
) -> None:
    """
    payload keys (as stashed in state):
      - per_q_disp: Dict[qcode, DataFrame]
      - per_q_metric_col: Dict[qcode, str]
      - per_q_metric_label: Dict[qcode, str]
      - pivot: DataFrame
      - tab_labels: List[qcode]
      - years: List[int]
      - demo_selection: str | None
      - sub_selection: str | None
      - code_to_text: Dict[qcode, text]
    """
    per_q_disp: Dict[str, pd.DataFrame] = payload["per_q_disp"]
    per_q_metric_col: Dict[str, str]   = payload["per_q_metric_col"]
    per_q_metric_label: Dict[str, str] = payload["per_q_metric_label"]
    pivot: pd.DataFrame                = payload["pivot"]
    tab_labels                         = payload["tab_labels"]
    years                              = payload["years"]
    demo_selection                     = payload["demo_selection"]
    sub_selection                      = payload["sub_selection"]
    code_to_text                       = payload["code_to_text"]

    # [stash payload for ai.py] Make question texts discoverable to build_overall_prompt()
    st.session_state["menu1_results_payload"] = {"code_to_text": code_to_text}

    # Build a stable "result signature" so we do NOT recompute AI on benign reruns
    ai_sig = {
        "tab_labels": tab_labels,
        "years": years,
        "demo_selection": demo_selection,
        "sub_selection": sub_selection,
        "metric_labels": {q: per_q_metric_label[q] for q in tab_labels},
        "pivot_sig": _hash_key(pivot),  # compact signature of numbers only
    }
    ai_key = "menu1_ai_" + _hash_key(ai_sig)

    # UI tabs: first Summary, then one per question, then Technical notes
    tabs = st.tabs(["Summary table"] + tab_labels + ["Technical notes"])

    # We will fill these and also stash them in session for the footer export
    per_q_narratives: Dict[str, str] = st.session_state.get("menu1_ai_narr_per_q", {})
    overall_narrative: Optional[str] = st.session_state.get("menu1_ai_narr_overall", None)

    # ------------------------- Summary tab -------------------------
    with tabs[0]:
        st.markdown("### Summary table")

        # Small list of questions and metric used (lightweight) — shown above the table
        if tab_labels:
            st.markdown("<div style='font-size:0.9rem; color:#444; margin-bottom:4px;'>"
                        "Questions & metrics included:</div>", unsafe_allow_html=True)
            for q in tab_labels:
                qtext = code_to_text.get(q, "")
                mlabel = per_q_metric_label.get(q, "% positive")
                st.markdown(
                    f"<div style='font-size:0.85rem; color:#555;'>"
                    f"<strong>{q}</strong>: {qtext} "
                    f"<span style='opacity:.85;'>[{mlabel}]</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # The table (rows are codes or code×demographic; columns are years)
        st.dataframe(pivot.reset_index(), use_container_width=True)

        # Source line directly under the tabulation
        _source_link_line(source_title, source_url)

        # ----- AI Summary area (compute & render) -----
        if ai_on:
            cached = _ai_cache_get(ai_key)
            if cached:
                per_q_narratives = cached.get("per_q", {})
                overall_narrative = cached.get("overall")
            else:
                per_q_narratives, overall_narrative = _compute_ai_narratives(
                    tab_labels=tab_labels,
                    per_q_disp=per_q_disp,
                    per_q_metric_col=per_q_metric_col,
                    per_q_metric_label=per_q_metric_label,
                    code_to_text=code_to_text,
                    demo_selection=demo_selection,
                    pivot=pivot,
                    build_overall_prompt=build_overall_prompt,
                    build_per_q_prompt=build_per_q_prompt,
                    call_openai_json=call_openai_json,
                )
                _ai_cache_put(ai_key, {"per_q": per_q_narratives, "overall": overall_narrative})

            # Save for the persistent footer exporter
            st.session_state["menu1_ai_narr_per_q"] = per_q_narratives
            st.session_state["menu1_ai_narr_overall"] = overall_narrative

            # Render AI section
            st.markdown("---")
            st.markdown("### AI Summary")
            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    st.markdown(f"**{q} — {code_to_text.get(q, '')}**")
                    st.write(txt)
            if overall_narrative and len(tab_labels) > 1:
                st.markdown("**Overall**")
                st.write(overall_narrative)

            # ================== FACT CHECK VALIDATION UI (DISABLED) ==================
            # (kept for later re-enablement)
            # ========================================================================

    # ---------------------- Per-question tabs ----------------------
    for idx, qcode in enumerate(tab_labels, start=1):
        with tabs[idx]:
            qtext = code_to_text.get(qcode, "")
            st.subheader(f"{qcode} — {qtext}")
            st.dataframe(per_q_disp[qcode], use_container_width=True)
            _source_link_line(source_title, source_url)

    # ---------------------- Technical notes tab (always last) ----------------------
    with tabs[-1]:
        st.markdown("### Technical notes")
        st.markdown(
            "- **Summary results** are mainly shown as “positive answers” reflecting the affirmative responses. "
            "Positive answers are calculated by removing the \"Don't know\" and \"Not applicable\" responses from the total responses.\n"
            "- **Non-response adjustment:** Results have been adjusted for non-response to better represent the target population. "
            "Therefore, percentages should not be used to determine the number of respondents within a response category.\n"
            "- **Rounding:** Due to rounding, percentages may not add to 100.\n"
            "- **Suppression:** Results were suppressed for questions with low respondent counts (under 10) and for low response category counts."
        )

    # ---------------------- Persistent footer (all tabs) ----------------------
    st.markdown("---")
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
    col_dl, col_new = st.columns([1, 1])

    with col_dl:
        # Ensure we have narratives for export if AI is ON (use cache if possible)
        export_per_q = st.session_state.get("menu1_ai_narr_per_q")
        export_overall = st.session_state.get("menu1_ai_narr_overall")

        if ai_on and export_per_q is None:
            cached = _ai_cache_get(ai_key)
            if cached:
                export_per_q = cached.get("per_q", {})
                export_overall = cached.get("overall")
            else:
                # compute on demand (user may not have visited the Summary tab)
                export_per_q, export_overall = _compute_ai_narratives(
                    tab_labels=tab_labels,
                    per_q_disp=per_q_disp,
                    per_q_metric_col=per_q_metric_col,
                    per_q_metric_label=per_q_metric_label,
                    code_to_text=code_to_text,
                    demo_selection=demo_selection,
                    pivot=pivot,
                    build_overall_prompt=build_overall_prompt,
                    build_per_q_prompt=build_per_q_prompt,
                    call_openai_json=call_openai_json,
                )
                _ai_cache_put(ai_key, {"per_q": export_per_q, "overall": export_overall})

        _render_excel_download(
            pivot=pivot,
            per_q_disp=per_q_disp,
            tab_labels=tab_labels,
            per_q_narratives=(export_per_q or {}),
            overall_narrative=(export_overall if (export_overall and len(tab_labels) > 1) else None),
        )

    with col_new:
        if st.button("Start a new search", key="menu1_new_search"):
            # Full reset: core params/state
            try:
                from .. import state  # local import to avoid circulars
                state.reset_menu1_state()
            except Exception:
                for k in [
                    "menu1_selected_codes", "menu1_multi_questions",
                    "menu1_ai_toggle", "menu1_show_diag",
                    "select_all_years", "demo_main", "last_query_info",
                ]:
                    st.session_state.pop(k, None)

            # ALSO clear keyword-search area so no stale warning appears
            for k in [
                "menu1_hits", "menu1_hit_codes_selected",
                "menu1_search_done", "menu1_last_search_query",
                "menu1_kw_query",
            ]:
                st.session_state.pop(k, None)

            # Remove dynamic checkbox keys from previous searches/selections
            for k in list(st.session_state.keys()):
                if k.startswith("kwhit_") or k.startswith("sel_"):
                    st.session_state.pop(k, None)

            # Clear AI cache and saved narratives
            st.session_state.pop("menu1_ai_cache", None)
            st.session_state.pop("menu1_ai_narr_per_q", None)
            st.session_state.pop("menu1_ai_narr_overall", None)

            # Rerun
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

# ----- Excel export with AI Summary sheet ------------------------------------

def _render_excel_download(
    *,
    pivot: pd.DataFrame,
    per_q_disp: Dict[str, pd.DataFrame],
    tab_labels: List[str],
    per_q_narratives: Dict[str, str],
    overall_narrative: Optional[str],
) -> None:
    # Build the workbook in-memory
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            # Summary
            pivot.reset_index().to_excel(writer, sheet_name="Summary_Table", index=False)
            # Per-question sheets
            for q, df_disp in per_q_disp.items():
                safe = q[:28]
                df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)
            # AI Summary sheet
            rows = []
            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    rows.append({"Section": q, "Narrative": txt})
            if overall_narrative and len(tab_labels) > 1:
                rows.append({"Section": "Overall", "Narrative": overall_narrative})
            ai_df = pd.DataFrame(rows, columns=["Section", "Narrative"])
            ai_df.to_excel(writer, sheet_name="AI Summary", index=False)
        data = buf.getvalue()

    st.download_button(
        label="Download data and AI summaries",
        data=data,
        file_name="PSES_results_with_AI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="menu1_excel_download",
    )
