# menu1/render/results.py
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
            payload = df_to_cache_csv(obj)
        else:
            payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def df_to_cache_csv(df: pd.DataFrame) -> str:
    # keep index; represent NAs as empty string to stabilize signature without coercion
    return df.to_csv(index=True, na_rep="")

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

# ====================== FACT-CHECK VALIDATOR HELPERS (advisory) ======================

_INT_RE = re.compile(r"-?\d+")

def _is_year_like(n: int) -> bool:
    return 1900 <= n <= 2100

def _is_year_label(col) -> bool:
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
    Original safe integer caster used elsewhere in this file.
    """
    try:
        if v is None:
            return None
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

# ---- NA-tolerant int for the fact-check path only ---------------------------
def _safe_int(x: Any) -> Optional[int]:
    """
    Return a plain int when possible; otherwise None.
    Handles 'none', 'n/a', '', whitespace, pandas NA/NaN, 9999 (treated as NA).
    """
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("", "na", "n/a", "none", "nan", "null"):
                return None
        if x == 9999:
            return None
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return int(round(float(v)))
    except Exception:
        return None
# -----------------------------------------------------------------------------

def _pick_display_metric(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
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

    Hardened to never raise on mixed/odd values. Any non-coercible token is skipped.
    """
    if df is None or df.empty:
        return set(), set()

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
            return set(), set()
    else:
        work = df.copy()

    # Normalize types (robustly)
    work["__Year__"] = pd.to_numeric(work.get("Year"), errors="coerce").astype("Int64")
    if "Demographic" not in work.columns:
        work["Demographic"] = "All respondents"

    if metric_col_work not in work.columns:
        return set(), set()

    # Safe metric integerization (NA/9999/etc → None)
    work["__Val__"] = work[metric_col_work].apply(_safe_int)

    # Visible years set (ints only)
    years_list = []
    for y in work["__Year__"].tolist():
        yi = _safe_int(y)
        if yi is not None and _is_year_like(yi):
            years_list.append(yi)
    years: Set[int] = set(years_list)

    # Base allowed values (ints only)
    allowed: Set[int] = set()
    for v in work["__Val__"].tolist():
        vi = _safe_int(v)
        if vi is not None:
            allowed.add(vi)

    # Prepare a cleaned working frame for diffs/gaps
    gdf_work = work.dropna(subset=["__Year__"]).copy()
    gdf_work["__YearI__"] = gdf_work["__Year__"].apply(_safe_int)
    gdf_work["__ValI__"] = gdf_work["__Val__"].apply(_safe_int)
    gdf_work = gdf_work[gdf_work["__YearI__"].notna() & gdf_work["__ValI__"].notna()]

    # YoY diffs (within each group label)
    for _, gdf in gdf_work.groupby("Demographic", dropna=False):
        seq = [int(v) for v in gdf.sort_values("__YearI__")["__ValI__"].tolist()]
        n = len(seq)
        for i in range(n):
            for j in range(i + 1, n):
                allowed.add(abs(seq[j] - seq[i]))

    # Latest-year gaps and gap-over-time
    if years:
        latest = max(years)

        ydf = gdf_work[gdf_work["__YearI__"] == latest]
        vals = list(ydf[["Demographic", "__ValI__"]].itertuples(index=False, name=None))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                vi = int(vals[i][1]); vj = int(vals[j][1])
                allowed.add(abs(vi - vj))

        # Build year->val maps per group
        groups = sorted(gdf_work["Demographic"].astype(str).unique().tolist())
        maps: Dict[str, Dict[int, int]] = {}
        for g in groups:
            gmap: Dict[int, int] = {}
            sub = gdf_work[gdf_work["Demographic"].astype(str) == g]
            for _, r in sub.iterrows():
                yi = r["__YearI__"]; vi = r["__ValI__"]
                if pd.notna(yi) and pd.notna(vi):
                    gmap[int(yi)] = int(vi)
            if gmap:
                maps[g] = gmap

        # Gap-over-time deltas
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                m1 = maps.get(g1, {}); m2 = maps.get(g2, {})
                if latest not in m1 or latest not in m2:
                    continue
                gap_latest = abs(m1[latest] - m2[latest])
                for y in sorted(years):
                    if y >= latest:
                        continue
                    if y in m1 and y in m2:
                        gap_prev = abs(m1[y] - m2[y])
                        allowed.add(abs(gap_latest - gap_prev))

    return allowed, years

def _extract_datapoint_integers_with_sentences(text: str) -> List[Tuple[int, str]]:
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
    if not narrative:
        return {"ok": True, "bad_numbers": set(), "problems": []}
    pairs = _extract_datapoint_integers_with_sentences(narrative)
    bad: Set[int] = set()
    problems: List[str] = []
    for n, sentence in pairs:
        if _is_year_like(n):
            continue
        if n not in allowed:
            bad.add(n)
            problems.append(f"{n} — {sentence}")
    return {"ok": len(bad) == 0, "bad_numbers": bad, "problems": problems[:5]}

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
    per_q_narratives: Dict[str, str] = {}
    overall_narrative: Optional[str] = None

    # Per-question
    for q in tab_labels:
        df_disp = per_q_disp[q]
        metric_col = per_q_metric_col[q]
        metric_label = per_q_metric_label[q]
        qtext = code_to_text.get(q, "")
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

    # Overall
    if len(tab_labels) > 1:
        content, _hint = call_openai_json(
            system=AI_SYSTEM_PROMPT,
            user=build_overall_prompt(
                tab_labels=tab_labels,
                pivot_df=pivot,
                q_to_metric={q: per_q_metric_label[q] for q in tab_labels},
                code_to_text=code_to_text,
            )
        )
        try:
            j = json.loads(content) if content else {}
            overall_narrative = (j.get("narrative") or "").strip()
        except Exception:
            overall_narrative = None

    return per_q_narratives, overall_narrative

# ----- AI Data Validation (subsection renderer) -------------------------------

def _render_data_validation_subsection(
    *,
    tab_labels: List[str],
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    per_q_narratives: Dict[str, str],
) -> None:
    """
    Renders a subsection under AI Summary:
      • Title line: bold text (matches “Select from the list” format)
      • One-sentence outcome with ✅ (or ❌ if issues)
      • Dropdown to view per-question details
    """
    any_issue = False
    details: List[Tuple[str, str]] = []  # (level, message)

    for q in tab_labels:
        try:
            df_disp = per_q_disp.get(q)
            if not isinstance(df_disp, pd.DataFrame) or df_disp.empty:
                details.append(("caption", f"{q}: validation skipped (no table available)."))
                continue

            metric_col = per_q_metric_col.get(q) or _pick_display_metric(df_disp)
            if not metric_col:
                details.append(("caption", f"{q}: validation skipped (no metric column)."))
                continue

            allowed, years = _allowed_numbers_from_disp(df_disp.copy(deep=True), metric_col)
            narrative = per_q_narratives.get(q, "") or ""
            res = _validate_narrative(narrative, allowed, years)

            if not res["ok"]:
                any_issue = True
                nums = ", ".join(str(x) for x in sorted(res["bad_numbers"]))
                if nums:
                    details.append(("warning", f"{q}: potential mismatches detected ({nums})."))
                else:
                    details.append(("warning", f"{q}: potential mismatches detected."))
            else:
                details.append(("caption", f"{q}: no numeric inconsistencies detected."))
        except Exception as e:
            details.append(("caption", f"{q}: validation skipped ({type(e).__name__})."))

    # --- Scoped style to enforce "bold body text" (not a header) --------------
    st.markdown(
        """
        <style>
          #ai_data_validation_title {
            font-size: 1rem;
            line-height: 1.25;
            font-weight: 600;
            margin: 0.25rem 0 0.25rem 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div id='ai_data_validation_title'>AI Data Validation</div>", unsafe_allow_html=True)

    if not any_issue:
        st.markdown("✅ The data points in the summaries have been validated and correspond to the data provided.")
    else:
        st.markdown("❌ Some AI statements may not match the tables. Review the details below.")

    with st.expander("View per-question validation details", expanded=False):
        for level, msg in details:
            if level == "warning":
                st.warning(msg)
            else:
                st.caption(msg)

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
    payload keys:
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

    # Build a stable "result signature"
    ai_sig = {
        "tab_labels": tab_labels,
        "years": years,
        "demo_selection": demo_selection,
        "sub_selection": sub_selection,
        "metric_labels": {q: per_q_metric_label[q] for q in tab_labels},
        "pivot_sig": _hash_key(pivot),
    }
    ai_key = "menu1_ai_" + _hash_key(ai_sig)

    # ---- (1) Title above the tabulations (UX) ----
    st.header("Results")

    # Tabs: Summary + per-question + Technical notes (at end)
    tab_titles = ["Summary table"] + tab_labels + ["Technical notes"]
    tabs = st.tabs(tab_titles)

    # Cached AI outputs (may be populated later in AI section)
    per_q_narratives: Dict[str, str] = st.session_state.get("menu1_ai_narr_per_q", {})
    overall_narrative: Optional[str] = st.session_state.get("menu1_ai_narr_overall", None)

    # ------------------------- Summary tab -------------------------
    with tabs[0]:
        st.markdown("### Summary table")

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

        st.dataframe(pivot.reset_index(), use_container_width=True)
        _source_link_line(source_title, source_url)

    # ---------------------- Per-question tabs ----------------------
    for idx, qcode in enumerate(tab_labels, start=1):
        with tabs[idx]:
            qtext = code_to_text.get(qcode, "")
            # ---- (2) Title above each results table (UX) ----
            st.subheader(f"{qcode} — {qtext}")
            st.dataframe(per_q_disp[qcode], use_container_width=True)
            _source_link_line(source_title, source_url)

    # ---------------------- Technical notes tab --------------------
    tech_tab_index = len(tab_titles) - 1
    with tabs[tech_tab_index]:
        st.markdown("### Technical notes")
        st.markdown(
            """
1. **Summary results** are mainly shown as “positive answers,” reflecting the affirmative responses. Positive answers are calculated by removing the "Don't know" and "Not applicable" responses from the total responses.  
2. **Weights/adjustment:** Results have been adjusted for non-response to better represent the target population. Therefore, percentages should not be used to determine the number of respondents within a response category.  
3. **Rounding:** Due to rounding, percentages may not add to 100.  
4. **Suppression:** Results were suppressed for questions with low respondent counts (under 10) and for low response category counts.
            """
        )

    # ---------------------- AI section (unchanged) -----------------------
    if ai_on:
        st.markdown("---")
        st.markdown("## AI Summary")

        cached = _ai_cache_get(ai_key)
        if cached:
            per_q_narratives = cached.get("per_q", {}) or {}
            overall_narrative = cached.get("overall")
            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    st.markdown(f"**{q} — {code_to_text.get(q, '')}**")
                    st.write(txt)
            if overall_narrative and len(tab_labels) > 1:
                st.markdown("**Overall**")
                st.write(overall_narrative)
        else:
            per_q_narratives = {}
            for q in tab_labels:
                dfq = per_q_disp.get(q)
                metric_col = per_q_metric_col.get(q)
                metric_label = per_q_metric_label.get(q)
                qtext = code_to_text.get(q, "")

                with st.spinner(f"AI — analyzing {q}…"):
                    try:
                        content, _hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=build_per_q_prompt(
                                question_code=q,
                                question_text=qtext,
                                df_disp=(dfq.copy(deep=True) if isinstance(dfq, pd.DataFrame) else dfq),
                                metric_col=metric_col,
                                metric_label=metric_label,
                                category_in_play=(demo_selection != "All respondents")
                            )
                        )
                        j = json.loads(content) if content else {}
                        txt = (j.get("narrative") or "").strip()
                    except Exception as e:
                        txt = ""
                        st.warning(f"AI skipped for {q} due to an internal error ({type(e).__name__}).")
                per_q_narratives[q] = txt
                if txt:
                    st.markdown(f"**{q} — {qtext}**")
                    st.write(txt)

            overall_narrative = None
            if len(tab_labels) > 1:
                with st.spinner("AI — synthesizing overall pattern…"):
                    try:
                        content, _hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=build_overall_prompt(
                                tab_labels=tab_labels,
                                pivot_df=pivot.copy(deep=True),
                                q_to_metric={q: per_q_metric_label[q] for q in tab_labels},
                                code_to_text=code_to_text,
                            )
                        )
                    except Exception as e:
                        content = None
                        st.warning(f"AI skipped for Overall due to an internal error ({type(e).__name__}).")
                    try:
                        j = json.loads(content) if content else {}
                        overall_narrative = (j.get("narrative") or "").strip()
                    except Exception:
                        overall_narrative = None

                if overall_narrative:
                    st.markdown("**Overall**")
                    st.write(overall_narrative)

            _ai_cache_put(ai_key, {"per_q": per_q_narratives, "overall": overall_narrative})
            st.session_state["menu1_ai_narr_per_q"] = per_q_narratives
            st.session_state["menu1_ai_narr_overall"] = overall_narrative

        # --- AI Data Validation subsection (under AI Summary) ---
        try:
            _render_data_validation_subsection(
                tab_labels=tab_labels,
                per_q_disp=per_q_disp,
                per_q_metric_col=per_q_metric_col,
                per_q_narratives=per_q_narratives,
            )
        except Exception:
            st.caption("AI Data Validation is unavailable for this run.")

    # ---------------------- Footer: Export + Start new ----------------------
    st.markdown("---")
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
    col_dl, col_new = st.columns([1, 1])

    with col_dl:
        export_per_q = st.session_state.get("menu1_ai_narr_per_q")
        export_overall = st.session_state.get("menu1_ai_narr_overall")

        if ai_on and export_per_q is None:
            cached = _ai_cache_get(ai_key)
            if cached:
                export_per_q = cached.get("per_q", {})
                export_overall = cached.get("overall")
            else:
                try:
                    per_q_disp_ai2: Dict[str, pd.DataFrame] = {}
                    for q in tab_labels:
                        dfq = per_q_disp.get(q)
                        per_q_disp_ai2[q] = (dfq.copy(deep=True) if isinstance(dfq, pd.DataFrame) else dfq)
                    export_per_q, export_overall = _compute_ai_narratives(
                        tab_labels=tab_labels,
                        per_q_disp=per_q_disp_ai2,
                        per_q_metric_col=per_q_metric_col,
                        per_q_metric_label=per_q_metric_label,
                        code_to_text=code_to_text,
                        demo_selection=demo_selection,
                        pivot=pivot.copy(deep=True),
                        build_overall_prompt=build_overall_prompt,
                        build_per_q_prompt=build_per_q_prompt,
                        call_openai_json=call_openai_json,
                    )
                    _ai_cache_put(ai_key, {"per_q": export_per_q, "overall": export_overall})
                except Exception:
                    export_per_q, export_overall = {}, None

        _render_excel_download(
            pivot=pivot,
            per_q_disp=per_q_disp,
            tab_labels=tab_labels,
            per_q_narratives=(export_per_q or {}),
            overall_narrative=(export_overall if (export_overall and len(tab_labels) > 1) else None),
        )

    with col_new:
        if st.button("Start a new search", key="menu1_new_search"):
            _prev_ai_toggle = st.session_state.get("menu1_ai_toggle")

            try:
                from .. import state
                state.reset_menu1_state()
            except Exception:
                for k in [
                    "menu1_selected_codes", "menu1_multi_questions",
                    # "menu1_ai_toggle",  # do NOT clear here; we preserve
                    "menu1_show_diag",
                    "select_all_years", "demo_main", "last_query_info",
                ]:
                    st.session_state.pop(k, None)
            for k in [
                "menu1_hits", "menu1_hit_codes_selected",
                "menu1_search_done", "menu1_last_search_query",
                "menu1_kw_query",
            ]:
                st.session_state.pop(k, None)
            for k in list(st.session_state.keys()):
                if k.startswith("kwhit_") or k.startswith("sel_"):
                    st.session_state.pop(k, None)
            st.session_state.pop("menu1_global_hits_selected", None)
            st.session_state.pop("menu1_ai_cache", None)
            st.session_state.pop("menu1_ai_narr_per_q", None)
            st.session_state.pop("menu1_ai_narr_overall", None)

            if _prev_ai_toggle is not None:
                st.session_state["menu1_ai_toggle"] = _prev_ai_toggle

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
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            pivot.reset_index().to_excel(writer, sheet_name="Summary_Table", index=False)
            for q, df_disp in per_q_disp.items():
                safe = q[:28]
                df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)
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
