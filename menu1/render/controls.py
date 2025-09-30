# menu1/render/controls.py
from __future__ import annotations
from typing import List, Optional, Set, Dict
import re
import pandas as pd
import streamlit as st

__all__ = [
    "question_picker",
    "year_picker",
    "demographic_picker",
    "search_button_enabled",
]

# Try to import the hybrid search (your local util). Safe fallback if missing.
try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore

# -----------------------------
# Session-state keys (Menu 1)
# -----------------------------
K_MULTI_QUESTIONS   = "menu1_multi_questions"        # List[str] (display labels) chosen in the dropdown
K_SELECTED_CODES    = "menu1_selected_codes"         # Ordered List[str] of final choices (max 5)
K_KW_QUERY          = "menu1_kw_query"               # Text in Box 2
K_HITS              = "menu1_hits"                   # List[{"code","text","display","score"}]
K_LAST_QUERY        = "menu1_last_search_query"      # Last searched string
K_PREV_MULTI        = "menu1_prev_multi_snapshot"    # Snapshot of Box 1 display list
K_REQ_CLEAR_KW      = "menu1_request_clear_kw"       # Flag: clear Box 2 input before rendering
K_EMPTY_WARN        = "menu1_empty_search_warn"      # Show “enter keywords” warning
K_PENDING_REMOVE    = "menu1_pending_remove_from_multi"  # Displays to remove next run
K_TO_UNCHECK_HITS   = "menu1_to_uncheck_hit_keys"    # Hit-checkbox keys to uncheck next run

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Threshold / limits
MIN_SCORE = 0.40
MAX_Q = 5

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _dedupe_hits(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["code", "text", "display", "score"])
    out = df.copy()
    if "score" not in out.columns:
        out["score"] = 0.0
    out["code"] = out["code"].astype(str)
    out = out.sort_values(["score", "code"], ascending=[False, True], kind="mergesort")
    out = out.drop_duplicates(subset=["code"], keep="first")
    try:
        out = out[out["score"] > MIN_SCORE]
    except Exception:
        pass
    return out.reset_index(drop=True)

def _run_keyword_search(qdf: pd.DataFrame, query: str, top_k: int = 120) -> pd.DataFrame:
    if callable(hybrid_question_search):
        try:
            hits_df = hybrid_question_search(qdf, query, top_k=top_k, min_score=MIN_SCORE)  # type: ignore
            if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                return _dedupe_hits(hits_df).head(top_k)
        except Exception:
            pass
    return pd.DataFrame(columns=["code", "text", "display", "score"])

# -----------------------------------------------------------------------------
# Main control: Box 1 + Box 2 + Selected section
# -----------------------------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    # Initialize state BEFORE widgets render
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_LAST_QUERY, "")
    st.session_state.setdefault(K_PREV_MULTI, None)
    st.session_state.setdefault(K_REQ_CLEAR_KW, False)
    st.session_state.setdefault(K_EMPTY_WARN, False)
    st.session_state.setdefault(K_PENDING_REMOVE, [])
    st.session_state.setdefault(K_TO_UNCHECK_HITS, [])

    # Apply any scheduled unchecks for hit checkboxes BEFORE rendering them
    if st.session_state[K_TO_UNCHECK_HITS]:
        for key in list(st.session_state[K_TO_UNCHECK_HITS]):
            st.session_state[key] = False
        st.session_state[K_TO_UNCHECK_HITS] = []

    # If requested, clear the keyword input BEFORE creating the text_input
    if st.session_state.get(K_REQ_CLEAR_KW, False):
        st.session_state[K_KW_QUERY] = ""
        st.session_state[K_REQ_CLEAR_KW] = False

    # If Box 1 exists and there are pending removals, apply them BEFORE drawing the widget
    if K_MULTI_QUESTIONS in st.session_state and st.session_state[K_PENDING_REMOVE]:
        current = st.session_state.get(K_MULTI_QUESTIONS, [])
        if isinstance(current, list):
            remove_set = set(st.session_state[K_PENDING_REMOVE])
            st.session_state[K_MULTI_QUESTIONS] = [d for d in current if d not in remove_set]
        st.session_state[K_PENDING_REMOVE] = []

    # Build maps
    code_to_text = dict(zip(qdf["code"], qdf["text"]))
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # Small style for subtitles
    st.markdown("""
        <style>
            .sub-title { font-weight: 700; font-size: 0.95rem; margin: 0.25rem 0 0.25rem 0; }
            .tight-gap { margin-top: 0.15rem; margin-bottom: 0.35rem; }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Box 1: multiselect ----------
    st.markdown('<div class="sub-title">Choose a question from the list below</div>', unsafe_allow_html=True)

    # Prepare list with Q01 first if present
    all_displays = qdf["display"].tolist()
    q01_disp = code_to_display.get("Q01") or code_to_display.get("Q1")
    if q01_disp and q01_disp in all_displays:
        all_displays.remove(q01_disp)
        all_displays.insert(0, q01_disp)
    first_disp = all_displays[0] if all_displays else ""

    st.multiselect(
        "Choose one or more from the official list",
        all_displays,
        max_selections=MAX_Q,
        label_visibility="collapsed",
        key=K_MULTI_QUESTIONS,
        placeholder=first_disp,  # not preselected, just a hint
    )

    # Track codes from Box 1
    current_multi_displays: List[str] = st.session_state.get(K_MULTI_QUESTIONS, [])
    current_multi_codes = [display_to_code[d] for d in current_multi_displays if d in display_to_code]

    # Detect removal from Box 1 and reflect in selected list
    prev_multi = st.session_state.get(K_PREV_MULTI)
    if prev_multi is None:
        st.session_state[K_PREV_MULTI] = list(current_multi_displays)
    elif prev_multi != current_multi_displays:
        removed_displays = set(prev_multi) - set(current_multi_displays)
        if removed_displays:
            removed_codes = {display_to_code.get(d) for d in removed_displays if d in display_to_code}
            st.session_state[K_SELECTED_CODES] = [c for c in st.session_state[K_SELECTED_CODES] if c not in removed_codes]
        # Reset Box 2 state when Box 1 changes
        st.session_state[K_LAST_QUERY] = ""
        st.session_state[K_EMPTY_WARN] = False
        st.session_state[K_REQ_CLEAR_KW] = True
        st.session_state[K_PREV_MULTI] = list(current_multi_displays)

    # ---------- OR ----------
    st.markdown('<div class="sub-title">or</div>', unsafe_allow_html=True)

    # ---------- Box 2: search ----------
    st.markdown('<div class="sub-title tight-gap">Search questionnaire by keywords or theme</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Enter keywords",
        key=K_KW_QUERY,
        label_visibility="collapsed",
        placeholder='Type keywords like “career advancement”, “harassment”, “recognition”…',
    )

    def _on_click_search():
        q = (st.session_state.get(K_KW_QUERY, "") or "").strip()
        if not q:
            st.session_state[K_EMPTY_WARN] = True
            return
        hits_df = _run_keyword_search(qdf, q, top_k=120)
        st.session_state[K_LAST_QUERY] = q
        st.session_state[K_HITS] = hits_df[["code", "text", "display", "score"]].to_dict(orient="records") \
                                   if isinstance(hits_df, pd.DataFrame) and not hits_df.empty else []
        st.session_state[K_EMPTY_WARN] = False
        st.session_state[K_REQ_CLEAR_KW] = True  # clear input next render

    st.button("Search the questionnaire", key="menu1_do_kw_search", on_click=_on_click_search)

    if st.session_state.get(K_EMPTY_WARN, False):
        st.warning("Please enter one or more keywords to search the questionnaire.")

    # Render hits; allow multiple checks
    hits = st.session_state.get(K_HITS, [])
    selected_from_hits: List[str] = []
    if hits:
        st.write(f"Top {len(hits)} matches meeting the quality threshold:")
        for rec in hits:
            code = rec["code"]; text = rec["text"]
            key = f"kwhit_{code}"
            checked = st.checkbox(f"{code} — {text}", key=key)
            if checked:
                selected_from_hits.append(code)

    # ---------- Merge and enforce 5-cap ----------
    prev_selected: List[str] = list(st.session_state.get(K_SELECTED_CODES, []))
    merged: List[str] = list(prev_selected)

    # Add Box 1 selections
    for c in current_multi_codes:
        if c not in merged:
            merged.append(c)
    # Add Box 2 checks
    for c in selected_from_hits:
        if c not in merged:
            merged.append(c)

    if len(merged) > MAX_Q:
        # Keep existing; allow only remaining new ones; revert others
        remaining = MAX_Q - len(prev_selected)
        remaining = max(0, remaining)
        new_box1 = [c for c in current_multi_codes if c not in prev_selected]
        new_hits = [c for c in selected_from_hits if c not in prev_selected]
        accepted = set(new_box1[:remaining])
        if remaining > len(accepted):
            accepted |= set(new_hits[: (remaining - len(accepted))])
        final_list = list(prev_selected)
        overflow = []

        for c in new_box1 + new_hits:
            if c in accepted and c not in final_list:
                final_list.append(c)
            elif c not in accepted:
                overflow.append(c)

        # Queue UI reverts for overflow
        #  - hit boxes → uncheck next run
        st.session_state[K_TO_UNCHECK_HITS] = list(set(st.session_state.get(K_TO_UNCHECK_HITS, [])) |
                                                   {f"kwhit_{c}" for c in overflow})
        #  - multiselect → remove displays next run
        for c in overflow:
            # find display label
            disp = None
            for d, cc in code_to_display.items():
                if cc == c:
                    disp = d; break
            if disp:
                st.session_state[K_PENDING_REMOVE] = list(set(st.session_state.get(K_PENDING_REMOVE, [])) | {disp})

        st.session_state[K_SELECTED_CODES] = final_list
        if overflow:
            st.warning(f"Limit is {MAX_Q} questions; extra selections were ignored.")
    else:
        st.session_state[K_SELECTED_CODES] = merged

    # If user checked any hit, clear hit list so they can do a fresh search next
    if selected_from_hits:
        st.session_state[K_HITS] = []

    # ---------- Selected list with quick unselect ----------
    if st.session_state[K_SELECTED_CODES]:
        st.markdown('<div class="sub-title">Selected questions</div>', unsafe_allow_html=True)
        updated = list(st.session_state[K_SELECTED_CODES])
        cols = st.columns(min(MAX_Q, len(updated)))
        for idx, code in enumerate(list(updated)):
            with cols[idx % len(cols)]:
                # Find its display label
                disp = next((d for d, cc in code_to_display.items() if cc == code), code)
                keep = st.checkbox(disp, value=True, key=f"sel_{code}")
                if not keep:
                    updated = [c for c in updated if c != code]
                    # Uncheck any hit box for that code next run
                    st.session_state[K_TO_UNCHECK_HITS] = list(set(st.session_state.get(K_TO_UNCHECK_HITS, [])) | {f"kwhit_{code}"})
                    # Remove from multiselect next run
                    st.session_state[K_PENDING_REMOVE] = list(set(st.session_state.get(K_PENDING_REMOVE, [])) | {disp})

        if updated != st.session_state[K_SELECTED_CODES]:
            st.session_state[K_SELECTED_CODES] = updated
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    return st.session_state[K_SELECTED_CODES]

# -----------------------------------------------------------------------------
# Years selector
# -----------------------------------------------------------------------------
def year_picker() -> List[int]:
    st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
    st.session_state.setdefault(K_SELECT_ALL_YEARS, True)
    select_all = st.checkbox("All years", key=K_SELECT_ALL_YEARS)

    selected_years: List[int] = []
    year_cols = st.columns(len(DEFAULT_YEARS))
    for idx, yr in enumerate(DEFAULT_YEARS):
        with year_cols[idx]:
            default_checked = True if select_all else st.session_state.get(f"year_{yr}", False)
            if st.checkbox(str(yr), value=default_checked, key=f"year_{yr}"):
                selected_years.append(yr)
    return sorted(selected_years)

# -----------------------------------------------------------------------------
# Demographic picker
# -----------------------------------------------------------------------------
def demographic_picker(demo_df: pd.DataFrame):
    st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
    st.session_state.setdefault("demo_main", "All respondents")
    demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

    sub_selection = None
    if demo_selection != "All respondents":
        st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
        sub_items = demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL].dropna().astype(str).unique().tolist()
        sub_items = sorted(sub_items)
        sub_key = f"sub_{demo_selection.replace(' ', '_')}"
        sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=sub_key, label_visibility="collapsed")
        if sub_selection == "":
            sub_selection = None

    # Resolve codes + display map
    demcodes, disp_map, category_in_play = _resolve_demcodes(demo_df, demo_selection, sub_selection)
    return demo_selection, sub_selection, demcodes, disp_map, category_in_play

def _resolve_demcodes(demo_df: pd.DataFrame, category_label: str, subgroup_label: Optional[str]):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            r = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not r.empty:
                code = str(r.iloc[0][code_col])
                return [code], {code: subgroup_label}, True
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        return labels, {l: l for l in labels}, True

    return [None], {None: "All respondents"}, False

# -----------------------------------------------------------------------------
# Query button enablement
# -----------------------------------------------------------------------------
def search_button_enabled(question_codes: List[str], years: List[int]) -> bool:
    return bool(question_codes) and bool(years)
