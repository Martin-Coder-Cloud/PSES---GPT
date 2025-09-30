# menu1/render/controls.py
"""
Controls for Menu 1:
- Question picker: dropdown multi-select (authoritative) + keyword search (hybrid)
- Selected list with quick unselect checkboxes
- Years selector
- Demographic category & subgroup selector
- Search button enablement helper

Behavior (fixed/polished):
  • Box 1: shows first item's label (Q01 if present) as placeholder (not selected)
  • Box 2: Search button is ALWAYS visible; empty click warns; no Enter needed
  • Multiple searches can be performed; selections persist across searches
  • Selected questions section always visible, feeds the query; max 5 enforced
  • If limit exceeded, overflow items are ignored with a warning; any hit
    checkboxes for overflow are auto-unchecked on next run; any Box 1 overflow
    is auto-removed from the multiselect on next run
  • Unselecting in Box 1 removes from Selected questions
  • Unselecting in Selected questions removes from Box 1 and unchecks any hit
  • Avoids “cannot be modified after widget is instantiated” via pre-render flags
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import re
import pandas as pd
import streamlit as st

# Try to import the hybrid search; provide a robust fallback as well
try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore

# -----------------------------
# Session-state keys (Menu 1)
# -----------------------------
K_MULTI_QUESTIONS    = "menu1_multi_questions"        # List[str] (display labels) chosen in the dropdown
K_SELECTED_CODES     = "menu1_selected_codes"         # Ordered List[str] of codes (persistent)
K_KW_QUERY           = "menu1_kw_query"               # str
K_HITS               = "menu1_hits"                   # List[{"code","text","display","score"}]
K_LAST_QUERY         = "menu1_last_search_query"      # str
K_PREV_MULTI         = "menu1_prev_multi_snapshot"    # snapshot of Box 1 value for change detection
K_REQ_CLEAR_KW       = "menu1_request_clear_kw"       # bool: clear KW input before widget renders
K_EMPTY_WARN         = "menu1_empty_search_warn"      # bool: show “enter keywords” warning after click
K_PENDING_REMOVE     = "menu1_pending_remove_from_multi"  # List[str] displays to remove from Box 1 next run
K_TO_UNCHECK_HITS    = "menu1_to_uncheck_hit_keys"    # List[str] checkbox keys to uncheck next run
K_PREV_HIT_KEYS      = "menu1_prev_hit_keys"          # Set[str] previous hit checkbox keys (optional)

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Threshold (fixed)
MIN_SCORE = 0.40  # keep > 0.40 filter in both engines
MAX_Q = 5

# -----------------------------------------------------------------------------
# Internal regex/token helpers (fallback search)
# -----------------------------------------------------------------------------
_word_re = re.compile(r"[a-z0-9']+")

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _tokens(s: str) -> List[str]:
    return _word_re.findall(_normalize(s))

# -----------------------------------------------------------------------------
# Utilities for dedupe and wrapped call
# -----------------------------------------------------------------------------
def _dedupe_hits(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["code", "text", "display", "score"])
    out = df.copy()
    if "score" not in out.columns:
        out["score"] = 0.0
    out["code"] = out["code"].astype(str)
    out = out.sort_values("score", ascending=False, kind="mergesort")
    out = out.drop_duplicates(subset=["code"], keep="first")
    try:
        out = out[out["score"] > MIN_SCORE]
    except Exception:
        pass
    return out.reset_index(drop=True)

def _run_keyword_search(qdf: pd.DataFrame, query: str, top_k: int = 120) -> pd.DataFrame:
    """
    Calls utils.hybrid_search.hybrid_question_search with fixed threshold.
    (That function can use embeddings if available; no external API.)
    """
    if callable(hybrid_question_search):
        try:
            hits_df = hybrid_question_search(qdf, query, top_k=top_k, min_score=MIN_SCORE)  # type: ignore
            if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                return _dedupe_hits(hits_df).head(top_k)
        except Exception:
            pass
    return pd.DataFrame(columns=["code", "text", "display", "score"])

# -----------------------------------------------------------------------------
# Question picker (dropdown + keyword search) → returns List[str] (codes)
# -----------------------------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    """
    Returns ordered list of selected question codes (max 5).
    """
    # ---- Ensure session defaults BEFORE widgets ----
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_LAST_QUERY, "")
    st.session_state.setdefault(K_PREV_MULTI, None)
    st.session_state.setdefault(K_REQ_CLEAR_KW, False)
    st.session_state.setdefault(K_EMPTY_WARN, False)
    st.session_state.setdefault(K_PENDING_REMOVE, [])
    st.session_state.setdefault(K_TO_UNCHECK_HITS, [])
    st.session_state.setdefault(K_PREV_HIT_KEYS, set())

    # Apply any pending unchecks for hit checkboxes BEFORE rendering them
    if st.session_state[K_TO_UNCHECK_HITS]:
        for key in list(st.session_state[K_TO_UNCHECK_HITS]):
            st.session_state[key] = False
        st.session_state[K_TO_UNCHECK_HITS] = []

    # If Box 1 has been created before and there are pending removals, apply them BEFORE rendering the widget
    if K_MULTI_QUESTIONS in st.session_state and st.session_state[K_PENDING_REMOVE]:
        current = st.session_state.get(K_MULTI_QUESTIONS, [])
        if isinstance(current, list):
            to_remove = set(st.session_state[K_PENDING_REMOVE])
            st.session_state[K_MULTI_QUESTIONS] = [d for d in current if d not in to_remove]
        st.session_state[K_PENDING_REMOVE] = []

    # Clear the keyword input BEFORE rendering if requested
    if st.session_state.get(K_REQ_CLEAR_KW, False):
        st.session_state[K_KW_QUERY] = ""
        st.session_state[K_REQ_CLEAR_KW] = False

    # Mappings
    code_to_text = dict(zip(qdf["code"], qdf["text"]))
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # Scoped styles
    st.markdown("""
        <style>
            .sub-title { font-weight: 700; font-size: 0.95rem; margin: 0.25rem 0 0.25rem 0; }
            .tight-gap { margin-top: 0.15rem; margin-bottom: 0.35rem; }
        </style>
    """, unsafe_allow_html=True)

    # ---------- 1) Box 1: Dropdown multi-select ----------
    st.markdown('<div class="sub-title">Choose a question from the list below</div>', unsafe_allow_html=True)

    # Build list with Q01 forced to the top if present (not auto-selecting)
    all_displays = qdf["display"].tolist()
    q01_disp = code_to_display.get("Q01") or code_to_display.get("Q1")
    if q01_disp and q01_disp in all_displays:
        all_displays.remove(q01_disp)
        all_displays.insert(0, q01_disp)

    # Placeholder shows the first item (Q01 if present) WITHOUT selecting it
    first_disp = all_displays[0] if all_displays else ""
    st.multiselect(
        "Choose one or more from the official list",
        all_displays,
        max_selections=MAX_Q,
        label_visibility="collapsed",
        key=K_MULTI_QUESTIONS,                 # Streamlit holds value in session_state
        placeholder=first_disp,                # show first item label (not selected)
        # no 'default=' to avoid conflicts/warnings
    )

    # Codes from current Box 1 selection
    current_multi_displays: List[str] = st.session_state.get(K_MULTI_QUESTIONS, [])
    current_multi_codes: List[str] = []
    for d in current_multi_displays:
        c = display_to_code.get(d)
        if c:
            current_multi_codes.append(c)

    # If the Box 1 selection changed, drop any codes that were removed from Box 1
    prev_multi = st.session_state.get(K_PREV_MULTI)
    if prev_multi is None:
        st.session_state[K_PREV_MULTI] = list(current_multi_displays)
    elif prev_multi != current_multi_displays:
        removed_displays = set(prev_multi) - set(current_multi_displays)
        if removed_displays:
            removed_codes = {display_to_code.get(d) for d in removed_displays if d in display_to_code}
            st.session_state[K_SELECTED_CODES] = [c for c in st.session_state[K_SELECTED_CODES] if c not in removed_codes]
        # Clearing search input/results when Box 1 changed is fine
        st.session_state[K_LAST_QUERY] = ""
        st.session_state[K_EMPTY_WARN] = False
        st.session_state[K_REQ_CLEAR_KW] = True
        st.session_state[K_PREV_MULTI] = list(current_multi_displays)

    # ---------- "or" ----------
    st.markdown('<div class="sub-title">or</div>', unsafe_allow_html=True)

    # ---------- 2) Box 2: Search input + always-visible button ----------
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
        st.session_state[K_REQ_CLEAR_KW] = True  # clear input before next render

    st.button("Search the questionnaire", key="menu1_do_kw_search", on_click=_on_click_search)

    if st.session_state.get(K_EMPTY_WARN, False):
        st.warning("Please enter one or more keywords to search the questionnaire.")

    # Render hits; allow multiple to be checked before applying the 5-cap
    hits = st.session_state.get(K_HITS, [])
    selected_from_hits: List[str] = []
    overflow_hit_keys: List[str] = []

    if hits:
        st.write(f"Top {len(hits)} matches meeting the quality threshold:")
        for rec in hits:
            code = rec["code"]; text = rec["text"]
            label = f"{code} — {text}"
            key = f"kwhit_{code}"   # stable key per code
            checked = st.checkbox(label, key=key)
            if checked:
                selected_from_hits.append(code)

    # ---------- Merge & enforce the cap of 5 (persistent selection) ----------
    prev_selected: List[str] = list(st.session_state.get(K_SELECTED_CODES, []))

    # Start from existing order; add Box 1 codes first (as authoritative add source)
    merged: List[str] = [c for c in prev_selected if c in prev_selected]  # copy
    # Add codes newly present in Box 1
    for c in current_multi_codes:
        if c not in merged:
            merged.append(c)
    # Add codes checked from hits
    for c in selected_from_hits:
        if c not in merged:
            merged.append(c)

    # Enforce max
    if len(merged) > MAX_Q:
        # Determine which were newly added this run (in order) beyond remaining slots
        remaining = MAX_Q - len(prev_selected)
        if remaining < 0:
            remaining = 0
        # Identify the true “new candidates”: Box1 adds (not in prev), then hit adds (not in prev)
        new_candidates: List[str] = [c for c in current_multi_codes if c not in prev_selected]
        new_candidates += [c for c in selected_from_hits if c not in prev_selected and c not in new_candidates]

        # Accept only up to 'remaining' new items
        accept_set: Set[str] = set(new_candidates[:remaining])
        # Build final list: keep all prev_selected, then only accepted new ones
        final_list: List[str] = list(prev_selected)
        for c in new_candidates:
            if c in accept_set and c not in final_list:
                final_list.append(c)
            elif c not in accept_set:
                # prepare uncheck/removal for overflow:
                #  - if came from hits, uncheck its checkbox next run
                overflow_hit_keys.append(f"kwhit_{c}")
                #  - if came from Box 1, queue its display label for removal next run
                #    (so the multiselect doesn't show it as chosen)
                # we can infer if it came from Box 1 by presence in current_multi_codes
                if c in current_multi_codes:
                    # find display label
                    # build display map locally
                    # (we have display_to_code above)
                    # reverse map:
                    # NOTE: multiple displays cannot map to same code here
                    for d, code in display_to_code.items():
                        if code == c:
                            st.session_state[K_PENDING_REMOVE] = list(set(st.session_state.get(K_PENDING_REMOVE, [])) | {d})
                            break
        st.session_state[K_SELECTED_CODES] = final_list
        if new_candidates and len(new_candidates) > remaining:
            st.warning(f"Limit is {MAX_Q} questions; extra selections were ignored.")
        # queue unchecks
        if overflow_hit_keys:
            st.session_state[K_TO_UNCHECK_HITS] = list(set(st.session_state.get(K_TO_UNCHECK_HITS, [])) | set(overflow_hit_keys))
    else:
        st.session_state[K_SELECTED_CODES] = merged

    # ---------- Selected list with quick unselect ----------
    if st.session_state[K_SELECTED_CODES]:
        st.markdown('<div class="sub-title">Selected questions</div>', unsafe_allow_html=True)
        updated = list(st.session_state[K_SELECTED_CODES])
        cols = st.columns(min(MAX_Q, len(updated)))
        for idx, code in enumerate(list(updated)):
            with cols[idx % len(cols)]:
                # display label
                disp = None
                for d, c in display_to_code.items():
                    if c == code:
                        disp = d
                        break
                label = disp or code
                keep = st.checkbox(label, value=True, key=f"sel_{code}")
                if not keep:
                    # remove from selected list
                    updated = [c for c in updated if c != code]
                    # uncheck hit checkbox (next run)
                    st.session_state[K_TO_UNCHECK_HITS] = list(set(st.session_state.get(K_TO_UNCHECK_HITS, [])) | {f"kwhit_{code}"})
                    # remove from multiselect (next run) if present
                    if disp:
                        st.session_state[K_PENDING_REMOVE] = list(set(st.session_state.get(K_PENDING_REMOVE, [])) | {disp})

        if updated != st.session_state[K_SELECTED_CODES]:
            st.session_state[K_SELECTED_CODES] = updated
            # also clear any empty warnings; keep hits as-is so the user can add more if needed
            st.session_state[K_EMPTY_WARN] = False
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    return st.session_state[K_SELECTED_CODES]

# -----------------------------------------------------------------------------
# Years selector → returns List[int]
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
# Demographic picker → returns (demo_selection, sub_selection, demcodes, disp_map, category_in_play)
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

    # Resolve codes
    demcodes, disp_map, category_in_play = _resolve_demcodes(demo_df, demo_selection, sub_selection)
    return demo_selection, sub_selection, demcodes, disp_map, category_in_play

# -----------------------------------------------------------------------------
# Internal helper: resolve demographic codes & display map
# -----------------------------------------------------------------------------
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
# Search button enabled?
# -----------------------------------------------------------------------------
def search_button_enabled(question_codes: List[str], years: List[int]) -> bool:
    return bool(question_codes) and bool(years)
