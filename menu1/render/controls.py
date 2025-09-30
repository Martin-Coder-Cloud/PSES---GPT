# menu1/render/controls.py
"""
Controls for Menu 1:
- Question picker: dropdown multi-select (authoritative) + keyword search (hybrid)
- Selected list with quick unselect checkboxes
- Years selector
- Demographic category & subgroup selector
- Search button enablement helper

UI tweaks kept:
  • Subtitle above the multiselect: "Choose a question from the list below"
  • Multiselect placeholder is empty (doesn't duplicate the subtitle)
  • Keyword search button is always enabled; warns if textbox is empty
  • "or" label for visual separation
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
K_MULTI_QUESTIONS = "menu1_multi_questions"   # List[str] of "display" labels picked in the dropdown
K_SELECTED_CODES  = "menu1_selected_codes"    # Ordered List[str] of codes (multi + hits merged)
K_KW_QUERY        = "menu1_kw_query"          # str
K_HITS            = "menu1_hits"              # List[{"code","text","display","score"}]
K_FIND_HITS_BTN   = "menu1_find_hits"         # button key
K_SEARCH_DONE     = "menu1_search_done"       # bool: did user click Search the questionnaire?
K_LAST_QUERY      = "menu1_last_search_query" # str: what query was used last time
K_HITS_SELECTED   = "menu1_hit_codes_selected"# List[str]: codes ticked from hits (persistent)

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Threshold (fixed)
MIN_SCORE = 0.40  # keep > 0.40 filter in both engines

# -----------------------------------------------------------------------------
# Internal regex/token helpers (fallbacks)
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
    out = out.sort_values("score", ascending=False)
    out = out.drop_duplicates(subset=["code"], keep="first")
    try:
        out = out[out["score"] > MIN_SCORE]  # strictly greater than 0.40
    except Exception:
        pass
    return out

def _run_keyword_search(qdf: pd.DataFrame, query: str, top_k: int = 120) -> pd.DataFrame:
    """
    Calls utils.hybrid_search.hybrid_question_search with fixed threshold.
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
    UI:
      1) Dropdown multi-select (authoritative, max 5) with separate subtitle above the box
      2) Keyword search section with persistent results + multi-tick checkboxes
      3) "Selected questions" list with quick unselect checkboxes

    Returns ordered list of selected question codes (max 5).
    """
    # Ensure session defaults (initialize BEFORE widget, never assign after render)
    st.session_state.setdefault(K_MULTI_QUESTIONS, [])
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_SEARCH_DONE, False)
    st.session_state.setdefault(K_LAST_QUERY, "")
    st.session_state.setdefault(K_HITS_SELECTED, [])  # persist checked hits

    # Mappings
    code_to_text = dict(zip(qdf["code"], qdf["text"]))
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # ---------- 1) Dropdown multi-select ----------
    st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)
    # Subtitle above the multiselect; increase top margin for better separation
    st.markdown('<div style="margin: 8px 0 4px 0; font-weight:600; color:#222;">Choose a question from the list below</div>', unsafe_allow_html=True)

    all_displays = qdf["display"].tolist()
    st.multiselect(
        "Choose one or more from the official list",
        all_displays,
        max_selections=5,
        label_visibility="collapsed",
        key=K_MULTI_QUESTIONS,
        placeholder="",  # empty to avoid duplicating the subtitle inside the box
    )
    selected_from_multi: Set[str] = set(display_to_code[d] for d in st.session_state[K_MULTI_QUESTIONS] if d in display_to_code)

    # ---------- Divider: "or" ----------
    st.markdown("""
        <div style="
            margin: .1rem 0 .1rem .5rem;
            font-size: 0.9rem; font-weight: 600;
            color: rgba(49,51,63,.8); font-family: inherit;
        ">or</div>
    """, unsafe_allow_html=True)

    # ---------- 2) Keyword search ----------
    # Title above the input
    st.markdown("<div class='field-label'>Search questionnaire by keywords or theme</div>", unsafe_allow_html=True)

    # Search input + button (button ALWAYS enabled; warns if empty)
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input(
            "Enter keywords (e.g., harassment, recognition, onboarding)",
            key=K_KW_QUERY,
            label_visibility="collapsed",
            placeholder="Type keywords like “career advancement”, “harassment”, “recognition”…",
        )
    with c2:
        if st.button("Search the questionnaire", key=K_FIND_HITS_BTN):
            q = (query or "").strip()
            if not q:
                st.warning("Please enter at least one keyword to search the questionnaire.")
            else:
                hits_df = _run_keyword_search(qdf, q, top_k=120)
                st.session_state[K_SEARCH_DONE] = True
                st.session_state[K_LAST_QUERY] = q
                st.session_state[K_HITS] = hits_df[["code", "text", "display", "score"]].to_dict(orient="records") \
                                           if isinstance(hits_df, pd.DataFrame) and not hits_df.empty else []
                # Keep only still-valid checked hits after a new search
                current_codes = {h["code"] for h in st.session_state[K_HITS]}
                st.session_state[K_HITS_SELECTED] = [c for c in st.session_state[K_HITS_SELECTED] if c in current_codes]

    # Results area (persistent across reruns until a new search or reset)
    hits = st.session_state.get(K_HITS, [])
    if st.session_state.get(K_SEARCH_DONE, False):
        if not hits:
            q = (st.session_state.get(K_LAST_QUERY) or "").strip()
            safe_q = q if q else "your search"
            st.warning(
                f"No questions matched “{safe_q}”. "
                "Try broader or different keywords (e.g., synonyms), split phrases (e.g., “career advancement” → “career”), "
                "or search by a question code like “Q01”."
            )
        else:
            st.write(f"Top {len(hits)} matches meeting the quality threshold:")

    # Multi-tick checkboxes for hits (do NOT clear on additional selections)
    selected_from_hits: Set[str] = set()
    if hits:
        for rec in hits:
            code = rec["code"]; text = rec["text"]
            label = f"{code} — {text}"
            key = f"kwhit_{code}"
            default_checked = (code in st.session_state[K_HITS_SELECTED]) or (code in selected_from_multi)
            checked = st.checkbox(label, value=default_checked, key=key)
            # Maintain persistent selection list explicitly
            if checked and code not in st.session_state[K_HITS_SELECTED]:
                st.session_state[K_HITS_SELECTED].append(code)
            if (not checked) and code in st.session_state[K_HITS_SELECTED]:
                st.session_state[K_HITS_SELECTED] = [c for c in st.session_state[K_HITS_SELECTED] if c != code]

        selected_from_hits = set(st.session_state[K_HITS_SELECTED])

    # ---------- Merge selections (dropdown first, then hits), cap at 5 ----------
    combined_order: List[str] = []
    for d in st.session_state.get(K_MULTI_QUESTIONS, []):
        c = display_to_code.get(d)
        if c and c not in combined_order:
            combined_order.append(c)
    for c in selected_from_hits:
        if c not in combined_order:
            combined_order.append(c)
    if len(combined_order) > 5:
        combined_order = combined_order[:5]
        st.warning("Limit is 5 questions; extra selections were ignored.")
    st.session_state[K_SELECTED_CODES] = combined_order

    # ---------- 3) Selected list with quick unselect ----------
    if st.session_state[K_SELECTED_CODES]:
        st.markdown('<div class="field-label">Selected questions:</div>', unsafe_allow_html=True)
        updated = list(st.session_state[K_SELECTED_CODES])
        cols = st.columns(min(5, len(updated)))
        for idx, code in enumerate(list(updated)):
            with cols[idx % len(cols)]:
                label = code_to_display.get(code, code)
                keep = st.checkbox(label, value=True, key=f"sel_{code}")
                if not keep:
                    # remove from current selection
                    updated = [c for c in updated if c != code]
                    # uncheck corresponding search hit and remove from persistent list
                    hk = f"kwhit_{code}"
                    if hk in st.session_state:
                        st.session_state[hk] = False
                    if code in st.session_state[K_HITS_SELECTED]:
                        st.session_state[K_HITS_SELECTED] = [c for c in st.session_state[K_HITS_SELECTED] if c != code]
                    # remove from dropdown selected list
                    disp = code_to_display.get(code)
                    if disp:
                        st.session_state[K_MULTI_QUESTIONS] = [d for d in st.session_state[K_MULTI_QUESTIONS] if d != disp]
        if updated != st.session_state[K_SELECTED_CODES]:
            st.session_state[K_SELECTED_CODES] = updated

    return st.session_state[K_SELECTED_CODES]

# -----------------------------------------------------------------------------
# Years selector → returns List[int]
# -----------------------------------------------------------------------------
def year_picker() -> List[int]:
    """
    Avoids passing value= to checkboxes that also use st.session_state,
    preventing the 'default value + Session State' warning.
    """
    st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)

    # Master toggle
    st.session_state.setdefault(K_SELECT_ALL_YEARS, True)
    select_all = st.checkbox("All years", key=K_SELECT_ALL_YEARS)

    # Establish per-year defaults BEFORE rendering year checkboxes
    if select_all:
        # When "All years" is on, force all True (setdefault + explicit True)
        for yr in DEFAULT_YEARS:
            st.session_state.setdefault(f"year_{yr}", True)
            st.session_state[f"year_{yr}"] = True
    else:
        # When "All years" is off, ensure keys exist but do not force a value
        for yr in DEFAULT_YEARS:
            st.session_state.setdefault(f"year_{yr}", False)

    # Render the year checkboxes WITHOUT a value= param
    selected_years: List[int] = []
    year_cols = st.columns(len(DEFAULT_YEARS))
    for idx, yr in enumerate(DEFAULT_YEARS):
        with year_cols[idx]:
            st.checkbox(str(yr), key=f"year_{yr}")
            if st.session_state.get(f"year_{yr}", False):
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
