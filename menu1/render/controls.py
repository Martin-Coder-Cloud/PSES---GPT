# app/menu1/render/controls.py
"""
Menu 1 input controls:
- Question picker (authoritative multiselect + optional keyword search)
- Year picker (with 'All years' master checkbox)
- Demographic picker (category + optional subgroup), with DEMCODE resolution
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import streamlit as st

# Local imports (stable keys & constants)
from ..state import (
    K_SELECTED_CODES, K_MULTI_QUESTIONS, K_KW_QUERY, K_HITS, K_FIND_HITS_BTN,
    K_SELECT_ALL_YEARS, K_DEMO_MAIN, YEAR_KEYS, SELECTED_PREFIX, HIT_PREFIX, SUBGROUP_PREFIX
)
from ..constants import MAX_QUESTIONS, DEFAULT_YEARS

# Optional hybrid search (fallback included)
try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    def hybrid_question_search(qdf: pd.DataFrame, query: str, top_k: int = 120, min_score: float = 0.40) -> pd.DataFrame:
        """Simple fallback: case-insensitive substring + token overlap scorer."""
        if not query or not str(query).strip():
            return pd.DataFrame(columns=["code", "text", "display", "score"])
        q = str(query).strip().lower()
        tokens = {t for t in q.replace(",", " ").split() if t}
        scores = []
        for _, r in qdf.iterrows():
            text = f"{r['code']} {r['text']}".lower()
            base = 1.0 if q in text else 0.0
            overlap = sum(1 for t in tokens if t in text) / max(len(tokens), 1)
            score = 0.6 * overlap + 0.4 * base
            scores.append(score)
        out = qdf.copy()
        out["score"] = scores
        out = out.sort_values("score", ascending=False)
        out = out[out["score"] >= min_score]
        return out.head(top_k)


# =============================================================================
# Question Picker
# =============================================================================
def question_picker(qdf: pd.DataFrame) -> List[str]:
    """
    Renders:
      1) Authoritative multiselect (display = "Qxx – text")
      2) Optional keyword search expander with checkboxes for hits
      3) 'Selected questions' quick unselect checkboxes
    Returns ordered list of selected QUESTION CODES (max MAX_QUESTIONS).
    """
    st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)

    # Display strings and lookup maps
    all_displays = qdf["display"].tolist()
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # --- 1) Dropdown multiselect (authoritative order) ---
    multi_choices = st.multiselect(
        "Choose one or more from the official list",
        all_displays,
        default=st.session_state.get(K_MULTI_QUESTIONS, []),
        max_selections=MAX_QUESTIONS,
        label_visibility="collapsed",
        key=K_MULTI_QUESTIONS,
    )
    selected_from_multi: List[str] = [display_to_code[d] for d in multi_choices if d in display_to_code]

    # --- 2) Keyword search (expander) ---
    with st.expander("Search by keywords or theme (optional)"):
        search_query = st.text_input(
            "Enter keywords (e.g., harassment, recognition, onboarding)",
            key=K_KW_QUERY
        )
        if st.button("Search questions", key=K_FIND_HITS_BTN):
            hits_df = hybrid_question_search(qdf, search_query, top_k=120, min_score=0.40)
            st.session_state[K_HITS] = hits_df[["code", "text"]].to_dict(orient="records") if not hits_df.empty else []

        selected_from_hits: Set[str] = set()
        hits = st.session_state.get(K_HITS, [])
        if hits:
            st.write(f"Top {len(hits)} matches meeting the quality threshold:")
            for rec in hits:
                code = rec["code"]; text = rec["text"]
                label = f"{code} – {text}"
                key = f"{HIT_PREFIX}{code}"
                default_checked = st.session_state.get(key, False) or (code in selected_from_multi)
                checked = st.checkbox(label, value=default_checked, key=key)
                if checked:
                    selected_from_hits.add(code)
        else:
            st.info('Enter keywords and click "Search questions" to see matches.')

    # --- Merge ordered selections: multiselect first, then hits; cap MAX_QUESTIONS ---
    combined_order: List[str] = []
    for d in st.session_state.get(K_MULTI_QUESTIONS, []):
        c = display_to_code.get(d)
        if c and c not in combined_order:
            combined_order.append(c)
    for c in selected_from_hits:
        if c not in combined_order:
            combined_order.append(c)

    trimmed = False
    if len(combined_order) > MAX_QUESTIONS:
        combined_order = combined_order[:MAX_QUESTIONS]
        trimmed = True

    if trimmed:
        st.warning(f"Limit is {MAX_QUESTIONS} questions; extra selections were ignored.")

    st.session_state[K_SELECTED_CODES] = combined_order

    # --- 3) Quick unselect list (preserves order) ---
    if combined_order:
        st.markdown('<div class="field-label">Selected questions:</div>', unsafe_allow_html=True)
        updated = list(combined_order)
        cols = st.columns(min(MAX_QUESTIONS, len(updated)))
        for idx, code in enumerate(list(updated)):
            with cols[idx % len(cols)]:
                label = code_to_display.get(code, code)
                keep = st.checkbox(label, value=True, key=f"{SELECTED_PREFIX}{code}")
                if not keep:
                    updated = [c for c in updated if c != code]
                    # Uncheck any keyword hit box
                    hk = f"{HIT_PREFIX}{code}"
                    if hk in st.session_state:
                        st.session_state[hk] = False
                    # Remove from multiselect default list
                    disp = code_to_display.get(code)
                    if disp:
                        st.session_state[K_MULTI_QUESTIONS] = [d for d in st.session_state[K_MULTI_QUESTIONS] if d != disp]
        if updated != combined_order:
            st.session_state[K_SELECTED_CODES] = updated

    return list(st.session_state.get(K_SELECTED_CODES, []))


# =============================================================================
# Year Picker
# =============================================================================
def year_picker() -> List[int]:
    """
    Renders an 'All years' checkbox and one checkbox per year.
    Returns a sorted list of selected years.
    """
    st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
    st.session_state.setdefault(K_SELECT_ALL_YEARS, True)

    select_all = st.checkbox("All years", key=K_SELECT_ALL_YEARS)
    selected_years: List[int] = []

    year_cols = st.columns(len(DEFAULT_YEARS))
    for idx, yr in enumerate(DEFAULT_YEARS):
        with year_cols[idx]:
            key = f"year_{yr}"
            default_checked = True if select_all else st.session_state.get(key, False)
            if st.checkbox(str(yr), value=default_checked, key=key):
                selected_years.append(yr)

    return sorted(selected_years)


# =============================================================================
# Demographic Picker
# =============================================================================
def demographic_picker(
    demo_df: pd.DataFrame,
) -> Tuple[str, Optional[str], List[Optional[str]], Dict[str, str], bool]:
    """
    Renders demographic category and optional subgroup picker.
    Returns:
      (demo_selection, sub_selection, demcodes, disp_map, category_in_play)
    """
    st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)

    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    demo_categories = ["All respondents"] + sorted(
        demo_df.get(DEMO_CAT_COL, pd.Series([], dtype="object")).dropna().astype(str).unique().tolist()
    )

    st.session_state.setdefault(K_DEMO_MAIN, "All respondents")
    demo_selection = st.selectbox("Demographic category", demo_categories, key=K_DEMO_MAIN, label_visibility="collapsed")

    # Optional subgroup
    sub_selection = None
    if demo_selection != "All respondents":
        st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
        sub_items = demo_df.loc[demo_df.get(DEMO_CAT_COL) == demo_selection, LABEL_COL].dropna().astype(str).unique().tolist()
        sub_items = sorted(sub_items)
        sub_key = f"{SUBGROUP_PREFIX}{demo_selection.replace(' ', '_')}"
        sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=sub_key, label_visibility="collapsed")
        if sub_selection == "":
            sub_selection = None

    # Resolve demcodes + display map
    demcodes, disp_map, category_in_play = _resolve_demcodes(demo_df, demo_selection, sub_selection)

    return demo_selection, sub_selection, demcodes, disp_map, category_in_play


# -----------------------------------------------------------------------------
# Helper: DEMCODE resolution
# -----------------------------------------------------------------------------
def _resolve_demcodes(
    demo_df: pd.DataFrame,
    category_label: str,
    subgroup_label: Optional[str],
) -> Tuple[List[Optional[str]], Dict[str, str], bool]:
    """
    Map the selected category/subgroup to DEMCODE values used in data queries.
    Returns:
      demcodes (list[Optional[str]]), display_map (dict), category_in_play (bool)
    """
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    # overall (no category)
    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    # find code column
    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    # single subgroup chosen
    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            r = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not r.empty:
                code = str(r.iloc[0][code_col])
                return [code], {code: subgroup_label}, True
        # fallback (if code not found)
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # no subgroup -> all codes in category
    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    # fallback
    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        return labels, {l: l for l in labels}, True

    return [None], {None: "All respondents"}, False


# =============================================================================
# Misc
# =============================================================================
def search_button_enabled(question_codes: List[str], years: List[int]) -> bool:
    """Return True if we have enough inputs to enable the Search action."""
    return bool(question_codes) and bool(years)
