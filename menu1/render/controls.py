# menu1/render/controls.py
"""
Controls for Menu 1:
- Question picker: dropdown multi-select (authoritative) + keyword search (hybrid)
- Selected list with quick unselect checkboxes
- Years selector
- Demographic category & subgroup selector
- Search button enablement helper

Behavior:
- Fixed min score threshold (> 0.40) for BOTH the external hybrid search and the local fallback
- Multi-keyword support with partial/typo tolerance in the fallback
- Clear "no results" feedback
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import re
import pandas as pd
import streamlit as st

# Try to import the hybrid search; we'll wrap it and also provide a robust fallback
try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore

# -----------------------------
# Session-state keys (Menu 1)
# -----------------------------
K_MULTI_QUESTIONS = "menu1_multi_questions"  # List[str] of "display" labels picked in the dropdown
K_SELECTED_CODES  = "menu1_selected_codes"   # Ordered List[str] of codes (multi + hits merged)
K_KW_QUERY        = "menu1_kw_query"         # str
K_HITS            = "menu1_hits"             # List[{"code","text"}] from last search
K_FIND_HITS_BTN   = "menu1_find_hits"        # button key
K_SEARCH_DONE     = "menu1_search_done"      # bool: did user click Search questions?
K_LAST_QUERY      = "menu1_last_search_query"# str: what query was used last time

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Threshold (fixed)
MIN_SCORE = 0.40  # keep > 0.40 filter in both engines

# -----------------------------------------------------------------------------
# Internal helpers: tokenization & scoring for fallback search
# -----------------------------------------------------------------------------
_word_re = re.compile(r"[a-z0-9']+")

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _tokens(s: str) -> List[str]:
    return _word_re.findall(_normalize(s))

def _fallback_search(qdf: pd.DataFrame, query: str, top_k: int = 120) -> pd.DataFrame:
    """
    Lightweight multi-keyword search with partial/typo tolerance.
    Scoring:
      - token_coverage: matched tokens / total tokens (substring/prefix tolerant)
      - phrase_hit: 1 if full query substring appears
      - bigram_hit: 1 if any two consecutive tokens appear as a phrase
    score = 0.6*token_coverage + 0.3*phrase_hit + 0.1*bigram_hit
    Filter: score > MIN_SCORE (0.40)
    """
    q = _normalize(query)
    if not q:
        return pd.DataFrame(columns=["code", "text", "display", "score"])

    toks = _tokens(q)
    if not toks:
        return pd.DataFrame(columns=["code", "text", "display", "score"])

    phrases = [" ".join(toks[i:i+2]) for i in range(len(toks) - 1)]  # bigrams

    rows = []
    for _, r in qdf.iterrows():
        code = str(r["code"])
        text = str(r["text"])
        display = f"{code} — {text}"

        t = _normalize(f"{code} {text}")
        t_words = set(_tokens(t))

        # token matches (substring/prefix tolerant)
        matched = 0
        for tok in toks:
            if (tok in t) or any(w.startswith(tok) or tok.startswith(w) for w in t_words):
                matched += 1
        token_coverage = matched / max(len(toks), 1)

        # phrase & bigram hits
        phrase_hit = 1.0 if q in t else 0.0
        bigram_hit = 1.0 if any(p in t for p in phrases) else 0.0

        score = 0.6 * token_coverage + 0.3 * phrase_hit + 0.1 * bigram_hit
        if score > MIN_SCORE:
            rows.append({"code": code, "text": text, "display": display, "score": score})

    if not rows:
        return pd.DataFrame(columns=["code", "text", "display", "score"])

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out.head(top_k)


def _run_keyword_search(qdf: pd.DataFrame, query: str, top_k: int = 120) -> pd.DataFrame:
    """
    Wrapper that:
      1) Calls external hybrid_question_search if available (min_score=MIN_SCORE)
      2) Falls back to our tolerant multi-keyword search
    """
    # Try external search first (if available)
    if callable(hybrid_question_search):
        try:
            hits_df = hybrid_question_search(qdf, query, top_k=top_k, min_score=MIN_SCORE)  # type: ignore
            if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                return hits_df
        except Exception:
            pass  # ignore and fallback

    # Fallback (multi-keyword tolerant, fixed threshold)
    return _fallback_search(qdf, query, top_k=top_k)


# -----------------------------------------------------------------------------
# Internal helper: resolve demographic codes & display map
# -----------------------------------------------------------------------------
def _resolve_demcodes(demo_df: pd.DataFrame, category_label: str, subgroup_label: Optional[str]):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    # overall
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
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # no subgroup -> take all codes in category
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


# -----------------------------------------------------------------------------
# Question picker (dropdown + keyword search) → returns List[str] (codes)
# -----------------------------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    """
    UI:
      1) Dropdown multi-select (authoritative, max 5)
      2) Expander: keyword search; checkboxes to select hits
      3) "Selected questions" list with quick unselect checkboxes

    Returns ordered list of selected question codes (max 5).
    """
    # Ensure session defaults
    st.session_state.setdefault(K_MULTI_QUESTIONS, [])
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_SEARCH_DONE, False)
    st.session_state.setdefault(K_LAST_QUERY, "")

    # Mappings
    code_to_text = dict(zip(qdf["code"], qdf["text"]))
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # ---------- 1) Dropdown multi-select ----------
    st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)
    all_displays = qdf["display"].tolist()
    multi_choices = st.multiselect(
        "Choose one or more from the official list",
        all_displays,
        default=st.session_state.get(K_MULTI_QUESTIONS, []),
        max_selections=5,
        label_visibility="collapsed",
        key=K_MULTI_QUESTIONS,
    )
    selected_from_multi: Set[str] = set(display_to_code[d] for d in multi_choices if d in display_to_code)

    # ---------- 2) Hybrid keyword search ----------
    with st.expander("Search by keywords or theme (optional)"):
        query = st.text_input(
            "Enter keywords (e.g., harassment, recognition, onboarding)",
            key=K_KW_QUERY
        )
        if st.button("Search questions", key=K_FIND_HITS_BTN):
            hits_df = _run_keyword_search(qdf, query, top_k=120)
            st.session_state[K_SEARCH_DONE] = True
            st.session_state[K_LAST_QUERY] = query
            st.session_state[K_HITS] = hits_df[["code", "text"]].to_dict(orient="records") if isinstance(hits_df, pd.DataFrame) and not hits_df.empty else []

        selected_from_hits: Set[str] = set()
        hits = st.session_state.get(K_HITS, [])

        if st.session_state.get(K_SEARCH_DONE, False):
            if not hits:
                q = (st.session_state.get(K_LAST_QUERY) or "").strip()
                safe_q = q if q else "your search"
                st.warning(
                    f"No questions matched “{safe_q}”. "
                    "Try broader or different keywords (e.g., synonyms), split phrases (e.g., “overall satisfaction” → “satisfaction”), "
                    "or search by a question code like “Q01”."
                )
            else:
                st.write(f"Top {len(hits)} matches meeting the quality threshold:")
        else:
            st.info('Enter keywords and click "Search questions" to see matches.')

        if hits:
            for rec in hits:
                code = rec["code"]; text = rec["text"]
                label = f"{code} — {text}"
                key = f"kwhit_{code}"
                default_checked = st.session_state.get(key, False) or (code in selected_from_multi)
                checked = st.checkbox(label, value=default_checked, key=key)
                if checked:
                    selected_from_hits.add(code)

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
                    # uncheck corresponding search hit, if any
                    hk = f"kwhit_{code}"
                    if hk in st.session_state:
                        st.session_state[hk] = False
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

    demcodes, disp_map, category_in_play = _resolve_demcodes(demo_df, demo_selection, sub_selection)
    return demo_selection, sub_selection, demcodes, disp_map, category_in_play


# -----------------------------------------------------------------------------
# Search button enabled?
# -----------------------------------------------------------------------------
def search_button_enabled(question_codes: List[str], years: List[int]) -> bool:
    return bool(question_codes) and bool(years)
