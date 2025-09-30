# menu1/render/controls.py
from __future__ import annotations
from typing import List, Optional, Set
import re
import pandas as pd
import streamlit as st

# Optional hybrid search import
try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore

# ---- Session-state keys -----------------------------------------------------
K_MULTI_QUESTIONS = "menu1_multi_questions"     # Multiselect (display strings)
K_SELECTED_CODES  = "menu1_selected_codes"      # Final ordered codes (max 5)
K_KW_QUERY        = "menu1_kw_query"            # Keyword text
K_HITS            = "menu1_hits"                # Search hits (list[dict])
K_FIND_HITS_BTN   = "menu1_find_hits"           # Button key
K_SEARCH_DONE     = "menu1_search_done"         # Bool: did search run?
K_LAST_QUERY      = "menu1_last_search_query"   # Last query (string)

# Legacy (no longer used for persistence; kept for compatibility only)
K_HITS_SELECTED   = "menu1_hit_codes_selected"

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Threshold
MIN_SCORE = 0.40  # strictly greater than 0.40

# ---- Helpers ----------------------------------------------------------------
_word_re = re.compile(r"[a-z0-9']+")

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _dedupe_hits(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["code", "text", "display", "score"])
    out = df.copy()
    if "score" not in out.columns:
        out["score"] = 0.0
    out["code"] = out["code"].astype(str)
    out = out.sort_values("score", ascending=False).drop_duplicates(subset=["code"], keep="first")
    try:
        out = out[out["score"] > MIN_SCORE]
    except Exception:
        pass
    return out

def _run_keyword_search(qdf: pd.DataFrame, query: str, top_k: int = 120) -> pd.DataFrame:
    if callable(hybrid_question_search):
        try:
            hits_df = hybrid_question_search(qdf, query, top_k=top_k, min_score=MIN_SCORE)  # type: ignore
            if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                return _dedupe_hits(hits_df).head(top_k)
        except Exception:
            pass
    return pd.DataFrame(columns=["code", "text", "display", "score"])

# ---- Main controls ----------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    # seed session (no mutation-after-render)
    st.session_state.setdefault(K_MULTI_QUESTIONS, [])
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_SEARCH_DONE, False)
    st.session_state.setdefault(K_LAST_QUERY, "")

    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # Step 1
    st.markdown('<div class="field-label">Step 1: Pick up to 5 survey questions:</div>', unsafe_allow_html=True)

    # Indented area for the two sub-options
    col_spacer, col_main = st.columns([0.08, 0.92])
    with col_main:
        # Subtitle + multiselect
        st.markdown(
            '<div style="margin: 8px 0 4px 0; font-weight:600; color:#222;">Choose a question from the list below</div>',
            unsafe_allow_html=True
        )
        all_displays = qdf["display"].tolist()
        st.multiselect(
            "Choose one or more from the official list",
            all_displays,
            max_selections=5,
            label_visibility="collapsed",
            key=K_MULTI_QUESTIONS,
            placeholder="",
        )

        # "or"
        st.markdown(
            """
            <div style="
                margin: .1rem 0 .1rem .5rem;
                font-size: 0.9rem; font-weight: 600;
                color: rgba(49,51,63,.8); font-family: inherit;
            ">or</div>
            """,
            unsafe_allow_html=True
        )

        # Keyword header + input
        st.markdown("<div class='field-label'>Search questionnaire by keywords or theme</div>", unsafe_allow_html=True)
        query = st.text_input(
            "Enter keywords (e.g., harassment, recognition, onboarding)",
            key=K_KW_QUERY,
            label_visibility="collapsed",
            placeholder='Type keywords like “career advancement”, “harassment”, “recognition”…',
        )

        # ⬇️ Button directly under the text box (as requested; no styling change)
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

        # Results list
        hits = st.session_state.get(K_HITS, [])
        if st.session_state.get(K_SEARCH_DONE, False):
            if not hits:
                q = (st.session_state.get(K_LAST_QUERY) or "").strip()
                safe_q = q if q else "your search"
                st.warning(
                    f'No questions matched “{safe_q}”. '
                    'Try broader/different keywords (e.g., synonyms), split phrases, '
                    'or search by a question code like “Q01”.'
                )
            else:
                st.write(f"Top {len(hits)} matches meeting the quality threshold:")
                # Render checkboxes for hits; rely ONLY on per-code checkbox keys (so reset works)
                for rec in hits:
                    code = rec["code"]; text = rec["text"]
                    key = f"kwhit_{code}"  # matches state.HIT_PREFIX
                    # default checked if already chosen via multiselect
                    default_checked = code in [display_to_code.get(d) for d in st.session_state.get(K_MULTI_QUESTIONS, [])]
                    st.session_state.setdefault(key, default_checked)
                    st.checkbox(f"{code} — {text}", key=key)

    # Merge selections (multiselect first, then currently checked hits), cap 5
    combined_order: List[str] = []
    for d in st.session_state.get(K_MULTI_QUESTIONS, []):
        c = display_to_code.get(d)
        if c and c not in combined_order:
            combined_order.append(c)
    for rec in st.session_state.get(K_HITS, []):
        code = rec["code"]; key = f"kwhit_{code}"
        if st.session_state.get(key, False) and code not in combined_order:
            combined_order.append(code)
    if len(combined_order) > 5:
        combined_order = combined_order[:5]
        st.warning("Limit is 5 questions; extra selections were ignored.")
    st.session_state[K_SELECTED_CODES] = combined_order

    # Not indented: Selected questions (quick unselect)
    if st.session_state[K_SELECTED_CODES]:
        st.markdown('<div class="field-label">Selected questions:</div>', unsafe_allow_html=True)
        updated = list(st.session_state[K_SELECTED_CODES])
        cols = st.columns(min(5, len(updated)))
        for idx, code in enumerate(list(updated)):
            with cols[idx % len(cols)]:
                label = code_to_display.get(code, code)
                keep = st.checkbox(label, value=True, key=f"sel_{code}")
                if not keep:
                    # remove from selection
                    updated = [c for c in updated if c != code]
                    # uncheck corresponding hit checkbox and remove from multiselect
                    hk = f"kwhit_{code}"
                    if hk in st.session_state:
                        st.session_state[hk] = False
                    disp = code_to_display.get(code)
                    if disp:
                        st.session_state[K_MULTI_QUESTIONS] = [d for d in st.session_state[K_MULTI_QUESTIONS] if d != disp]
        if updated != st.session_state[K_SELECTED_CODES]:
            st.session_state[K_SELECTED_CODES] = updated

    return st.session_state[K_SELECTED_CODES]

# ---- Years ------------------------------------------------------------------
def year_picker() -> List[int]:
    st.markdown('<div class="field-label">Step 2: Select survey year(s):</div>', unsafe_allow_html=True)
    st.session_state.setdefault(K_SELECT_ALL_YEARS, True)
    select_all = st.checkbox("All years", key=K_SELECT_ALL_YEARS)

    # Pre-set year keys before rendering widgets
    if select_all:
        for yr in DEFAULT_YEARS:
            st.session_state.setdefault(f"year_{yr}", True)
            st.session_state[f"year_{yr}"] = True
    else:
        for yr in DEFAULT_YEARS:
            st.session_state.setdefault(f"year_{yr}", False)

    selected_years: List[int] = []
    cols = st.columns(len(DEFAULT_YEARS))
    for i, yr in enumerate(DEFAULT_YEARS):
        with cols[i]:
            st.checkbox(str(yr), key=f"year_{yr}")  # no value= (prevents double-default warning)
            if st.session_state.get(f"year_{yr}", False):
                selected_years.append(yr)
    return sorted(selected_years)

# ---- Demographics -----------------------------------------------------------
def demographic_picker(demo_df: pd.DataFrame):
    st.markdown('<div class="field-label">Step 3: Select a demographic category (optional):</div>', unsafe_allow_html=True)
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
    st.session_state.setdefault("demo_main", "All respondents")
    demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

    sub_selection: Optional[str] = None
    if demo_selection != "All respondents":
        st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
        sub_items = demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL].dropna().astype(str).unique().tolist()
        sub_items = sorted(sub_items)
        sub_key = f"sub_{demo_selection.replace(' ', '_')}"
        sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=sub_key, label_visibility="collapsed")
        if sub_selection == "":
            sub_selection = None

    # Resolve demcodes (mirrors earlier helper)
    if not demo_selection or demo_selection == "All respondents":
        return demo_selection, sub_selection, [None], {None: "All respondents"}, False

    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == demo_selection] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return demo_selection, sub_selection, [None], {None: "All respondents"}, False

    if sub_selection:
        if code_col and LABEL_COL in df_cat.columns:
            r = df_cat[df_cat[LABEL_COL] == sub_selection]
            if not r.empty:
                code = str(r.iloc[0][code_col])
                return demo_selection, sub_selection, [code], {code: sub_selection}, True
        return demo_selection, sub_selection, [sub_selection], {sub_selection: sub_selection}, True

    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return demo_selection, sub_selection, codes, disp_map, True

    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        return demo_selection, sub_selection, labels, {l: l for l in labels}, True

    return demo_selection, sub_selection, [None], {None: "All respondents"}, False

# ---- Enable search? ---------------------------------------------------------
def search_button_enabled(question_codes: List[str], years: List[int]) -> bool:
    return bool(question_codes) and bool(years)
