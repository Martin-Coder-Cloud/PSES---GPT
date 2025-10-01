# menu1/render/controls.py
from __future__ import annotations
from typing import List, Optional
import re
import pandas as pd
import streamlit as st

try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore

# ---- Session-state keys -----------------------------------------------------
K_MULTI_QUESTIONS = "menu1_multi_questions"
K_SELECTED_CODES  = "menu1_selected_codes"
K_KW_QUERY        = "menu1_kw_query"
K_HITS            = "menu1_hits"                  # list[dict] with 'origin'
K_FIND_HITS_BTN   = "menu1_find_hits"
K_SEARCH_DONE     = "menu1_search_done"
K_LAST_QUERY      = "menu1_last_search_query"
K_HITS_PAGE_LEX   = "menu1_hits_page_lex"         # page index for lexical tab
K_HITS_PAGE_SEM   = "menu1_hits_page_sem"         # page index for semantic tab

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Threshold
MIN_SCORE = 0.40
PAGE_SIZE = 10

# ---- Helpers ----------------------------------------------------------------
_word_re = re.compile(r"[a-z0-9']+")

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _dedupe_hits(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["code", "text", "display", "score", "origin"])
    out = df.copy()
    if "score" not in out.columns:
        out["score"] = 0.0
    if "origin" not in out.columns:
        out["origin"] = "lex"
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
    return pd.DataFrame(columns=["code", "text", "display", "score", "origin"])

# ---- Main controls ----------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    # seed session (no mutation-after-render)
    st.session_state.setdefault(K_MULTI_QUESTIONS, [])
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_SEARCH_DONE, False)
    st.session_state.setdefault(K_LAST_QUERY, "")
    st.session_state.setdefault(K_HITS_PAGE_LEX, 0)
    st.session_state.setdefault(K_HITS_PAGE_SEM, 0)

    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # ---------- Step 1 ----------
    st.markdown('<div class="field-label">Step 1: Pick up to 5 survey questions:</div>', unsafe_allow_html=True)

    col_spacer, col_main = st.columns([0.08, 0.92])
    with col_main:
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

        st.markdown("<div class='field-label'>Search questionnaire by keywords or theme</div>", unsafe_allow_html=True)
        query = st.text_input(
            "Enter keywords (e.g., harassment, recognition, onboarding)",
            key=K_KW_QUERY,
            label_visibility="collapsed",
            placeholder='Type keywords like “career advancement”, “harassment”, “recognition”…',
        )

        # Localized "Searching…" spinner/status directly under the button
        status_placeholder = st.empty()

        if st.button("Search the questionnaire", key=K_FIND_HITS_BTN):
            q = (query or "").strip()
            if not q:
                st.warning("Please enter at least one keyword to search the questionnaire.")
            else:
                with status_placeholder:
                    with st.spinner("Searching…"):
                        st.toast("Searching…", icon="🔎")
                        hits_df = _run_keyword_search(qdf, q, top_k=120)

                st.session_state[K_SEARCH_DONE] = True
                st.session_state[K_LAST_QUERY] = q
                if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                    st.session_state[K_HITS] = hits_df[["code", "text", "display", "score", "origin"]].to_dict(orient="records")
                else:
                    st.session_state[K_HITS] = []
                # reset both tab pages on new search
                st.session_state[K_HITS_PAGE_LEX] = 0
                st.session_state[K_HITS_PAGE_SEM] = 0
                status_placeholder.empty()  # remove spinner/status

    # ---------- Results (indented) ----------
    hits = st.session_state.get(K_HITS, [])
    if st.session_state.get(K_SEARCH_DONE, False):
        col_spacer2, col_results = st.columns([0.08, 0.92])
        with col_results:
            if not hits:
                q = (st.session_state.get(K_LAST_QUERY) or "").strip()
                safe_q = q if q else "your search"
                st.warning(
                    f'No questions matched “{safe_q}”. '
                    'Try broader/different keywords (e.g., synonyms), split phrases, '
                    'or search by a question code like “Q01”.'
                )
            else:
                # split into lexical vs semantic
                lex_hits = [r for r in hits if r.get("origin","lex") == "lex"]
                sem_hits = [r for r in hits if r.get("origin","lex") == "sem"]

                tabs = st.tabs(["Lexical matches", "Other matches (semantic)"])

                # --- Tab 1: Lexical ---
                with tabs[0]:
                    total = len(lex_hits)
                    page  = int(st.session_state.get(K_HITS_PAGE_LEX, 0)) or 0
                    max_page = max(0, (total - 1) // PAGE_SIZE) if total else 0
                    page = max(0, min(page, max_page))
                    start = page * PAGE_SIZE
                    end   = min(total, start + PAGE_SIZE)

                    if total == 0:
                        st.warning("No lexical matches.")
                    else:
                        st.write(f"Results {start + 1}–{end} of {total} lexical matches meeting the quality threshold:")
                        for rec in lex_hits[start:end]:
                            code = rec["code"]; text = rec["text"]
                            key = f"kwhit_{code}"
                            default_checked = code in [display_to_code.get(d) for d in st.session_state.get(K_MULTI_QUESTIONS, [])]
                            st.session_state.setdefault(key, default_checked)
                            st.checkbox(f"{code} — {text}", key=key)

                        col_prev, col_next = st.columns(2)
                        with col_prev:
                            if st.button("Prev", disabled=(page <= 0), key="menu1_hits_prev_lex"):
                                if page > 0:
                                    st.session_state[K_HITS_PAGE_LEX] = page - 1
                        with col_next:
                            if st.button("Next", disabled=(page >= max_page), key="menu1_hits_next_lex"):
                                if page < max_page:
                                    st.session_state[K_HITS_PAGE_LEX] = page + 1

                # --- Tab 2: Semantic ---
                with tabs[1]:
                    total = len(sem_hits)
                    page  = int(st.session_state.get(K_HITS_PAGE_SEM, 0)) or 0
                    max_page = max(0, (total - 1) // PAGE_SIZE) if total else 0
                    page = max(0, min(page, max_page))
                    start = page * PAGE_SIZE
                    end   = min(total, start + PAGE_SIZE)

                    if total == 0:
                        st.warning("No other (semantic) matches.")
                    else:
                        st.write(f"Results {start + 1}–{end} of {total} other (semantic) matches meeting the quality threshold:")
                        for rec in sem_hits[start:end]:
                            code = rec["code"]; text = rec["text"]
                            key = f"kwhit_{code}"
                            default_checked = code in [display_to_code.get(d) for d in st.session_state.get(K_MULTI_QUESTIONS, [])]
                            st.session_state.setdefault(key, default_checked)
                            st.checkbox(f"{code} — {text}", key=key)

                        col_prev, col_next = st.columns(2)
                        with col_prev:
                            if st.button("Prev", disabled=(page <= 0), key="menu1_hits_prev_sem"):
                                if page > 0:
                                    st.session_state[K_HITS_PAGE_SEM] = page - 1
                        with col_next:
                            if st.button("Next", disabled=(page >= max_page), key="menu1_hits_next_sem"):
                                if page < max_page:
                                    st.session_state[K_HITS_PAGE_SEM] = page + 1

    # ---------- Merge selections (multiselect first, then checked hits), cap 5 ----------
    combined_order: List[str] = []
    for d in st.session_state.get(K_MULTI_QUESTIONS, []):
        c = display_to_code.get(d)
        if c and c not in combined_order:
            combined_order.append(c)
    # include checked hits from either tab
    for rec in st.session_state.get(K_HITS, []):
        code = rec["code"]; key = f"kwhit_{code}"
        if st.session_state.get(key, False) and code not in combined_order:
            combined_order.append(code)
    if len(combined_order) > 5:
        combined_order = combined_order[:5]
        st.warning("Limit is 5 questions; extra selections were ignored.")
    st.session_state[K_SELECTED_CODES] = combined_order

    # ---------- Selected list ----------
    if st.session_state[K_SELECTED_CODES]:
        st.markdown('<div class="field-label">Selected questions:</div>', unsafe_allow_html=True)
        updated = list(st.session_state[K_SELECTED_CODES])
        cols = st.columns(min(5, len(updated)))
        for idx, code in enumerate(list(updated)):
            with cols[idx % len(cols)]:
                label = code_to_display.get(code, code)
                keep = st.checkbox(label, value=True, key=f"sel_{code}")
                if not keep:
                    updated = [c for c in updated if c != code]
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
            st.checkbox(str(yr), key=f"year_{yr}")
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

    if not demo_selection or demo_selection == "All respondents":
        return demo_selection, sub_selection, [None], {None: "All respondents"}, False

    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    if "DEMCODE Category" in demo_df.columns:
        df_cat = demo_df[demo_df["DEMCODE Category"] == demo_selection]
    else:
        df_cat = demo_df.copy()

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
