# menu1/render/controls.py
from __future__ import annotations
from typing import List, Optional
import re
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # for a tiny one-time scrollIntoView

try:
    from utils.hybrid_search import hybrid_question_search, get_embedding_status, get_last_search_metrics  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore
    def get_embedding_status(): return {}
    def get_last_search_metrics(): return {}

# ---- Session-state keys -----------------------------------------------------
K_SELECTED_CODES  = "menu1_selected_codes"
K_KW_QUERY        = "menu1_kw_query"
K_HITS            = "menu1_hits"
K_FIND_HITS_BTN   = "menu1_find_hits"
K_SEARCH_DONE     = "menu1_search_done"
K_LAST_QUERY      = "menu1_last_search_query"
K_HITS_PAGE_LEX   = "menu1_hits_page_lex"
K_HITS_PAGE_SEM   = "menu1_hits_page_sem"
K_SEEN_NONCE      = "menu1_seen_nonce"
K_GLOBAL_HITS_SELECTED = "menu1_global_hits_selected"

K_AI_ENGINE       = "menu1_ai_engine"
K_AI_METRICS      = "menu1_ai_metrics"

K_DO_CLEAR        = "menu1_do_clear"
K_SCROLL_TO_STEP2 = "menu1_scroll_to_step2"

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

MIN_SCORE = 0.40
PAGE_SIZE = 10

_word_re = re.compile(r"[a-z0-9']+")


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


def _clear_menu1_state():
    st.session_state[K_SELECTED_CODES]  = []
    st.session_state[K_KW_QUERY]        = ""
    st.session_state[K_HITS]            = []
    st.session_state[K_SEARCH_DONE]     = False
    st.session_state[K_LAST_QUERY]      = ""
    st.session_state[K_HITS_PAGE_LEX]   = 0
    st.session_state[K_HITS_PAGE_SEM]   = 0
    st.session_state[K_GLOBAL_HITS_SELECTED] = {}
    for k in list(st.session_state.keys()):
        if k.startswith("kwhit_") or k.startswith("sel_"):
            try:
                del st.session_state[k]
            except Exception:
                st.session_state[k] = False


def _maybe_auto_reset_on_mount():
    nonce = st.session_state.get("menu1_mount_nonce", None)
    seen  = st.session_state.get(K_SEEN_NONCE, None)
    if nonce is not None and nonce != seen:
        _clear_menu1_state()
        st.session_state[K_SEEN_NONCE] = nonce


# ---- Main controls ----------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    # seed session
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_SEARCH_DONE, False)
    st.session_state.setdefault(K_LAST_QUERY, "")
    st.session_state.setdefault(K_HITS_PAGE_LEX, 0)
    st.session_state.setdefault(K_HITS_PAGE_SEM, 0)
    st.session_state.setdefault(K_SEEN_NONCE, None)
    st.session_state.setdefault(K_DO_CLEAR, False)
    st.session_state.setdefault(K_AI_ENGINE, {})
    st.session_state.setdefault(K_AI_METRICS, {})
    st.session_state.setdefault(K_SCROLL_TO_STEP2, False)
    st.session_state.setdefault(K_GLOBAL_HITS_SELECTED, {})

    if st.session_state.get(K_DO_CLEAR, False):
        _clear_menu1_state()
        st.session_state[K_DO_CLEAR] = False

    _maybe_auto_reset_on_mount()

    code_to_display = dict(zip(qdf["code"], qdf["display"]))

    # ---------- Step 1 (search only) ----------
    st.markdown('<div class="field-label">Step 1: Search the questionnaire to select a question (max. 5 questions)</div>', unsafe_allow_html=True)

    # indent wrapper for step 1
    st.markdown("<div id='menu1_indent' style='margin-left:8%'>", unsafe_allow_html=True)

    # search label
    st.markdown("**Search questionnaire by keywords or theme**")
    query = st.text_input(
        "Enter keywords (e.g., harassment, recognition, onboarding)",
        key=K_KW_QUERY,
        label_visibility="collapsed",
        placeholder='Type keywords like “career advancement”, “harassment”, “recognition”…',
    )

    bcol1, bcol2 = st.columns([0.5, 0.5])
    with bcol1:
        if st.button("Search the questionnaire", key=K_FIND_HITS_BTN):
            q = (query or "").strip()
            st.session_state[K_GLOBAL_HITS_SELECTED] = {}
            if not q:
                st.session_state[K_SEARCH_DONE] = True
                st.session_state[K_LAST_QUERY] = ""
                st.session_state[K_HITS] = []
                st.session_state[K_HITS_PAGE_LEX] = 0
                st.session_state[K_HITS_PAGE_SEM] = 0
            else:
                t0 = time.time()
                hits_df = _run_keyword_search(qdf, q, top_k=120)
                st.session_state[K_AI_ENGINE]  = get_embedding_status()
                st.session_state[K_AI_METRICS] = get_last_search_metrics()
                st.session_state[K_SEARCH_DONE] = True
                st.session_state[K_LAST_QUERY] = q
                if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                    st.session_state[K_HITS] = hits_df[["code", "text", "display", "score", "origin"]].to_dict(orient="records")
                else:
                    st.session_state[K_HITS] = []
                st.session_state[K_HITS_PAGE_LEX] = 0
                st.session_state[K_HITS_PAGE_SEM] = 0

                try:
                    from .diagnostics import mark_last_query  # type: ignore
                    metrics = st.session_state[K_AI_METRICS] or {}
                    extra = {
                        "query": q,
                        "results_total": int(metrics.get("total", len(st.session_state[K_HITS]) or 0)),
                        "results_lex": int(metrics.get("count_lex", 0)),
                        "results_sem": int(metrics.get("count_sem", 0)),
                        "t_total_ms": int(metrics.get("t_total_ms", (time.time() - t0) * 1000)),
                        "semantic_active": bool(metrics.get("semantic_active", False)),
                        "sem_floor": metrics.get("sem_floor"),
                        "jaccard_cutoff": metrics.get("jaccard_cutoff"),
                    }
                    mark_last_query(started_ts=t0, finished_ts=time.time(), engine="hybrid_search", extra=extra)
                except Exception:
                    pass

    with bcol2:
        if st.button("Clear search & selections", key="menu1_clear_all"):
            st.session_state[K_DO_CLEAR] = True
            st.experimental_rerun()

    if st.session_state.get(K_SEARCH_DONE, False):
        n_total = len(st.session_state.get(K_HITS, []) or [])
        if n_total > 0:
            st.info(f"Found {n_total} total matches meeting the quality threshold.")
        else:
            last_q = (st.session_state.get(K_LAST_QUERY) or "").strip()
            safe_q = last_q if last_q else "your search"
            st.warning(
                f'No questions matched “{safe_q}”. '
                'Try broader/different keywords (e.g., synonyms), split phrases, '
                'or search by a question code like “Q01”.'
            )

    # search results
    hits = st.session_state.get(K_HITS, [])
    if st.session_state.get(K_SEARCH_DONE, False) and hits:
        lex_hits = [r for r in hits if r.get("origin","lex") == "lex"]
        sem_hits = [r for r in hits if r.get("origin","lex") == "sem"]

        tabs = st.tabs(["Lexical matches", "Other matches (semantic)"])

        # lexical tab
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
                gmap = dict(st.session_state.get(K_GLOBAL_HITS_SELECTED, {}))
                for rec in lex_hits[start:end]:
                    code = rec["code"]; text = rec["text"]
                    key = f"kwhit_{code}"
                    label = f"{code} — {text}"
                    if key in st.session_state:
                        checked = st.checkbox(label, key=key)
                    else:
                        desired = bool(gmap.get(code, False))
                        checked = st.checkbox(label, key=key, value=desired)
                    gmap[code] = bool(checked)
                st.session_state[K_GLOBAL_HITS_SELECTED] = gmap

                pcol, ncol = st.columns([0.5, 0.5])
                with pcol:
                    st.button("Prev", disabled=(page <= 0), key="menu1_hits_prev_lex",
                              on_click=lambda: st.session_state.update({K_HITS_PAGE_LEX: max(0, page - 1)}))
                with ncol:
                    st.button("Next", disabled=(page >= max_page), key="menu1_hits_next_lex",
                              on_click=lambda: st.session_state.update({K_HITS_PAGE_LEX: min(max_page, page + 1)}))

        # semantic tab
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
                gmap = dict(st.session_state.get(K_GLOBAL_HITS_SELECTED, {}))
                for rec in sem_hits[start:end]:
                    code = rec["code"]; text = rec["text"]; score = rec.get("score", 0.0)
                    label = f"{code} — {text}  (score: {score:.2f})"
                    key = f"kwhit_{code}"
                    if key in st.session_state:
                        checked = st.checkbox(label, key=key)
                    else:
                        desired = bool(gmap.get(code, False))
                        checked = st.checkbox(label, key=key, value=desired)
                    gmap[code] = bool(checked)
                st.session_state[K_GLOBAL_HITS_SELECTED] = gmap

                pcol, ncol = st.columns([0.5, 0.5])
                with pcol:
                    st.button("Prev", disabled=(page <= 0), key="menu1_hits_prev_sem",
                              on_click=lambda: st.session_state.update({K_HITS_PAGE_SEM: max(0, page - 1)}))
                with ncol:
                    st.button("Next", disabled=(page >= max_page), key="menu1_hits_next_sem",
                              on_click=lambda: st.session_state.update({K_HITS_PAGE_SEM: min(max_page, page + 1)}))

    # close indent
    st.markdown("</div>", unsafe_allow_html=True)

    # merge selections (from hits only), cap 5
    combined_order: List[str] = []
    gmap_all = st.session_state.get(K_GLOBAL_HITS_SELECTED, {})
    for rec in st.session_state.get(K_HITS, []):
        code = rec["code"]
        if gmap_all.get(code, False) and code not in combined_order:
            combined_order.append(code)
    if len(combined_order) > 5:
        combined_order = combined_order[:5]
        st.warning("Limit is 5 questions; extra selections were ignored.")
    st.session_state[K_SELECTED_CODES] = combined_order

    # show selected
    if st.session_state[K_SELECTED_CODES]:
        st.markdown('<div class="field-label">Selected questions:</div>', unsafe_allow_html=True)
        updated = list(st.session_state[K_SELECTED_CODES])
        for code in list(updated):
            label = code_to_display.get(code, code)
            keep = st.checkbox(label, value=True, key=f"sel_{code}")
            if not keep:
                updated = [c for c in updated if c != code]
                hk = f"kwhit_{code}"
                if hk in st.session_state:
                    st.session_state[hk] = False
                gmap = dict(st.session_state.get(K_GLOBAL_HITS_SELECTED, {}))
                if code in gmap:
                    gmap[code] = False
                    st.session_state[K_GLOBAL_HITS_SELECTED] = gmap
        if updated != st.session_state[K_SELECTED_CODES]:
            st.session_state[K_SELECTED_CODES] = updated

    # separator after step 1
    st.markdown("<hr style='margin-top:1rem;margin-bottom:1rem;'>", unsafe_allow_html=True)

    return st.session_state[K_SELECTED_CODES]


# ---- Years (Step 2) ---------------------------------------------------------
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
            st.session_state[f"year_{yr}"] = False

    selected_years: List[int] = []
    cols = st.columns(len(DEFAULT_YEARS))
    for i, yr in enumerate(DEFAULT_YEARS):
        with cols[i]:
            st.checkbox(str(yr), key=f"year_{yr}")
            if st.session_state.get(f"year_{yr}", False):
                selected_years.append(yr)

    # separator after step 2
    st.markdown("<hr style='margin-top:1rem;margin-bottom:1rem;'>", unsafe_allow_html=True)

    return sorted(selected_years)


# ---- Demographics (Step 3) --------------------------------------------------
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

    if "DEMCODE Category" in demo_df.columns:
        df_cat = demo_df[demo_df["DEMCODE Category"] == demo_selection]
    else:
        df_cat = demo_df.copy()

    if df_cat.empty:
        return demo_selection, sub_selection, [None], {None: "All respondents"}, False

    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

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
