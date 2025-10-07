# menu1/render/controls.py
from __future__ import annotations
from typing import List, Optional
import re
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # for a tiny one-time scrollIntoView

# ─────────────────────────────────────────────────────────────────────────────
#  Typography & indentation — ultra-tight, minimalist spacing
# ─────────────────────────────────────────────────────────────────────────────
def ensure_pses_styles():
    st.markdown(
        """
        <style>
          /* Title 2 (Step 1–3) */
          .pses-h2 {
            font-size: 1.08rem;
            font-weight: 600;
            margin: 0.55em 0 0.15em 0; /* compact */
            color: #222;
          }
          /* Step 2 pulled close to Step 1 (~one row gap) */
          .pses-h2-tight-top {
            margin-top: 0.10em !important;
            margin-bottom: 0.15em !important;
          }
          /* Title 3 (sub-sections) */
          .pses-h3 {
            font-size: 1.0rem;
            font-weight: 550;
            margin: 0.35em 0 0.15em 0; /* compact */
            color: #333;
          }
          /* Indented content block for Title 3 + its contents */
          .pses-block {
            margin-left: 2.2cm !important;  /* clear indent */
            padding-left: 0.1cm;
            margin-top: 0;
            margin-bottom: 0;
          }
          /* Utilities */
          .pses-no-top    { margin-top: 0 !important; }
          .pses-no-bottom { margin-bottom: 0 !important; }
          .pses-zero      { margin: 0 !important; padding: 0 !important; line-height: 1 !important; }
          .pses-nudge-up  { margin-top: -0.15em !important; }
          .pses-note      { font-size: 0.9rem; color: #666; }

          /* --- Streamlit widget spacing overrides (target minimal gaps) --- */
          /* Slightly tighten default vertical rhythm */
          div[data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
          /* No space below multiselect (keeps "or" flush) */
          div[data-testid="stMultiSelect"]   { margin-bottom: 0 !important; }
          /* No space above/between selectboxes (Step 3 & subgroup) */
          div[data-testid="stSelectbox"]     { margin-top: 0 !important; }
          /* Tighten text input spacing (search box under "or") */
          div[data-testid="stTextInput"]     { margin-top: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

try:
    from utils.hybrid_search import hybrid_question_search, get_embedding_status, get_last_search_metrics  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore
    def get_embedding_status(): return {}
    def get_last_search_metrics(): return {}

# ---- Session-state keys -----------------------------------------------------
K_MULTI_QUESTIONS = "menu1_multi_questions"
K_SELECTED_CODES  = "menu1_selected_codes"
K_KW_QUERY        = "menu1_kw_query"
K_HITS            = "menu1_hits"                  # list[dict] with 'origin'
K_FIND_HITS_BTN   = "menu1_find_hits"
K_SEARCH_DONE     = "menu1_search_done"
K_LAST_QUERY      = "menu1_last_search_query"
K_HITS_PAGE_LEX   = "menu1_hits_page_lex"         # pagination per tab
K_HITS_PAGE_SEM   = "menu1_hits_page_sem"
K_SEEN_NONCE      = "menu1_seen_nonce"            # remember last mount nonce

# Global persistence for hit selections across pages/tabs
K_GLOBAL_HITS_SELECTED = "menu1_global_hits_selected"  # Dict[str,bool]

# Diagnostics snapshots (optional consumers: Diagnostics tab)
K_AI_ENGINE       = "menu1_ai_engine"
K_AI_METRICS      = "menu1_ai_metrics"

# Deferred actions to avoid "cannot be modified after widget is instantiated"
K_DO_CLEAR        = "menu1_do_clear"              # bool: clear everything next run
K_SYNC_MULTI      = "menu1_sync_multi"            # List[str]: display labels to set for multiselect next run

# One-time scroll trigger to reveal Step 2 after selection from list
K_SCROLL_TO_STEP2 = "menu1_scroll_to_step2"

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Thresholds / paging
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
    """Reset Menu 1 selections and search artifacts (do not alter SYNC/flags here)."""
    st.session_state[K_MULTI_QUESTIONS] = []
    st.session_state[K_SELECTED_CODES]  = []
    st.session_state[K_KW_QUERY]        = ""
    st.session_state[K_HITS]            = []
    st.session_state[K_SEARCH_DONE]     = False
    st.session_state[K_LAST_QUERY]      = ""
    st.session_state[K_HITS_PAGE_LEX]   = 0
    st.session_state[K_HITS_PAGE_SEM]   = 0
    # clear global paginated selections
    st.session_state[K_GLOBAL_HITS_SELECTED] = {}

    # remove dynamic checkboxes
    for k in list(st.session_state.keys()):
        if k.startswith("kwhit_") or k.startswith("sel_"):
            try:
                del st.session_state[k]
            except Exception:
                st.session_state[k] = False

def _maybe_auto_reset_on_mount():
    """If router set 'menu1_mount_nonce', clear once (no rerun to avoid loops)."""
    nonce = st.session_state.get("menu1_mount_nonce", None)
    seen  = st.session_state.get(K_SEEN_NONCE, None)
    if nonce is not None and nonce != seen:
        _clear_menu1_state()
        st.session_state[K_SEEN_NONCE] = nonce
        # no rerun here

# ---- Main controls ----------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    ensure_pses_styles()

    # seed session
    st.session_state.setdefault(K_MULTI_QUESTIONS, [])
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_SEARCH_DONE, False)
    st.session_state.setdefault(K_LAST_QUERY, "")
    st.session_state.setdefault(K_HITS_PAGE_LEX, 0)
    st.session_state.setdefault(K_HITS_PAGE_SEM, 0)
    st.session_state.setdefault(K_SEEN_NONCE, None)
    st.session_state.setdefault(K_DO_CLEAR, False)
    st.session_state.setdefault(K_SYNC_MULTI, None)
    st.session_state.setdefault(K_AI_ENGINE, {})
    st.session_state.setdefault(K_AI_METRICS, {})
    st.session_state.setdefault(K_SCROLL_TO_STEP2, False)
    st.session_state.setdefault(K_GLOBAL_HITS_SELECTED, {})

    # 1) Apply deferred CLEAR before any widgets are created
    if st.session_state.get(K_DO_CLEAR, False):
        _clear_menu1_state()
        st.session_state[K_DO_CLEAR] = False  # consume

    # 2) Auto-reset on mount
    _maybe_auto_reset_on_mount()

    # 3) Apply deferred MULTISET sync before the multiselect is created
    if st.session_state.get(K_SYNC_MULTI) is not None:
        st.session_state[K_MULTI_QUESTIONS] = list(st.session_state[K_SYNC_MULTI])  # type: ignore
        st.session_state[K_SYNC_MULTI] = None  # consume

    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # ============================ Step 1 (Title 2 / H2, compact) ==============================
    st.markdown("<div class='pses-h2'>Step 1: Pick up to 5 survey questions</div>", unsafe_allow_html=True)

    # ---- Single indented block: Select → or → Search (Title 3s, zero extra spacing) ----------
    st.markdown("<div class='pses-block'>", unsafe_allow_html=True)

    # Select from the list (nudged up to reduce space under Step 1 title)
    st.markdown("<div class='pses-h3 pses-no-top pses-nudge-up'>Select from the list</div>", unsafe_allow_html=True)

    def _on_list_change_scroll_step2():
        st.session_state[K_SCROLL_TO_STEP2] = True

    st.multiselect(
        "Choose one or more from the official list",
        qdf["display"].tolist(),
        max_selections=5,
        label_visibility="collapsed",
        key=K_MULTI_QUESTIONS,
        placeholder="",
        on_change=_on_list_change_scroll_step2,
    )

    # "or" (left-aligned, absolutely no spacing above/below)
    st.markdown("<div class='pses-h3 pses-zero'>or</div>", unsafe_allow_html=True)

    # Search questionnaire by keywords or theme (flush under "or")
    st.markdown("<div class='pses-h3 pses-no-top'>Search questionnaire by keywords or theme</div>", unsafe_allow_html=True)
    query = st.text_input(
        "Enter keywords (e.g., harassment, recognition, onboarding)",
        key=K_KW_QUERY,
        label_visibility="collapsed",
        placeholder='Type keywords like “career advancement”, “harassment”, “recognition”…',
    )

    # Buttons row (Search & Clear)
    bcol1, bcol2 = st.columns([0.5, 0.5])
    with bcol1:
        if st.button("Search the questionnaire", key=K_FIND_HITS_BTN):
            q = (query or "").strip()
            # Reset paginated hit selections on a *new* search universe
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

                # Diagnostics snapshots
                st.session_state[K_AI_ENGINE]  = get_embedding_status()
                st.session_state[K_AI_METRICS] = get_last_search_metrics()

                # Record last query + results
                st.session_state[K_SEARCH_DONE] = True
                st.session_state[K_LAST_QUERY] = q
                if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                    st.session_state[K_HITS] = hits_df[["code", "text", "display", "score", "origin"]].to_dict(orient="records")
                else:
                    st.session_state[K_HITS] = []
                st.session_state[K_HITS_PAGE_LEX] = 0
                st.session_state[K_HITS_PAGE_SEM] = 0

                # Mark timing in diagnostics (best-effort)
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

    # Close the unified indented block
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Search results (Title 3 content: EACH TAB CONTENT IS INDENTED) ----------------------
    hits = st.session_state.get(K_HITS, [])
    if st.session_state.get(K_SEARCH_DONE, False):
        if hits:
            lex_hits = [r for r in hits if r.get("origin","lex") == "lex"]
            sem_hits = [r for r in hits if r.get("origin","lex") == "sem"]

            tabs = st.tabs(["Lexical matches", "Other matches (semantic)"])

            # Lexical tab (indent INSIDE the tab)
            with tabs[0]:
                st.markdown("<div class='pses-block'>", unsafe_allow_html=True)

                total = len(lex_hits)
                page  = int(st.session_state.get(K_HITS_PAGE_LEX, 0)) or 0
                max_page = max(0, (total - 1) // PAGE_SIZE) if total else 0
                page = max(0, min(page, max_page))
                start = page * PAGE_SIZE
                end   = min(total, start + PAGE_SIZE)

                if total == 0:
                    st.warning("No lexical matches.")
                else:
                    st.info(f"Found {len(hits)} total matches meeting the quality threshold.")
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
                st.markdown("</div>", unsafe_allow_html=True)

            # Semantic tab (indent INSIDE the tab)
            with tabs[1]:
                st.markdown("<div class='pses-block'>", unsafe_allow_html=True)

                total = len(sem_hits)
                page  = int(st.session_state.get(K_HITS_PAGE_SEM, 0)) or 0
                max_page = max(0, (total - 1) // PAGE_SIZE) if total else 0
                page = max(0, min(page, max_page))
                start = page * PAGE_SIZE
                end   = min(total, start + PAGE_SIZE)

                if total == 0:
                    st.warning("No other (semantic) matches.")
                else:
                    gmap = dict(st.session_state.get(K_GLOBAL_HITS_SELECTED, {}))
                    for rec in sem_hits[start:end]:
                        code = rec["code"]; text = rec["text"]; score = rec.get("score", 0.0)
                        label = f"{code} — {text}  _(score: {score:.2f})_"
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
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            last_q = (st.session_state.get(K_LAST_QUERY) or "").strip()
            safe_q = last_q if last_q else "your search"
            st.warning(
                f'No questions matched “{safe_q}”. '
                'Try broader/different keywords (e.g., synonyms), split phrases, '
                'or search by a question code like “Q01”.'
            )

    # ============================ Step 2 (Title 2, pulled closer) ============================
    # anchor for smooth scroll
    st.markdown('<div id="step2_anchor"></div>', unsafe_allow_html=True)

    if st.session_state.get(K_SCROLL_TO_STEP2, False):
        components.html(
            """
            <script>
              const el = parent.document.getElementById('step2_anchor');
              if (el && el.scrollIntoView) { el.scrollIntoView({behavior:'smooth', block:'start'}); }
            </script>
            """,
            height=0,
        )
        st.session_state[K_SCROLL_TO_STEP2] = False

    # Tight top margin for Step 2 to sit ~one row below previous section
    st.markdown("<div class='pses-h2 pses-h2-tight-top'>Step 2: Select survey year(s)</div>", unsafe_allow_html=True)
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
    years_sorted = sorted(selected_years)  # retained if needed by caller

    # ============================ Step 3 (Title 2, compact; no gap to dropdown) ==============
    st.markdown("<div class='pses-h2'>Step 3: Select a demographic category (optional)</div>", unsafe_allow_html=True)
    # Note: demographic_picker() renders the dropdowns; CSS above removes the gap before them.

    # Return selected codes (API unchanged)
    return st.session_state[K_SELECTED_CODES]

# ---- Years (unchanged signature; used by caller) ----------------------------
def year_picker() -> List[int]:
    selected_years: List[int] = []
    for yr in DEFAULT_YEARS:
        if st.session_state.get(f"year_{yr}", False):
            selected_years.append(yr)
    return sorted(selected_years)

# ---- Demographics (no H2 here to avoid duplicate Step 3; only H3 when needed) --------------
def demographic_picker(demo_df: pd.DataFrame):
    ensure_pses_styles()

    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
    st.session_state.setdefault("demo_main", "All respondents")
    demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

    sub_selection: Optional[str] = None
    if demo_selection != "All respondents":
        # Title 3 + content block (indented)
        st.markdown("<div class='pses-block'>", unsafe_allow_html=True)
        st.markdown(f"<div class='pses-h3'>Subgroup ({demo_selection}) (optional)</div>", unsafe_allow_html=True)
        sub_items = demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL].dropna().astype(str).unique().tolist()
        sub_items = sorted(sub_items)
        sub_key = f"sub_{demo_selection.replace(' ', '_')}"
        sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=sub_key, label_visibility="collapsed")
        if sub_selection == "":
            sub_selection = None
        st.markdown("</div>", unsafe_allow_html=True)

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
        disp_map = {c: l for c, _ in keep}
        return demo_selection, sub_selection, codes, disp_map, True

    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        return demo_selection, sub_selection, labels, {l: l for l in labels}, True

    return demo_selection, sub_selection, [None], {None: "All respondents"}, False

# ---- Enable search? ---------------------------------------------------------
def search_button_enabled(question_codes: List[str], years: List[int]) -> bool:
    return bool(question_codes) and bool(years)
