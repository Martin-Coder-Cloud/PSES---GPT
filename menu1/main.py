# menu1/main.py
from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Local modules (relative to the menu1 package)
from .constants import (
    PAGE_TITLE,          # kept for parity with your imports (not used directly here)
    CENTER_COLUMNS,
    SOURCE_URL,
    SOURCE_TITLE,
)
from . import state
from .metadata import load_questions, load_scales, load_demographics
from .render import layout, controls, diagnostics, results
from .queries import fetch_per_question, normalize_results
from .formatters import drop_suppressed, scale_pairs, format_display, detect_metric
from .ai import build_overall_prompt, build_per_q_prompt, call_openai_json  # direct imports


def _build_summary_pivot(
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    years: List[int],
    demo_selection: Optional[str],
    sub_selection: Optional[str],
) -> pd.DataFrame:
    """Create the Summary tabulation."""
    if not per_q_disp:
        return pd.DataFrame()

    long_rows = []
    for qcode, df_disp in per_q_disp.items():
        metric_col = per_q_metric_col.get(qcode)
        if not metric_col or metric_col not in df_disp.columns:
            continue

        t = df_disp.copy()
        t["QuestionLabel"] = qcode
        t["Year"] = pd.to_numeric(t["Year"], errors="coerce").astype("Int64")
        if "Demographic" not in t.columns:
            t["Demographic"] = None

        t = t.rename(columns={metric_col: "Value"})
        long_rows.append(t[["QuestionLabel", "Demographic", "Year", "Value"]])

    if not long_rows:
        return pd.DataFrame()

    long_df = pd.concat(long_rows, ignore_index=True)

    if (
        (demo_selection is not None)
        and (demo_selection != "All respondents")
        and (sub_selection is None)
        and long_df["Demographic"].notna().any()
    ):
        idx_cols = ["QuestionLabel", "Demographic"]
    else:
        idx_cols = ["QuestionLabel"]

    pivot = long_df.pivot_table(index=idx_cols, columns="Year", values="Value", aggfunc="mean")
    pivot = pivot.reindex(years, axis=1)
    return pivot


def _clear_keyword_search_state() -> None:
    """Remove all keys related to the keyword search so no stale warnings remain."""
    for k in [
        "menu1_hits",
        "menu1_search_done",
        "menu1_last_search_query",
        "menu1_kw_query",
    ]:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if k.startswith("kwhit_") or k.startswith("sel_"):
            st.session_state.pop(k, None)


def run() -> None:
    # NOTE: st.set_page_config() is intentionally NOT called here
    # to avoid double-calling it (root main.py calls it once).

    # Scoped CSS: BOTH bottom action buttons (Search / Clear) red & white
    st.markdown(
        """
        <style>
          .action-row { margin-top: .25rem; margin-bottom: .35rem; }

          /* Solid red for Search and Clear buttons at the bottom */
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button,
          [data-testid="stAppViewContainer"] .block-container #menu1-reset-btn .stButton > button {
            background-color: #e03131 !important;  /* bright red */
            color: #ffffff !important;              /* white text */
            border: 1px solid #c92a2a !important;
            font-weight: 700 !important;
          }
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button:hover,
          [data-testid="stAppViewContainer"] .block-container #menu1-reset-btn .stButton > button:hover {
            background-color: #c92a2a !important;   /* darker on hover */
            border-color: #a61e1e !important;
            color: #ffffff !important;
          }
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button:active,
          [data-testid="stAppViewContainer"] .block-container #menu1-reset-btn .stButton > button:active {
            background-color: #a61e1e !important;   /* deepest red on click */
            border-color: #8c1a1a !important;
            color: #ffffff !important;
          }
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button:disabled {
            opacity: 0.50 !important;
            filter: saturate(0.85);
            color: #ffffff !important;
            background-color: #e03131 !important;
            border-color: #c92a2a !important;
          }

          #menu1-reset-btn { text-align: left; }
        </style>
        """,
        unsafe_allow_html=True
    )

    left, center, right = layout.centered_page(CENTER_COLUMNS)
    with center:
        # Header
        layout.banner()
        layout.title("PSES Explorer Search")
        ai_on, show_diag = layout.toggles()

        _prev_ai = st.session_state.get("menu1_ai_prev", ai_on)
        if _prev_ai != ai_on:
            st.session_state["menu1_ai_prev"] = ai_on
            st.session_state["menu1_ai_toggle_dirty"] = True
        else:
            if "menu1_ai_prev" not in st.session_state:
                st.session_state["menu1_ai_prev"] = ai_on

        layout.instructions()

        # Reset when arriving fresh from another menu
        if state.get_last_active_menu() != "menu1":
            state.reset_menu1_state()
            _clear_keyword_search_state()
        state.set_last_active_menu("menu1")
        state.set_defaults()

        qdf = load_questions()
        sdf = load_scales()
        demo_df = load_demographics()

        # Diagnostics (tabs)
        if show_diag:
            diag_tab, ai_diag_tab = st.tabs(["Diagnostics", "AI diagnostics"])
            with diag_tab:
                diagnostics.render_diagnostics_tabs(qdf, sdf, demo_df)
            with ai_diag_tab:
                ai_diag = st.session_state.get("menu1_ai_diag")
                if ai_diag:
                    st.json(ai_diag)
                else:
                    st.caption("No AI diagnostics captured yet. Run a search with AI turned on.")

        # Controls
        question_codes = controls.question_picker(qdf)
        years = controls.year_picker()
        demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

        # Action row: Search / Clear (side-by-side)
        st.markdown("<div class='action-row'>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1], gap="small")

        with colA:
            can_search = controls.search_button_enabled(question_codes, years)
            st.markdown("<div id='menu1-run-btn' style='text-align:left;'>", unsafe_allow_html=True)
            run_clicked = st.button("Search the survey results", key="menu1_run_query", disabled=not can_search)
            st.markdown("</div>", unsafe_allow_html=True)

            if run_clicked:
                t0 = time.time()
                per_q_disp: Dict[str, pd.DataFrame] = {}
                per_q_metric_col: Dict[str, str] = {}
                per_q_metric_label: Dict[str, str] = {}

                for qcode in question_codes:
                    df_all = fetch_per_question(qcode, years, demcodes)
                    if df_all is None or df_all.empty:
                        continue

                    df_all = normalize_results(df_all)
                    df_all = drop_suppressed(df_all)

                    spairs = scale_pairs(sdf, qcode)
                    df_disp = format_display(
                        df_slice=df_all,
                        dem_disp_map=disp_map,
                        category_in_play=category_in_play,
                        scale_pairs=spairs,
                    )
                    if df_disp.empty:
                        continue

                    det = detect_metric(df_disp, spairs)
                    per_q_disp[qcode] = df_disp
                    per_q_metric_col[qcode] = det["metric_col"]
                    per_q_metric_label[qcode] = det["metric_label"]

                if per_q_disp:
                    pivot = _build_summary_pivot(
                        per_q_disp=per_q_disp,
                        per_q_metric_col=per_q_metric_col,
                        years=years,
                        demo_selection=demo_selection,
                        sub_selection=sub_selection,
                    )
                    code_to_text = dict(zip(qdf["code"], qdf["text"]))
                    state.stash_results({
                        "per_q_disp": per_q_disp,
                        "per_q_metric_col": per_q_metric_col,
                        "per_q_metric_label": per_q_metric_label,
                        "pivot": pivot,
                        "tab_labels": [qc for qc in question_codes if qc in per_q_disp],
                        "years": years,
                        "demo_selection": demo_selection,
                        "sub_selection": sub_selection,
                        "code_to_text": code_to_text,
                    })

                diagnostics.mark_last_query(
                    started_ts=t0,
                    finished_ts=time.time(),
                    extra={"notes": "Menu 1 query run"},
                )
                st.session_state["menu1_ai_toggle_dirty"] = False

        with colB:
            st.markdown("<div id='menu1-reset-btn'>", unsafe_allow_html=True)
            if st.button("Clear parameters", key="menu1_reset_all"):
                state.reset_menu1_state()
                _clear_keyword_search_state()
                st.session_state.pop("menu1_ai_cache", None)
                st.session_state.pop("menu1_ai_narr_per_q", None)
                st.session_state.pop("menu1_ai_narr_overall", None)
                st.session_state.pop("menu1_ai_toggle_dirty", None)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if state.has_results():
            if st.session_state.get("menu1_ai_toggle_dirty", False):
                st.info("AI setting changed — click **Search** to refresh results.")
            else:
                payload = state.get_results()
                results.tabs_summary_and_per_q(
                    payload=payload,
                    ai_on=ai_on,
                    build_overall_prompt=build_overall_prompt,
                    build_per_q_prompt=build_per_q_prompt,
                    call_openai_json=call_openai_json,
                    source_url=SOURCE_URL,
                    source_title=SOURCE_TITLE,
                )


if __name__ == "__main__":
    run()

def run_menu1():
    return run()
