# menu1/main.py — two-pass scroll (fixed): anchor lives just below controls; pre-scroll fires there
from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # for smooth scroll

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
from .ai import build_overall_prompt, build_per_q_prompt, call_openai_json


def _build_summary_pivot(
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    years: List[int],
    demo_selection: Optional[str],
    sub_selection: Optional[str],
) -> pd.DataFrame:
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

    if (demo_selection is not None) and (demo_selection != "All respondents") and (sub_selection is None) and long_df["Demographic"].notna().any():
        idx_cols = ["QuestionLabel", "Demographic"]
    else:
        idx_cols = ["QuestionLabel"]

    pivot = long_df.pivot_table(index=idx_cols, columns="Year", values="Value", aggfunc="mean")
    pivot = pivot.reindex(years, axis=1)
    return pivot


def _clear_keyword_search_state() -> None:
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
    # Scoped CSS
    st.markdown(
        """
        <style>
          .action-row { margin-top: .25rem; margin-bottom: .35rem; }
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button {
            background-color: #e03131 !important;
            color: #ffffff !important;
            border: 1px solid #c92a2a !important;
            font-weight: 700 !important;
          }
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button:hover {
            background-color: #c92a2a !important;
            border-color: #a61e1e !important;
          }
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button:disabled {
            opacity: 0.50 !important;
            filter: saturate(0.85);
            color: #ffffff !important;
            background-color: #e03131 !important;
            border-color: #c92a2a !important;
          }
          #menu1-reset-btn { text-align: left; }

          /* brief pulse highlight for results after search */
          @keyframes pulseFade {
            0%   { background: rgba(255, 230, 120, 0.55); }
            100% { background: transparent; }
          }
          .pulse-highlight {
            animation: pulseFade 1400ms ease-out 1;
            border-radius: 10px;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    left, center, right = layout.centered_page(CENTER_COLUMNS)
    with center:
        layout.banner()
        layout.title("PSES Explorer Search")
        ai_on, show_diag = layout.toggles()

        # Track AI toggle changes
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

        # Metadata (cached)
        qdf = load_questions()
        sdf = load_scales()
        demo_df = load_demographics()

        # Diagnostics (tabs)
        if show_diag:
            diagnostics.render_diagnostics_tabs(qdf, sdf, demo_df)

        # Controls
        question_codes = controls.question_picker(qdf)
        years = controls.year_picker()
        demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

        # Action row: Search / Clear
        st.markdown("<div class='action-row'>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1], gap="small")

        with colA:
            can_search = controls.search_button_enabled(question_codes, years)
            st.markdown("<div id='menu1-run-btn' style='text-align:left;'>", unsafe_allow_html=True)
            run_clicked = st.button("Search the survey results", key="menu1_run_query", disabled=not can_search)
            st.markdown("</div>", unsafe_allow_html=True)

            if run_clicked:
                # Request pre-scroll (handled just below, after the anchor is emitted)
                st.session_state["_pre_scroll_results"] = True
                st.experimental_rerun()

        with colB:
            st.markdown("<div id='menu1-reset-btn'>", unsafe_allow_html=True)
            if st.button("Clear parameters", key="menu1_reset_all"):
                state.reset_menu1_state()
                _clear_keyword_search_state()
                st.session_state.pop("menu1_ai_cache", None)
                st.session_state.pop("menu1_ai_narr_per_q", None)
                st.session_state.pop("menu1_ai_narr_overall", None)
                st.session_state.pop("_focus_results", None)
                st.session_state.pop("_pre_scroll_results", None)
                st.session_state.pop("_do_results_search", None)
                st.session_state.pop("menu1_ai_toggle_dirty", None)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # end action row

        # ── Anchor lives JUST BELOW the controls; this is where we want to scroll to ──
        st.markdown("<span id='results-anchor'></span>", unsafe_allow_html=True)

        # If a pre-scroll was requested, fire it RIGHT HERE (downward), then do the real search next run
        if st.session_state.get("_pre_scroll_results"):
            components.html(
                """
                <script>
                  const go = () => {
                    const el = window.parent.document.querySelector('span#results-anchor');
                    if (el) el.scrollIntoView({behavior:'smooth', block:'start', inline:'nearest'});
                  };
                  go(); setTimeout(go, 200); setTimeout(go, 600);
                </script>
                """,
                height=0, width=0
            )
            st.session_state["_pre_scroll_results"] = False
            st.session_state["_do_results_search"] = True
            st.experimental_rerun()

        # ── Run the actual search after the pre-scroll ──
        if st.session_state.pop("_do_results_search", False):
            st.session_state["_focus_results"] = True  # for the highlight

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

        # ── Results ──
        if state.has_results():
            if st.session_state.get("menu1_ai_toggle_dirty", False):
                st.info("AI setting changed — click **Search** to refresh results.")
            else:
                payload = state.get_results()
                wrap_cls = "pulse-highlight" if st.session_state.get("_focus_results") else ""
                st.markdown(f"<div class='{wrap_cls}'>", unsafe_allow_html=True)

                results.tabs_summary_and_per_q(
                    payload=payload,
                    ai_on=ai_on,
                    build_overall_prompt=build_overall_prompt,
                    build_per_q_prompt=build_per_q_prompt,
                    call_openai_json=call_openai_json,
                    source_url=SOURCE_URL,
                    source_title=SOURCE_TITLE,
                )

                st.markdown("</div>", unsafe_allow_html=True)
                if st.session_state.get("_focus_results"):
                    st.session_state["_focus_results"] = False


if __name__ == "__main__":
    run()

# Back-compat alias for older loaders
def run_menu1():
    return run()
