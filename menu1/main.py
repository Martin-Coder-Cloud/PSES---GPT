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
    """
    Create the Summary tabulation:
      - Row index: Question code only (or Question×Demographic if a category is selected without a specific subgroup)
      - Columns: selected years
      - Values: detected metric per question (mean across demo rows when needed)
    """
    if not per_q_disp:
        return pd.DataFrame()

    long_rows = []
    for qcode, df_disp in per_q_disp.items():
        metric_col = per_q_metric_col.get(qcode)
        if not metric_col or metric_col not in df_disp.columns:
            continue

        t = df_disp.copy()
        t["QuestionLabel"] = qcode  # code only
        t["Year"] = pd.to_numeric(t["Year"], errors="coerce").astype("Int64")
        if "Demographic" not in t.columns:
            t["Demographic"] = None

        t = t.rename(columns={metric_col: "Value"})
        long_rows.append(t[["QuestionLabel", "Demographic", "Year", "Value"]])

    if not long_rows:
        return pd.DataFrame()

    long_df = pd.concat(long_rows, ignore_index=True)

    # If a demographic category is selected (and no single subgroup), preserve the demo rows;
    # otherwise, index only by question code.
    if (demo_selection is not None) and (demo_selection != "All respondents") and (sub_selection is None) and long_df["Demographic"].notna().any():
        idx_cols = ["QuestionLabel", "Demographic"]
    else:
        idx_cols = ["QuestionLabel"]

    pivot = long_df.pivot_table(index=idx_cols, columns="Year", values="Value", aggfunc="mean")
    pivot = pivot.reindex(years, axis=1)  # ensure column order matches selected years
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
    # Clear dynamic checkbox keys from previous hits and selections
    for k in list(st.session_state.keys()):
        if k.startswith("kwhit_") or k.startswith("sel_"):
            st.session_state.pop(k, None)


def run() -> None:
    # NOTE: st.set_page_config() is intentionally NOT called here
    # to avoid double-calling it (root main.py calls it once).

    # Scoped CSS: ONLY the main Search button is red/white. All others use your global/default style.
    # Use very high specificity so hosted themes cannot override it.
    st.markdown(
        """
        <style>
          .action-row { margin-top: .25rem; margin-bottom: .35rem; }

          /* Solid red for ONLY the Search button inside #menu1-run-btn */
          [data-testid="stAppViewContainer"] .block-container #menu1-run-btn .stButton > button {
            background-color: #e03131 !important;  /* red */
            color: #ffffff !important;             /* white text */
            border: 1px solid #c92a2a !important;  /* red border */
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

          /* Keep the reset/clear button left-aligned; use default theme */
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

        # [AI-toggle gate] Track toggle changes without triggering rebuilds
        _prev_ai = st.session_state.get("menu1_ai_prev", ai_on)
        if _prev_ai != ai_on:
            st.session_state["menu1_ai_prev"] = ai_on
            st.session_state["menu1_ai_toggle_dirty"] = True
        else:
            # initialize on first load
            if "menu1_ai_prev" not in st.session_state:
                st.session_state["menu1_ai_prev"] = ai_on

        layout.instructions()

        # Reset when arriving fresh from another menu
        if state.get_last_active_menu() != "menu1":
            state.reset_menu1_state()
            _clear_keyword_search_state()  # also clear keyword UI state on first arrival
        state.set_last_active_menu("menu1")
        state.set_defaults()  # idempotent

        # Metadata (cached)
        qdf = load_questions()
        sdf = load_scales()
        demo_df = load_demographics()

        # Diagnostics (tabs)
        if show_diag:
            diagnostics.render_diagnostics_tabs(qdf, sdf, demo_df)

            # --- ADDED: show AI diagnostics captured in results.py ---
            ai_diag = st.session_state.get("menu1_ai_diag")
            if ai_diag:
                st.markdown("#### AI diagnostics")
                st.json(ai_diag)
            # --- END ADD ---

        # Controls
        question_codes = controls.question_picker(qdf)  # -> List[str] (codes)
        years = controls.year_picker()                  # -> List[int]
        demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

        # Action row: Search / Clear (side-by-side, aligned left)
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

                # Build per-question display tables
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

                # Build pivot & stash results for centered rendering
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

                # Mark diagnostics timing
                diagnostics.mark_last_query(
                    started_ts=t0,
                    finished_ts=time.time(),
                    extra={"notes": "Menu 1 query run"},
                )

                # [AI-toggle gate] A fresh Search clears the dirty flag
                st.session_state["menu1_ai_toggle_dirty"] = False

        with colB:
            st.markdown("<div id='menu1-reset-btn'>", unsafe_allow_html=True)
            # Label per your UX spec (“Clear parameters” beside Search)
            if st.button("Clear parameters", key="menu1_reset_all"):
                # Reset core menu state
                state.reset_menu1_state()
                # Also clear keyword-search UI state so no stale "No questions matched…" persists
                _clear_keyword_search_state()
                # Clear AI caches (to prevent reruns from showing stale narratives)
                st.session_state.pop("menu1_ai_cache", None)
                st.session_state.pop("menu1_ai_narr_per_q", None)
                st.session_state.pop("menu1_ai_narr_overall", None)
                # [AI-toggle gate] Clearing parameters also clears the dirty flag
                st.session_state.pop("menu1_ai_toggle_dirty", None)
                # Rerun
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Results (center area)
        if state.has_results():
            # [AI-toggle gate] If the AI toggle changed since last Search, do not render results
            if st.session_state.get("menu1_ai_toggle_dirty", False):
                st.info("AI setting changed — click **Search** to refresh results.")
            else:
                payload = state.get_results()
                results.tabs_summary_and_per_q(
                    payload=payload,
                    ai_on=ai_on,
                    build_overall_prompt=build_overall_prompt,  # pass directly
                    build_per_q_prompt=build_per_q_prompt,      # pass directly
                    call_openai_json=call_openai_json,          # pass directly
                    source_url=SOURCE_URL,
                    source_title=SOURCE_TITLE,
                )


if __name__ == "__main__":
    run()

# --- keep this at the end of menu1/main.py ---
def run_menu1():
    # backward-compat alias for older loaders that expect run_menu1
    return run()
