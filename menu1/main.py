# menu1/main.py
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Local modules
from .constants import (
    PAGE_TITLE,
    CENTER_COLUMNS,
    SOURCE_URL,
    SOURCE_TITLE,
)
from . import state
from .metadata import load_questions, load_scales, load_demographics
from .render import layout, controls, diagnostics, results
from .queries import fetch_per_question, normalize_results
from .formatters import drop_suppressed, scale_pairs, format_display, detect_metric
from .ai import build_overall_prompt, build_per_q_prompt, call_openai_json  # <-- direct imports


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
        t["QuestionLabel"] = qcode  # IMPORTANT: code only (no text) per your requirement
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
    # Ensure our column order matches the selected years
    pivot = pivot.reindex(years, axis=1)
    return pivot


def run() -> None:
    # Page config + centered layout
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    left, center, right = layout.centered_page(CENTER_COLUMNS)
    with center:
        # Header
        layout.banner()
        layout.title("PSES Explorer Search")
        ai_on, show_diag = layout.toggles()
        layout.instructions()

        # Reset when arriving fresh from another menu
        if state.get_last_active_menu() != "menu1":
            state.reset_menu1_state()
        state.set_last_active_menu("menu1")
        state.set_defaults()  # idempotent

        # Metadata (cached)
        qdf = load_questions()
        sdf = load_scales()
        demo_df = load_demographics()

        # Optional diagnostics
        if show_diag:
            diagnostics.parameters_preview(qdf, demo_df)
            diagnostics.backend_info_panel(qdf, sdf, demo_df)
            diagnostics.last_query_panel()

        # Controls
        question_codes = controls.question_picker(qdf)  # -> List[str] (codes)
        years = controls.year_picker()                  # -> List[int]
        demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

        # Action row: Search / Reset
        st.markdown("<div class='action-row'>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1])
        with colA:
            can_search = controls.search_button_enabled(question_codes, years)
            if st.button("Search", disabled=not can_search):
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

        with colB:
            if st.button("Reset all parameters"):
                state.reset_menu1_state()
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Results (center area)
        if state.has_results():
            payload = state.get_results()
            results.tabs_summary_and_per_q(
                payload=payload,
                ai_on=ai_on,
                build_overall_prompt=build_overall_prompt,  # <-- pass directly
                build_per_q_prompt=build_per_q_prompt,      # <-- pass directly
                call_openai_json=call_openai_json,          # <-- pass directly
                source_url=SOURCE_URL,
                source_title=SOURCE_TITLE,
            )


if __name__ == "__main__":
    run()

# --- keep this at the end of menu1/main.py ---
def run_menu1():
    # backward-compat alias for older loaders that expect run_menu1
    return run()
