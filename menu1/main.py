# menu1/main.py
from __future__ import annotations

import time
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

# your existing imports (kept from your previous version)
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
from .formatters import (
    drop_suppressed,
    scale_pairs,
    format_display,
    detect_metric,
)

# ─────────────────────────────────────────
# Small helper: inject scroll JS on rerun
# ─────────────────────────────────────────
def _scroll_if_needed() -> None:
    """
    Looks for st.session_state["_menu1_scroll_target"] and, if present,
    injects a tiny JS snippet to scroll to that div.
    Then clears the target.
    """
    target = st.session_state.get("_menu1_scroll_target")
    if not target:
        return

    # try both inside app and outer frame
    st.components.v1.html(
        f"""
        <script>
        const go = () => {{
            const el = document.getElementById("{target}") || window.parent.document.getElementById("{target}");
            if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        }};
        // slight delay to let streamlit finish layout
        setTimeout(go, 200);
        </script>
        """,
        height=0,
    )
    # reset so it doesn’t keep scrolling every rerun
    st.session_state["_menu1_scroll_target"] = None


def run() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")

    # init scroll target once
    if "_menu1_scroll_target" not in st.session_state:
        st.session_state["_menu1_scroll_target"] = None

    # run scroll listener at the very top of each rerun
    _scroll_if_needed()

    # load metadata (your original code did this at start)
    questions_df = load_questions()
    scales_df = load_scales()
    demo_df = load_demographics()

    # header / title
    layout.header(title=PAGE_TITLE, source_title=SOURCE_TITLE, source_url=SOURCE_URL)

    # ─────────────────────────────────────────
    # PART 1 — QUERY PARAMETERS
    # we put an anchor here for “after questionnaire search”
    # ─────────────────────────────────────────
    st.markdown('<div id="menu1-part1"></div>', unsafe_allow_html=True)

    # Render the controls
    params: Dict[str, Any] = controls.render(
        questions_df=questions_df,
        demographics_df=demo_df,
        scales_df=scales_df,
    )

    # 1) detect QUESTION change → scroll to rest of parameters (still part 1)
    current_q = params.get("question_code")
    last_q = st.session_state.get("_menu1_last_question")
    if current_q and current_q != last_q:
        # user just picked a question from “Search questionnaire …”
        st.session_state["_menu1_scroll_target"] = "menu1-part1-params"
    st.session_state["_menu1_last_question"] = current_q

    # this is a second anchor a bit lower in part 1 (right before years/demographics)
    st.markdown('<div id="menu1-part1-params"></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # PART 2 — RESULTS (we’ll scroll here after “Query and view results”)
    # ─────────────────────────────────────────
    st.markdown('<div id="menu1-part2-results"></div>', unsafe_allow_html=True)

    results_df: Optional[pd.DataFrame] = None
    ai_summary: Optional[str] = None

    # did the user click "Query and view results"?
    run_query = params.get("run_query", False)

    if run_query:
        # user explicitly asked for results → after this run, scroll to results section
        st.session_state["_menu1_scroll_target"] = "menu1-part2-results"

        # fetch data using your existing logic
        question_code = params.get("question_code")
        years = params.get("years", [])
        demo_codes = params.get("demo_codes", None) or params.get("demographics", None)

        raw = fetch_per_question(
            question_code=question_code,
            years=years,
            demographics=demo_codes,
        )

        if raw is not None and not raw.empty:
            # normalize / drop 999 / format as in your current app
            norm = normalize_results(raw)
            norm = drop_suppressed(norm)
            metric = detect_metric(norm)
            pairs = scale_pairs(scales_df, question_code)
            display_df = format_display(
                norm,
                demographics_map=params.get("demographics_map"),
                metric=metric,
                scale_pairs=pairs,
            )
            results_df = display_df
            results.render(display_df)
            diagnostics.render(raw, norm)
        else:
            st.info("No data found for the selected filters.")

    # ─────────────────────────────────────────
    # PART 3 — AI SUMMARY
    # ─────────────────────────────────────────
    st.markdown('<div id="menu1-part3-ai"></div>', unsafe_allow_html=True)

    run_ai = params.get("run_ai", False)

    if run_ai:
        # set the scroll target FIRST
        st.session_state["_menu1_scroll_target"] = "menu1-part3-ai"

        ai_summary = results.render_ai_summary(
            question_label=params.get("question_label"),
            results_df=results_df,
            years=params.get("years", []),
            demographics=params.get("demographics_map", {}),
        )

    # footer / navigation, if you have one
    layout.footer()


# add the name your app expects
def run_menu1() -> None:
    run()


if __name__ == "__main__":
    run_menu1()
