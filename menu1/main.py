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

    st.components.v1.html(
        f"""
        <script>
        const go = () => {{
            const el = document.getElementById("{target}") || window.parent.document.getElementById("{target}");
            if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        }};
        setTimeout(go, 200);
        </script>
        """,
        height=0,
    )
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

    # ─────────────────────────────────────────
    # HEADER / TITLE (safe version)
    # ─────────────────────────────────────────
    if hasattr(layout, "header"):
        # if your layout module actually has it in the future
        layout.header(title=PAGE_TITLE, source_title=SOURCE_TITLE, source_url=SOURCE_URL)
    else:
        # safe fallback
        st.markdown(f"## {PAGE_TITLE}")
        if SOURCE_TITLE and SOURCE_URL:
            st.markdown(f"[{SOURCE_TITLE}]({SOURCE_URL})")

    # ─────────────────────────────────────────
    # PART 1 — QUERY PARAMETERS
    # ─────────────────────────────────────────
    st.markdown('<div id="menu1-part1"></div>', unsafe_allow_html=True)

    params: Dict[str, Any] = controls.render(
        questions_df=questions_df,
        demographics_df=demo_df,
        scales_df=scales_df,
    )

    # detect QUESTION change → scroll to rest of parameters
    current_q = params.get("question_code")
    last_q = st.session_state.get("_menu1_last_question")
    if current_q and current_q != last_q:
        st.session_state["_menu1_scroll_target"] = "menu1-part1-params"
    st.session_state["_menu1_last_question"] = current_q

    # anchor lower in part 1
    st.markdown('<div id="menu1-part1-params"></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # PART 2 — RESULTS
    # ─────────────────────────────────────────
    st.markdown('<div id="menu1-part2-results"></div>', unsafe_allow_html=True)

    results_df: Optional[pd.DataFrame] = None

    run_query = params.get("run_query", False)

    if run_query:
        # scroll to results on next render
        st.session_state["_menu1_scroll_target"] = "menu1-part2-results"

        question_code = params.get("question_code")
        years = params.get("years", [])
        demo_codes = params.get("demo_codes", None) or params.get("demographics", None)

        raw = fetch_per_question(
            question_code=question_code,
            years=years,
            demographics=demo_codes,
        )

        if raw is not None and not raw.empty:
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
        st.session_state["_menu1_scroll_target"] = "menu1-part3-ai"

        _ = results.render_ai_summary(
            question_label=params.get("question_label"),
            results_df=results_df,
            years=params.get("years", []),
            demographics=params.get("demographics_map", {}),
        )

    # ─────────────────────────────────────────
    # FOOTER (safe)
    # ─────────────────────────────────────────
    if hasattr(layout, "footer"):
        layout.footer()
    else:
        # minimal footer or nothing
        pass


# app expects run_menu1
def run_menu1() -> None:
    run()


if __name__ == "__main__":
    run_menu1()
