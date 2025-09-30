# menu1/main.py
from __future__ import annotations
import streamlit as st

# Interfaces you listed (signatures only)
from layout import centered_page, banner, title, toggles, instructions
from constants import PAGE_TITLE, CENTER_COLUMNS, SOURCE_URL, SOURCE_TITLE
from state import (
    set_defaults,
    reset_menu1_state,
    get_last_active_menu,
    set_last_active_menu,
    stash_results,
    has_results,
    get_results,
)
from metadata import load_questions, load_demographics
from menu1.render import controls
from menu1.render import results as menu1_results

MENU_KEY = "menu1"

def _inject_primary_search_button_css() -> None:
    """
    Style ONLY the 'Search the survey results' button.
    We scope the style to a unique wrapper so NOTHING else is affected.
    """
    st.markdown(
        """
        <style>
          /* Only affect the button inside this specific wrapper */
          #pses-primary-search-btn button {
            background: #d90429 !important;       /* red */
            color: #ffffff !important;            /* white text */
            border: 1px solid #d90429 !important; /* red border */
          }
          #pses-primary-search-btn button:hover {
            filter: brightness(0.92);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_menu() -> None:
    # Page chrome
    centered_page(cols=CENTER_COLUMNS)
    banner()
    title(PAGE_TITLE)
    toggles()
    instructions()

    # Init and mark this menu as active
    set_defaults()
    set_last_active_menu(MENU_KEY)

    # Load metadata for controls
    qdf = load_questions()        # expects columns: code, text, display
    demo_df = load_demographics()

    # ----- Step 1 / 2 / 3 controls (unchanged behavior) -----
    question_codes = controls.question_picker(qdf)     # ordered list[str], max 5
    years         = controls.year_picker()             # list[int]
    demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

    # ----- Action row: Search + Clear (left-aligned on the same row) -----
    _inject_primary_search_button_css()

    # Whether a search is allowed (unchanged rule)
    can_search = controls.search_button_enabled(question_codes, years)

    # Two narrow columns + a large spacer keeps them LEFT and prevents centering
    c1, c2, _spacer = st.columns([1, 1, 8])

    with c1:
        # Wrap ONLY this button for the red/white style
        st.markdown('<div id="pses-primary-search-btn">', unsafe_allow_html=True)
        search_clicked = st.button("Search the survey results", key="menu1_run_search", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        reset_clicked = st.button("Clear parameters", key="menu1_clear_params", use_container_width=True)

    # Guard: enforce enablement (same behavior as before)
    if search_clicked and not can_search:
        st.warning("Please select at least one question and at least one year before searching.")
        search_clicked = False

    # ---- Handle actions (drop your existing logic in these spots) ----
    if search_clicked:
        # TODO: insert your existing search logic here, e.g.:
        # data = run_your_query(question_codes, years, demcodes)
        # stash_results(data)
        pass

    if reset_clicked:
        # Fully reset Menu 1 state (this should also clear any warnings/results)
        reset_menu1_state()
        # Force a rerun so the UI reflects the cleared state immediately
        try:
            st.rerun()  # Streamlit >= 1.30
        except Exception:
            st.experimental_rerun()  # Back-compat

    # ---- Render results (unchanged) ----
    if has_results():
        menu1_results.render(
            get_results(),
            years,
            demcodes,
            disp_map,
            category_in_play,
            source_title=SOURCE_TITLE,
            source_url=SOURCE_URL,
        )

def main():
    try:
        render_menu()
    except Exception as e:
        st.error(f"Menu 1 is unavailable: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
