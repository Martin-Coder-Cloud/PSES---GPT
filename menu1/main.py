# menu1/main.py
from __future__ import annotations
import streamlit as st

# Your layout lives under menu1/render/layout.py
from menu1.render.layout import centered_page, banner, title, toggles, instructions

# Constants live under menu1/constants.py (fixes import path)
from menu1.constants import PAGE_TITLE, CENTER_COLUMNS, SOURCE_URL, SOURCE_TITLE

# State & data helpers (paths based on your existing project layout)
from state import (
    set_defaults,
    reset_menu1_state,
    set_last_active_menu,
    has_results,
    get_results,
    stash_results,  # kept for compatibility if you already use it in your search flow
)

from metadata import load_questions, load_demographics

# Controls & results
from menu1.render import controls
from menu1.render import results as menu1_results

# Optional: call a helper if you expose one in menu1/ai.py. Safe if missing.
try:
    from menu1.ai import do_menu1_search  # expected: do_menu1_search(question_codes, years, demcodes) -> any
except Exception:
    do_menu1_search = None  # type: ignore

MENU_KEY = "menu1"

def _inject_primary_search_button_css() -> None:
    """
    Style ONLY the 'Search the survey results' button via a tightly-scoped wrapper.
    Other buttons remain unaffected.
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
    # --- Page chrome ---
    centered_page(cols=CENTER_COLUMNS)
    banner()
    title(PAGE_TITLE)
    toggles()
    instructions()

    # --- Init & mark active ---
    set_defaults()
    set_last_active_menu(MENU_KEY)

    # --- Data for controls ---
    qdf = load_questions()        # expects columns: code, text, display
    demo_df = load_demographics()

    # --- Step 1 / 2 / 3 (handled by controls.py; unchanged behavior) ---
    question_codes = controls.question_picker(qdf)     # ordered list[str], max 5
    years         = controls.year_picker()             # list[int]
    demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

    # --- Action row: Search + Clear (left-aligned, same row) ---
    _inject_primary_search_button_css()

    can_search = controls.search_button_enabled(question_codes, years)

    # Keep both buttons on the left: two narrow columns + a spacer to prevent centering
    c1, c2, _spacer = st.columns([1, 1, 8])

    with c1:
        # Wrap ONLY this button so the scoped CSS applies to it (and nothing else)
        st.markdown('<div id="pses-primary-search-btn">', unsafe_allow_html=True)
        search_clicked = st.button("Search the survey results", key="menu1_run_search")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        reset_clicked = st.button("Clear parameters", key="menu1_clear_params")

    # Enforce enablement (unchanged rule)
    if search_clicked and not can_search:
        st.warning("Please select at least one question and at least one year before searching.")
        search_clicked = False

    # --- Actions (no behavioral changes beyond where you already handle them) ---
    if search_clicked:
        # If you expose a helper, we call it; otherwise your existing flow can remain elsewhere.
        if callable(do_menu1_search):
            try:
                data = do_menu1_search(question_codes, years, demcodes)  # type: ignore
                if data is not None:
                    stash_results(data)
            except Exception as e:
                st.error(f"Search failed: {type(e).__name__}: {e}")
        # If your app already performs the search in another layer, leave as-is.

    if reset_clicked:
        # Fully reset Menu 1 UI and state; also clears warnings like "no matches"
        reset_menu1_state()
        # Immediate refresh so the cleared UI shows right away
        try:
            st.rerun()  # Streamlit >= 1.30
        except Exception:
            st.experimental_rerun()

    # --- Results rendering (unchanged) ---
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
