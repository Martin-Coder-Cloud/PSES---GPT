# menu1/main.py
from __future__ import annotations
import streamlit as st

# Layout lives under menu1/render/layout.py
from menu1.render.layout import centered_page, banner, title, toggles, instructions

# Constants live under menu1/constants.py
from menu1.constants import PAGE_TITLE, CENTER_COLUMNS, SOURCE_URL, SOURCE_TITLE

# These are root-level modules in your project
from state import (
    set_defaults,
    reset_menu1_state,
    set_last_active_menu,
    has_results,
    get_results,
    stash_results,  # keep if your search flow uses it
)
from metadata import load_questions, load_demographics

# Controls & results under menu1/render/
from menu1.render import controls
from menu1.render import results as menu1_results

# Optional helper (safe if missing)
try:
    from menu1.ai import do_menu1_search  # expected: (question_codes, years, demcodes) -> any
except Exception:
    do_menu1_search = None  # type: ignore

MENU_KEY = "menu1"

def _inject_primary_search_button_css() -> None:
    """
    Style ONLY the 'Search the survey results' button via a tightly-scoped wrapper.
    Other buttons remain unchanged.
    """
    st.markdown(
        """
        <style>
          /* Only the button inside this specific wrapper gets the red/white style */
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

def _render_menu1() -> None:
    # --- Page chrome (signature takes no args) ---
    centered_page()
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

    # --- Step 1 / 2 / 3 (behavior unchanged; handled by controls.py) ---
    question_codes = controls.question_picker(qdf)     # ordered list[str], max 5
    years         = controls.year_picker()             # list[int]
    demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

    # --- Action row: Search + Clear (left-aligned, same row) ---
    _inject_primary_search_button_css()

    can_search = controls.search_button_enabled(question_codes, years)

    # Two narrow columns + a wide spacer keeps both buttons left-aligned (not centered)
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

    # --- Actions (plug your existing logic here; nothing else changed) ---
    if search_clicked:
        # If you expose a helper, call it; otherwise keep your existing flow.
        if callable(do_menu1_search):
            try:
                data = do_menu1_search(question_codes, years, demcodes)  # type: ignore
                if data is not None:
                    stash_results(data)
            except Exception as e:
                st.error(f"Search failed: {type(e).__name__}: {e}")
        # If your app already performs the search elsewhere, leave as-is.

    if reset_clicked:
        # Fully reset Menu 1 UI/state, including warnings like "no matches"
        reset_menu1_state()
        # Immediate refresh so the cleared UI shows right away
        try:
            st.rerun()              # Streamlit ≥ 1.30
        except Exception:
            st.experimental_rerun() # Back-compat

    # --- Results (unchanged) ---
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

def run_menu1() -> None:
    """Public entrypoint expected by the app (restored)."""
    try:
        _render_menu1()
    except Exception as e:
        st.error(f"Menu 1 is unavailable: {type(e).__name__}: {e}")

# Optional: allow running this module directly
def main():
    run_menu1()

if __name__ == "__main__":
    main()
