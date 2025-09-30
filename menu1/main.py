# menu1/main.py
from __future__ import annotations
import streamlit as st

# Layout lives under menu1/render/layout.py per your structure
from menu1.render.layout import centered_page, banner, title, toggles, instructions

# Shared modules (these were listed as project interfaces)
from constants import PAGE_TITLE, CENTER_COLUMNS, SOURCE_URL, SOURCE_TITLE
from state import (
    set_defaults,
    reset_menu1_state,
    set_last_active_menu,
    has_results,
    get_results,
    stash_results,  # kept for backward compatibility if you use it in your search logic
)

from metadata import load_questions, load_demographics

# Controls & results for Menu 1
from menu1.render import controls
from menu1.render import results as menu1_results

# Optional search helper (called if present; otherwise we won’t error)
try:
    from menu1.ai import do_menu1_search  # expected signature: (question_codes, years, demcodes) -> any
except Exception:
    do_menu1_search = None  # type: ignore

MENU_KEY = "menu1"

def _inject_primary_search_button_css() -> None:
    """
    Style ONLY the 'Search the survey results' button via a tightly-scoped wrapper.
    Other buttons are unaffected by this CSS.
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
    # --- Chrome ---
    centered_page(cols=CENTER_COLUMNS)
    banner()
    title(PAGE_TITLE)
    toggles()
    instructions()

    # --- Init & mark active ---
    set_defaults()
    set_last_active_menu(MENU_KEY)

    # --- Metadata for controls ---
    qdf = load_questions()        # expects columns: code, text, display
    demo_df = load_demographics()

    # --- Step 1 / 2 / 3 (unchanged behavior; handled by controls.py) ---
    question_codes = controls.question_picker(qdf)     # ordered list[str], max 5
    years         = controls.year_picker()             # list[int]
    demo_selection, sub_selection, demcodes, disp_map, category_in_play = controls.demographic_picker(demo_df)

    # --- Action row: Search + Clear (left-aligned, same row) ---
    _inject_primary_search_button_css()

    # Leave your enablement rule intact
    can_search = controls.search_button_enabled(question_codes, years)

    # Keep the buttons left by using two narrow columns + a spacer
    c1, c2, _spacer = st.columns([1, 1, 8])

    with c1:
        # Wrap ONLY this button so the scoped CSS applies to it (and nothing else)
        st.markdown('<div id="pses-primary-search-btn">', unsafe_allow_html=True)
        search_clicked = st.button("Search the survey results", key="menu1_run_search", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        reset_clicked = st.button("Clear parameters", key="menu1_clear_params", use_container_width=True)

    # Enforce enablement (unchanged)
    if search_clicked and not can_search:
        st.warning("Please select at least one question and at least one year before searching.")
        search_clicked = False

    # --- Handle actions (kept flexible so your existing logic keeps working) ---
    if search_clicked:
        # If you already have a search routine, it will run here.
        # 1) Prefer a provided helper if present:
        if callable(do_menu1_search):
            try:
                data = do_menu1_search(question_codes, years, demcodes)  # type: ignore
                if data is not None:
                    stash_results(data)
            except Exception as e:
                st.error(f"Search failed: {type(e).__name__}: {e}")
        # 2) Otherwise assume your app already stashes results elsewhere in this call stack.
        #    If you previously handled search in this file, you can move that code here.

    if reset_clicked:
        # Fully reset Menu 1 state (this should also clear any warnings/results)
        reset_menu1_state()
        # Force a rerun so the cleared UI shows up immediately
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

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

def main():
    try:
        render_menu()
    except Exception as e:
        st.error(f"Menu 1 is unavailable: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
