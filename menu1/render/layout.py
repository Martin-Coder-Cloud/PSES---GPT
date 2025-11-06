# menu1/render/layout.py
"""
Layout helpers for Menu 1:
- CSS injection and centered page container
- Banner, title, instructions
- Top-row toggles (AI + Diagnostics)
"""

from __future__ import annotations
from typing import List, Tuple
import streamlit as st

from ..constants import (
    BASE_CSS,
    BANNER_URL,
    INSTRUCTION_HTML,
    CENTER_COLUMNS,
)
from ..state import (
    K_AI_TOGGLE,
    K_DIAG_TOGGLE,
)

# ---------------------------------------------------------------------------
# CSS / Centering
# ---------------------------------------------------------------------------
def inject_base_css() -> None:
    """Injects base CSS for consistent look/feel across Menu 1."""
    st.markdown(BASE_CSS, unsafe_allow_html=True)

def centered_page(columns: List[int] | None = None, *, with_css: bool = True):
    """
    Returns a (left, center, right) 3-column layout.
    Use the 'center' column as the primary content area to keep widths consistent.
    """
    if with_css:
        inject_base_css()
    col_spec = columns or CENTER_COLUMNS
    left, center, right = st.columns(col_spec)
    return left, center, right

# ---------------------------------------------------------------------------
# Header elements
# ---------------------------------------------------------------------------
def banner(src: str = BANNER_URL) -> None:
    """Top banner image (full width within the center column)."""
    st.markdown(f"<img class='menu-banner' src='{src}'>", unsafe_allow_html=True)

def title(text: str) -> None:
    """Large page title using the shared CSS class."""
    st.markdown(f"<div class='custom-header'>{text}</div>", unsafe_allow_html=True)

def instructions(html: str = INSTRUCTION_HTML) -> None:
    """
    Render the instruction as a Title-2-like line, left aligned, no box.
    """
    # use markdown heading level 3 to keep it simple and left aligned
    st.markdown(f"### {html}")

# ---------------------------------------------------------------------------
# Toggles row
# ---------------------------------------------------------------------------
def toggles() -> Tuple[bool, bool]:
    """
    Renders the AI and Diagnostics toggles in a single row and
    returns their ON/OFF states as (ai_on, show_diag).
    Keys are defined in state.py to keep names consistent.
    """
    # Ensure defaults before rendering widgets: AI ON, Diagnostics OFF
    st.session_state.setdefault(K_AI_TOGGLE, True)
    st.session_state.setdefault(K_DIAG_TOGGLE, False)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.toggle(
            "🧠 Enable AI analysis",
            key=K_AI_TOGGLE,
            help="Include the AI-generated analysis alongside the tables.",
        )
    with c2:
        st.toggle(
            "🔧 Show technical parameters & diagnostics",
            key=K_DIAG_TOGGLE,
            help="Show current parameters, app setup status, AI status, and last query timings.",
        )

    ai_on = bool(st.session_state.get(K_AI_TOGGLE, True))
    show_diag = bool(st.session_state.get(K_DIAG_TOGGLE, False))
    return ai_on, show_diag
