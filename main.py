# main.py — App entry & routing (Menu 1 default)
from __future__ import annotations
import os
import time
import streamlit as st

# Page config early
st.set_page_config(page_title="PSES AI Explorer", layout="wide")

# ===== Warmup (metadata + PS-wide data) =====
try:
    from utils.data_loader import prewarm_all, get_backend_info
    prewarm_all()  # cached; safe to call on every rerun
    _backend = get_backend_info()
except Exception as _e:
    _backend = {"last_engine": "error", "inmem_mode": "none", "inmem_rows": 0, "metadata_counts": {}}
    st.warning(f"Startup warmup encountered an issue: {type(_e).__name__}: {str(_e)}")

# ===== Simple style =====
st.markdown("""
<style>
  .banner-wrap { text-align:center; margin-bottom: 10px; }
  .banner-wrap img { width: 75%; max-width: 740px; height: auto; }
  .app-title { font-size: 26px; font-weight: 700; margin: 6px 0 2px; text-align:center; }
  .app-sub  { font-size: 15px; color:#333; text-align:center; margin-bottom: 14px; }
  .status    { font-size: 12px; color:#555; text-align:center; margin-top: 4px; }
  .center-col { max-width: 880px; margin: 0 auto; }
  .big-button button { font-size: 16px; padding: 0.7em 1.6em; }
</style>
""", unsafe_allow_html=True)

# ===== Helpers =====
def _status_ribbon():
    mc = _backend.get("metadata_counts", {}) or {}
    meta_txt = f"Q={mc.get('questions', 0)}, Scales={mc.get('scales', 0)}"
    info = (
        f"Engine: {_backend.get('last_engine','?')} • "
        f"In-mem: {_backend.get('inmem_mode','none')} ({_backend.get('inmem_rows',0)} rows) • "
        f"Meta: {meta_txt} • PS-wide only"
    )
    st.markdown(f"<div class='status'>{info}</div>", unsafe_allow_html=True)

def _banner():
    st.markdown(
        "<div class='banner-wrap'>"
        "<img src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'/>"
        "</div>",
        unsafe_allow_html=True,
    )

def _home():
    _banner()
    st.markdown("<div class='app-title'>PSES AI Explorer — Home</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-sub'>Use the button below to search Public Service–wide PSES results by survey question.</div>",
        unsafe_allow_html=True,
    )
    _status_ribbon()
    st.divider()
    c = st.container()
    with c:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<div class='center-col'>", unsafe_allow_html=True)
            # ✅ For testing: go straight to Menu 1
            if st.button("🔎 Start your search", type="primary", use_container_width=True):
                st.session_state.run_menu = "menu1"
                st.rerun()
            # Optional dev link to Wizard (won’t be used unless clicked)
            with st.expander("Developer options"):
                if st.button("🧪 Open Wizard (dev)"):
                    st.session_state.run_menu = "wizard"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

def show_return_then_run(run_callable):
    """Render a sub-app then offer a Return button."""
    try:
        run_callable()
    except Exception as e:
        st.error(f"An error occurred running this module: {type(e).__name__}: {e}")
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.rerun()

# ===== Router =====
def main():
    # Initialize route flag once
    if "run_menu" not in st.session_state:
        st.session_state.run_menu = None

    # Menu selection
    if st.session_state.run_menu == "menu1":
        try:
            from menu1.main import run_menu1
        except Exception as e:
            st.error(f"Could not import Menu 1: {type(e).__name__}: {e}")
            st.session_state.run_menu = None
            _home()
            return
        show_return_then_run(run_menu1)
        return

    if st.session_state.run_menu == "wizard":
        try:
            from wizard.main import run_wizard
        except Exception as e:
            st.error(f"Could not import Wizard: {type(e).__name__}: {e}")
            st.session_state.run_menu = None
            _home()
            return
        show_return_then_run(run_wizard)
        return

    # Default home
    _home()

if __name__ == "__main__":
    main()
