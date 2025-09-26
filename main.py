# main.py — Home-first router; uses utils.data_loader.prewarm_all + get_backend_info
from __future__ import annotations
import streamlit as st

st.set_page_config(layout="wide")

# Loader hooks (match your utils/data_loader.py)
try:
    from utils.data_loader import prewarm_all, get_backend_info
except Exception:
    prewarm_all = None
    get_backend_info = None

# ── tiny nav helper ──────────────────────────────────────────────────────────
def goto(page: str):
    st.session_state["_nav"] = page
    st.rerun()

# ── shared: compact status ribbon (uses get_backend_info) ────────────────────
def _status_ribbon():
    info = {}
    try:
        info = (get_backend_info() or {})
    except Exception:
        pass
    mc = info.get("metadata_counts", {}) or {}
    q = mc.get("questions", 0); s = mc.get("scales", 0); d = mc.get("demographics", 0)
    engine = info.get("last_engine", "?")
    inmem  = info.get("inmem_mode", "none")
    rows   = info.get("inmem_rows", 0)
    pswide = "Yes" if info.get("pswide_only") else "No"
    st.markdown(
        f"""
        <div style="text-align:center; font-size:12px; color:#f5f5f5; margin-top:10px; opacity:.95;">
          Engine: <b>{engine}</b> • In-mem: <b>{inmem}</b> ({rows:,} rows) •
          Meta: Q={q}, Scales={s}, Demos={d} • PS-wide only: <b>{pswide}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Home view (injects background CSS locally) ───────────────────────────────
def render_home():
    # Home background + typography (scoped to this view only)
    st.markdown("""
        <style>
            .block-container {
                padding-top: 100px !important;
                padding-left: 300px !important;
                padding-bottom: 300px !important;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center top;
                color: white;
            }
            .main-section { margin-left: 200px; max-width: 820px; text-align: left; }
            .main-title { font-size: 42px; font-weight: 800; margin-bottom: 16px; }
            .subtitle { font-size: 22px; line-height: 1.4; margin-bottom: 18px; opacity: 0.95; max-width: 700px; }
            .context { font-size: 18px; line-height: 1.55; margin-top: 8px; margin-bottom: 36px; opacity: 0.95; max-width: 700px; text-align: left; }
            .single-button { display: flex; flex-direction: column; gap: 16px; }
            div.stButton > button {
                background-color: rgba(255,255,255,0.08) !important; color: white !important;
                border: 2px solid rgba(255, 255, 255, 0.35) !important;
                font-size: 30px !important; font-weight: 700 !important;
                padding: 26px 34px !important; width: 420px !important; min-height: 88px !important;
                border-radius: 14px !important; text-align: left !important; backdrop-filter: blur(2px);
            }
            div.stButton > button:hover { border-color: white !important; background-color: rgba(255, 255, 255, 0.14) !important; }
            div[data-testid="stExpander"] > details > summary { color: #fff; font-size: 16px; }
        </style>
    """, unsafe_allow_html=True)

    # One-time warmup here (metadata + PS-wide in-memory)
    if prewarm_all is not None:
        try:
            with st.spinner("Preparing data backend (one-time)…"):
                prewarm_all()
        except Exception as e:
            st.warning(f"Warmup notice: {type(e).__name__}: {e}")

    st.markdown("<div class='main-section'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='main-title'>Welcome to the AI-powered Explorer of the Public Service Employee Survey (PSES)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>This app provides Public Service-wide survey results and analysis for the previous 4 survey cycles (2019, 2020, 2022, and 2024)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class='context'>
        The 2024 Public Service Employee Survey (PSES) helps departments and agencies strengthen people management by highlighting areas such as employee engagement, equity and inclusion, anti-racism, and workplace well-being. It provides employees with a voice to share their experiences, supporting workplace improvements that benefit both public servants and Canadians. Results are tracked over time to guide and refine organizational action plans.
        <br><br>
        Each survey cycle combines recurring questions for trend analysis with new ones to reflect emerging priorities. In 2024, Employment Equity demographics were updated to advance diversity and inclusion, and hybrid work questions were streamlined to stay relevant post-pandemic. Statistics Canada, in partnership with the Office of the Chief Human Resources Officer, ran the survey from October 28 to December 31, 2024. The PSES will continue on a two-year cycle.
        </div>
        """,
        unsafe_allow_html=True
    )

    _status_ribbon()

    st.markdown("<div class='single-button'>", unsafe_allow_html=True)
    if st.button("▶️ Start your search", key="menu_start_button"):
        goto("menu1")
    st.markdown("</div>", unsafe_allow_html=True)

    # Optional: legacy link for Menu 2
    with st.expander("Advanced: open classic menus (for testing / legacy flows)"):
        if st.button("🧩 Menu 2 — Search by Keywords/Theme"):
            goto("menu2")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Menu wrappers (clear background before rendering) ────────────────────────
def _clear_bg_css():
    st.markdown("""
        <style>
            .block-container {
                background-image: none !important;
                background: none !important;
                color: inherit !important;
                padding-top: 1.25rem !important;
                padding-left: 1.25rem !important;
                padding-bottom: 2rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

def render_menu1():
    _clear_bg_css()
    try:
        from menu1.main import run_menu1
        run_menu1()
    except Exception as e:
        st.error(f"Menu 1 is unavailable: {type(e).__name__}: {e}")
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        goto("home")

def render_menu2():
    _clear_bg_css()
    try:
        from menu2.main import run_menu2
        run_menu2()
    except Exception as e:
        st.error(f"Menu 2 is unavailable: {type(e).__name__}: {e}")
    st.markdown("---")
    if st.button("🔙 Return to Main Menu", key="back2"):
        goto("home")

# ── Entry ────────────────────────────────────────────────────────────────────
def main():
    # Ensure old router flags can't hijack navigation
    if "run_menu" in st.session_state:
        st.session_state.pop("run_menu")
    if "_nav" not in st.session_state:
        st.session_state["_nav"] = "home"

    page = st.session_state["_nav"]
    if page == "menu1":
        render_menu1()
    elif page == "menu2":
        render_menu2()
    else:
        render_home()

if __name__ == "__main__":
    main()
