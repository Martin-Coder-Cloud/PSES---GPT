# main.py — homepage (Menu 1 on click only; no router/state)
from __future__ import annotations
import streamlit as st

# Optional: prewarm metadata + PS-wide data only on the home view
try:
    from utils.data_loader import prewarm_all, get_backend_info
except Exception:
    prewarm_all = None
    get_backend_info = None

st.set_page_config(layout="wide")

# ── Home background + typography (your original CSS) ─────────────────────────
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
        /* Content rail */
        .main-section {
            margin-left: 200px;      /* shared left indent */
            max-width: 820px;        /* content width */
            text-align: left;        /* left-align text */
        }
        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 16px;
        }
        .subtitle {
            font-size: 22px;
            line-height: 1.4;
            margin-bottom: 18px;
            opacity: 0.95;
            max-width: 700px;
        }
        .context {
            font-size: 18px;
            line-height: 1.55;
            margin-top: 8px;
            margin-bottom: 36px;
            opacity: 0.95;
            max-width: 700px;
            text-align: left;
        }
        .single-button { display: flex; flex-direction: column; gap: 16px; }
        div.stButton > button {
            background-color: rgba(255,255,255,0.08) !important;
            color: white !important;
            border: 2px solid rgba(255, 255, 255, 0.35) !important;
            font-size: 30px !important; font-weight: 700 !important;
            padding: 26px 34px !important;
            width: 420px !important; min-height: 88px !important;
            border-radius: 14px !important;
            text-align: left !important;
            backdrop-filter: blur(2px);
        }
        div.stButton > button:hover {
            border-color: white !important;
            background-color: rgba(255, 255, 255, 0.14) !important;
        }
        div[data-testid="stExpander"] > details > summary {
            color: #fff; font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ── Home view (always rendered first) ────────────────────────────────────────
def render_home():
    # One-time warmup here (no routing state anywhere)
    if prewarm_all is not None:
        try:
            with st.spinner("Preparing data backend (one-time)…"):
                prewarm_all()
            if get_backend_info is not None:
                info = get_backend_info() or {}
                engine = info.get("last_engine", "unknown")
                inmem = info.get("inmem_mode", "none")
                inmem_rows = info.get("inmem_rows", 0)
                parquet_dir = info.get("parquet_dir")
                csv_path = info.get("csv_path")
                if engine.startswith("inmem"):
                    st.caption(f"🧠 In-memory store ready ({inmem}, {inmem_rows:,} rows).")
                elif engine == "parquet":
                    st.caption(f"✅ Parquet ready at: {parquet_dir}" if parquet_dir else "✅ Parquet backend initialized.")
                elif engine == "csv":
                    st.caption(f"⚠️ Using CSV fallback. (CSV: {csv_path})")
                else:
                    st.caption("ℹ️ Data backend initialized.")
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

    st.markdown("<div class='single-button'>", unsafe_allow_html=True)
    open_menu1 = st.button("▶️ Start your search", key="menu_start_button")
    st.markdown("</div>", unsafe_allow_html=True)

    # If clicked, render Menu 1 immediately (no rerun, no state)
    if open_menu1:
        render_menu1()
    else:
        # Optional: legacy menu links for testing
        with st.expander("Advanced: open classic menus (for testing / legacy flows)"):
            c1, c2 = st.columns([1,1])
            if c1.button("🔍 Menu 1 — Search by Question"):
                render_menu1()
            if c2.button("🧩 Menu 2 — Search by Keywords/Theme"):
                render_menu2()

    st.markdown("</div>", unsafe_allow_html=True)

# ── Menu wrappers (called only when the user clicks) ─────────────────────────
def render_menu1():
    try:
        from menu1.main import run_menu1
    except Exception as e:
        st.error(f"Menu 1 is unavailable: {type(e).__name__}: {e}")
        return
    run_menu1()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.experimental_rerun()

def render_menu2():
    try:
        from menu2.main import run_menu2
    except Exception as e:
        st.error(f"Menu 2 is unavailable: {type(e).__name__}: {e}")
        return
    run_menu2()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu", key="back2"):
        st.experimental_rerun()

# ── Entry ────────────────────────────────────────────────────────────────────
def main():
    # Important: do NOT read or write any routing keys; no query params either.
    render_home()

if __name__ == "__main__":
    main()
