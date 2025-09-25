# main.py — homepage (routes to Wizard; Menu 4 removed; legacy Menus 1 & 2 in expander)
import streamlit as st

# Keep prewarm imports
try:
    from utils.data_loader import prewarm_fastpath, get_backend_info
except Exception:
    prewarm_fastpath = None
    get_backend_info = None

st.set_page_config(layout="wide")

def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_set_query_params()
        st.rerun()

def main():
    if "run_menu" not in st.session_state:
        st.session_state.run_menu = None

    # ✅ Prewarm backend only on home page
    if prewarm_fastpath is not None and st.session_state.run_menu is None:
        with st.spinner("Preparing data backend (one-time)…"):
            backend = prewarm_fastpath()
        if get_backend_info is not None:
            info = get_backend_info() or {}
            store = info.get("store")
            if backend == "memory_csv" or store == "in_memory_csv":
                st.caption("🧠 In-memory data store is ready — queries will run from RAM.")
            elif backend == "csv":
                st.caption(f"⚠️ Using CSV fallback. Parquet unavailable. (CSV: {info.get('csv_path')})")
            else:
                parquet_dir = info.get("parquet_dir")
                st.caption(f"✅ Parquet ready at: {parquet_dir}" if parquet_dir else "ℹ️ Data backend initialized.")

    # ===== Routing (Wizard + legacy Menus 1 & 2 only) =====
    if st.session_state.run_menu == "wizard":
        from wizard.main import run_wizard
        show_return_then_run(run_wizard)
        return
    elif st.session_state.run_menu == "1":
        from menu1.main import run_menu1
        show_return_then_run(run_menu1)
        return
    elif st.session_state.run_menu == "2":
        from menu2.main import run_menu2
        show_return_then_run(run_menu2)
        return

    # ===== Homepage styling =====
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
            .main-section { margin-left: 200px; max-width: 820px; }
            .main-title { font-size: 42px; font-weight: 800; margin-bottom: 16px; }
            .subtitle { font-size: 22px; line-height: 1.4; margin-bottom: 18px; opacity: 0.95; }
            .context { font-size: 18px; line-height: 1.55; margin-top: 8px; margin-bottom: 36px; opacity: 0.95; }
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
            div[data-testid="stExpander"] > details > summary { color: #fff; font-size: 16px; }
        </style>
    """, unsafe_allow_html=True)

    # ===== Homepage content =====
    st.markdown("<div class='main-section'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='main-title'>Welcome to the AI-powered Explorer of the Public Service Employee Survey (PSES)</div>",
        unsafe_allow_html=True
    )

    # Updated subtitle (exact wording requested)
    st.markdown(
        "<div class='subtitle'>This app provides Public Service-wide survey results and analysis for the previous 4 survey cycles (2019, 2020, 2022, and 2024)</div>",
        unsafe_allow_html=True
    )

    # Context block (exact text requested)
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

    # Primary CTA → Wizard
    st.markdown("<div class='single-button'>", unsafe_allow_html=True)
    if st.button("▶️ Start your search", key="menu_start_button"):
        st.session_state.run_menu = "wizard"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Legacy Menus 1 & 2 (for testing / fallback) — Menu 4 removed
    with st.expander("Advanced: open classic menus (for testing / legacy flows)"):
        c1, c2 = st.columns([1,1])
        if c1.button("🔍 Menu 1 — Search by Question"):
            st.session_state.run_menu = "1"; st.rerun()
        if c2.button("🧩 Menu 2 — Search by Keywords/Theme"):
            st.session_state.run_menu = "2"; st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
