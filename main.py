import streamlit as st

# NEW: prewarm imports (safe if utils not available yet)
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
        st.experimental_rerun()

def main():
    if "run_menu" not in st.session_state:
        st.session_state.run_menu = None

    # NEW: prewarm Parquet/CSV backend on home page before any menu loads
    if prewarm_fastpath is not None and st.session_state.run_menu is None:
        with st.spinner("Preparing data backend (one-time)…"):
            backend = prewarm_fastpath()
        if get_backend_info is not None:
            info = get_backend_info()
            if backend == "csv":
                st.caption(f"⚠️ Using CSV fallback. Parquet unavailable. (CSV: {info.get('csv_path')})")
            else:
                st.caption(f"✅ Parquet ready at: {info.get('parquet_dir')}")

    # Menu routing (Menu 3 removed)
    if st.session_state.run_menu == "1":
        from menu1.main import run_menu1
        show_return_then_run(run_menu1)
        return
    elif st.session_state.run_menu == "2":
        from menu2.main import run_menu2
        show_return_then_run(run_menu2)
        return
    elif st.session_state.run_menu == "4":
        from menu4.main import run_menu4
        show_return_then_run(run_menu4)
        return

    # Landing page
    st.markdown("""
        <style>
            /* Keep general padding + text color, remove global background image */
            .block-container {
                padding-top: 100px !important;
                padding-left: 300px !important;
                padding-bottom: 300px !important;
                color: white;
            }
            /* Landing-only wrapper carries the background to avoid flicker */
            .landing-bg {
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center top;
                background-attachment: scroll;
                padding: 60px 0 200px 0;
                width: 100%;
            }
            .main-section {
                margin-left: 200px;
                max-width: 700px;
            }
            .main-title {
                font-size: 42px;
                font-weight: bold;
                margin-bottom: 20px;
                color: white;
                line-height: 1.2;
            }
            .subtitle {
                font-size: 24px;
                margin-bottom: 0px;
                color: white;
            }
            .survey-years {
                font-size: 20px;
                margin-bottom: 40px;
                color: white;
            }
            div.stButton > button {
                background-color: transparent !important;
                color: white !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                font-size: 32px !important;
                font-weight: 600 !important;
                padding: 28px 36px !important;
                width: 420px !important;
                min-height: 90px !important;
                line-height: 1.2 !important;
                border-radius: 12px !important;
                transition: 0.3s ease-in-out;
                text-align: left !important;
                overflow: visible !important;
                height: auto !important;
                display: block !important;
            }
            div.stButton > button:hover {
                border-color: white !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
            }
            .menu-grid {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='landing-bg'>", unsafe_allow_html=True)
    st.markdown("<div class='main-section'>", unsafe_allow_html=True)

    # UPDATED: Title & subtitle
    st.markdown(
        "<div class='main-title'>Welcome to the AI-powered Explorer of the Public Service Employee Survey (PSES)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>This AI-powered app provides Public Service-wide survey results and analysis</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    st.markdown("<div class='menu-grid'>", unsafe_allow_html=True)

    # RENAMED: Menu 1
    if st.button("🔍 Search by Survey Question", key="menu1_button"):
        st.session_state.run_menu = "1"   # no explicit rerun; Streamlit will rerun automatically

    # RENAMED: Menu 2
    if st.button("🧩 Search by keywords or theme", key="menu2_button"):
        st.session_state.run_menu = "2"

    # Menu 3 removed

    if st.button("📋 View Questionnaire", key="menu4_button"):
        st.session_state.run_menu = "4"

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # close landing-bg

if __name__ == "__main__":
    main()
