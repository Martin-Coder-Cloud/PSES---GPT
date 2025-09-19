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
            info = get_backend_info()
            if backend == "csv":
                st.caption(f"⚠️ Using CSV fallback. Parquet unavailable. (CSV: {info.get('csv_path')})")
            else:
                st.caption(f"✅ Parquet ready at: {info.get('parquet_dir')}")

    # ===== Routing =====
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
            .main-section { margin-left: 200px; max-width: 700px; }
            .main-title { font-size: 42px; font-weight: bold; margin-bottom: 20px; }
            .subtitle { font-size: 24px; margin-bottom: 0px; }
            .survey-years { font-size: 20px; margin-bottom: 40px; }
            div.stButton > button {
                background-color: transparent !important;
                color: white !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                font-size: 32px !important; font-weight: 600 !important;
                padding: 28px 36px !important;
                width: 420px !important; min-height: 90px !important;
                border-radius: 12px !important;
                text-align: left !important;
            }
            div.stButton > button:hover {
                border-color: white !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
            }
            .menu-grid { display: flex; flex-direction: column; gap: 20px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-section'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Welcome to the AI-powered Explorer of the Public Service Employee Survey (PSES)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>This AI-powered app provides Public Service-wide survey results and analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    st.markdown("<div class='menu-grid'>", unsafe_allow_html=True)

    # ✅ Single-click menu navigation (no wireframe flash)
    if st.button("🔍 Search by Survey Question", key="menu1_button"):
        st.session_state.run_menu = "1"
        st.rerun()

    if st.button("🧩 Search by keywords or theme", key="menu2_button"):
        st.session_state.run_menu = "2"
        st.rerun()

    if st.button("📋 View Questionnaire", key="menu4_button"):
        st.session_state.run_menu = "4"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
