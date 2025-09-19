import streamlit as st

# Prewarm imports (safe if utils not available yet)
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

    # Prewarm Parquet/CSV backend on home page before any menu loads
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

    # ───────────────── Landing page ─────────────────
    st.markdown("""
        <style>
            /* Keep sane padding; do NOT paint global background or global text color */
            .block-container {
                padding-top: 80px !important;
                padding-left: 24px !important;
                padding-right: 24px !important;
                padding-bottom: 120px !important;
            }

            /* Landing-only background wrapper + dark overlay for readability */
            .landing-bg {
                position: relative;
                width: 100%;
                min-height: 100vh;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center top;
                background-attachment: scroll;
                background-color: #0a2540; /* solid fallback if image fails/partial */
            }
            .landing-bg::before {
                content: "";
                position: absolute;
                inset: 0;
                background: rgba(0,0,0,0.35); /* overlay so white text always readable */
            }

            /* Content container sits above overlay */
            .landing-content {
                position: relative;
                z-index: 1;
                max-width: 880px;
                margin-left: clamp(24px, 8vw, 200px);
                margin-right: 24px;
                color: #ffffff;
            }

            .main-title {
                font-size: clamp(28px, 4vw, 42px);
                font-weight: 800;
                margin-bottom: 16px;
                line-height: 1.2;
            }
            .subtitle {
                font-size: clamp(18px, 2.5vw, 24px);
                margin-bottom: 6px;
            }
            .survey-years {
                font-size: clamp(16px, 2vw, 20px);
                margin-bottom: 28px;
                opacity: 0.95;
            }

            /* Buttons */
            div.stButton > button {
                background-color: transparent !important;
                color: white !important;
                border: 2px solid rgba(255, 255, 255, 0.35) !important;
                font-size: 28px !important;
                font-weight: 700 !important;
                padding: 24px 28px !important;
                width: 420px !important;
                min-height: 84px !important;
                line-height: 1.2 !important;
                border-radius: 12px !important;
                transition: 0.25s ease-in-out;
                text-align: left !important;
                height: auto !important;
                display: block !important;
            }
            div.stButton > button:hover {
                border-color: #ffffff !important;
                background-color: rgba(255, 255, 255, 0.12) !important;
            }

            .menu-grid {
                display: flex;
                flex-direction: column;
                gap: 18px;
                margin-top: 8px;
                margin-bottom: 24px;
            }

            /* Small screens: make buttons full-width */
            @media (max-width: 768px) {
                div.stButton > button { width: 100% !important; }
            }
        </style>
    """, unsafe_allow_html=True)

    # Background wrapper + content
    st.markdown("<div class='landing-bg'>", unsafe_allow_html=True)
    st.markdown("<div class='landing-content'>", unsafe_allow_html=True)

    # Updated title & subtitle (as requested)
    st.markdown(
        "<div class='main-title'>Welcome to the AI-powered Explorer of the Public Service Employee Survey (PSES)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>This AI-powered app provides Public Service-wide survey results and analysis</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    # Menu buttons
    st.markdown("<div class='menu-grid'>", unsafe_allow_html=True)

    # Menu 1 (renamed)
    if st.button("🔍 Search by Survey Question", key="menu1_button"):
        st.session_state.run_menu = "1"   # no explicit rerun; Streamlit reruns automatically

    # Menu 2 (renamed)
    if st.button("🧩 Search by keywords or theme", key="menu2_button"):
        st.session_state.run_menu = "2"

    # Menu 3 removed

    # Menu 4 (kept)
    if st.button("📋 View Questionnaire", key="menu4_button"):
        st.session_state.run_menu = "4"

    st.markdown("</div>", unsafe_allow_html=True)   # close landing-content
    st.markdown("</div>", unsafe_allow_html=True)   # close landing-bg

if __name__ == "__main__":
    main()
