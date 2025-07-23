import streamlit as st

st.set_page_config(layout="wide")

# ✅ Helper: show menu and return button
def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_set_query_params()
        st.experimental_rerun()

def main():
    # ✅ Fullscreen background and layout
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
                background-attachment: scroll;
                color: white;
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
            .menu-grid {
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            /* New: style Streamlit buttons like menu-button links */
            button[kind="primary"] {
                display: flex !important;
                align-items: center !important;
                justify-content: flex-start !important;
                gap: 14px !important;
                width: 320px !important;
                padding: 16px 24px !important;
                font-size: 20px !important;
                font-weight: 600 !important;
                color: white !important;
                background-color: rgba(255,255,255,0.12) !important;
                border-radius: 12px !important;
                text-decoration: none !important;
                transition: background 0.3s ease !important;
                border: none !important;
            }
            button[kind="primary"]:hover {
                background-color: rgba(255,255,255,0.25) !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Menu routing logic
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    if "run_menu" in st.session_state:
        if st.session_state.run_menu == "1":
            from menu1.main import run_menu1
            show_return_then_run(run_menu1)
        elif st.session_state.run_menu == "2":
            from menu2.main import run_menu2
            show_return_then_run(run_menu2)
        elif st.session_state.run_menu == "3":
            show_return_then_run(lambda: st.info("📊 Analyze Data is under construction."))
        elif st.session_state.run_menu == "4":
            show_return_then_run(lambda: st.info("📋 View Questionnaire is under construction."))
        return

    # ✅ Render landing content and menu
    st.markdown("<div class='main-section'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>This AI app provides Public Service-wide survey results and analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    # ✅ Render styled button-style menu using Streamlit buttons
    st.markdown("<div class='menu-grid'>", unsafe_allow_html=True)

    if st.button("🔍 Search by Question", key="menu1_button"):
        st.session_state.run_menu = "1"
        st.experimental_rerun()

    if st.button("🧩 Search by Theme", key="menu2_button"):
        st.session_state.run_menu = "2"
        st.experimental_rerun()

    if st.button("📊 Analyze Data", key="menu3_button"):
        st.session_state.run_menu = "3"
        st.experimental_rerun()

    if st.button("📋 View Questionnaire", key="menu4_button"):
        st.session_state.run_menu = "4"
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
