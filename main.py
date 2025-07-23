import streamlit as st

st.set_page_config(layout="wide")

# ✅ Helper: show menu and return button
def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_rerun()

def main():
    # ✅ Apply visual and background styling
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
            .menu-button {
                display: flex;
                align-items: center;
                justify-content: flex-start;
                gap: 14px;
                width: 320px;
                padding: 16px 24px;
                font-size: 20px;
                font-weight: 600;
                color: white !important;
                background-color: rgba(255,255,255,0.12);
                border-radius: 12px;
                text-decoration: none !important;
                transition: background 0.3s ease;
                cursor: pointer;
            }
            .menu-button:hover {
                background-color: rgba(255,255,255,0.25);
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Menu routing logic
    if "run_menu" not in st.session_state:
        st.session_state.run_menu = None

    if st.session_state.run_menu == "1":
        from menu1.main import run_menu1
        show_return_then_run(run_menu1)
        return
    elif st.session_state.run_menu == "2":
        from menu2.main import run_menu2
        show_return_then_run(run_menu2)
        return
    elif st.session_state.run_menu == "3":
        show_return_then_run(lambda: st.info("📊 Analyze Data is under construction."))
        return
    elif st.session_state.run_menu == "4":
        show_return_then_run(lambda: st.info("📋 View Questionnaire is under construction."))
        return

    # ✅ Render home page layout
    st.markdown("<div class='main-section'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>This AI app provides Public Service-wide survey results and analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    # ✅ Render styled visual menu using Streamlit buttons (same look, native logic)
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
