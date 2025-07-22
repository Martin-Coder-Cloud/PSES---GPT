import streamlit as st

st.set_page_config(layout="wide")

# ✅ Helper: show menu and return button
def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_set_query_params()  # ✅ Clear URL param
        st.experimental_rerun()

def main():
    # === Layout and background reset ===
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0px !important;
                margin-top: 0px !important;
            }
            body {
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            .main-content {
                color: white;
                text-align: left;
                margin-left: 60px;
                max-width: 600px;
                padding-top: 30px;
                font-family: "Segoe UI", sans-serif;
            }
            .menu-option {
                font-size: 20px;
                margin: 15px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .menu-option a {
                color: white;
                text-decoration: none;
                font-weight: bold;
            }
            .menu-option a:hover {
                text-decoration: underline;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Read selection from URL
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    # ✅ Render selected menu page
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
        return  # ✅ Prevent showing main menu content below

    # ✅ Main menu content with white text over background
    st.markdown("""
        <div class='main-content'>
            <h1>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</h1>
            <h3>This AI app provides Public Service-wide survey results and analysis from 2019, 2020, 2022, and 2024.</h3>

            <div class='menu-option'>🔍 <a href="?menu=1">Search by Question</a></div>
            <div class='menu-option'>🧩 <a href="?menu=2">Search by Theme</a></div>
            <div class='menu-option'>📊 <a href="?menu=3">Analyze Data</a></div>
            <div class='menu-option'>📋 <a href="?menu=4">View Questionnaire</a></div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
