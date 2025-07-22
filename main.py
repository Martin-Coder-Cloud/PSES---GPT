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
    # === Layout + background image with white text
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 80px !important;
            padding-left: 60px !important;
            background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
            color: white !important;
        }
        h1, h3, p, a {
            color: white !important;
        }
        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            max-width: 800px;
            line-height: 1.3;
        }
        .subtitle {
            font-size: 20px;
            font-weight: 400;
            margin-bottom: 35px;
            max-width: 850px;
            word-wrap: break-word;
        }
        .menu-option {
            font-size: 20px;
            font-weight: bold;
            margin: 14px 0;
            padding: 12px 22px;
            background-color: rgba(255,255,255,0.08);
            border-radius: 12px;
            display: inline-block;
            text-decoration: none !important;
            transition: background 0.3s;
        }
        .menu-option:hover {
            background-color: rgba(255,255,255,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # === Handle menu selection from query
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    # === Load menu page if selected
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

    # === Render title, subtitle and menu buttons
    st.markdown("<div class='main-title'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>This AI app provides Public Service-wide survey results and analysis from 2019, 2020, 2022, and 2024.</div>", unsafe_allow_html=True)

    for label, icon, menu_id in [
        ("Search by Question", "🔍", "1"),
        ("Search by Theme", "🧩", "2"),
        ("Analyze Data", "📊", "3"),
        ("View Questionnaire", "📋", "4"),
    ]:
        st.markdown(
            f"<a class='menu-option' href='?menu={menu_id}'>{icon} {label}</a>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
