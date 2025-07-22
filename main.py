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
    # ✅ Set full-height background and white text
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem !important;
            padding-left: 3rem !important;
            margin: 0 !important;
            background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
            color: white !important;
        }
        h1, h3, p, a {
            color: white !important;
        }
        .menu-option {
            font-size: 20px;
            font-weight: bold;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: rgba(0,0,0,0.3);
            border-radius: 10px;
            display: inline-block;
            text-decoration: none !important;
            transition: background 0.3s;
        }
        .menu-option:hover {
            background-color: rgba(255,255,255,0.2);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ✅ Routing logic
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    # ✅ Launch menu page
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

    # ✅ Main content (left-aligned with padding)
    st.markdown("## Welcome to the AI Explorer of the Public Service Employee Survey (PSES)")
    st.markdown("This AI app provides Public Service-wide survey results and analysis from 2019, 2020, 2022, and 2024.")

    st.markdown("---")
    st.markdown("[🔍 Search by Question](?menu=1)", unsafe_allow_html=True)
    st.markdown("[🧩 Search by Theme](?menu=2)", unsafe_allow_html=True)
    st.markdown("[📊 Analyze Data](?menu=3)", unsafe_allow_html=True)
    st.markdown("[📋 View Questionnaire](?menu=4)", unsafe_allow_html=True)

    # Wrap with styled divs
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
