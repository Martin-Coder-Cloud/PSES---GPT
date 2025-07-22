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
    # === Background image and white text styling
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 0rem;
                margin-top: 0rem;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center;
                color: white;
            }
            h1, h3, a {
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # === Routing logic from URL
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    # === Launch corresponding menu page
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

    # === Main menu content
    st.markdown("### Welcome to the AI Explorer of the Public Service Employee Survey (PSES)")
    st.markdown("This AI app provides Public Service-wide survey results and analysis from 2019, 2020, 2022, and 2024.")
    st.markdown("")

    st.markdown("### Menu")
    st.markdown("🔍 [Search by Question](?menu=1)")
    st.markdown("🧩 [Search by Theme](?menu=2)")
    st.markdown("📊 [Analyze Data](?menu=3)")
    st.markdown("📋 [View Questionnaire](?menu=4)")

if __name__ == "__main__":
    main()
