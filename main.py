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
    # ✅ Routing logic
    if "run_menu" not in st.session_state:
        st.session_state.run_menu = None

    # ✅ If a menu is active, show it
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

    # ✅ Home page content
    st.title("Welcome to the AI Explorer of the Public Service Employee Survey (PSES)")
    st.subheader("This AI app provides Public Service-wide survey results and analysis")
    st.markdown("**Available survey years:** 2019, 2020, 2022, and 2024")

    st.markdown("---")
    st.subheader("📋 Menu Options")

    # ✅ Simple, functional menu buttons
    if st.button("🔍 Search by Question"):
        st.session_state.run_menu = "1"
        st.experimental_rerun()

    if st.button("🧩 Search by Theme"):
        st.session_state.run_menu = "2"
        st.experimental_rerun()

    if st.button("📊 Analyze Data"):
        st.session_state.run_menu = "3"
        st.experimental_rerun()

    if st.button("📋 View Questionnaire"):
        st.session_state.run_menu = "4"
        st.experimental_rerun()

if __name__ == "__main__":
    main()
