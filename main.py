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
    # === Detect selection from query param
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    # === Run specific menu
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

    # === Layout starts here ===
    st.markdown("<br>", unsafe_allow_html=True)

    # ✅ Use Streamlit columns to center content
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("###", unsafe_allow_html=True)
        st.markdown("<h1 style='color:white; font-size: 38px;'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:white; font-size: 22px;'>This AI app provides Public Service-wide survey results and analysis</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:white; font-size: 20px;'>(2019, 2020, 2022, and 2024)</h4>", unsafe_allow_html=True)
        st.markdown("---")

        # ✅ Menu buttons
        for label, icon, menu_id in [
            ("Search by Question", "🔍", "1"),
            ("Search by Theme", "🧩", "2"),
            ("Analyze Data", "📊", "3"),
            ("View Questionnaire", "📋", "4"),
        ]:
            if st.button(f"{icon} {label}", key=menu_id):
                st.session_state.run_menu = menu_id
                st.experimental_set_query_params(menu=menu_id)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
