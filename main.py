import streamlit as st

st.set_page_config(layout="wide")

# ✅ Helper function: show menu and return button below
def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_rerun()

def main():
    # === Layout reset ===
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0px !important;
                margin-top: 0px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Centered banner
    st.markdown("""
        <div style='text-align: center; max-width: 1100px; margin: auto; margin-top: 30px; margin-bottom: 20px;'>
            <img src='https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/06e8805a54c2c28ed7e1528676e2dc5f750cca62/PSES%20email%20banner.png?raw=true' width='960'>
        </div>
    """, unsafe_allow_html=True)

    # === Show main menu only if no selection has been made ===
    if "run_menu" not in st.session_state:

        # ✅ Title + Subtitle
        st.markdown("""
            <div style='text-align: center; max-width: 1100px; margin: auto;'>
                <h1 style='margin-top: 10px; font-size: 26px;'>
                    Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.
                </h1>
                <h3 style='font-weight: normal; margin-bottom: 25px; font-size: 18px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
                    This AI app provides Public Service-wide survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).
                </h3>
            </div>
        """, unsafe_allow_html=True)

        # ✅ Inject button tile styles
        st.markdown("""
            <style>
                div.stButton > button {
                    height: 240px;
                    width: 240px;
                    border-radius: 20px;
                    border: 2px solid #0d6efd;
                    background-color: #f8f9fa;
                    color: #0d6efd;
                    font-size: 18px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                }
                div.stButton > button:hover {
                    background-color: #0d6efd;
                    color: white;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
            </style>
        """, unsafe_allow_html=True)

        # ✅ Render 4 button tiles in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🔍\nSearch by Question"):
                st.session_state.run_menu = "1"
                st.experimental_rerun()
        with col2:
            if st.button("🧩\nSearch by Theme"):
                st.session_state.run_menu = "2"
                st.experimental_rerun()
        with col3:
            if st.button("📊\nAnalyze Data"):
                st.session_state.run_menu = "3"
                st.experimental_rerun()
        with col4:
            if st.button("📋\nView Questionnaire"):
                st.session_state.run_menu = "4"
                st.experimental_rerun()

    # === Routing logic
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

if __name__ == "__main__":
    main()
