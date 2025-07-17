import streamlit as st

st.set_page_config(layout="wide")

# ✅ Add this new helper function
def show_return_then_run(run_func):
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_rerun()
    run_func()

def main():
    # === Remove top margin above banner ===
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # === Always display banner ===
    st.image("assets/ANC006-PSES_banner825x200_EN.png", use_column_width=True)

    # === Show title/subtitle/instruction ONLY if no menu is selected ===
    if "run_menu" not in st.session_state:
        # === Title & Subtitle ===
        st.markdown("""
            <h1 style='text-align: center; margin-top: 10px; font-size: 28px;'>
                Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.
            </h1>
            <h3 style='text-align: center; font-weight: normal; margin-bottom: 25px; font-size: 20px;'>
                This AI app provides Public Service-wide survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).
            </h3>
        """, unsafe_allow_html=True)

        # === Instruction line ===
        st.markdown("""
            <div style="max-width: 950px; margin: auto; text-align: left; font-size: 16px; margin-bottom: 20px;">
                To start your analysis, please select one of the menu options below:
            </div>
        """, unsafe_allow_html=True)

        # === Button CSS ===
        st.markdown("""
            <style>
                .menu-wrapper {
                    display: flex;
                    justify-content: center;
                    gap: 30px;
                    flex-wrap: wrap;
                    margin-bottom: 40px;
                }
                .menu-tile {
                    width: 240px;
                    height: 240px;
                    background-color: #f8f9fa;
                    border: 2px solid #0d6efd;
                    border-radius: 20px;
                    text-align: center;
                    font-size: 18px;
                    font-weight: 600;
                    color: #0d6efd;
                    padding-top: 45px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                .menu-tile span {
                    display: block;
                    font-size: 60px;
                    margin-bottom: 10px;
                }
                .menu-tile:hover {
                    background-color: #0d6efd;
                    color: white;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                .menu-form {
                    margin: 0;
                }
            </style>
        """, unsafe_allow_html=True)

        # === Render Menu Buttons Using Streamlit Buttons ===
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔍\nSearch by Question", use_container_width=True):
                st.session_state.run_menu = "1"
                st.experimental_rerun()

        with col2:
            if st.button("🧩\nSearch by Theme", use_container_width=True):
                st.session_state.run_menu = "2"
                st.experimental_rerun()

        with col3:
            if st.button("📊\nAnalyze Data", use_container_width=True):
                st.session_state.run_menu = "3"
                st.experimental_rerun()

        with col4:
            if st.button("📋\nView Questionnaire", use_container_width=True):
                st.session_state.run_menu = "4"
                st.experimental_rerun()

    # === Route to selected menu ===
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
