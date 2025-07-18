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
    # === Layout padding reset
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0px !important;
                margin-top: 0px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Centered banner from GitHub
    st.markdown("""
        <div style='text-align: center; max-width: 1100px; margin: auto; margin-top: 30px; margin-bottom: 20px;'>
            <img src='https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/06e8805a54c2c28ed7e1528676e2dc5f750cca62/PSES%20email%20banner.png?raw=true' width='960'>
        </div>
    """, unsafe_allow_html=True)

    # === Show menu if not yet selected
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

        # ✅ Style for menu tiles
        st.markdown("""
            <style>
                .menu-wrapper {
                    display: flex;
                    justify-content: center;
                    gap: 30px;
                    flex-wrap: wrap;
                    margin-top: 20px;
                    margin-bottom: 60px;
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
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.2s ease;
                    cursor: pointer;
                }
                .menu-tile:hover {
                    background-color: #0d6efd;
                    color: white;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                .menu-icon {
                    font-size: 60px;
                    margin-bottom: 10px;
                }
                .menu-link {
                    text-decoration: none;
                }
            </style>
        """, unsafe_allow_html=True)

        # ✅ Render menu buttons as styled tiles
        st.markdown("""
            <div class="menu-wrapper">
                <a class="menu-link" href="?menu=1">
                    <div class="menu-tile">
                        <div class="menu-icon">🔍</div>
                        Search by Question
                    </div>
                </a>
                <a class="menu-link" href="?menu=2">
                    <div class="menu-tile">
                        <div class="menu-icon">🧩</div>
                        Search by Theme
                    </div>
                </a>
                <a class="menu-link" href="?menu=3">
                    <div class="menu-tile">
                        <div class="menu-icon">📊</div>
                        Analyze Data
                    </div>
                </a>
                <a class="menu-link" href="?menu=4">
                    <div class="menu-tile">
                        <div class="menu-icon">📋</div>
                        View Questionnaire
                    </div>
                </a>
            </div>
        """, unsafe_allow_html=True)

    # ✅ Read menu selection from URL
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    # ✅ Run menu page
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
