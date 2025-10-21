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
    # ✅ Fullscreen background and layout
    st.markdown("""
        <style>
            html, body {
                height: 100%;
                margin: 0;
                padding: 0;
            }
            .block-container {
                padding-top: 100px !important;
                padding-left: 0px !important;
                padding-buttom: 100px !important;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-size: cover;
                background-position: center top;
                background-repeat: no-repeat;
                background-attachment: fixed;
                min-height: 100vh;
                color: white;
            }
            .main-section {
                padding-left: 200px;  /* Shift content to center-left */
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
            .menu-option {
                font-size: 20px;
                font-weight: 600;
                margin: 16px 0;
                padding: 16px 28px;
                background-color: rgba(255,255,255,0.12);
                border-radius: 12px;
                display: inline-block;
                text-decoration: none;
                transition: background 0.3s;
                color: white !important;
            }
            .menu-option:hover {
                background-color: rgba(255,255,255,0.25);
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Menu routing logic (Menu 1 only)
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    if "run_menu" in st.session_state:
        if st.session_state.run_menu == "1":
            from menu1.main import run_menu1
            show_return_then_run(run_menu1)
        return

    # ✅ Render landing content
    st.markdown("<div class='main-section'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</div>", unsafe_allow_html=True)
    # Subtitle intentionally removed
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    # 🔹 New intro + semantic search + concise results text (PS-wide only)
    st.markdown("""
        <div style="font-size:18px; line-height:1.6;">
          <p>
            The <strong>PSES AI Explorer</strong> is an interactive tool designed to help users navigate, analyze, and interpret the
            <strong>Public Service Employee Survey (PSES)</strong> results from <strong>2019 to 2024</strong>.
            It combines open data with AI-assisted insights to help identify trends, challenges, and opportunities for action across the federal public service.
          </p>

          <p>
            A key feature of the PSES AI Explorer is its <strong>AI-powered questionnaire search</strong>. Using <strong>semantic search</strong>,
            the system goes beyond simple keyword matching to understand the meaning and context of a query. It recognizes related concepts and phrasing,
            helping you find questions that reflect the same ideas even when the wording differs.
          </p>

          <p>
            Once a question is selected, you can explore results by <strong>year</strong> and <strong>demographic group</strong> through a simple, guided interface.
            Results are presented in a standardized format with summary percentages and a brief AI-generated narrative highlighting the main trends.
          </p>

          <p><em>Together, these features make the PSES AI Explorer a powerful, user-friendly platform for transforming survey data into actionable insights.</em></p>
        </div>
    """, unsafe_allow_html=True)

    # 🔹 Single entry point (legacy menu links removed)
    st.markdown(
        f"<a class='menu-option' href='?menu=1'>🔍 Search the Survey Results</a>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
