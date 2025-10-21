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
    # ✅ Fullscreen background and layout (fixed bottom padding typo; added box-sizing)
    st.markdown("""
        <style>
            html, body { height: 100%; margin: 0; padding: 0; }
            .block-container, .block-container * { box-sizing: border-box; }
            .block-container {
                padding-top: 100px !important;
                padding-left: 0 !important;
                padding-bottom: 100px !important;  /* fixed spelling */
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-size: cover;
                background-position: center top;
                background-repeat: no-repeat;
                background-attachment: fixed;
                min-height: 100vh;
                color: white;
            }

            /* Title centered across the page */
            .hero-title, .survey-years { text-align: center; width: 100%; }
            .main-title {
                font-size: 42px;
                font-weight: bold;
                margin-bottom: 12px;
                color: white;
                line-height: 1.2;
            }
            .survey-years {
                font-size: 20px;
                margin-bottom: 28px;
                color: white;
            }

            /* Left-centered content column that never crosses mid-screen */
            .landing-container {
                /* nudge off the hard left and cap width so it stays in the clear zone */
                margin-left: 200px;                 /* same offset you liked originally */
                max-width: 560px;                   /* hard cap: prevents overflow into busy right side */
            }

            /* Clean, unboxed text wrapper — remove translucent backgrounds to avoid artifacts */
            .intro-text {
                font-size: 18px;
                line-height: 1.6;
                word-wrap: break-word;
                overflow-wrap: break-word;
                margin-bottom: 14px;
            }
            .intro-text p { margin: 0 0 14px 0; }

            /* Primary button styling (unchanged label) */
            div.stButton > button {
                background: rgba(255, 255, 255, 0.15) !important;
                color: #ffffff !important;
                border: 2px solid rgba(255, 255, 255, 0.6) !important;
                font-size: 22px !important;
                font-weight: 700 !important;
                padding: 14px 24px !important;
                border-radius: 14px !important;
                box-shadow: 0 6px 18px rgba(0,0,0,0.25) !important;
                min-width: 260px;
            }
            div.stButton > button:hover {
                background: rgba(255, 255, 255, 0.25) !important;
                border-color: #ffffff !important;
                box-shadow: 0 8px 22px rgba(0,0,0,0.35) !important;
            }

            /* Responsive nudges: keep text left and narrow on smaller screens */
            @media (max-width: 1100px) {
                .landing-container { margin-left: 140px; max-width: 540px; }
            }
            @media (max-width: 900px) {
                .landing-container { margin-left: 80px; max-width: 54ch; }
            }
            @media (max-width: 700px) {
                .landing-container { margin-left: 24px; max-width: 56ch; }
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Menu routing logic (unchanged)
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
        elif st.session_state.run_menu == "2":
            from menu2.main import run_menu2
            show_return_then_run(run_menu2)
        elif st.session_state.run_menu == "3":
            show_return_then_run(lambda: st.info("📊 Analyze Data is under construction."))
        elif st.session_state.run_menu == "4":
            show_return_then_run(lambda: st.info("📋 View Questionnaire is under construction."))
        return

    # ✅ Title centered
    st.markdown("<div class='hero-title'><div class='main-title'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    # ✅ Intro text kept to a boxed width and left offset
    st.markdown("<div class='landing-container'>", unsafe_allow_html=True)

    # (TEXT BELOW IS UNCHANGED — only layout around it is updated)
    st.markdown("""
        <div class="intro-text">
          <p>
            The <strong>PSES AI Explorer</strong> is an interactive tool designed to help users navigate, analyze, and interpret the
            <a href="https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey.html" target="_blank" style="color:#fff; text-decoration: underline;">
              Public Service Employee Survey (PSES)
            </a>
            results from <strong>2019 to 2024</strong>. It combines open data with AI-assisted insights to help identify trends, challenges,
            and opportunities for action across the federal public service.
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

    if st.button("Start your search", key="start_search"):
        st.session_state.run_menu = "1"
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # close .landing-container

if __name__ == "__main__":
    main()
