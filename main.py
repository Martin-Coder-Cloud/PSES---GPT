import streamlit as st

st.set_page_config(layout="wide")

def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_set_query_params()
        st.experimental_rerun()

def main():
    if "run_menu" not in st.session_state:
        st.session_state.run_menu = None

    if st.session_state.run_menu == "1":
        from menu1.main import run_menu1
        show_return_then_run(run_menu1)
        return

    st.markdown("""
        <style>
            .block-container {
                padding-top: 100px !important;
                padding-left: 300px !important;
                padding-bottom: 300px !important;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center top;
                background-attachment: scroll;
                color: white;
            }
            .main-section {
                margin-left: 200px;
                max-width: 700px;
            }
            .main-title {
                font-size: 42px;
                font-weight: bold;
                margin-bottom: 20px;
                color: white;
                line-height: 1.2;
            }
            .survey-years {
                font-size: 20px;
                margin-bottom: 40px;
                color: white;
            }

            div.stButton > button {
                background-color: transparent !important;
                color: white !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
                font-size: 32px !important;
                font-weight: 600 !important;
                padding: 28px 36px !important;
                width: 420px !important;
                min-height: 90px !important;
                line-height: 1.2 !important;
                border-radius: 12px !important;
                transition: 0.3s ease-in-out;
                text-align: left !important;
                overflow: visible !important;
                height: auto !important;
                display: block !important;
            }
            div.stButton > button:hover {
                border-color: white !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
            }

            .menu-grid {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-section'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</div>", unsafe_allow_html=True)
    st.markdown("<div class='survey-years'>(2019, 2020, 2022, and 2024)</div>", unsafe_allow_html=True)

    # Updated introduction text
    st.markdown("""
        <div style="font-size:18px; line-height:1.6;">
        The <strong>PSES AI Explorer</strong> is an interactive tool designed to help users navigate, analyze, and interpret 
        the <strong>Public Service Employee Survey (PSES)</strong> results from <strong>2019 to 2024</strong>. 
        It combines open data with AI-assisted insights to help identify trends, challenges, and opportunities for action across the federal public service.
        <br><br>
        A key feature of the PSES AI Explorer is its <strong>AI-powered questionnaire search</strong>. 
        Using <strong>semantic search</strong>, the system goes beyond simple keyword matching to understand the meaning and context of a query. 
        It can recognize related concepts and phrases, allowing users to find questions that capture similar ideas, even when the wording differs.
        <br><br>
        Once a question is selected, users can explore results by <strong>year</strong> and <strong>demographic group</strong> through a simple, guided interface. 
        The data are displayed in a clear, standardized format with response distributions, summary percentages, and a brief AI-generated narrative that highlights the main trends.
        <br><br>
        Together, these features make the PSES AI Explorer a powerful, user-friendly platform for transforming survey data into actionable insights.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='menu-grid'>", unsafe_allow_html=True)

    if st.button("🔍 Search the Survey Results", key="menu1_button"):
        st.session_state.run_menu = "1"
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
