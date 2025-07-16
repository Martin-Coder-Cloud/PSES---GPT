import streamlit as st
from menu1_question.main import run_menu1

# Page setup
st.set_page_config(page_title="PSES Explorer", layout="wide")

# Banner
st.image("assets/ANC006-PSES_banner825x200_EN.png", use_column_width=True)

# Title and subtitle
st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <h2>Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.</h2>
        <p style='font-size:18px; color:#555; max-width: 800px; margin: 0 auto;'>
            This AI app provides survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).
        </p>
    </div>
""", unsafe_allow_html=True)

# CSS for tile menu
st.markdown("""
    <style>
        .main-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 40px;
            flex-wrap: wrap;
        }
        .menu-tile {
            background-color: #f1f3f6;
            border-radius: 12px;
            padding: 40px 20px;
            width: 220px;
            height: 220px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            color: #222;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        .menu-tile:hover {
            background-color: #e0ecf8;
            border-color: #5b9bd5;
            transform: scale(1.05);
        }
        .menu-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Store selected menu in session
if "menu_selection" not in st.session_state:
    st.session_state.menu_selection = None

# Handle click actions with Streamlit buttons
def render_menu():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔍\nSearch by Question"):
            st.session_state.menu_selection = "menu_1"
    with col2:
        if st.button("🧩\nSearch by Theme"):
            st.session_state.menu_selection = "menu_2"
    with col3:
        if st.button("📊\nAnalyze Data"):
            st.session_state.menu_selection = "menu_3"
    with col4:
        if st.button("📋\nView Questionnaire"):
            st.session_state.menu_selection = "menu_4"

# Show main menu if nothing is selected
if not st.session_state.menu_selection:
    render_menu()

# Route to selected menu
elif st.session_state.menu_selection == "menu_1":
    run_menu1()

elif st.session_state.menu_selection == "menu_2":
    st.info("🧩 Search by Theme is under construction.")
elif st.session_state.menu_selection == "menu_3":
    st.info("📊 Analyze Data is under construction.")
elif st.session_state.menu_selection == "menu_4":
    st.info("📋 View Questionnaire is under construction.")

# Optional: allow return to home
if st.session_state.menu_selection:
    if st.button("⬅️ Back to Main Menu"):
        st.session_state.menu_selection = None
        st.experimental_rerun()
