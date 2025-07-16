import streamlit as st

# Set up page
st.set_page_config(page_title="PSES Explorer", layout="wide")

# Initialize session state
if "menu" not in st.session_state:
    st.session_state.menu = None

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

# Custom tile button styling
st.markdown("""
    <style>
        .menu-button {
            height: 220px;
            width: 100%;
            font-size: 20px;
            font-weight: 600;
            border-radius: 12px;
            border: 2px solid #ccc;
            background-color: #f1f3f6;
            transition: 0.3s;
        }
        .menu-button:hover {
            background-color: #e0ecf8;
            border-color: #5b9bd5;
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# === Main Menu View ===
if st.session_state.menu is None:
    st.markdown("### Select a menu below:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔍\nSearch by Question", key="menu1", use_container_width=True):
            st.session_state.menu = "menu_1"

    with col2:
        if st.button("🧩\nSearch by Theme", key="menu2", use_container_width=True):
            st.session_state.menu = "menu_2"

    with col3:
        if st.button("📊\nAnalyze Data", key="menu3", use_container_width=True):
            st.session_state.menu = "menu_3"

    with col4:
        if st.button("📋\nView Questionnaire", key="menu4", use_container_width=True):
            st.session_state.menu = "menu_4"

# === Menu Routing Logic ===
else:
    if st.session_state.menu == "menu_1":
        from menu1_question.main import run_menu1
        run_menu1()

    elif st.session_state.menu == "menu_2":
        st.info("🧩 Search by Theme is under construction.")

    elif st.session_state.menu == "menu_3":
        st.info("📊 Analyze Data is under construction.")

    elif st.session_state.menu == "menu_4":
        st.info("📋 View Questionnaire is under construction.")

    st.markdown("⬅️ [Return to Main Menu](#)", unsafe_allow_html=True)
    if st.button("Back to Main Menu"):
        st.session_state.menu = None
