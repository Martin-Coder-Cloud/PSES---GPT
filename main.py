# main.py

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

# Custom CSS for tile buttons
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

# === Main Menu View ===
if st.session_state.menu is None:
    st.markdown("""
        <div class="main-container">
            <div class="menu-tile" onclick="selectMenu('menu_1')">
                <div class="menu-icon">🔍</div>
                Search by Question
            </div>
            <div class="menu-tile" onclick="selectMenu('menu_2')">
                <div class="menu-icon">🧩</div>
                Search by Theme
            </div>
            <div class="menu-tile" onclick="selectMenu('menu_3')">
                <div class="menu-icon">📊</div>
                Analyze Data
            </div>
            <div class="menu-tile" onclick="selectMenu('menu_4')">
                <div class="menu-icon">📋</div>
                View Questionnaire
            </div>
        </div>
        <script>
            function selectMenu(menuName) {
                window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: menuName}, '*');
            }
        </script>
    """, unsafe_allow_html=True)

    # Fallback buttons (invisible to users, necessary for routing)
    if st.button("Go to Menu 1"):
        st.session_state.menu = "menu_1"
    if st.button("Go to Menu 2"):
        st.session_state.menu = "menu_2"
    if st.button("Go to Menu 3"):
        st.session_state.menu = "menu_3"
    if st.button("Go to Menu 4"):
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
