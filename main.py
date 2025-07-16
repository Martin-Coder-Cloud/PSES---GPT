# main.py (at root level of your app)

import streamlit as st

# === Page Setup ===
st.set_page_config(page_title="PSES Explorer", layout="wide")

# === Custom CSS for Layout ===
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

# === Banner ===
st.image("assets/ANC006-PSES_banner825x200_EN.png", use_column_width=True)

# === Title and Subtitle ===
st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <h2>Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.</h2>
        <p style='font-size:18px; color:#555; max-width: 800px; margin: 0 auto;'>
            This AI app provides survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# === Menu Navigation ===
menu_selection = st.selectbox("Select a menu option", [
    "🔍 Search by Question",
    "🧩 Search by Theme",
    "📊 Analyze Data",
    "📋 View Questionnaire"
])

# === Route to Menu Logic ===
if menu_selection == "🔍 Search by Question":
    from menu1_question.main import run_menu1
    run_menu1()

elif menu_selection == "🧩 Search by Theme":
    st.info("🧩 Theme search under development.")

elif menu_selection == "📊 Analyze Data":
    st.info("📊 Analysis tools are in development.")

elif menu_selection == "📋 View Questionnaire":
    st.info("📋 Questionnaire viewer coming soon.")
