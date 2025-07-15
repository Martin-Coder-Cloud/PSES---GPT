import streamlit as st
from metadata_loader import load_required_metadata, validate_dataset

# === Page Setup ===
st.set_page_config(page_title="PSES Explorer", layout="wide")

# === Custom CSS for clean UI ===
st.markdown("""
    <style>
        .menu-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 30px;
            margin-top: 30px;
        }
        .menu-button {
            background-color: #F0F2F6;
            border: 2px solid #DEE2E6;
            border-radius: 12px;
            padding: 30px 20px;
            text-align: center;
            width: 220px;
            font-size: 18px;
            transition: all 0.3s ease;
            cursor: pointer;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
        }
        .menu-button:hover {
            background-color: #E0ECF8;
            border-color: #4682B4;
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# === Banner (Optional) ===
st.image("assets/ANC006-PSES_banner825x200_EN.png", use_column_width=True)

# === Title and Subtitle ===
st.markdown("""
    ## Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.
    <div style='font-size:18px; color:#555; margin-top:-10px'>
        This AI app provides survey results and analysis on the latest iterations of the survey 
        (2018, 2019, 2020, 2022, 2024).
    </div>
""", unsafe_allow_html=True)

# === Menu Buttons ===
st.markdown('<div class="menu-container">', unsafe_allow_html=True)

if st.button("🔍\nSearch by Question", key="menu_1"):
    st.session_state.menu_selection = "menu_1"

if st.button("🧩\nSearch by Theme", key="menu_2"):
    st.session_state.menu_selection = "menu_2"

if st.button("📊\nAnalyze Data", key="menu_3"):
    st.session_state.menu_selection = "menu_3"

if st.button("📋\nView Questionnaire", key="menu_4"):
    st.session_state.menu_selection = "menu_4"

st.markdown('</div>', unsafe_allow_html=True)

# === Route to Menu Logic ===
menu = st.session_state.get("menu_selection", None)

if menu == "menu_1":
    st.success("➡ You selected: Search by Question")
    # Call menu 1 logic

elif menu == "menu_2":
    st.success("➡ You selected: Search by Theme")
    # Call menu 2 logic

elif menu == "menu_3":
    st.success("➡ You selected: Analyze Data")
    # Call menu 3 logic

elif menu == "menu_4":
    st.success("➡ You selected: View Questionnaire")
    # Call menu 4 logic
