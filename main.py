import streamlit as st
from metadata_loader import load_required_metadata, validate_dataset

# === Page Setup ===
st.set_page_config(page_title="PSES Explorer", layout="wide")

# === Custom CSS Styling ===
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

st.markdown("<br>", unsafe_allow_html=True)

# === Menu Grid ===
st.markdown("""
    <div class="main-container">
        <div class="menu-tile" onclick="window.parent.postMessage({ type: 'select', key: 'menu_1' }, '*')">
            <div class="menu-icon">🔍</div>
            Search by Question
        </div>
        <div class="menu-tile" onclick="window.parent.postMessage({ type: 'select', key: 'menu_2' }, '*')">
            <div class="menu-icon">🧩</div>
            Search by Theme
        </div>
        <div class="menu-tile" onclick="window.parent.postMessage({ type: 'select', key: 'menu_3' }, '*')">
            <div class="menu-icon">📊</div>
            Analyze Data
        </div>
        <div class="menu-tile" onclick="window.parent.postMessage({ type: 'select', key: 'menu_4' }, '*')">
            <div class="menu-icon">📋</div>
            View Questionnaire
        </div>
    </div>
""", unsafe_allow_html=True)

# === Session Fallback (manual click override) ===
selected = st.session_state.get("menu_selection", None)

if selected == "menu_1":
    st.success("➡ You selected: Search by Question")
    # Menu 1 logic here
elif selected == "menu_2":
    st.success("➡ You selected: Search by Theme")
    # Menu 2 logic here
elif selected == "menu_3":
    st.success("➡ You selected: Analyze Data")
    # Menu 3 logic here
elif selected == "menu_4":
    st.success("➡ You selected: View Questionnaire")
    # Menu 4 logic here
