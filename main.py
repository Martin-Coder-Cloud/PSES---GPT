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

    <script>
        const streamlitEvents = window.parent || window;
        document.addEventListener("DOMContentLoaded", () => {
            const tiles = document.querySelectorAll(".menu-tile");
            tiles.forEach(tile => {
                tile.addEventListener("click", () => {
                    const key = tile.getAttribute("data-menu");
                    streamlitEvents.postMessage({ type: "streamlit:setComponentValue", value: key }, "*");
                });
            });
        });
    </script>
""", unsafe_allow_html=True)

# === Banner (Optional) ===
st.image("assets/ANC006-PSES_banner825x200_EN.png", use_column_width=True)

# === Title and Subtitle ===
st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <h2>Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.</h2>
        <p style='font-size:18px; color:#555; max-width: 800px; margin: 0 auto;'>
            This AI app provides survey results and analysis on the latest iterations of the survey 
            (2019, 2020, 2022, 2024).
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# === Menu Grid ===
st.markdown("""
    <div class="main-container">
        <div class="menu-tile" data-menu="menu_1">
            <div class="menu-icon">🔍</div>
            Search by Question
        </div>
        <div class="menu-tile" data-menu="menu_2">
            <div class="menu-icon">🧩</div>
            Search by Theme
        </div>
        <div class="menu-tile" data-menu="menu_3">
            <div class="menu-icon">📊</div>
            Analyze Data
        </div>
        <div class="menu-tile" data-menu="menu_4">
            <div class="menu-icon">📋</div>
            View Questionnaire
        </div>
    </div>
""", unsafe_allow_html=True)

# === Interactive Menu Logic ===
selected = st.experimental_get_query_params().get("menu", [None])[0] or st.session_state.get("menu_selection")

if selected == "menu_1":
    from menu1_question.main import run_menu1
    run_menu1()

elif selected == "menu_2":
    st.info("🔧 Theme search under construction.")

elif selected == "menu_3":
    st.info("📊 Analysis view coming soon.")

elif selected == "menu_4":
    st.info("📄 Questionnaire viewer will be available shortly.")
