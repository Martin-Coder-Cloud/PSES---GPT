import streamlit as st
from urllib.parse import urlencode
from PIL import Image

# Set page config
st.set_page_config(page_title="PSES Explorer", layout="wide")

# === Banner ===
# Load and resize image to reduce height
banner_img = Image.open("assets/ANC006-PSES_banner825x200_EN.png")
banner_resized = banner_img.resize((banner_img.width, 120))  # Adjust height as needed

# Display full-width resized banner
st.image(banner_resized, use_column_width=True)

# === Get menu param from URL query param ===
query_params = st.experimental_get_query_params()
menu = query_params.get("menu", [None])[0]

# === CSS Styling ===
st.markdown("""
    <style>
        .tile-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 40px;
            flex-wrap: wrap;
        }
        .tile {
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
            text-decoration: none;
        }
        .tile:hover {
            background-color: #e0ecf8;
            border-color: #5b9bd5;
            transform: scale(1.05);
            text-decoration: none;
        }
        .icon {
            font-size: 48px;
            margin-bottom: 15px;
            display: block;
        }
        .instruction-text {
            font-size: 18px;
            color: #444;
            text-align: left;
            margin-left: 15%;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# === Main Menu View ===
if not menu:
    # Title and subtitle
    st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <h2>Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.</h2>
            <p style='font-size:18px; color:#555; max-width: 1000px; margin: 0 auto; white-space: nowrap;'>
                This AI app provides survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Aligned instruction line (closer to tile start)
    st.markdown("<div class='instruction-text'>To start your analysis, please select one of the menu options below:</div>", unsafe_allow_html=True)

    # Menu tiles
    st.markdown(f"""
        <div class="tile-container">
            <a href="?menu=1" class="tile" target="_self">
                <span class="icon">🔎</span>
                Search by Question
            </a>
            <a href="?menu=2" class="tile" target="_self">
                <span class="icon">🧩</span>
                Search by Theme
            </a>
            <a href="?menu=3" class="tile" target="_self">
                <span class="icon">📈</span>
                Analyze Data
            </a>
            <a href="?menu=4" class="tile" target="_self">
                <span class="icon">📄</span>
                View Questionnaire
            </a>
        </div>
    """, unsafe_allow_html=True)

# === Menu Routing Logic ===
else:
    if menu == "1":
        import menu1_question.main as menu1
        menu1.run_menu1()

    elif menu == "2":
        st.info("🧩 Search by Theme is under construction.")

    elif menu == "3":
        st.info("📊 Analyze Data is under construction.")

    elif menu == "4":
        st.info("📋 View Questionnaire is under construction.")

    # Back to Main Menu
    st.markdown("---")
    if st.button("⬅️ Back to Main Menu"):
        st.experimental_set_query_params()
        st.experimental_rerun()
