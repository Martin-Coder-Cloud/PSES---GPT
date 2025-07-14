# main.py

import streamlit as st
import pandas as pd
import tempfile
import os

# ==== UI SETUP ====

st.set_page_config(page_title="PSES Explorer", layout="wide")
st.title("📋 Welcome to the Public Service Employee Survey (PSES) Explorer")

# ==== FILE UPLOAD SECTION ====

st.subheader("🔁 Upload Required Files")

uploaded_csv = st.file_uploader("1️⃣ Upload PSES Dataset (.csv)", type="csv")
uploaded_layout = st.file_uploader("2️⃣ Upload filelayout.xlsx", type="xlsx")
uploaded_questions = st.file_uploader("3️⃣ Upload Survey Questions.xlsx", type="xlsx")
uploaded_themes = st.file_uploader("4️⃣ Upload Survey Themes.xlsx", type="xlsx")
uploaded_scales = st.file_uploader("5️⃣ Upload Survey Scales.xlsx", type="xlsx")
uploaded_demographics = st.file_uploader("6️⃣ Upload Demographics.xlsx", type="xlsx")

files_ready = all([
    uploaded_csv,
    uploaded_layout,
    uploaded_questions,
    uploaded_themes,
    uploaded_scales,
    uploaded_demographics,
])

if not files_ready:
    st.warning("Please upload all 6 required files to enable the menu system.")
    st.stop()

st.success("✅ All required files uploaded.")

# ==== TEMP STORAGE FOR LARGE CSV ====

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(uploaded_csv.read())
    dataset_path = tmp.name

st.session_state["DATASET_PATH"] = dataset_path

# ==== LOAD METADATA FILES ====

@st.cache_data
def load_metadata():
    try:
        layout = pd.read_excel(uploaded_layout)
        questions = pd.read_excel(uploaded_questions)
        themes = pd.read_excel(uploaded_themes)
        scales = pd.read_excel(uploaded_scales)
        demographics = pd.read_excel(uploaded_demographics)
        return layout, questions, themes, scales, demographics
    except Exception as e:
        st.error(f"❌ Metadata loading error: {e}")
        st.stop()

filelayout_df, questions_df, themes_df, scales_df, demo_df = load_metadata()

# ==== MAIN MENU ====

st.header("🧭 Main Menu")
menu_option = st.radio(
    "Please select an option:",
    [
        "1. 🔍 Search by Question",
        "2. 🧩 Search by Theme",
        "3. 📊 Analyze Data",
        "4. 📜 View the Questionnaire",
    ]
)

# ==== ROUTE TO MENUS ====

if menu_option.startswith("1"):
    st.subheader("🔍 Menu 1: Search by Question")
    st.info("Feature coming soon – will allow filtering by question number, year, and demographics.")

elif menu_option.startswith("2"):
    st.subheader("🧩 Menu 2: Search by Theme")
    st.info("Feature coming soon – will allow browsing and querying by survey themes and sub-themes.")

elif menu_option.startswith("3"):
    st.subheader("📊 Menu 3: Analyze Data")
    st.info("Feature coming soon – will support trend analysis and demographic comparisons.")

elif menu_option.startswith("4"):
    st.subheader("📜 Menu 4: View the Questionnaire")
    st.info("PDF viewer and section browser to be implemented.")

# ==== FOOTER ====
st.markdown("---")
st.caption("Type `menu` to return to the main menu. PSES Explorer v0.1 © 2025")
