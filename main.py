# main.py

import streamlit as st
import pandas as pd
from metadata_loader import load_required_metadata, validate_dataset

# === Page Setup ===
st.set_page_config(page_title="PSES Explorer", layout="wide")

# === Top Banner ===
st.image("ANC006-PSES_banner825x200_EN.png", use_column_width=True)

# === Title ===
st.title("📋 Welcome to the Public Service Employee Survey (PSES) Explorer")

# === Load Metadata and Dataset ===
try:
    metadata = load_required_metadata()
    dataset_path = validate_dataset(metadata["layout"])
except RuntimeError as err:
    st.error(f"❌ Startup error: {err}")
    st.stop()

st.success("✅ All metadata and dataset loaded successfully.")

# === Main Menu ===
st.header("🧭 Main Menu")
menu_option = st.radio(
    "Please select an option to begin:",
    [
        "1. 🔍 Search by Question",
        "2. 🧩 Search by Theme",
        "3. 📊 Analyze Data",
        "4. 📜 View the Questionnaire",
    ]
)

# === Route to Menu Pages ===
if menu_option.startswith("1"):
    st.subheader("🔍 Menu 1: Search by Question")
    st.info("This feature will let you view results for a specific survey question across years and demographics.")

elif menu_option.startswith("2"):
    st.subheader("🧩 Menu 2: Search by Theme")
    st.info("This feature will allow exploration of questions grouped by theme.")

elif menu_option.startswith("3"):
    st.subheader("📊 Menu 3: Analyze Data")
    st.info("This feature will let you analyze trends by year, group, or region.")

elif menu_option.startswith("4"):
    st.subheader("📜 Menu 4: View the Questionnaire")
    st.info("This will provide a browsable list of all PSES questions by year.")

# === Footer Visual (Optional Promo Image) ===
st.markdown("---")
st.image("ANC006-PSES-SM_EN.png", width=600)
st.caption("Source: Government of Canada – Public Service Employee Survey 2024")

# === End Marker ===
st.caption("Type `menu` to return to the main menu. PSES Explorer v0.1 © 2025")
