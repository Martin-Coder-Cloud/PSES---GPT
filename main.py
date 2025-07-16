# menu1_question/main.py

import streamlit as st
import pandas as pd

# Load demographics metadata
demo_df = pd.read_excel("metadata/Demographics.xlsx")
demo_df.columns = [col.strip() for col in demo_df.columns]  # Normalize headers

# Normalize column name
DEMO_CAT_COL = "DEMCODE Category"
LABEL_COL = "DESCRIP_E"

# Define which categories require a second-level dropdown
long_list_categories = {
    "2SLGBTQIA+ sub group",
    "Ethnic origins",
    "Occ. Group and Level",
    "Occupational group",
    "Person with a disability sub group",
    "Racial sub group",
    "Work Community"
}

def run_menu1():
    # --- Page Setup ---
    st.markdown("<h1 style='text-align: center;'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>This AI app provides survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Menu 1: Instructions ---
    st.subheader("🔍 Search by Question")

    st.markdown("""
    Use this menu if you already know the specific survey question you wish to explore (e.g., **Q58**).

    You can:
    - Use the dropdown menus below to select the year and demographic category
    - Or use natural language to describe your request

    The system will confirm your query before retrieving official PSES results.
    """)

    st.markdown("""
    - 📜 [View the list of survey questions (2024)](https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey/2024-25/2024-25-public-service-employee-survey.html)
    """)

    # --- Input Controls ---
    year = st.multiselect("Select survey year(s):", [2024, 2022, 2020, 2019], default=[2024])

    demo_categories = sorted(demo_df[DEMO_CAT_COL].dropna().unique().tolist())
    demo_selection = st.selectbox("Select a demographic category (optional):", ["All respondents"] + demo_categories)

    sub_selection = None
    if demo_selection in long_list_categories:
        sub_items = demo_df[demo_df[DEMO_CAT_COL] == demo_selection][LABEL_COL].dropna().unique().tolist()
        if len(sub_items) > 25:
            sub_selection = st.text_input(f"Search or enter a {demo_selection} value:")
        else:
            sub_selection = st.selectbox(f"Select a {demo_selection} value:", sub_items)

    # --- Question and Prompt ---
    question_input = st.text_input("Enter a specific question number (e.g., Q58):")
    prompt_text = st.text_area("Or describe what you're looking for:")

    # --- Trigger Search ---
    if st.button("Search"):
        st.markdown("🔄 *Processing your request...*")
        # Debug preview
        st.write("Selected Year(s):", year)
        st.write("Demographic Category:", demo_selection)
        if sub_selection:
            st.write("Sub-category value:", sub_selection)
        st.write("Question:", question_input)
        st.write("Prompt:", prompt_text)

        st.success("✅ Query received. (Back-end connection coming soon)")
