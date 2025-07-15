# menu1_question/main.py

import streamlit as st

# --- Page Setup ---
st.set_page_config(page_title="Search by Question – PSES Explorer", layout="wide")

# --- Title & Introduction ---
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

# --- Input Controls ---
year = st.selectbox("Select a survey year:", [2024, 2022, 2020, 2019])

demo_categories = [
    "All respondents",
    "2SLGBTQIA+",
    "Age",
    "Bilingual region",
    "Departmental experience",
    "Ethnic origins",
    "First official language",
    "Gender",
    "Hybrid work",
    "Indigenous",
    "Language requirements of position",
    "Occupational group",
    "Person with a disability",
    "Public Service Experience",
    "Racial group",
    "Region of work",
    "Service to public",
    "Shift work",
    "Supervisory role",
    "Tenure",
    "Work Community",
    "Work schedule"
]
demo_selection = st.selectbox("Select a demographic category (optional):", demo_categories)

question_input = st.text_input("Enter a specific question number (e.g., Q58):")
prompt_text = st.text_area("Or describe what you're looking for:")

# --- Helpful Resources ---
st.markdown("""
- 📜 [View the list of survey questions (2024)](https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey/2024-25/2024-25-public-service-employee-survey.html)
- 👥 [Demographic groups explained](https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey/about.html#demographics)
""")

# --- Trigger Search ---
if st.button("Search"):
    st.markdown("🔄 *Processing your request...*")
    # Placeholder until interpreter and backend are connected
    st.success("✅ Query received. (Back-end connection coming soon)")
