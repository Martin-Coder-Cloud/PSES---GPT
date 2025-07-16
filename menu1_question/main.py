import streamlit as st
import pandas as pd

# === Load Metadata ===
demo_df = pd.read_excel("metadata/Demographics.xlsx")
demo_df.columns = [col.strip() for col in demo_df.columns]

# === Constants ===
DEMO_CAT_COL = "DEMCODE Category"
LABEL_COL = "DESCRIP_E"

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
    # --- CSS Styling ---
    st.markdown("""
        <style>
            .custom-header {
                font-size: 30px !important;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .custom-instruction {
                font-size: 18px !important;
                line-height: 1.6;
                margin-bottom: 20px;
                color: #333;
            }
            .field-label {
                font-size: 18px !important;
                font-weight: 600 !important;
                margin-top: 25px !important;
                margin-bottom: 6px !important;
                color: #222 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Centered Layout ---
    left, center, right = st.columns([1, 3, 1])
    with center:
        # --- Header ---
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)

        # --- Instructions ---
        st.markdown("""
            <div class="custom-instruction">
                Use this menu if you already know the specific survey question you wish to explore (e.g., <b>Q58</b>).<br><br>
                The list of survey questions is available here: 
                <a href="https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey/2024-25/2024-25-public-service-employee-survey.html" target="_blank">
                View the 2024 Questionnaire</a>.<br><br>
                You can:
                <ul>
                    <li>Enter a question number</li>
                    <li>Select year(s) of interest</li>
                    <li>Apply optional demographic filters</li>
                </ul>
                The system will confirm your query before retrieving official PSES results.
            </div>
        """, unsafe_allow_html=True)

        # === Question Number First ===
        st.markdown('<div class="field-label">Enter a specific question number (e.g., Q58):</div>', unsafe_allow_html=True)
        question_input = st.text_input("")

        # === Year Selection with Checkboxes ===
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)

        all_years = [2024, 2022, 2020, 2019]
        select_all = st.checkbox("Select all years", value=True)

        selected_years = []
        for year in all_years:
            checked = True if select_all else False
            selected = st.checkbox(str(year), value=checked, key=f"year_{year}")
            if selected:
                selected_years.append(year)

        # === Prompt Input ===
        st.markdown('<div class="field-label">Or describe what you’re looking for:</div>', unsafe_allow_html=True)
        prompt_text = st.text_area("")

        # === Demographic Category ===
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        demo_categories = sorted(demo_df[DEMO_CAT_COL].dropna().unique().tolist())
        demo_selection = st.selectbox("", ["All respondents"] + demo_categories)

        # === Sub-category Selection (if applicable) ===
        sub_selection = None
        if demo_selection in long_list_categories:
            sub_items = demo_df[demo_df[DEMO_CAT_COL] == demo_selection][LABEL_COL].dropna().unique().tolist()
            if len(sub_items) > 25:
                st.markdown(f'<div class="field-label">Search or enter a {demo_selection} value:</div>', unsafe_allow_html=True)
                sub_selection = st.text_input("")
            else:
                st.markdown(f'<div class="field-label">Select a {demo_selection} value:</div>', unsafe_allow_html=True)
                sub_selection = st.selectbox("", sub_items)

        # === Search Button ===
        if st.button("Search"):
            st.markdown("🔄 *Processing your request...*")
            st.write("Selected Question:", question_input)
            st.write("Selected Year(s):", selected_years)
            st.write("Demographic Category:", demo_selection)
            if sub_selection:
                st.write("Sub-category value:", sub_selection)
            st.write("Prompt:", prompt_text)
            st.success("✅ Query received. (Back-end connection coming soon)")
