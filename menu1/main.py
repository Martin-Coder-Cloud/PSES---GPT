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
    # === Styling ===
    st.markdown("""
        <style>
            .custom-header {
                font-size: 30px !important;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .custom-instruction {
                font-size: 17px !important;
                line-height: 1.5;
                margin-bottom: 15px;
                color: #333;
            }
            .field-label {
                font-size: 18px !important;
                font-weight: 600 !important;
                margin-top: 12px !important;
                margin-bottom: 2px !important;
                color: #222 !important;
            }
            .big-button button {
                font-size: 18px !important;
                padding: 0.75em 2em !important;
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # === Layout ===
    left, center, right = st.columns([1, 3, 1])
    with center:
        # === Header ===
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)

        # === Instructions ===
        st.markdown("""
            <div class="custom-instruction">
                Please note that only Public Service–wide results are available in this tool. Departmental data is not included.<br><br>
                Use this menu to explore a specific survey question (e.g., <b>Q58</b>).<br>
                A valid query must include both a <b>question number</b> and at least one <b>year</b>.<br><br>
                <b>The list of survey questions is available here:</b><br>
                <a href="https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey/2024-25/2024-25-public-service-employee-survey.html" target="_blank">
                2024 Public Service Employee Survey Questionnaire</a>
            </div>
        """, unsafe_allow_html=True)

        # === Question ===
        st.markdown('<div class="field-label">Enter a specific question number (e.g., Q58):</div>', unsafe_allow_html=True)
        question_input = st.text_input("", key="question_number")

        # === Year Selection ===
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        select_all = st.checkbox("All years", value=True, key="select_all_years")
        all_years = [2024, 2022, 2020, 2019]
        selected_years = []
        year_cols = st.columns(len(all_years))
        for idx, year in enumerate(all_years):
            with year_cols[idx]:
                is_checked = True if select_all else False
                if st.checkbox(str(year), value=is_checked, key=f"year_{year}"):
                    selected_years.append(year)

        # === Demographics ===
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        demo_categories = sorted(demo_df[DEMO_CAT_COL].dropna().unique().tolist())
        demo_selection = st.selectbox("", ["All respondents"] + demo_categories, key="demo_main")

        sub_selection = None
        sub_required = False
        if demo_selection in long_list_categories:
            sub_items = demo_df[demo_df[DEMO_CAT_COL] == demo_selection][LABEL_COL].dropna().unique().tolist()
            sub_required = True
            st.markdown(f'<div class="field-label">Please select one option for {demo_selection}:</div>', unsafe_allow_html=True)
            sub_selection = st.selectbox("", sub_items, key=f"sub_{demo_selection.replace(' ', '_')}")

        # === Natural Prompt ===
        st.markdown('<div class="field-label">Or describe what you’re looking for:</div>', unsafe_allow_html=True)
        prompt_text = st.text_area("", key="search_prompt")

        # === Search Button ===
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                # === Validation ===
                if sub_required and not sub_selection:
                    st.warning(f"⚠️ Please select a value for {demo_selection} before proceeding.")
                else:
                    st.markdown("🔄 *Processing your request...*")
                    st.write("Selected Question:", question_input)
                    st.write("Selected Year(s):", selected_years)
                    st.write("Demographic Category:", demo_selection)
                    if sub_selection:
                        st.write("Sub-category value:", sub_selection)
                    st.write("Prompt:", prompt_text)
                    st.success("✅ Query received. (Back-end connection coming soon)")
            st.markdown('</div>', unsafe_allow_html=True)
