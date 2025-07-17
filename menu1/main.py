import streamlit as st
import pandas as pd

# === Load Metadata ===
demo_df = pd.read_excel("metadata/Demographics.xlsx")
demo_df.columns = [col.strip() for col in demo_df.columns]  # Normalize headers

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
    # === Custom Styling ===
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
            .checkbox-inline label {
                display: inline-block;
                margin-right: 15px;
            }
            .big-button button {
                font-size: 18px !important;
                padding: 0.75em 2em !important;
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # === Layout Container ===
    left, center, right = st.columns([1, 3, 1])
    with center:
        # --- Page Header ---
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)

        # --- Instructions ---
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

        # === QUESTION INPUT ===
        st.markdown('<div class="field-label">Enter a specific question number (e.g., Q58):</div>', unsafe_allow_html=True)
        question_input = st.text_input("", key="question_number")

        # === YEAR SELECTION ===
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        select_all = st.checkbox("All years", value=True, key="select_all_years")
        all_years = [2024, 2022, 2020, 2019]
        selected_years = []
        year_checkbox_cols = st.columns(len(all_years))
        for idx, year in enumerate(all_years):
            with year_checkbox_cols[idx]:
                is_checked = True if select_all else False
                if st.checkbox(str(year), value=is_checked, key=f"year_{year}"):
                    selected_years.append(year)

        # === DEMOGRAPHIC SELECTION ===
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        demo_categories = sorted(demo_df[DEMO_CAT_COL].dropna().unique().tolist())
        demo_selection = st.selectbox("", ["All respondents"] + demo_categories, key="demo_main")

        sub_selection = None
        if demo_selection in long_list_categories:
            sub_items = demo_df[demo_df[DEMO_CAT_COL] == demo_selection][LABEL_COL].dropna().unique().tolist()
            widget_key = f"subselect_{demo_selection.replace(' ', '_')}"
            if len(sub_items) > 50:
                st.markdown(f'<div class="field-label">Select a {demo_selection} value:</div>', unsafe_allow_html=True)
                sub_select = st.multiselect("", options=sub_items, max_selections=1, key=f"multiselect_{widget_key}")
                if sub_select:
                    sub_selection = sub_select[0]
            else:
                st.markdown(f'<div class="field-label">Select a {demo_selection} value:</div>', unsafe_allow_html=True)
                sub_selection = st.selectbox("", sub_items, key=f"selectbox_{widget_key}")

        # === NATURAL LANGUAGE PROMPT AT END ===
        st.markdown('<div class="field-label">Or describe what you’re looking for:</div>', unsafe_allow_html=True)
        prompt_text = st.text_area("", key="search_prompt")

        # === SUBMIT BUTTON ===
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                st.markdown("🔄 *Processing your request...*")
                st.write("Selected Question:", question_input)
                st.write("Selected Year(s):", selected_years)
                st.write("Demographic Category:", demo_selection)
                if sub_selection:
                    st.write("Sub-category value:", sub_selection)
                st.write("Prompt:", prompt_text)
                st.success("✅ Query received. (Back-end connection coming soon)")
            st.markdown('</div>', unsafe_allow_html=True)
