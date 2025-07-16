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

# === Menu 1 Function ===
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
                margin-bottom: 25px;
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Centered Layout using Streamlit columns ---
    left, center, right = st.columns([1, 3, 1])

    with center:
        # Title
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)

        # Instructions
        st.markdown("""
            <div class="custom-instruction">
                Use this menu if you already know the specific survey question you wish to explore (e.g., <b>Q58</b>).<br><br>
                You can:
                <ul>
                    <li>Select the year and demographic category using dropdowns</li>
                    <li>Or describe your question in plain language</li>
                </ul>
                The system will confirm your query before retrieving official PSES results.
            </div>
        """, unsafe_allow_html=True)

        # Question list link
        st.markdown("📜 [View the list of survey questions (2024)](https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey/2024-25/2024-25-public-service-employee-survey.html)")

        # Input controls
        st.markdown("### Survey Filters")

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

        question_input = st.text_input("Enter a specific question number (e.g., Q58):")
        prompt_text = st.text_area("Or describe what you're looking for:")

        if st.button("Search"):
            st.markdown("🔄 *Processing your request...*")
            st.write("Selected Year(s):", year)
            st.write("Demographic Category:", demo_selection)
            if sub_selection:
                st.write("Sub-category value:", sub_selection)
            st.write("Question:", question_input)
            st.write("Prompt:", prompt_text)
            st.success("✅ Query received. (Back-end connection coming soon)")
