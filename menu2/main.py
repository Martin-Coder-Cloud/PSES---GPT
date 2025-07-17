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

def run_menu2():
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
        st.markdown('<div class="custom-header">🧩 Search by Theme</div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="custom-instruction">
                Use this menu to explore survey results by thematic areas (e.g., harassment, leadership, equity).<br><br>
                Once a theme is selected, the tool will display all questions associated with that theme. You can then choose a question to view results.<br><br>
                A theme and at least one year must be selected to proceed.
            </div>
        """, unsafe_allow_html=True)

        # === THEME DROPDOWN ===
        st.markdown('<div class="field-label">Select a predefined survey theme:</div>', unsafe_allow_html=True)
        themes = ["Select a theme", "Workplace Culture", "Discrimination", "Career Development", "Wellbeing"]
        theme_selection = st.selectbox("", themes, key="theme_selector")

        # === YEAR SELECTION ===
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        select_all = st.checkbox("All years", value=True, key="select_all_years_2")
        all_years = [2024, 2022, 2020, 2019]
        selected_years = []
        year_cols = st.columns(len(all_years))
        for i, year in enumerate(all_years):
            with year_cols[i]:
                is_checked = True if select_all else False
                if st.checkbox(str(year), value=is_checked, key=f"year2_{year}"):
                    selected_years.append(year)

        # === DEMOGRAPHICS ===
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        demo_categories = sorted(demo_df[DEMO_CAT_COL].dropna().unique().tolist())
        demo_selection = st.selectbox("", ["All respondents"] + demo_categories, key="demo_theme_main")

        sub_selection = None
        if demo_selection in long_list_categories:
            sub_items = demo_df[demo_df[DEMO_CAT_COL] == demo_selection][LABEL_COL].dropna().unique().tolist()
            widget_key = f"subselect2_{demo_selection.replace(' ', '_')}"

            if len(sub_items) > 100:
                st.markdown(f'<div class="field-label">Enter a {demo_selection} value:</div>', unsafe_allow_html=True)
                sub_selection = st.text_input("", key=f"textinput_{widget_key}")
                if sub_selection and sub_selection not in sub_items:
                    st.warning(f"⚠️ '{sub_selection}' is not a valid {demo_selection}. Please select a recognized value.")
                    sub_selection = None
            else:
                st.markdown(f'<div class="field-label">Select a {demo_selection} value:</div>', unsafe_allow_html=True)
                sub_selection = st.selectbox("", sub_items, key=f"selectbox_{widget_key}")

        # === NATURAL LANGUAGE PROMPT ===
        st.markdown('<div class="field-label">Or describe your theme or concern in your own words:</div>', unsafe_allow_html=True)
        prompt_text = st.text_area("", key="theme_prompt")

        # === SUBMIT BUTTON ===
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                st.markdown("🔄 *Processing your request...*")
                st.write("Selected Theme:", theme_selection)
                st.write("Selected Year(s):", selected_years)
                st.write("Demographic Category:", demo_selection)
                if sub_selection:
                    st.write("Sub-category value:", sub_selection)
                st.write("Prompt:", prompt_text)
                st.success("✅ Query received. Theme-to-question matching will be implemented once data is loaded.")
            st.markdown('</div>', unsafe_allow_html=True)
