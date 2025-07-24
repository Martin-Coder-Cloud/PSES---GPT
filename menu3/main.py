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

def run_menu3():
    # === Styling and background reset ===
    st.markdown("""
        <style>
            body {
                background-image: none !important;
                background-color: white !important;
            }
            .block-container {
                padding-top: 1rem !important;
            }
            .menu-banner {
                width: 100%;
                height: auto;
                display: block;
                margin-top: 0px;
                margin-bottom: 20px;
            }
            .custom-header {
                font-size: 30px !important;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .custom-instruction {
                font-size: 16px !important;
                line-height: 1.4;
                margin-bottom: 16px;
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
        # === Banner Image ===
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )

        # === Header ===
        st.markdown('<div class="custom-header">📊 Analyze Data</div>', unsafe_allow_html=True)

        # === Instructions (HTML now rendered properly) ===
        st.markdown("""
            <div class="custom-instruction">
                This section provides a narrative summary based on trend and comparative analysis of survey results.
                It focuses exclusively on the percentage of positive responses (combined “Strongly agree” and “Agree”)
                for all opinion-based questions.<br><br>

                Please describe your area of interest in the text box below. For example:<br>
                <em>“Career development and recognition for employees with disabilities between 2020 and 2024.”</em><br><br>

                Optionally, you may specify one or more survey years, as well as a demographic category 
                to focus your comparative analysis.
            </div>
        """, unsafe_allow_html=True)

        # === Prompt Text Box ===
        st.markdown('<div class="field-label">Describe your area of interest:</div>', unsafe_allow_html=True)
        prompt_text = st.text_area("", key="menu3_prompt")

        # === Year Selection ===
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        select_all = st.checkbox("All years", value=True, key="menu3_all_years")
        all_years = [2024, 2022, 2020, 2019]
        selected_years = []
        year_cols = st.columns(len(all_years))
        for idx, year in enumerate(all_years):
            with year_cols[idx]:
                is_checked = True if select_all else False
                if st.checkbox(str(year), value=is_checked, key=f"menu3_year_{year}"):
                    selected_years.append(year)

        # === Demographic Selection ===
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        demo_categories = sorted(demo_df[DEMO_CAT_COL].dropna().unique().tolist())
        demo_selection = st.selectbox("", ["All respondents"] + demo_categories, key="menu3_demo")

        sub_selection = None
        if demo_selection in long_list_categories:
            sub_items = demo_df[demo_df[DEMO_CAT_COL] == demo_selection][LABEL_COL].dropna().unique().tolist()
            st.markdown(f'<div class="field-label">Select a {demo_selection} value:</div>', unsafe_allow_html=True)
            sub_selection = st.selectbox("", sub_items, key=f"menu3_sub_{demo_selection.replace(' ', '_')}")

        # === Submit Button ===
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                if not prompt_text.strip():
                    st.warning("⚠️ Please describe your area of interest before proceeding.")
                else:
                    st.markdown("🔄 *Analyzing your request...*")
                    st.write("Narrative prompt:", prompt_text)
                    st.write("Selected Year(s):", selected_years)
                    st.write("Demographic Category:", demo_selection)
                    if sub_selection:
                        st.write("Sub-category value:", sub_selection)
                    st.success("✅ Request submitted. (Narrative engine under development)")
            st.markdown('</div>', unsafe_allow_html=True)
