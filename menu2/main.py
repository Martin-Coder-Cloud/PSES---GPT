import streamlit as st
import pandas as pd

# === Load Metadata ===
demo_df = pd.read_excel("metadata/Demographics.xlsx")
demo_df.columns = [col.strip() for col in demo_df.columns]

theme_df = pd.read_excel("metadata/Survey Themes.xlsx")
theme_df.columns = [col.strip() for col in theme_df.columns]  # Preserve original casing

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
    # === Styling ===
    st.markdown("""
        <style>
            .custom-header {
                font-size: 30px !important;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .custom-instruction {
                font-size: 16px !important;
                line-height: 1.4;
                margin-bottom: 10px;
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
        st.markdown('<div class="custom-header">🧩 Search by Theme</div>', unsafe_allow_html=True)

        # === Instructions ===
        st.markdown("""
            <div class="custom-instruction">
                Use this menu to explore survey results by thematic areas (e.g., harassment, leadership, equity).<br>
                You can choose a pre-defined survey theme from the list below, or describe your topic using keywords in the search box.
            </div>
        """, unsafe_allow_html=True)

        # === Main Theme Dropdown ===
        main_themes = sorted(theme_df["INDICATORENG"].dropna().unique().tolist())
        st.markdown('<div class="field-label">Select a main survey theme (optional):</div>', unsafe_allow_html=True)
        selected_main_theme = st.selectbox("", [""] + main_themes, key="main_theme")

        # === Sub-theme Dropdown (conditional) ===
        sub_theme = None
        if selected_main_theme:
            sub_options = sorted(
                theme_df[theme_df["INDICATORENG"] == selected_main_theme]["SUBINDICATORENG"].dropna().unique().tolist()
            )
            st.markdown('<div class="field-label">Select a sub-theme:</div>', unsafe_allow_html=True)
            sub_theme = st.selectbox("", [""] + sub_options, key="sub_theme")

        # === Prompt (after dropdowns) ===
        st.markdown('<div class="field-label">Or describe your theme using keywords:</div>', unsafe_allow_html=True)
        prompt_text = st.text_area("", key="theme_prompt")

        # === Year Selection (all shown, all selected by default) ===
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        select_all = st.checkbox("All years", value=True, key="select_all_years_2")
        all_years = [2024, 2022, 2020, 2019]
        selected_years = []
        year_cols = st.columns(len(all_years))
        for idx, year in enumerate(all_years):
            with year_cols[idx]:
                is_checked = True if select_all else False
                if st.checkbox(str(year), value=is_checked, key=f"year_{year}"):
                    selected_years.append(year)

        # === Demographic Selection (default to All respondents) ===
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        demo_categories = sorted(demo_df[DEMO_CAT_COL].dropna().unique().tolist())
        demo_selection = st.selectbox("", ["All respondents"] + demo_categories, key="demo_theme_main")

        sub_selection = None
        if demo_selection in long_list_categories:
            sub_items = demo_df[demo_df[DEMO_CAT_COL] == demo_selection][LABEL_COL].dropna().unique().tolist()
            st.markdown(f'<div class="field-label">Select a {demo_selection} value:</div>', unsafe_allow_html=True)
            sub_selection = st.selectbox("", sub_items, key=f"selectbox_{demo_selection.replace(' ', '_')}")

        # === Submit Button & Validation ===
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                if not prompt_text.strip() and (not selected_main_theme or not sub_theme):
                    st.warning("⚠️ Please enter keywords or select both a main theme and sub-theme.")
                else:
                    st.markdown("🔄 *Processing your request...*")
                    if prompt_text.strip():
                        st.write("🔹 Prompt override used:", prompt_text)
                    else:
                        st.write("Main Theme:", selected_main_theme)
                        st.write("Sub-Theme:", sub_theme)
                    st.write("Selected Year(s):", selected_years)
                    st.write("Demographic Category:", demo_selection)
                    if sub_selection:
                        st.write("Sub-category value:", sub_selection)
                    st.success("✅ Theme search submitted. (Back-end coming soon)")
            st.markdown('</div>', unsafe_allow_html=True)
