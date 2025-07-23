import streamlit as st
import pandas as pd

# === Load Metadata ===
demo_df = pd.read_excel("metadata/Demographics.xlsx")
demo_df.columns = [col.strip() for col in demo_df.columns]

# === Load Survey Questions Metadata ===
question_df = pd.read_excel("metadata/Survey Questions.xlsx")
question_df.columns = [col.strip().lower() for col in question_df.columns]
question_df = question_df.rename(columns={"question": "code", "english": "text"})
question_df["qnum"] = question_df["code"].str.extract(r'Q?(\d+)').astype(int)
question_df = question_df.sort_values("qnum")
question_df["display"] = question_df["code"] + " – " + question_df["text"]

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
    # === Reset background to white and insert banner image ===
    st.markdown("""
        <style>
            body {
                background-image: none !important;
                background-color: white !important;
            }
            .menu-banner {
                width: 100%;
                max-width: 1000px;
                height: auto;
                display: block;
                margin: 0 auto 20px auto;
            }

            }
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
        <img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>
    """, unsafe_allow_html=True)

    # === Layout ===
    left, center, right = st.columns([1, 3, 1])
    with center:
        # === Header ===
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)

        # === Instructions (compact) ===
        st.markdown("""
            <div class="custom-instruction">
                Use this menu to explore results for a specific survey question.<br>
                <br>
                Select a question from the list below to begin or enter a keyword to search a question.
            </div>
        """, unsafe_allow_html=True)

        # === Question Selection ===
        st.markdown('<div class="field-label">Select a survey question:</div>', unsafe_allow_html=True)
        question_options = question_df["display"].tolist()
        selected_label = st.selectbox("Choose from the official list (type Q# or keywords to filter):", question_options, key="question_dropdown")
        question_input = question_df[question_df["display"] == selected_label]["code"].values[0]

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

        # === Search Button ===
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                if not question_input:
                    st.warning("⚠️ Please select a question from the list.")
                elif sub_required and not sub_selection:
                    st.warning(f"⚠️ Please select a value for {demo_selection} before proceeding.")
                else:
                    st.markdown("🔄 *Processing your request...*")
                    st.write("Selected Question:", question_input)
                    st.write("Selected Year(s):", selected_years)
                    st.write("Demographic Category:", demo_selection)
                    if sub_selection:
                        st.write("Sub-category value:", sub_selection)
                    st.success("✅ Query received. (Back-end connection coming soon)")
            st.markdown('</div>', unsafe_allow_html=True)
