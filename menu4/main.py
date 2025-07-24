import streamlit as st

def run_menu4():
    # === Page styling ===
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
                line-height: 1.6;
                margin-bottom: 20px;
                color: #333;
            }
            .field-label {
                font-size: 18px !important;
                font-weight: 600;
                margin-top: 12px;
                margin-bottom: 6px;
                color: #222;
            }
        </style>
    """, unsafe_allow_html=True)

    # === Layout ===
    left, center, right = st.columns([1, 3, 1])
    with center:
        # === Banner ===
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )

        # === Header ===
        st.markdown('<div class="custom-header">📋 View Questionnaire</div>', unsafe_allow_html=True)

        # === Instructions ===
        st.markdown(
            """
            <div class="custom-instruction">
                You can search the 2024 PSES questionnaire by entering keywords in the text box below.<br><br>

                Or, you can browse the full questionnaire by section directly on the official site:<br>
                <a href="https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey/2024-25/2024-25-public-service-employee-survey.html" target="_blank">
                🌐 View the 2024-25 PSES Questionnaire online</a>.<br><br>

                A downloadable PDF version is available here:<br>
                <a href="https://github.com/Martin-Coder-Cloud/PSES---GPT/raw/main/2024%20PSES%20Questionnaire_English.pdf" target="_blank">📄 Download 2024 Questionnaire (English)</a><br><br>

                📄 French version coming soon.
            </div>
            """,
            unsafe_allow_html=True
        )

        # === Search Text Box ===
        st.markdown('<div class="field-label">Search questions by keyword:</div>', unsafe_allow_html=True)
        query = st.text_input("", placeholder="e.g. harassment, onboarding, senior leadership", key="menu4_query")

        if st.button("🔍 Search"):
            if not query.strip():
                st.warning("⚠️ Please enter a keyword to search.")
            else:
                st.markdown("🔄 *Searching questionnaire (functionality coming soon)...*")
