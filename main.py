import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

def main():
    # === Banner ===
    st.image("assets/ANC006-PSES_banner825x200_EN.png")

    # === Title and Subtitle ===
    st.title("Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.")
    st.subheader("This AI app provides Public Service-wide survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).")

    # === Instruction (moved and styled) ===
    st.markdown("""
        <div style="margin-left: 20%; margin-top: -10px; margin-bottom: 5px; font-size: 16px;">
            To start your analysis, please select one of the menu options below:
        </div>
    """, unsafe_allow_html=True)

    # === Menu Options ===
    menu = option_menu(
        menu_title=None,
        options=["1", "2", "3", "4"],
        icons=["search", "puzzle", "bar-chart", "clipboard"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#f0f2f6"
            },
            "icon": {"color": "#0d6efd", "font-size": "24px"},
            "nav-link": {
                "font-size": "18px",
                "font-weight": "600",
                "text-align": "center",
                "margin": "0 10px",
                "--hover-color": "#eee"
            },
            "nav-link-selected": {
                "background-color": "#0d6efd",
                "color": "white"
            }
        }
    )

    # === Route Logic ===
    if menu == "1":
        from menu1.main import run_menu1
        run_menu1()

    elif menu == "2":
        from menu2.main import run_menu2
        run_menu2()

    elif menu == "3":
        st.info("📊 Analyze Data is under construction.")

    elif menu == "4":
        st.info("📋 View Questionnaire is under construction.")

if __name__ == "__main__":
    main()
