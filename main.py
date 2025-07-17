import streamlit as st

st.set_page_config(layout="wide")

def main():
    # === Banner ===
    st.image("assets/ANC006-PSES_banner825x200_EN.png", use_column_width=True)

    # === Title & Subtitle (centered) ===
    st.markdown("""
        <h1 style='text-align: center; margin-top: 20px;'>
            Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.
        </h1>
        <h3 style='text-align: center; font-weight: normal; margin-bottom: 40px;'>
            This AI app provides Public Service-wide survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).
        </h3>
    """, unsafe_allow_html=True)

    # === Instruction line (left-aligned, centered container) ===
    st.markdown("""
        <div style="max-width: 950px; margin: auto; text-align: left; font-size: 16px; margin-bottom: 30px;">
            To start your analysis, please select one of the menu options below:
        </div>
    """, unsafe_allow_html=True)

    # === Inject custom CSS for the menu buttons ===
    st.markdown("""
        <style>
            .menu-wrapper {
                display: flex;
                justify-content: center;
                gap: 40px;
                flex-wrap: wrap;
                margin-bottom: 50px;
            }
            .menu-tile {
                width: 160px;
                height: 160px;
                background-color: #f8f9fa;
                border: 2px solid #0d6efd;
                border-radius: 16px;
                text-align: center;
                font-size: 16px;
                font-weight: 600;
                color: #0d6efd;
                padding-top: 28px;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .menu-tile span {
                display: block;
                font-size: 42px;
                margin-bottom: 10px;
            }
            .menu-tile:hover {
                background-color: #0d6efd;
                color: white;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .menu-form {
                margin: 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # === Render the menu tiles as buttons ===
    st.markdown("""
        <div class="menu-wrapper">
            <form class="menu-form" action="" method="post">
                <button name="menu_button" value="1" class="menu-tile" type="submit">
                    <span>🔍</span>Search by Question
                </button>
            </form>
            <form class="menu-form" action="" method="post">
                <button name="menu_button" value="2" class="menu-tile" type="submit">
                    <span>🧩</span>Search by Theme
                </button>
            </form>
            <form class="menu-form" action="" method="post">
                <button name="menu_button" value="3" class="menu-tile" type="submit">
                    <span>📊</span>Analyze Data
                </button>
            </form>
            <form class="menu-form" action="" method="post">
                <button name="menu_button" value="4" class="menu-tile" type="submit">
                    <span>📋</span>View Questionnaire
                </button>
            </form>
        </div>
    """, unsafe_allow_html=True)

    # === Capture which menu was clicked ===
    selected = st.session_state.get("menu_button", None)
    if "menu_button" in st.session_state:
        selected = st.session_state.menu_button
    if selected is None:
        selected = st.experimental_get_query_params().get("menu_button", [None])[0]
    if selected is None:
        selected = st.experimental_get_query_params().get("menu", [None])[0]

    # Use a workaround to capture form POST
    if st.experimental_get_query_params():
        params = st.experimental_get_query_params()
        if "menu_button" in params:
            selected = params["menu_button"][0]

    # === Run the appropriate menu ===
    if st.session_state.get("run_menu") is None:
        st.session_state.run_menu = ""

    if st.session_state.run_menu != selected:
        st.session_state.run_menu = selected

    if st.session_state.run_menu == "1":
        from menu1.main import run_menu1
        run_menu1()

    elif st.session_state.run_menu == "2":
        from menu2.main import run_menu2
        run_menu2()

    elif st.session_state.run_menu == "3":
        st.info("📊 Analyze Data is under construction.")

    elif st.session_state.run_menu == "4":
        st.info("📋 View Questionnaire is under construction.")

if __name__ == "__main__":
    main()
