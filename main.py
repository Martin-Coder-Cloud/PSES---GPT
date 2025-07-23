import streamlit as st

st.set_page_config(layout="wide")

# ✅ Helper: show menu and return button
def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_set_query_params()
        st.experimental_rerun()

def main():
    # ✅ Background and style setup
    st.markdown("""
        <style>
            .block-container {
                padding-top: 100px !important;
                padding-left: 0px !important;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-size: cover;
                background-position: top left;
                background-repeat: no-repeat;
                background-attachment: fixed;
                min-height: 100vh;
                overflow-x: hidden;
                color: white !important;
            }
            .left-column {
                max-width: 700px;
                margin-left: 120px; /* ✅ pushed more toward center */
            }
            h1, h3, p, a {
                color: white !important;
            }
            .main-title {
                font-size: 36px;
                font-weight: 700;
                margin-bottom: 20px;
                line-height: 1.3;
                word-wrap: break-word;
            }
            .subtitle {
                font-size: 22px;
                font-weight: 400;
                margin-bottom: 45px;
                word-wrap: break-word;
            }
            .menu-option {
                font-size: 20px;
                font-weight: 600;
                margin: 14px 0;
                padding: 16px 28px;
                background-color: rgba(255,255,255,0.12);
                border-radius: 12px;
                display: inline-block;
                text-decoration: none;
                transition: background 0.3s;
            }
            .menu-option:hover {
                background-color: rgba(255,255,255,0.25);
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Menu routing
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

    # ✅ Run selected menu
    if "run_menu" in st.session_state:
        if st.session_state.run_menu == "1":
            from menu1.main import run_menu1
            show_return_then_run(run_menu1)
        elif st.session_state.run_menu == "2":
            from menu2.main import run_menu2
            show_return_then_run(run_menu2)
        elif st.session_state.run_menu == "3":
            show_return_then_run(lambda: st.info("📊 Analyze Data is under construction."))
        elif st.session_state.run_menu == "4":
            show_return_then_run(lambda: st.info("📋 View Questionnaire is under construction."))
        return

    # ✅ Render title, subtitle, and vertical menu
    st.markdown("<div class='left-column'>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Welcome to the AI Explorer of the Public Service Employee Survey (PSES)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>This AI app provides Public Service-wide survey results and analysis (2019, 2020, 2022, and 2024).</div>", unsafe_allow_html=True)

    for label, icon, menu_id in [
        ("Search by Question", "🔍", "1"),
        ("Search by Theme", "🧩", "2"),
        ("Analyze Data", "📊", "3"),
        ("View Questionnaire", "📋", "4"),
    ]:
        st.markdown(
            f"<a class='menu-option' href='?menu={menu_id}'>{icon} {label}</a>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
