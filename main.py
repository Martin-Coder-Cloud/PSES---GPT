import streamlit as st

st.set_page_config(layout="wide")

# ✅ Helper function: show menu and return button below
def show_return_then_run(run_func):
    run_func()
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        st.session_state.run_menu = None
        st.experimental_rerun()

def main():
    # === Fix banner layout: tight margin, centered image ===
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0px !important;
                margin-top: -30px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Banner centered using columns
    left, center, right = st.columns([1, 3, 1])
    with center:
        st.image("assets/ANC006-PSES_banner825x200_EN.png", width=750)

    # === Show main menu only if no selection has been made ===
    if "run_menu" not in st.session_state:

        # === Title & Subtitle ===
        st.markdown("""
            <h1 style='text-align: center; margin-top: 10px; font-size: 28px;'>
                Welcome to the AI Explorer of the Public Service Employee Survey (PSES) results.
            </h1>
            <h3 style='text-align: center; font-weight: normal; margin-bottom: 25px; font-size: 20px;'>
                This AI app provides Public Service-wide survey results and analysis on the latest iterations of the survey (2019, 2020, 2022, 2024).
            </h3>
        """, unsafe_allow_html=True)

        # === Button CSS (tile style with icon on top) ===
        st.markdown("""
            <style>
                .menu-wrapper {
                    display: flex;
                    justify-content: center;
                    gap: 30px;
                    flex-wrap: wrap;
                    margin-top: 20px;
                    margin-bottom: 60px;
                }
                .menu-tile {
                    width: 240px;
                    height: 240px;
                    background-color: #f8f9fa;
                    border: 2px solid #0d6efd;
                    border-radius: 20px;
                    text-align: center;
                    font-size: 18px;
                    font-weight: 600;
                    color: #0d6efd;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.2s ease;
                    cursor: pointer;
                }
                .menu-tile:hover {
                    background-color: #0d6efd;
                    color: white;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                .menu-icon {
                    font-size: 60px;
                    margin-bottom: 10px;
                }
                .menu-link {
                    text-decoration: none;
                }
            </style>
        """, unsafe_allow_html=True)

        # === Menu Buttons in Tile Layout ===
        st.markdown("""
            <div class="menu-wrapper">
                <a href="#" onclick="window.location.reload();" class="menu-link" target="_self">
                    <div onclick="window.parent.postMessage({type: 'streamlit:sendMessage', data: '1'}, '*')" class="menu-tile">
                        <div class="menu-icon">🔍</div>
                        Search by Question
                    </div>
                </a>
                <a href="#" onclick="window.location.reload();" class="menu-link" target="_self">
                    <div onclick="window.parent.postMessage({type: 'streamlit:sendMessage', data: '2'}, '*')" class="menu-tile">
                        <div class="menu-icon">🧩</div>
                        Search by Theme
                    </div>
                </a>
                <a href="#" onclick="window.location.reload();" class="menu-link" target="_self">
                    <div onclick="window.parent.postMessage({type: 'streamlit:sendMessage', data: '3'}, '*')" class="menu-tile">
                        <div class="menu-icon">📊</div>
                        Analyze Data
                    </div>
                </a>
                <a href="#" onclick="window.location.reload();" class="menu-link" target="_self">
                    <div onclick="window.parent.postMessage({type: 'streamlit:sendMessage', data: '4'}, '*')" class="menu-tile">
                        <div class="menu-icon">📋</div>
                        View Questionnaire
                    </div>
                </a>
            </div>
        """, unsafe_allow_html=True)

        # === Capture tile clicks using JavaScript message (fallback) ===
        js = """
        <script>
        window.addEventListener("message", (event) => {
            if (event.data === "1") {
                window.parent.postMessage({type: "streamlit:setComponentValue", value: 1}, "*");
                window.location.href = window.location.href + "?menu=1";
            }
            if (event.data === "2") {
                window.parent.postMessage({type: "streamlit:setComponentValue", value: 2}, "*");
                window.location.href = window.location.href + "?menu=2";
            }
            if (event.data === "3") {
                window.parent.postMessage({type: "streamlit:setComponentValue", value: 3}, "*");
                window.location.href = window.location.href + "?menu=3";
            }
            if (event.data === "4") {
                window.parent.postMessage({type: "streamlit:setComponentValue", value: 4}, "*");
                window.location.href = window.location.href + "?menu=4";
            }
        });
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)

    # === Routing based on selection ===
    if "run_menu" in st.session_state:
        selection = st.session_state.run_menu
    else:
        params = st.experimental_get_query_params()
        selection = params.get("menu", [None])[0]
        if selection:
            st.session_state.run_menu = selection

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

if __name__ == "__main__":
    main()
