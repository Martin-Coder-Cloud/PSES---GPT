# main.py — Home-first router; preload on app load; Home-only background;
# Canada.ca link + white "Status" expander under CTA; robust loader imports.
from __future__ import annotations
import importlib
import streamlit as st

st.set_page_config(layout="wide")

# ── Import loader module once, pull functions via getattr so missing names don't break imports ──
_loader_err = ""
_dl = None
try:
    _dl = importlib.import_module("utils.data_loader")
except Exception as e:
    _loader_err = f"{type(e).__name__}: {e}"

def _fn(name, default=None):
    return getattr(_dl, name, default) if _dl else default

prewarm_all              = _fn("prewarm_all")
get_backend_info         = _fn("get_backend_info")
preload_pswide_dataframe = _fn("preload_pswide_dataframe")
get_last_query_diag      = _fn("get_last_query_diag")

# ── tiny nav helper ──────────────────────────────────────────────────────────
def goto(page: str):
    st.session_state["_nav"] = page
    st.rerun()

# ── background reset for non-Home pages ──────────────────────────────────────
def _clear_bg_css():
    st.markdown("""
        <style>
            .block-container {
                background-image: none !important;
                background: none !important;
                color: inherit !important;
                padding-top: 1.25rem !important;
                padding-left: 1.25rem !important;
                padding-bottom: 2rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

# ── Home view (hero background, CTA, Canada.ca link, Status expander) ────────
def render_home():
    # Scoped styles for Home view
    st.markdown("""
        <style>
            .block-container {
                padding-top: 100px !important;
                padding-left: 300px !important;
                padding-bottom: 300px !important;
                background-image: url('https://github.com/Martin-Coder-Cloud/PSES---GPT/blob/main/assets/Teams%20Background%20Tablet_EN.png?raw=true');
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center top;
                color: white;
            }
            .main-section { margin-left: 200px; max-width: 820px; text-align: left; }
            .main-title { font-size: 42px; font-weight: 800; margin-bottom: 16px; }
            .subtitle { font-size: 22px; line-height: 1.4; margin-bottom: 18px; opacity: 0.95; max-width: 700px; }
            .context { font-size: 18px; line-height: 1.55; margin-top: 8px; margin-bottom: 36px; opacity: 0.95; max-width: 700px; text-align: left; }
            .single-button { display: flex; flex-direction: column; gap: 16px; }
            div.stButton > button {
                background-color: rgba(255,255,255,0.08) !important; color: white !important;
                border: 2px solid rgba(255, 255, 255, 0.35) !important;
                font-size: 30px !important; font-weight: 700 !important;
                padding: 26px 34px !important; width: 420px !important; min-height: 88px !important;
                border-radius: 14px !important; text-align: left !important; backdrop-filter: blur(2px);
            }
            div.stButton > button:hover { border-color: white !important; background-color: rgba(255, 255, 255, 0.14) !important; }
            /* White expander summary like your "Advanced" look */
            div[data-testid="stExpander"] > details > summary { color: #fff; font-size: 16px; }
            /* Make links readable on the hero background */
            .main-section a { color: #fff !important; text-decoration: underline; }
            .status-lines { font-size: 14px; line-height: 1.4; }
            .status-subtle { font-size: 13px; opacity: 0.9; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-section'>", unsafe_allow_html=True)

    # Title & intro
    st.markdown(
        "<div class='main-title'>Welcome to the AI-powered Explorer of the Public Service Employee Survey (PSES)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>This app provides Public Service-wide survey results and analysis for the previous 4 survey cycles (2019, 2020, 2022, and 2024)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class='context'>
        The 2024 Public Service Employee Survey (PSES) helps departments and agencies strengthen people management by highlighting areas such as employee engagement, equity and inclusion, anti-racism, and workplace well-being. It provides employees with a voice to share their experiences, supporting workplace improvements that benefit both public servants and Canadians. Results are tracked over time to guide and refine organizational action plans.
        <br><br>
        Each survey cycle combines recurring questions for trend analysis with new ones to reflect emerging priorities. In 2024, Employment Equity demographics were updated to advance diversity and inclusion, and hybrid work questions were streamlined to stay relevant post-pandemic. Statistics Canada, in partnership with the Office of the Chief Human Resources Officer, ran the survey from October 28 to December 31, 2024. The PSES will continue on a two-year cycle.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Primary CTA → Menu 1
    st.markdown("<div class='single-button'>", unsafe_allow_html=True)
    if st.button("▶️ Start your search", key="menu_start_button"):
        goto("menu1")
    st.markdown("</div>", unsafe_allow_html=True)

    # Canada.ca link (directly under the CTA)
    st.markdown(
        "<div class='context'>"
        "<a href='https://www.canada.ca/en/treasury-board-secretariat/services/innovation/public-service-employee-survey.html' target='_blank'>"
        "Public Service Employee Survey - Canada.ca</a>"
        "</div>",
        unsafe_allow_html=True
    )

    # White "Status" expander (with richer diagnostics)
    with st.expander("Status", expanded=False):
        info = {}
        try:
            info = (get_backend_info() or {})
        except Exception:
            info = {}

        mc = info.get("metadata_counts", {}) or {}
        engine = info.get("last_engine", "?")
        inmem  = info.get("inmem_mode", "none")
        rows   = int(info.get("inmem_rows", 0) or 0)
        pswide = "Yes" if info.get("pswide_only") else "No"

        st.markdown("<div class='status-lines'>", unsafe_allow_html=True)
        st.markdown(f"- **Engine:** {engine}")
        st.markdown(f"- **In-memory:** {inmem}")
        st.markdown(f"- **Rows in memory:** {rows:,}")
        st.markdown(f"- **PS-wide only:** {pswide}")
        st.markdown(f"- **Questions (metadata):** {int(mc.get('questions', 0))}")
        st.markdown(f"- **Scales (metadata):** {int(mc.get('scales', 0))}")
        st.markdown(f"- **Demographics (metadata):** {int(mc.get('demographics', 0))}")
        if info.get("parquet_dir"):
            st.markdown(f"- **Parquet directory:** `{info.get('parquet_dir')}`")
        if info.get("csv_path"):
            st.markdown(f"- **CSV path:** `{info.get('csv_path')}`")

        # ── TEMP DIAGNOSTICS — confirm which functions actually imported
        imported = {
            "module_loaded": bool(_dl),
            "prewarm_all": bool(prewarm_all),
            "get_backend_info": bool(get_backend_info),
            "preload_pswide_dataframe": bool(preload_pswide_dataframe),
            "get_last_query_diag": bool(get_last_query_diag),
            "module_error": _loader_err,
        }
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='status-subtle'>Imports:</div>", unsafe_allow_html=True)
        st.json(imported)

        # Live in-memory verification (safe/cached)
        st.markdown("<div class='status-subtle'>Diagnostics (temporary):</div>", unsafe_allow_html=True)
        try:
            if preload_pswide_dataframe:
                df_mem = preload_pswide_dataframe()
                mem_rows = 0 if df_mem is None else int(getattr(df_mem, "shape", [0])[0])
                st.markdown(f"- In-memory DF rows (live check): **{mem_rows:,}**")
                if df_mem is not None and not df_mem.empty:
                    try:
                        uq = df_mem["question_code"].nunique()
                        st.markdown(f"- Unique questions in memory: **{int(uq)}**")
                    except Exception:
                        pass
                    try:
                        ymin = int(df_mem["year"].min())
                        ymax = int(df_mem["year"].max())
                        st.markdown(f"- Year range in memory: **{ymin}–{ymax}**")
                    except Exception:
                        pass
            else:
                st.caption("• preload_pswide_dataframe() not available.")
        except Exception as e:
            st.caption(f"• Preload verification failed: {type(e).__name__}: {e}")

# ── Menu wrappers (no hero background) ───────────────────────────────────────
def render_menu1():
    _clear_bg_css()
    try:
        from menu1.main import run_menu1
        run_menu1()
    except Exception as e:
        st.error(f"Menu 1 is unavailable: {type(e).__name__}: {e}")
    st.markdown("---")
    if st.button("🔙 Return to Main Menu"):
        goto("home")

def render_menu2():
    _clear_bg_css()
    try:
        from menu2.main import run_menu2
        run_menu2()
    except Exception as e:
        st.error(f"Menu 2 is unavailable: {type(e).__name__}: {e}")
    st.markdown("---")
    if st.button("🔙 Return to Main Menu", key="back2"):
        goto("home")

# ── Entry ────────────────────────────────────────────────────────────────────
def main():
    # Remove legacy router flags, if present
    if "run_menu" in st.session_state:
        st.session_state.pop("run_menu")

    # Always preload on app load (first run shows spinner; cached afterwards)
    if prewarm_all:
        if not st.session_state.get("_prewarmed", False):
            with st.spinner("Preparing data backend (one-time)…"):
                prewarm_all()
            st.session_state["_prewarmed"] = True
        else:
            prewarm_all()  # ensures cached resources available without spinner

    # Default to Home
    if "_nav" not in st.session_state:
        st.session_state["_nav"] = "home"

    # Route
    page = st.session_state["_nav"]
    if page == "menu1":
        render_menu1()
    elif page == "menu2":
        render_menu2()
    else:
        render_home()

if __name__ == "__main__":
    main()
