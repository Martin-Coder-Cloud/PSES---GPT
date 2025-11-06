# app/menu1/constants.py
"""
Centralized constants for Menu 1.
Keeping these here reduces churn in main logic and render modules.
"""

# --- Open Data source (shown under each table) ---
SOURCE_URL: str = "https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f"
SOURCE_TITLE: str = "Public Service Employee Survey Results – Open Government Portal"

# --- UI defaults ---
PAGE_TITLE: str = "PSES Explorer Search"
CENTER_COLUMNS: list[int] = [1, 3, 1]  # layout for left, center, right
MAX_QUESTIONS: int = 5  # cap for multi-select
DEFAULT_YEARS: list[int] = [2024, 2022, 2020, 2019]
DEFAULT_AI_TOGGLE: bool = True  # AI on by default
DEFAULT_DIAG_TOGGLE: bool = False  # Diagnostics off by default

# --- AI / Model settings (non-secret; key stays in st.secrets or env) ---
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"

# --- Analysis thresholds (points) ---
TREND_THRESHOLDS = {
    "stable": 1,   # ≤ 1 point
    "slight": 2,   # >1–2 points
    "notable": 999  # >2 points and above
}

GAP_THRESHOLDS = {
    "minimal": 2,  # ≤ 2 points
    "notable": 5,  # >2–5 points
    "important": 999
}

# --- Text/UI strings (keep here to avoid scattering copy) ---
BANNER_URL: str = (
    "https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/"
    "PSES%20email%20banner.png"
)

# Plain text, rendered in layout.py as a subtitle
INSTRUCTION_HTML: str = "To conduct your search, please follow the 3 steps below:"

# --- CSS Styling ---
BASE_CSS: str = """
<style>
  body { background-image: none !important; background-color: white !important; }
  .block-container { padding-top: 1rem !important; }
  .menu-banner { width: 100%; height: auto; display: block; margin-top: 0px; margin-bottom: 20px; }
  .custom-header { font-size: 30px !important; font-weight: 700; margin-bottom: 6px; }
  .custom-instruction { font-size: 16px !important; line-height: 1.4; margin-bottom: 10px; color: #333; }
  .field-label { font-size: 18px !important; font-weight: 600 !important; margin-top: 12px !important; margin-bottom: 2px !important; color: #222 !important; }
  .action-row { display:flex; gap:10px; align-items:center; }
  [data-testid="stSwitch"] div[role="switch"][aria-checked="true"] { background-color: #e03131 !important; }
  [data-testid="stSwitch"] div[role="switch"] { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1); }
  .tiny-note { font-size: 13px; color: #444; margin-bottom: 6px; }
  .diag-box { background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 10px 12px; }

  /* --- Step 1 buttons on Menu 1 --- */
  #menu1-step1-search-btn .stButton > button,
  #menu1-step1-clear-btn .stButton > button {
    background-color: #e03131 !important;   /* bright red */
    color: #ffffff !important;              /* white text */
    border: 1px solid #c92a2a !important;
    font-weight: 700 !important;
  }
  #menu1-step1-search-btn .stButton > button:hover,
  #menu1-step1-clear-btn .stButton > button:hover {
    background-color: #c92a2a !important;   /* darker red on hover */
    border-color: #a61e1e !important;
    color: #ffffff !important;
  }
  #menu1-step1-search-btn .stButton > button:active,
  #menu1-step1-clear-btn .stButton > button:active {
    background-color: #a61e1e !important;   /* deepest red on click */
    border-color: #8c1a1a !important;
    color: #ffffff !important;
  }
</style>
"""
