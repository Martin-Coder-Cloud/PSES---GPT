# wizard/main.py — single-page wizard with progress bar
WIZARD_CONTROLLER_VERSION = "progress-v1.0"

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional
import streamlit as st

STEP_KEYS = ["question", "years", "demo"]  # 3 steps before results

@dataclass
class StepResult:
    data: Dict[str, Any] = None
    is_valid: bool = False
    next_step: Optional[str] = None
    message: Optional[str] = None
    go_results: bool = False

def _import_callable(mod_name: str, fn_name: str):
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)
    except Exception as err:
        msg = f"{type(err).__name__}: {err}"
        def _placeholder(*_a, _msg=msg, **_kw):
            st.info("This section isn’t implemented yet.")
            st.caption(f"Missing: `{mod_name}.{fn_name}()`")
            st.caption(_msg)
            return StepResult(is_valid=False, message=_msg)
        return _placeholder

def _ensure_state():
    if "wizard" not in st.session_state:
        st.session_state.wizard = {
            "question": None,            # {"code","label"}
            "years": [],                 # ["2024","2022",...]
            "demo_category": "All respondents",
            "demo_subgroup": None,
            "demcodes": [None],
            "dem_disp_map": {None: "All respondents"},
            "category_in_play": False,
            "has_results": False,        # becomes True after we run the query
        }
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0  # 0,1,2 for steps; results appear below

def _header():
    st.markdown(
        """
        <style>
          .custom-header{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }
          .custom-instruction{ font-size: 15px; line-height: 1.45; margin-bottom: 10px; color: #333; }
          .field-label{ font-size: 16px; font-weight: 600; margin: 10px 0 6px; color: #222; }
          .tiny-note{ font-size: 12px; color: #666; margin-top: -4px; margin-bottom: 10px; }
          .pill{ padding:6px 10px;border-radius:9999px;border:1px solid #ddd; background:#f8f8f8; color:#444; }
          .pill.active{ background:#1f77b4; color:white; border-color:#1f77b4; }
          .pill.done{ background:#74c476; color:white; border-color:#5aa463; }
          .wizard-bar{ margin:8px 0 14px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<img style='width:75%;max-width:740px;height:auto;display:block;margin:0 auto 16px;' "
        "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="custom-header">🔍 PSES AI Explorer — Search Wizard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="custom-instruction">'
        'Please select a question, survey year(s), and (optionally) a demographic breakdown. '
        'The app will show results on this page after you run the query.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Wizard controller: {WIZARD_CONTROLLER_VERSION}")

def _progress_ui():
    step = st.session_state.wizard_step
    has_results = st.session_state.wizard.get("has_results", False)
    # 0..3 progress (3 steps → results). Once results are shown, show 100%.
    completed = step
    if has_results:
        completed = 3
    pct = int(round(100 * completed / 3))
    st.progress(pct, text=f"Progress: {pct}%")
    # Pills
    labels = ["1) Question", "2) Years", "3) Demographics"]
    cols = st.columns(3)
    for i, lab in enumerate(labels):
        cls = "pill"
        if i < step:
            cls += " done"
        elif i == step and not has_results:
            cls += " active"
        cols[i].markdown(f"<div class='{cls}'>{lab}</div>", unsafe_allow_html=True)

def _nav_buttons(is_valid: bool):
    left, mid, right = st.columns([1, 4, 1])
    with left:
        if st.button("◀ Back", disabled=(st.session_state.wizard_step == 0)):
            st.session_state.wizard_step = max(0, st.session_state.wizard_step - 1)
            st.session_state.wizard["has_results"] = False
            st.experimental_rerun()
    with right:
        label = "Next ▶"
        if st.session_state.wizard_step == 2:
            label = "🔎 Run query"
        if st.button(label, disabled=(not is_valid)):
            if st.session_state.wizard_step < 2:
                st.session_state.wizard_step += 1
                st.experimental_rerun()
            else:
                # Step 3 → run query with spinner and then show results below
                _run_and_show_results()

def _run_and_show_results():
    st.session_state.wizard["has_results"] = False
    with st.spinner("Processing data…"):
        render_results = _import_callable("wizard.results", "render")
        render_results(st.session_state.wizard)
    st.session_state.wizard["has_results"] = True
    st.experimental_rerun()

def render():
    _ensure_state()
    _header()
    _progress_ui()
    st.divider()

    step = st.session_state.wizard_step

    # Render the active step (single page)
    if step == 0:
        render_q = _import_callable("wizard.steps.step_question", "render")
        res = render_q(st.session_state.wizard)
        if isinstance(res, dict) and res.get("data"):
            st.session_state.wizard.update(res["data"])
        _nav_buttons(bool(res.get("is_valid")))
    elif step == 1:
        render_years = _import_callable("wizard.steps.step_years", "render")
        res = render_years(st.session_state.wizard)
        if isinstance(res, dict) and res.get("data"):
            st.session_state.wizard.update(res["data"])
        _nav_buttons(bool(res.get("is_valid")))
    elif step == 2:
        render_demo = _import_callable("wizard.steps.step_demo", "render")
        res = render_demo(st.session_state.wizard)
        if isinstance(res, dict) and res.get("data"):
            st.session_state.wizard.update(res["data"])
        _nav_buttons(bool(res.get("is_valid")))
    else:
        st.error("Unknown step.")

    st.markdown("---")

    # If we already ran a query in this session, keep the results visible at the bottom
    if st.session_state.wizard.get("has_results", False):
        st.subheader("Results")
        render_results = _import_callable("wizard.results", "render")
        render_results(st.session_state.wizard)

# Back-compat alias for your router:
def run_wizard():
    render()

if __name__ == "__main__":
    render()
