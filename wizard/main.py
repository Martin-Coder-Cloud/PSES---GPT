# wizard/main.py — compact controller for the multi-step Search Wizard
# Version tag for quick sanity check:
WIZARD_CONTROLLER_VERSION = "controller-v0.2"

from __future__ import annotations
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional
import streamlit as st

STEP_ORDER = ["question", "years", "demo", "review", "results"]
STEP_TITLES = {
    "question": "1) Select a Question",
    "years":    "2) Select Years",
    "demo":     "3) Choose a Demographic (optional)",
    "review":   "4) Review & Run",
    "results":  "Results",
}

@dataclass
class StepResult:
    data: Dict[str, Any] = None
    is_valid: bool = False
    next_step: Optional[str] = None
    message: Optional[str] = None
    go_results: bool = False

def _as_result(obj: Any) -> StepResult:
    if isinstance(obj, StepResult):
        return obj
    if isinstance(obj, dict):
        return StepResult(
            data=obj.get("data") or {},
            is_valid=bool(obj.get("is_valid", False)),
            next_step=obj.get("next_step"),
            message=obj.get("message"),
            go_results=bool(obj.get("go_results", False)),
        )
    return StepResult(data={}, is_valid=False, message="Step not ready yet.")

def _ensure_state() -> None:
    if "wizard" not in st.session_state:
        st.session_state.wizard = {
            "question": None,
            "years": [],
            "demo": None,
            "group": None,
        }
    if "wizard_page" not in st.session_state:
        st.session_state.wizard_page = "question"

def _reset_to_start() -> None:
    st.session_state.wizard = {
        "question": None,
        "years": [],
        "demo": None,
        "group": None,
    }
    st.session_state.wizard_page = "question"

def _go_to(step_key: str) -> None:
    if step_key in STEP_ORDER:
        st.session_state.wizard_page = step_key

def _prev_step(current: str) -> Optional[str]:
    try:
        i = STEP_ORDER.index(current)
        return STEP_ORDER[i - 1] if i > 0 else None
    except ValueError:
        return None

def _import_callable(module_name: str, fn_name: str):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, fn_name)
    except Exception as e:
        def _placeholder(*args, **kwargs):
            st.info("This page isn’t implemented yet. We’ll add it next.")
            st.caption(f"Missing: `{module_name}.{fn_name}()`")
            return StepResult(is_valid=False, message=str(e))
        return _placeholder

def _render_step(step_key: str) -> StepResult:
    mapping = {
        "question": ("wizard.steps.step_question", "render"),
        "years":    ("wizard.steps.step_years", "render"),
        "demo":     ("wizard.steps.step_demo", "render"),
        "review":   ("wizard.steps.step_review", "render"),
    }
    if step_key == "results":
        mod, fn = ("wizard.results", "render")
        render_results = _import_callable(mod, fn)
        render_results(st.session_state.wizard)  # draws results page
        return StepResult(is_valid=True)

    mod, fn = mapping.get(step_key, (None, None))
    if not mod:
        st.error(f"Unknown step: {step_key}")
        return StepResult(is_valid=False)
    render_callable = _import_callable(mod, fn)
    return _as_result(render_callable(st.session_state.wizard))

def _header():
    st.markdown(
        f"""
        <div style="padding:8px 0 16px 0;">
          <h2 style="margin:0;">PSES AI Explorer — Search Wizard</h2>
          <div style="opacity:.8;">
            Please select a question, survey year(s), and (optionally) a demographic breakdown.
            The app will send you to a separate <b>Results</b> page to run the query.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Wizard controller: {WIZARD_CONTROLLER_VERSION}")

def _breadcrumbs(current_key: str):
    cols = st.columns(len(STEP_ORDER))
    for i, key in enumerate(STEP_ORDER):
        label = STEP_TITLES.get(key, key.title())
        with cols[i]:
            if key == current_key:
                st.markdown(f"**{label}**")
            else:
                st.markdown(label)

def _top_actions(current_key: str):
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔙 Start Over", use_container_width=True):
            _reset_to_start()
            st.experimental_rerun()
    with c2:
        prev_key = _prev_step(current_key)
        disabled = prev_key is None or current_key == "results"
        if st.button("◀ Back", disabled=disabled, use_container_width=True):
            if prev_key:
                _go_to(prev_key)
                st.experimental_rerun()

def render():
    _ensure_state()
    current_key = st.session_state.wizard_page

    _header()
    _breadcrumbs(current_key)
    _top_actions(current_key)
    st.divider()

    result = _render_step(current_key)

    if result.data:
        st.session_state.wizard.update(result.data)

    if result.go_results or (result.next_step == "results"):
        _go_to("results")
        st.experimental_rerun()
    elif result.next_step and result.next_step in STEP_ORDER:
        _go_to(result.next_step)
        st.experimental_rerun()

# Back-compat alias (OK to keep even if router imports render-as-run_wizard)
def run_wizard():
    render()

if __name__ == "__main__":
    render()
