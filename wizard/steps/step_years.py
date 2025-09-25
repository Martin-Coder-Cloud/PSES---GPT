# wizard/steps/step_years.py — Step 2: Years
from typing import Any, Dict, List
import streamlit as st

def render(wizard_state: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
    all_years = ["2024", "2022", "2020", "2019"]

    # default: respect existing picks or "All years"
    pick_existing = wizard_state.get("years") or []
    select_all_default = (len(pick_existing) == 0)
    select_all = st.checkbox("All years", value=select_all_default, key="wiz_years_all")

    picked: List[str] = []
    cols = st.columns(len(all_years))
    for i, yr in enumerate(all_years):
        with cols[i]:
            default_checked = (yr in pick_existing) or select_all
            if st.checkbox(yr, value=default_checked, key=f"wiz_year_{yr}"):
                picked.append(yr)
    picked = sorted(set(picked))
    st.session_state.wizard["years"] = picked

    ok = bool(picked)
    if not ok:
        st.info("Select at least one year to continue.")

    return {"data": {"years": picked}, "is_valid": ok}
