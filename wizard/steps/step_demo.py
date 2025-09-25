# wizard/steps/step_demo.py — Step 3: Demographics (optional)
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_demographics_metadata() -> pd.DataFrame:
    df = pd.read_excel("metadata/Demographics.xlsx")
    df.columns = [c.strip() for c in df.columns]
    return df

def _find_demcode_col(demo_df: pd.DataFrame) -> str | None:
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            return c
    return None

def _four_digit(s: str) -> str:
    s = "".join(ch for ch in str(s) if ch is not None and ch.isdigit())
    return s.zfill(4) if s else ""

def resolve_demographic_codes_from_metadata(demo_df, category_label, subgroup_label):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    code_col = _find_demcode_col(demo_df)

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                raw_code = str(row.iloc[0][code_col])
                code4 = _four_digit(raw_code)
                code_final = code4 if code4 else raw_code
                return [code_final.strip()], {code_final.strip(): subgroup_label}, True
        return [str(subgroup_label).strip()], {str(subgroup_label).strip(): subgroup_label}, True

    if code_col and LABEL_COL in df_cat.columns:
        pairs = []
        for _, r in df_cat.iterrows():
            raw_code = str(r[code_col]); label = str(r[LABEL_COL])
            code4 = _four_digit(raw_code)
            if code4:
                pairs.append((code4.strip(), label))
        if pairs:
            demcodes = [c for c, _ in pairs]
            disp_map = {c: l for c, l in pairs}
            return demcodes, disp_map, True

    if LABEL_COL in df_cat.columns:
        labels = [str(l).strip() for l in df_cat[LABEL_COL].tolist()]
        return labels, {l: l for l in labels}, True

    return [None], {None: "All respondents"}, False

def render(wizard_state: Dict[str, Any]) -> Dict[str, Any]:
    demo_df = load_demographics_metadata()
    DEMO_CAT_COL = "DEMCODE Category"; LABEL_COL = "DESCRIP_E"

    st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
    categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
    cur_cat = wizard_state.get("demo_category") or "All respondents"
    cat = st.selectbox("Demographic category", categories, index=(categories.index(cur_cat) if cur_cat in categories else 0), label_visibility="collapsed", key="wiz_demo_cat")

    sub = None
    if cat != "All respondents":
        st.markdown(f'<div class="field-label">Subgroup ({cat}) (optional):</div>', unsafe_allow_html=True)
        sub_items = sorted(demo_df.loc[demo_df[DEMO_CAT_COL] == cat, LABEL_COL].dropna().astype(str).unique().tolist())
        prev_sub = wizard_state.get("demo_subgroup") or ""
        sub = st.selectbox("(leave blank to include all subgroups)", [""] + sub_items,
                           index=([""]+sub_items).index(prev_sub) if prev_sub in ([""]+sub_items) else 0,
                           label_visibility="collapsed", key=f"wiz_demo_sub_{cat.replace(' ','_')}")
        if sub == "": sub = None

    demcodes, disp_map, in_play = resolve_demographic_codes_from_metadata(demo_df, cat, sub)
    if in_play and (None not in demcodes):
        demcodes = [None] + demcodes  # include All respondents first

    st.session_state.wizard.update({
        "demo_category": cat,
        "demo_subgroup": sub,
        "demcodes": demcodes,
        "dem_disp_map": ({None: "All respondents"} | {str(k): v for k, v in (disp_map or {}).items()}),
        "category_in_play": bool(in_play),
    })

    # Demographic is optional → always valid
    return {"data": st.session_state.wizard, "is_valid": True}
