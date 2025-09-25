# wizard/steps/step_question.py — Step 1: Question (Menu-1-compatible metadata)
from typing import Any, Dict, List
import re
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def _load_questions_metadata_menu1_style() -> pd.DataFrame:
    qdf = pd.read_excel("metadata/Survey Questions.xlsx")
    qdf.columns = [c.strip().lower() for c in qdf.columns]
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["code"] = qdf["code"].astype(str).str.strip()
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"] + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]].reset_index(drop=True)

def _filter_qdf(qdf: pd.DataFrame, query: str, limit: int = 80) -> pd.DataFrame:
    if not query:
        return qdf.head(limit)
    q = query.strip().lower()
    mask = qdf["code"].str.lower().str.contains(re.escape(q), na=False) | qdf["text"].str.lower().str.contains(re.escape(q), na=False)
    hits = qdf[mask].copy()
    hits["rank"] = 100
    hits.loc[hits["code"].str.lower() == q, "rank"] = -2
    hits.loc[hits["code"].str.lower().str.startswith(q), "rank"] = -1
    hits.loc[hits["text"].str.lower().str.contains(re.escape(q), na=False), "rank"] = hits["rank"].clip(upper=10)
    return hits.sort_values(["rank", "qnum", "code"]).drop(columns=["rank"], errors="ignore").head(limit)

def render(wizard_state: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown('<div class="field-label">Select a survey question:</div>', unsafe_allow_html=True)

    if "_meta_questions" in st.session_state and isinstance(st.session_state["_meta_questions"], pd.DataFrame):
        qdf = st.session_state["_meta_questions"]
    else:
        qdf = _load_questions_metadata_menu1_style()
        st.session_state["_meta_questions"] = qdf

    pre_code = ""
    if isinstance(wizard_state.get("question"), dict):
        pre_code = str(wizard_state["question"].get("code") or "")

    q = st.text_input("Search", value=pre_code, placeholder="e.g., Q16 or “feedback”")
    filtered = _filter_qdf(qdf, q, limit=80)
    if filtered.empty:
        st.info("No matches. Try another code or keyword.")
        return {"data": {}, "is_valid": False}

    options: List[str] = filtered["display"].tolist()
    index = 0
    if pre_code:
        try:
            prev_label = filtered.loc[filtered["code"].astype(str) == pre_code, "display"].iloc[0]
            index = options.index(prev_label)
        except Exception:
            index = 0

    choice = st.selectbox("Question", options, index=index, label_visibility="collapsed", key="wiz_q_select")
    if choice:
        row = filtered.loc[filtered["display"] == choice].iloc[0]
        st.session_state.wizard["question"] = {"code": str(row["code"]), "label": str(row["text"])}

    return {"data": {"question": st.session_state.wizard.get("question")}, "is_valid": bool(choice)}
