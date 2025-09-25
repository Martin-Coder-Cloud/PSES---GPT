# wizard/steps/step_question.py
# Step 1 — Select a Question (reuses Menu 1 logic for question metadata)

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import re

import pandas as pd
import streamlit as st


@dataclass
class StepResult:
    data: Dict[str, Any] = None
    is_valid: bool = False
    next_step: Optional[str] = None
    message: Optional[str] = None
    go_results: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Load questions metadata — same logic as Menu 1 (path + normalization + sort)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_questions_metadata_menu1_style() -> pd.DataFrame:
    """
    Replicates Menu 1's load_questions_metadata:

    - Reads: metadata/Survey Questions.xlsx
    - Normalizes columns → code, text
    - Adds qnum (numeric) for sorting
    - Builds 'display' = "CODE – TEXT"
    - Returns: ["code", "text", "display"] sorted by (qnum, code)
    """
    qdf = pd.read_excel("metadata/Survey Questions.xlsx")
    qdf.columns = [c.strip().lower() for c in qdf.columns]
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    # Fallbacks if column names differ slightly
    if "code" not in qdf.columns:
        # try to guess a code-like column
        for c in qdf.columns:
            if qdf[c].astype(str).str.match(r"^\s*[Qq]?\d+[A-Za-z0-9_]*\s*$").mean() > 0.5:
                qdf = qdf.rename(columns={c: "code"})
                break
    if "text" not in qdf.columns:
        # pick the "longest-text" column as question text
        best_c, best_len = None, -1.0
        for c in qdf.columns:
            if qdf[c].dtype == object:
                avg = qdf[c].astype(str).str.len().mean()
                if avg > best_len:
                    best_len = avg
                    best_c = c
        if best_c:
            qdf = qdf.rename(columns={best_c: "text"})

    qdf["code"] = qdf["code"].astype(str).str.strip()
    # Extract numeric part for ordering (like Menu 1)
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"] + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight search helper (adds search on top of Menu 1 drop-down)
# ──────────────────────────────────────────────────────────────────────────────
def _filter_qdf(qdf: pd.DataFrame, query: str, limit: int = 80) -> pd.DataFrame:
    if not query:
        return qdf.head(limit)
    q = query.strip().lower()
    mask = (
        qdf["code"].str.lower().str.contains(re.escape(q), na=False)
        | qdf["text"].str.lower().str.contains(re.escape(q), na=False)
    )
    hits = qdf[mask].copy()
    # Light ranking: exact code match → startswith → text contains
    hits["rank"] = 100
    hits.loc[hits["code"].str.lower() == q, "rank"] = -2
    hits.loc[hits["code"].str.lower().str.startswith(q), "rank"] = -1
    hits.loc[hits["text"].str.lower().str.contains(re.escape(q), na=False), "rank"] = hits["rank"].clip(upper=10)
    return hits.sort_values(["rank", "qnum", "code"]).drop(columns=["rank"], errors="ignore").head(limit)


# ──────────────────────────────────────────────────────────────────────────────
# Public step renderer (contract: render(wizard_state) -> StepResult|dict)
# ──────────────────────────────────────────────────────────────────────────────
def render(wizard_state: Dict[str, Any]) -> Dict[str, Any]:
    st.subheader("Step 1 — Select a Question")
    st.caption("This list matches Menu 1. Tip: search by code (e.g., Q16) or keywords.")

    # Load once (cached); also store in session for other steps if needed
    if "_meta_questions" in st.session_state and isinstance(st.session_state["_meta_questions"], pd.DataFrame):
        qdf = st.session_state["_meta_questions"]
    else:
        try:
            qdf = _load_questions_metadata_menu1_style()
        except Exception as e:
            st.error(f"Could not read metadata/Survey Questions.xlsx ({type(e).__name__}: {e})")
            # Tiny fallback to keep the wizard usable
            qdf = pd.DataFrame(
                [
                    {"code": "Q01", "text": "I like my job.", "display": "Q01 – I like my job."},
                    {"code": "Q02", "text": "I am proud of the work I do.", "display": "Q02 – I am proud of the work I do."},
                    {"code": "Q16", "text": "I receive useful feedback on my work.", "display": "Q16 – I receive useful feedback on my work."},
                ]
            )
        st.session_state["_meta_questions"] = qdf

    # Restore previously selected question (if any)
    pre_code = None
    if isinstance(wizard_state.get("question"), dict):
        pre_code = str(wizard_state["question"].get("code") or "").strip()

    # Search + filtered results
    q = st.text_input("Search", value=pre_code or "", placeholder="e.g., Q16 or “feedback”")
    filtered = _filter_qdf(qdf, q, limit=80)
    if filtered.empty:
        st.info("No matches. Try another code or keyword.")
        return {"data": {}, "is_valid": False, "next_step": None}

    # Build selectbox options
    options: List[str] = filtered["display"].tolist()

    # Determine default index (keep prior selection if visible)
    index = 0
    if pre_code:
        try:
            prev_label = filtered.loc[filtered["code"].astype(str) == pre_code, "display"].iloc[0]
            index = options.index(prev_label)
        except Exception:
            index = 0

    # UI control
    choice = st.selectbox("Question", options, index=index, key="wiz_q_select", label_visibility="collapsed")

    # Footer actions
    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("Clear selection", on_click=lambda: _clear_selected_question())
    with c2:
        can_continue = bool(choice)
        go_next = st.button("Continue → Years", disabled=not can_continue)

    # Persist selection
    if choice:
        row = filtered.loc[filtered["display"] == choice].iloc[0]
        st.session_state.wizard["question"] = {"code": str(row["code"]), "label": str(row["text"])}

    if choice and go_next:
        return {"data": {"question": st.session_state.wizard["question"]}, "is_valid": True, "next_step": "years"}

    # Stay on this step
    return {"data": {"question": st.session_state.wizard.get("question")}, "is_valid": bool(choice)}


def _clear_selected_question():
    st.session_state.wizard["question"] = None
