# app/menu1/render/controls.py
"""
Controls for Menu 1:
- Question picker: dropdown multi-select (authoritative) + keyword search (hybrid)
- Selected list with quick unselect checkboxes
- Years selector
- Demographic category & subgroup selector
- Search button enablement helper

UI tweaks kept:
  • Multiselect placeholder: "Choose a question from the list below"
  • Search box header styled similarly
  • "or" label for visual separation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import re
import pandas as pd
import streamlit as st

# Try to import the hybrid search; provide a robust fallback as well
try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    hybrid_question_search = None  # type: ignore

# -----------------------------
# Session-state keys (Menu 1)
# -----------------------------
K_MULTI_QUESTIONS = "menu1_multi_questions"   # List[str] of "display" labels picked in the dropdown
K_SELECTED_CODES  = "menu1_selected_codes"    # Ordered List[str] of codes (multi + hits merged)
K_KW_QUERY        = "menu1_kw_query"          # str
K_HITS            = "menu1_hits"              # List[{"code","text","display","score"}]
K_FIND_HITS_BTN   = "menu1_find_hits"         # button key
K_SEARCH_DONE     = "menu1_search_done"       # bool: did user click Search the questionnaire?
K_LAST_QUERY      = "menu1_last_search_query" # str: what query was used last time
K_HITS_SELECTED   = "menu1_hit_codes_selected"# List[str]: codes ticked from hits (persistent)

# Years
DEFAULT_YEARS = [2024, 2022, 2020, 2019]
K_SELECT_ALL_YEARS = "select_all_years"

# Threshold (fixed)
MIN_SCORE = 0.40  # keep > 0.40 filter in both engines

# -----------------------------------------------------------------------------
# Internal regex/token helpers (fallbacks)
# -----------------------------------------------------------------------------
_word_re = re.compile(r"[a-z0-9']+")

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _tokens(s: str) -> List[str]:
    return _word_re.findall(_normalize(s))

# -----------------------------------------------------------------------------
# Utilities for dedupe and wrapped call
# -----------------------------------------------------------------------------
def _dedupe_hits(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["code", "text", "display", "score"])
    out = df.copy()
    if "score" not in out.columns:
        out["score"] = 0.0
    out["code"] = out["code"].astype(str)
    out = out.sort_values("score", ascending=False)
    out = out.drop_duplicates(subset=["code"], keep="first")
    try:
        out = out[out["score"] > MIN_SCORE]
    except Exception:
        pass
    return out

def _run_keyword_search(qdf: pd.DataFrame, query: str, top_k: int = 120) -> pd.DataFrame:
    """
    Calls utils.hybrid_search.hybrid_question_search with fixed threshold.
    """
    if callable(hybrid_question_search):
        try:
            hits_df = hybrid_question_search(qdf, query, top_k=top_k, min_score=MIN_SCORE)  # type: ignore
            if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
                return _dedupe_hits(hits_df).head(top_k)
        except Exception:
            pass
    return pd.DataFrame(columns=["code", "text", "display", "score"])

# -----------------------------------------------------------------------------
# Question picker (dropdown + keyword search) → returns List[str] (codes)
# -----------------------------------------------------------------------------
def question_picker(qdf: pd.DataFrame) -> List[str]:
    """
    UI:
      1) Dropdown multi-select (authoritative, max 5) with custom placeholder
      2) Keyword search section with persistent results + multi-tick checkboxes
      3) "Selected questions" list with quick unselect checkboxes

    Returns ordered list of selected question codes (max 5).
    """
    # Ensure session defaults (initialize BEFORE widget, never assign after render)
    st.session_state.setdefault(K_MULTI_QUESTIONS, [])
    st.session_state.setdefault(K_SELECTED_CODES, [])
    st.session_state.setdefault(K_KW_QUERY, "")
    st.session_state.setdefault(K_HITS, [])
    st.session_state.setdefault(K_SEARCH_DONE, False)
    st.session_state.setdefault(K_LAST_QUERY, "")
    st.session_state.setdefault(K_HITS_SELECTED, [])  # persist checked hits

    # Mappings
    code_to_text = dict(zip(qdf["code"], qdf["text"]))
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    # ---------- 1) Dropdown multi-select ----------
    st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)
    all_displays = qdf["display"].tolist()
    st.multiselect(
        "Choose one or more from the official list",
        all_displays,
        max_selections=5,
        label_visibility="collapsed",
        key=K_MULTI_QUESTIONS,
        placeholder="Choose a question from the list below",
    )
    selected_from_multi: Set[str] = set(display_to_code[d] for d in st.session_state[K_MULTI_QUESTIONS] if d in display_to_code)

    # ---------- Divider: "or" (tighter spacing) ----------
    st.markdown("""
        <div style="
            margin: .2rem 0 .2rem .5rem;
            font-size: 0.9rem; font-weight: 600;
            color: rgba(49,51,63,.8); font-family: inherit;
        ">or</div>
    """, unsafe_allow_html=True)

    # ---------- 2) Keyword search ----------
    # Header
    st.markdown("<div class='field-label'>Search questionnaire by keywords or theme</div>", unsafe_allow_html=True)

    # Search input + button (button always visible; disabled when empty)
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input(
            "Enter keywords (e.g., harassment, recognition, onboarding)",
            key=K_KW_QUERY,
            label_visibility="collapsed",
            placeholder="Type keywords like “career advancement”, “harassment”, “recognition”…",
        )
    with c2:
        btn_disabled = (not (query or "").strip())
        if st.button("Search the questionnaire", key=K_FIND_HITS_BTN, disabled=btn_disabled):
            hits_df = _run_keyword_search(qdf, query, top_k=120)
            st.session_state[K_SEARCH_DONE] = True
            st.session_state[K_LAST_QUERY] = query
            st.session_state[K_HITS] = hits_df[["code", "text", "display", "score"]].to_dict(orient="records") \
                                       if isinstance(hits_df, pd.DataFrame) and not hits_df.empty else []
            # Keep only ticks that still exist in the new results
            current_codes = {h["code"] for h in st.session_state[K_HITS]}
            st.session_state[K_HITS_SELECTED] = [c for c in st.session_state[K_HITS_SELECTED] if c in current_codes]

    # Results area (now hides warnings/results when input box is empty — fixes lingering message after reset)
    hits = st.session_state.get(K_HITS, [])
    current_query = (st.session_state.get(K_KW_QUERY) or "").strip()
    last_query = (st.session_state.get(K_LAST_QUERY) or "").strip()

    if not current_query:
        # After a reset the input is empty; do not show prior warnings/results
        st.info('Enter keywords and click "Search the questionnaire" to see matches.')
        show_hits_list = False
    else:
        # Show results/warning only if the displayed input matches the last executed query
        if st.session_state.get(K_SEARCH_DONE, False) and current_query == last_query:
            if not hits:
                st.warning(
                    f"No questions matched “{current_query}”. "
                    "Try broader or different keywords (e.g., synonyms), split phrases (e.g., “career advancement” → “career”), "
                    "or search by a question code like “Q01”."
                )
                show_hits_list = False
            else:
                st.write(f"Results for “{last_query}”: Top {len(hits)} matches meeting the quality threshold:")
                show_hits_list = True
        else:
            # User has typed/edited but not searched yet
            st.caption("Type your keywords, then click **Search the questionnaire**.")
            show_hits_list = False

    # Multi-tick checkboxes for hits (do NOT clear on additional selections)
    selected_from_hits: Set[str] = set()
    if show_hits_list:
        for rec in hits:
            code = rec["code"]; text = rec["text"]
            label = f"{code} — {text}"
            key = f"kwhit_{code}"
            default_checked = (code in st.session_state[K_HITS_SELECTED]) or (code in selected_from_multi)
            checked = st.checkbox(label, value=default_checked, key=key)
            # Maintain persistent selection list explicitly
            if checked and code not in st.session_state[K_HITS_SELECTED]:
                st.session_state[K_HITS_SELECTED].append(code)
            if (not checked) and code in st.session_state[K_HITS_SELECTED]:
                st.session_state[K_HITS_SELECTED] = [c for c in st.session_state[K_HITS_SELECTED] if c != code]

        selected_from_hits = set(st.session_state[K_HITS_SELECTED])

    # ---------- Merge selections (dropdown first, then hits), cap at 5 ----------
    combined_order: List[str] = []
    for d in st.session_state.get(K_MULTI_QUESTIONS, []):
        c = display_to_code.get(d)
        if c and c not in combined_order:
            combined_order.append(c)
    for c in selected_from_hits:
        if c not in combined_order:
            combined_order.append(c)
   
