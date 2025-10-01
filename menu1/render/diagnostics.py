# menu1/render/diagnostics.py
# Diagnostics panels for Menu 1.
# Change approved by user: make Parameters panel show CODE-FIRST values
# (question codes, years, and DEMCODEs), with labels only as optional hints.

from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import datetime
import json
import os

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Safe imports from local modules (avoid hard failures if a helper is missing)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ..constants import DEFAULT_YEARS
except Exception:
    DEFAULT_YEARS = [2019, 2020, 2022, 2024]

try:
    from ..state import (
        K_SELECTED_CODES,
        K_MULTI_QUESTIONS,
        K_DEMO_MAIN,
        K_SELECT_ALL_YEARS,
        SUBGROUP_PREFIX,
    )
except Exception:
    # Fallback key names if state module interface changes
    K_SELECTED_CODES = "selected_question_codes"
    K_MULTI_QUESTIONS = "multi_questions"
    K_DEMO_MAIN = "demo_main"
    K_SELECT_ALL_YEARS = "select_all_years"
    SUBGROUP_PREFIX = "sub_"

# Loader diagnostics (optional)
try:
    from utils.data_loader import (
        get_backend_info,        # returns dict about engine/memory/dataset
        get_last_query_diag,     # returns dict about last run (engine, elapsed_ms, etc.)
        set_last_query_info,     # store dict about current run (used by "Last query" tab)
    )
except Exception:
    def get_backend_info() -> Dict[str, Any]:
        return {}
    def get_last_query_diag() -> Dict[str, Any]:
        return {}
    def set_last_query_info(payload: Dict[str, Any]) -> None:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _json_box(payload: Dict[str, Any], title: str = "") -> None:
    if title:
        st.markdown(f"#### {title}")
    st.markdown(
        f"<div class='diag-box'><pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre></div>",
        unsafe_allow_html=True,
    )

def _detect_dem_code_column(demo_df: pd.DataFrame) -> Optional[str]:
    candidates = ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]
    for c in candidates:
        if c in demo_df.columns:
            return c
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 1) PARAMETERS PANEL (CODE-FIRST)
# ─────────────────────────────────────────────────────────────────────────────

def parameters_preview(qdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """
    Show the exact parameters that will be sent to the data layer.
    - Questions: codes (not labels)
    - Years: selected list
    - Demographics: resolved DEMCODE list (codes only; None for overall)
    Labels are shown only as small hints for quick cross-checks.
    """
    # 1) Question codes
    sel_q_codes: List[str] = st.session_state.get(K_SELECTED_CODES, []) or []
    # Optional display hint (if question metadata has columns 'code' and 'display')
    code_to_display: Dict[str, str] = {}
    try:
        if isinstance(qdf, pd.DataFrame) and {"code", "display"}.issubset(qdf.columns):
            code_to_display = dict(zip(qdf["code"].astype(str), qdf["display"].astype(str)))
    except Exception:
        pass
    q_display = [code_to_display.get(str(c), str(c)) for c in sel_q_codes]

    # 2) Years selected (respect "select all" toggle)
    years_selected: List[int] = []
    select_all = bool(st.session_state.get(K_SELECT_ALL_YEARS, True))
    for y in DEFAULT_YEARS:
        key = f"year_{y}"
        val = True if select_all else bool(st.session_state.get(key, False))
        if val:
            years_selected.append(int(y))

    # 3) Resolve DEMCODEs from Demographics.xlsx (codes only)
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL    = "DESCRIP_E"
    code_col     = _detect_dem_code_column(demo_df) if isinstance(demo_df, pd.DataFrame) else None

    demo_cat = st.session_state.get(K_DEMO_MAIN, "All respondents")
    if demo_cat and demo_cat != "All respondents":
        sub_key  = f"{SUBGROUP_PREFIX}{str(demo_cat).replace(' ', '_')}"
        subgroup = st.session_state.get(sub_key, "") or None
    else:
        subgroup = None

    demcodes: List[Optional[str]] = []
    dem_disp_map: Dict[Optional[str], str] = {}

    if not demo_cat or demo_cat == "All respondents":
        # Overall rows (blank DEMCODE in data)
        demcodes = [None]
        dem_disp_map = {None: "All respondents"}
    else:
        # Focus the selected category
        df_cat = demo_df.copy()
        try:
            if DEMO_CAT_COL in demo_df.columns:
                df_cat = demo_df[demo_df[DEMO_CAT_COL].astype(str) == str(demo_cat)]
        except Exception:
            pass

        # Subgroup chosen → single code (robust, trimmed, case-insensitive)
        if subgroup and code_col and LABEL_COL in df_cat.columns:
            try:
                r = df_cat[
                    df_cat[LABEL_COL].astype(str).str.strip().str.lower()
                    == str(subgroup).strip().lower()
                ]
                if not r.empty:
                    code = str(r.iloc[0][code_col]).strip()
                    demcodes = [code]
                    dem_disp_map = {code: str(subgroup)}
                else:
                    subgroup = None  # fall through to "all in category"
            except Exception:
                subgroup = None

        # Category only (no subgroup) → all codes in the category
        if not demcodes:
            if code_col and LABEL_COL in df_cat.columns and not df_cat.empty:
                try:
                    codes  = df_cat[code_col].astype(str).str.strip().tolist()
                    labels = df_cat[LABEL_COL].astype(str).tolist()
                    keep   = [(c, l) for c, l in zip(codes, labels) if c != ""]
                    demcodes   = [c for c, _ in keep]
                    dem_disp_map = {c: l for c, l in keep}
                except Exception:
                    demcodes = [None]
                    dem_disp_map = {None: "All respondents"}
            else:
                # Defensive fallback: never pass labels to data layer
                demcodes = [None]
                dem_disp_map = {None: "All respondents"}

    # 4) Emit CODE-FIRST diagnostics payload
    preview = {
        "Questions (codes)": sel_q_codes,                 # ← what hits the DB
        "Years (selected)": years_selected,               # ← what hits the DB
        "Demographic (category)": demo_cat or "All respondents",
        "Subgroup (label)": subgroup or ("All in category" if demo_cat != "All respondents" else "All respondents"),
        "Resolved DEMCODE(s)": demcodes,                  # ← what hits the DB (codes; None for overall)
        # Optional hints for quick visual check (not used by DB)
        "Questions (display hint)": q_display,
        "DEMCODE display (hint)": dem_disp_map,
    }

    st.markdown("#### Parameters preview (code-first)")
    _json_box(preview)

# ─────────────────────────────────────────────────────────────────────────────
# 2) BACKEND / APP SETUP PANEL  (unchanged behavior)
# ─────────────────────────────────────────────────────────────────────────────

def backend_info_panel(qdf: pd.DataFrame, sdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    info = {}
    try:
        bi = get_backend_info() or {}
        info.update(bi)
        # Add simple metadata counts (useful to sanity-check loaded metadata)
        try: info["metadata_questions"] = int(len(qdf))
        except Exception: pass
        try: info["metadata_scales"] = int(len(sdf))
        except Exception: pass
        try: info["metadata_demographics"] = int(len(demo_df))
        except Exception: pass
    except Exception:
        pass
    _json_box(info, "App setup status")

# ─────────────────────────────────────────────────────────────────────────────
# 3) AI STATUS PANEL  (kept lightweight; no behavior change)
# ─────────────────────────────────────────────────────────────────────────────

def ai_status_panel() -> None:
    payload = {}
    # If you store AI toggles or last prompt in session, show them here
    for k in ["ai_enabled", "ai_model", "ai_latency_ms", "ai_last_prompt"]:
        if k in st.session_state:
            payload[k] = st.session_state.get(k)
    _json_box(payload, "AI status")

# ─────────────────────────────────────────────────────────────────────────────
# 4) SEARCH STATUS PANEL  (no functional change; tiny guards only)
# ─────────────────────────────────────────────────────────────────────────────

def search_status_panel() -> None:
    diag = {}
    try:
        diag = get_last_query_diag() or {}
    except Exception:
        pass
    _json_box(diag, "Search engine status")

# ─────────────────────────────────────────────────────────────────────────────
# 5) LAST QUERY PANEL  (unchanged behavior; shows what you stored)
# ─────────────────────────────────────────────────────────────────────────────

def last_query_panel() -> None:
    payload = st.session_state.get("menu1_last_query_info", {})
    _json_box(payload, "Last query (app-level)")

def set_last_query(payload: Dict[str, Any]) -> None:
    """Optional helper your app may call after each run to persist a summary."""
    try:
        st.session_state["menu1_last_query_info"] = dict(payload)
    except Exception:
        pass
    try:
        set_last_query_info(payload)
    except Exception:
        # ignore if not available
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Entry point for diagnostics tab rendering (names kept unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def render_diagnostics_tabs(qdf: pd.DataFrame, sdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    tabs = st.tabs(["Parameters", "App setup", "AI status", "Search status", "Last query"])
    with tabs[0]:
        parameters_preview(qdf, demo_df)
    with tabs[1]:
        backend_info_panel(qdf, sdf, demo_df)
    with tabs[2]:
        ai_status_panel()
    with tabs[3]:
        search_status_panel()
    with tabs[4]:
        last_query_panel()
