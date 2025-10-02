# menu1/render/diagnostics.py
# (original file + approved changes)
#  • Parameters panel is code-first + shows Resolved DEMCODE(s)
#  • Raw data (debug) tab shows unsuppressed, unpivoted rows
#  • NEW section: DEMCODE scanner (PS-wide) with optional "Force CSV scan (PS-wide)"

from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import datetime
import json
import os
import importlib.util
from collections import Counter

import pandas as pd
import streamlit as st

from ..constants import DEFAULT_YEARS
from ..state import (
    K_SELECTED_CODES, K_MULTI_QUESTIONS, K_DEMO_MAIN, K_SELECT_ALL_YEARS,
    YEAR_KEYS, SUBGROUP_PREFIX, get_last_query_info, set_last_query_info,
)

# Optional backend info / preload (best-effort)
try:
    from utils.data_loader import get_backend_info  # type: ignore
except Exception:
    def get_backend_info() -> dict:  # type: ignore
        return {"engine": "csv.gz", "in_memory": False}

try:
    from utils.data_loader import preload_pswide_dataframe  # type: ignore
except Exception:
    def preload_pswide_dataframe():  # type: ignore
        return None

# Optional CSV path resolver for forced scan
try:
    from utils.data_loader import ensure_results2024_local  # type: ignore
except Exception:
    ensure_results2024_local = None  # type: ignore

# Optional query wrappers
_FETCH_FROM_QUERIES = True
try:
    # Preferred: use your existing wrapper that normalizes columns
    from ..queries import fetch_per_question  # type: ignore
except Exception:
    _FETCH_FROM_QUERIES = False
    # Fallback: call the loader directly if wrapper isn't available
    try:
        from utils.data_loader import load_results2024_filtered  # type: ignore
    except Exception:
        load_results2024_filtered = None  # type: ignore


def _json_box(obj: Dict[str, Any], title: str) -> None:
    st.markdown(f"#### {title}")
    st.markdown(
        f"<div class='diag-box'><pre>{json.dumps(obj, ensure_ascii=False, indent=2)}</pre></div>",
        unsafe_allow_html=True
    )

# --------------------------------------------------------------------------------------
# PARAMETERS — show codes (not labels) + Resolved DEMCODE(s)
# --------------------------------------------------------------------------------------
def parameters_preview(qdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """
    CHANGE (approved earlier): show codes, not labels; and include resolved DEMCODE(s).
    Everything else preserved.
    """
    code_to_display = dict(zip(qdf["code"], qdf["display"]))  # kept for cross-check, not displayed

    # Question CODES (what actually hits the DB)
    sel_codes: List[str] = st.session_state.get(K_SELECTED_CODES, [])  # ← codes

    # Years (unchanged logic)
    years_selected: List[int] = []
    select_all = bool(st.session_state.get(K_SELECT_ALL_YEARS, True))
    for y in DEFAULT_YEARS:
        key = f"year_{y}"
        val = True if select_all else bool(st.session_state.get(key, False))
        if val:
            years_selected.append(y)

    # Demographic category + subgroup (unchanged state reads)
    demo_cat = st.session_state.get(K_DEMO_MAIN, "All respondents")
    if demo_cat and demo_cat != "All respondents":
        sub_key = f"{SUBGROUP_PREFIX}{str(demo_cat).replace(' ', '_')}"
        subgroup = st.session_state.get(sub_key, "") or "All in category"
    else:
        subgroup = "All respondents"

    # Resolve DEMCODE(s) from Demographics metadata (codes only)
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL    = "DESCRIP_E"
    code_col     = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    def _resolve_demcodes() -> list:
        if (not demo_cat) or (demo_cat == "All respondents"):
            return [None]
        df_cat = demo_df
        if DEMO_CAT_COL in demo_df.columns:
            df_cat = demo_df[demo_df[DEMO_CAT_COL].astype(str) == str(demo_cat)]
        if subgroup not in (None, "", "All in category", "All respondents") and code_col and LABEL_COL in df_cat.columns:
            r = df_cat[df_cat[LABEL_COL].astype(str).str.strip().str.lower() == str(subgroup).strip().lower()]
            if not r.empty:
                return [str(r.iloc[0][code_col]).strip()]
        if code_col and LABEL_COL in df_cat.columns and not df_cat.empty:
            codes = df_cat[code_col].astype(str).str.strip().tolist()
            return [c for c in codes if c != ""]
        return [None]

    demcodes = _resolve_demcodes()

    preview = {
        "Selected questions": sel_codes,            # codes (DB-facing)
        "Years (selected)": years_selected,
        "Demographic category": demo_cat or "All respondents",
        "Subgroup": subgroup,
        "Resolved DEMCODE(s)": demcodes,            # codes (DB-facing; None for overall)
    }
    _json_box(preview, "Parameters preview")

# --------------------------------------------------------------------------------------
# RAW DATA (DEBUG) — unsuppressed, unpivoted rows + DEMCODE scanner
# --------------------------------------------------------------------------------------
def raw_data_debug_tab(qdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """
    Shows raw rows BEFORE any suppression, shaping, or tabulation, and
    adds a DEMCODE scanner (PS-wide) to list actual codes present for the current slice.
    """
    # 1) Read current selection
    sel_codes: List[str] = st.session_state.get(K_SELECTED_CODES, []) or []
    years_selected: List[int] = []
    select_all = bool(st.session_state.get(K_SELECT_ALL_YEARS, True))
    for y in DEFAULT_YEARS:
        key = f"year_{y}"
        val = True if select_all else bool(st.session_state.get(key, False))
        if val:
            years_selected.append(int(y))

    demo_cat = st.session_state.get(K_DEMO_MAIN, "All respondents")
    if demo_cat and demo_cat != "All respondents":
        sub_key = f"{SUBGROUP_PREFIX}{str(demo_cat).replace(' ', '_')}"
        subgroup = st.session_state.get(sub_key, "") or None
    else:
        subgroup = None

    # Resolve DEMCODE(s) — codes only
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL    = "DESCRIP_E"
    code_col     = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    def _resolve_demcodes_codes_only() -> List[Optional[str]]:
        if (not demo_cat) or (demo_cat == "All respondents"):
            return [None]
        df_cat = demo_df
        if DEMO_CAT_COL in demo_df.columns:
            df_cat = demo_df[demo_df[DEMO_CAT_COL].astype(str) == str(demo_cat)]
        if subgroup and code_col and LABEL_COL in df_cat.columns:
            r = df_cat[df_cat[LABEL_COL].astype(str).str.strip().str.lower() == str(subgroup).strip().lower()]
            if not r.empty:
                return [str(r.iloc[0][code_col]).strip()]
        if code_col and LABEL_COL in df_cat.columns and not df_cat.empty:
            codes = df_cat[code_col].astype(str).str.strip().tolist()
            return [c for c in codes if c != ""]
        return [None]

    demcodes: List[Optional[str]] = _resolve_demcodes_codes_only()

    # --- Raw rows (unsuppressed, unpivoted) ----------------------------------
    st.markdown("#### Raw data (debug)")
    if not sel_codes or not years_selected:
        st.info("Select at least one question and one year to view raw data.")
    else:
        parts: List[pd.DataFrame] = []
        try:
            if _FETCH_FROM_QUERIES:
                for q in sel_codes:
                    try:
                        df_q = fetch_per_question(q, years_selected, demcodes)  # type: ignore
                        if df_q is not None and not df_q.empty:
                            parts.append(df_q)
                    except Exception:
                        continue
            else:
                if load_results2024_filtered is None:
                    st.error("Raw data path unavailable: neither queries.fetch_per_question nor loader function is importable.")
                else:
                    for q in sel_codes:
                        for code in demcodes:
                            try:
                                df_part = load_results2024_filtered(  # type: ignore
                                    question_code=q,
                                    years=list(years_selected),
                                    group_value=(None if code in (None, "", "All") else str(code)),
                                )
                                if df_part is not None and not df_part.empty:
                                    parts.append(df_part)
                            except Exception:
                                continue
        except Exception as e:
            st.error(f"Error while fetching raw data: {e}")

        if parts:
            df_raw = pd.concat(parts, ignore_index=True)
            st.caption(f"Rows: {len(df_raw):,}  |  Columns: {list(df_raw.columns)}")
            sort_cols = [c for c in ["question_code", "year", "group_value"] if c in df_raw.columns]
            if sort_cols:
                try:
                    df_raw = df_raw.sort_values(sort_cols)
                except Exception:
                    pass
            st.dataframe(df_raw, use_container_width=True, hide_index=True)
        else:
            st.warning("No raw rows were returned for this selection.")
            st.caption(f"(Questions={sel_codes}; Years={years_selected}; DEMCODEs={demcodes})")

    # --- DEMCODE scanner (PS-wide) -------------------------------------------
    st.markdown("#### DEMCODE scanner (PS-wide)")
    st.caption(
        "Lists actual DEMCODE values present for the selected question(s) and year(s) "
        "at PS-wide (LEVEL1ID==0). Ignores the demographic filter. "
        "Use this to verify presence independent of the render pipeline
