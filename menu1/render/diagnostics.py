# menu1/render/diagnostics.py
"""
Diagnostics panels for Menu 1:
- Parameters preview snapshot
- App setup status (engine, cached data, metadata counts)
- AI status (key/model presence; simple checks)
- Last query timings
- Tabbed wrapper to show all of the above neatly
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime
import json
import os

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


def _json_box(obj: Dict[str, Any], title: str) -> None:
    st.markdown(f"#### {title}")
    st.markdown(f"<div class='diag-box'><pre>{json.dumps(obj, ensure_ascii=False, indent=2)}</pre></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Parameters preview
# --------------------------------------------------------------------------------------
def parameters_preview(qdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """Show a compact snapshot of currently selected inputs."""
    code_to_display = dict(zip(qdf["code"], qdf["display"]))

    # Selected questions (use display where available)
    sel_codes = st.session_state.get(K_SELECTED_CODES, [])
    selected_questions = [code_to_display.get(c, c) for c in sel_codes]

    # Years: respect 'All years' master toggle
    years_selected = []
    select_all = bool(st.session_state.get(K_SELECT_ALL_YEARS, True))
    for y in DEFAULT_YEARS:
        key = f"year_{y}"
        val = True if select_all else bool(st.session_state.get(key, False))
        if val:
            years_selected.append(y)

    # Demographics + optional subgroup
    demo_cat = st.session_state.get(K_DEMO_MAIN, "All respondents")
    if demo_cat and demo_cat != "All respondents":
        sub_key = f"{SUBGROUP_PREFIX}{str(demo_cat).replace(' ', '_')}"
        subgroup = st.session_state.get(sub_key, "") or "All in category"
    else:
        subgroup = "All respondents"

    preview = {
        "Selected questions": selected_questions,
        "Years (selected)": years_selected,
        "Demographic category": demo_cat or "All respondents",
        "Subgroup": subgroup,
    }
    _json_box(preview, "Parameters preview")


# --------------------------------------------------------------------------------------
# Backend info panel
# --------------------------------------------------------------------------------------
def backend_info_panel(qdf: pd.DataFrame, sdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """Show loader/engine info and quick dataset stats (best-effort, no hard deps)."""
    info: Dict[str, Any] = {}
    try:
        info = get_backend_info() or {}
    except Exception:
        info = {"engine": "csv.gz"}

    # In-memory PS-wide dataframe (optional)
    try:
        df_ps = preload_pswide_dataframe()
        if isinstance(df_ps, pd.DataFrame) and not df_ps.empty:
            ycol = "year" if "year" in df_ps.columns else ("SURVEYR" if "SURVEYR" in df_ps.columns else None)
            qcol = "question_code" if "question_code" in df_ps.columns else ("QUESTION" if "QUESTION" in df_ps.columns else None)
            yr_min = int(pd.to_numeric(df_ps[ycol], errors="coerce").min()) if ycol else None
            yr_max = int(pd.to_numeric(df_ps[ycol], errors="coerce").max()) if ycol else None
            info.update({
                "in_memory": True,
                "pswide_rows": int(len(df_ps)),
                "unique_questions": int(df_ps[qcol].astype(str).nunique()) if qcol else None,
                "year_range": f"{yr_min}–{yr_max}" if yr_min and yr_max else None,
            })
        else:
            info.update({"in_memory": False})
    except Exception:
        pass

    # Metadata counts
    try: info["metadata_questions"] = int(len(qdf))
    except Exception: pass
    try: info["metadata_scales"]   = int(len(sdf))
    except Exception: pass
    try: info["metadata_demographics"] = int(len(demo_df))
    except Exception: pass

    _json_box(info, "App setup status")


# --------------------------------------------------------------------------------------
# AI status panel
# --------------------------------------------------------------------------------------
def ai_status_panel() -> None:
    """
    Lightweight health check for AI configuration (no API calls).
    Shows presence of key/model and basic environment hints.
    """
    key = (st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", ""))
    model = (st.secrets.get("OPENAI_MODEL", "") or os.environ.get("OPENAI_MODEL", ""))
    status = {
        "api_key_present": bool(key),
        "model_name": model or "(default/unspecified)",
        "how_to_set_key": "Set OPENAI_API_KEY in Streamlit secrets or environment.",
        "how_to_set_model": "Optionally set OPENAI_MODEL to override the default model.",
    }
    # Present clearly but compactly
    _json_box(status, "AI status")


# --------------------------------------------------------------------------------------
# Last query panel + helper
# --------------------------------------------------------------------------------------
def last_query_panel() -> None:
    last = get_last_query_info() or {"status": "No query yet"}
    _json_box(last, "Last query")

def mark_last_query(
    *,
    started_ts: Optional[float] = None,
    finished_ts: Optional[float] = None,
    engine: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience helper to record the latest query timing.
    - Pass started_ts and finished_ts (epoch seconds) to compute elapsed.
    - 'engine' is optional (falls back to get_backend_info).
    - 'extra' can include any additional fields you want to display.
    """
    try:
        eng = engine or (get_backend_info() or {}).get("engine", "unknown")
    except Exception:
        eng = engine or "unknown"

    started = datetime.fromtimestamp(started_ts).strftime("%Y-%m-%d %H:%M:%S") if started_ts else None
    finished = datetime.fromtimestamp(finished_ts).strftime("%Y-%m-%d %H:%M:%S") if finished_ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = round((finished_ts - started_ts), 2) if (started_ts and finished_ts) else None

    payload = {
        "started": started or "(unknown)",
        "finished": finished,
        "elapsed_seconds": elapsed,
        "engine": eng,
    }
    if extra:
        payload.update(extra)

    set_last_query_info(payload)


# --------------------------------------------------------------------------------------
# Tabs wrapper
# --------------------------------------------------------------------------------------
def render_diagnostics_tabs(qdf: pd.DataFrame, sdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """
    Render diagnostics inside tabs:
      • Parameters • App setup • AI status • Last query
    """
    tabs = st.tabs(["Parameters", "App setup", "AI status", "Last query"])
    with tabs[0]:
        parameters_preview(qdf, demo_df)
    with tabs[1]:
        backend_info_panel(qdf, sdf, demo_df)
    with tabs[2]:
        ai_status_panel()
    with tabs[3]:
        last_query_panel()
