# menu1/render/diagnostics.py
# (original file restored; only two approved changes)
#  • Parameters panel is code-first + shows Resolved DEMCODE(s)
#  • NEW diagnostics tab: "Raw data (debug)" to show unsuppressed, unpivoted rows

from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import datetime
import json
import os
import importlib.util

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
# Approved change #1 — PARAMETERS: show codes (not labels) + Resolved DEMCODE(s)
# --------------------------------------------------------------------------------------
def parameters_preview(qdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """
    CHANGE (approved): show codes, not labels; and include resolved DEMCODE(s).
    Everything else preserved.
    """
    # We keep this mapping in case you want to cross-check display, but we output codes
    code_to_display = dict(zip(qdf["code"], qdf["display"]))

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
        # Overall → None indicates PS-wide rows in the loader
        if (not demo_cat) or (demo_cat == "All respondents"):
            return [None]
        df_cat = demo_df
        if DEMO_CAT_COL in demo_df.columns:
            df_cat = demo_df[demo_df[DEMO_CAT_COL].astype(str) == str(demo_cat)]
        # Subgroup chosen → single code
        if subgroup not in (None, "", "All in category", "All respondents") and code_col and LABEL_COL in df_cat.columns:
            r = df_cat[df_cat[LABEL_COL].astype(str).str.strip().str.lower() == str(subgroup).strip().lower()]
            if not r.empty:
                return [str(r.iloc[0][code_col]).strip()]
        # Category only → all codes in category
        if code_col and LABEL_COL in df_cat.columns and not df_cat.empty:
            codes = df_cat[code_col].astype(str).str.strip().tolist()
            return [c for c in codes if c != ""]
        # Fallback: overall
        return [None]

    demcodes = _resolve_demcodes()

    # Code-first preview payload
    preview = {
        "Selected questions": sel_codes,            # <-- codes (previously labels)
        "Years (selected)": years_selected,
        "Demographic category": demo_cat or "All respondents",
        "Subgroup": subgroup,
        "Resolved DEMCODE(s)": demcodes,            # <-- added
    }
    _json_box(preview, "Parameters preview")

# --------------------------------------------------------------------------------------
# NEW approved tab — Raw data (debug): unsuppressed, unpivoted rows
# --------------------------------------------------------------------------------------
def raw_data_debug_tab(qdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    """
    Re-executes the current selection and shows the raw rows BEFORE any suppression,
    shaping, or tabulation. Purely for diagnostics/validation.
    """
    # 1) Read current selection from state (same as Parameters panel)
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

    # 2) Resolve DEMCODE(s) from Demographics.xlsx (codes only)
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
            # if subgroup not found → fall through to all-in-category
        if code_col and LABEL_COL in df_cat.columns and not df_cat.empty:
            codes = df_cat[code_col].astype(str).str.strip().tolist()
            return [c for c in codes if c != ""]
        return [None]

    demcodes: List[Optional[str]] = _resolve_demcodes_codes_only()

    # 3) Fetch raw slices
    parts: List[pd.DataFrame] = []
    if not sel_codes or not years_selected:
        st.info("Select at least one question and one year to view raw data.")
        return

    try:
        if _FETCH_FROM_QUERIES:
            # Use your wrapper (loops over demcodes internally)
            for q in sel_codes:
                try:
                    df_q = fetch_per_question(q, years_selected, demcodes)  # type: ignore
                    if df_q is not None and not df_q.empty:
                        parts.append(df_q)
                except Exception:
                    continue
        else:
            # Direct loader fallback (no normalization changes)
            if load_results2024_filtered is None:
                st.error("Raw data path unavailable: neither queries.fetch_per_question nor loader function is importable.")
                return
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
        return

    if not parts:
        st.warning("No raw rows were returned for this selection.")
        st.caption(f"(Questions={sel_codes}; Years={years_selected}; DEMCODEs={demcodes})")
        return

    df_raw = pd.concat(parts, ignore_index=True)

    # 4) Present raw, unsuppressed, unpivoted rows
    st.markdown("#### Raw data (debug)")
    st.caption(f"Rows: {len(df_raw):,}  |  Columns: {list(df_raw.columns)}")
    # Sort for readability if normalized columns are present
    sort_cols = [c for c in ["question_code", "year", "group_value"] if c in df_raw.columns]
    if sort_cols:
        try:
            df_raw = df_raw.sort_values(sort_cols)
        except Exception:
            pass
    st.dataframe(df_raw, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# Unchanged panels (as in your original file)
# --------------------------------------------------------------------------------------
def backend_info_panel(qdf: pd.DataFrame, sdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    info: Dict[str, Any] = {}
    try:
        info = get_backend_info() or {}
    except Exception:
        info = {"engine": "csv.gz"}
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
                "year_range": f"{yr_min}â€“{yr_max}" if yr_min and yr_max else None,
            })
        else:
            info.update({"in_memory": False})
    except Exception:
        pass
    try: info["metadata_questions"] = int(len(qdf))
    except Exception: pass
    try: info["metadata_scales"]   = int(len(sdf))
    except Exception: pass
    try: info["metadata_demographics"] = int(len(demo_df))
    except Exception: pass
    _json_box(info, "App setup status")

def ai_status_panel() -> None:
    key = (st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", ""))
    model = (st.secrets.get("OPENAI_MODEL", "") or os.environ.get("OPENAI_MODEL", ""))
    status = {
        "api_key_present": bool(key),
        "model_name": model or "(default/unspecified)",
        "how_to_set_key": "Set OPENAI_API_KEY in Streamlit secrets or environment.",
        "how_to_set_model": "Optionally set OPENAI_MODEL to override the default model.",
    }
    _json_box(status, "AI status (summaries)")

def search_status_panel() -> None:
    """
    Show whether local sentence-transformer embeddings are available & active.
    Falls back to import check if the search module helper isn't present.
    """
    status: Dict[str, Any] = {}
    try:
        from utils.hybrid_search import get_embedding_status  # type: ignore
        status = get_embedding_status() or {}
    except Exception:
        status = {}
    if not status:
        try:
            have_st = importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            have_st = False
        status = {
            "sentence_transformers_installed": have_st,
            "model_loaded": False,
            "model_name": None,
            "catalogues_indexed": 0,
        }
    # Show both possible env var overrides for the model
    status["MENU1_EMBED_MODEL_env"] = os.environ.get("MENU1_EMBED_MODEL", None)
    status["PSES_EMBED_MODEL_env"]  = os.environ.get("PSES_EMBED_MODEL", None)
    _json_box(status, "Search status (semantic embeddings)")

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
    from datetime import datetime
    try:
        eng = engine or (get_backend_info() or {}).get("engine", "unknown")
    except Exception:
        eng = engine or "unknown"
    started = datetime.fromtimestamp(started_ts).strftime("%Y-%m-%d %H:%M:%S") if started_ts else None
    finished = datetime.fromtimestamp(finished_ts).strftime("%Y-%m-%d %H:%M:%S") if finished_ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = round((finished_ts - started_ts), 2) if (started_ts and finished_ts) else None
    payload: Dict[str, Any] = {
        "started": started or "(unknown)",
        "finished": finished,
        "elapsed_seconds": elapsed,
        "engine": eng,
    }
    if extra:
        payload.update(extra)
    set_last_query_info(payload)

def render_diagnostics_tabs(qdf: pd.DataFrame, sdf: pd.DataFrame, demo_df: pd.DataFrame) -> None:
    tabs = st.tabs(["Raw data (debug)", "Parameters", "App setup", "AI status", "Search status", "Last query"])
    with tabs[0]:
        raw_data_debug_tab(qdf, demo_df)
    with tabs[1]:
        parameters_preview(qdf, demo_df)
    with tabs[2]:
        backend_info_panel(qdf, sdf, demo_df)
    with tabs[3]:
        ai_status_panel()
    with tabs[4]:
        search_status_panel()
    with tabs[5]:
        last_query_panel()
