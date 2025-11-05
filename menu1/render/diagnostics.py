# menu1/render/diagnostics.py
# (original file + approved changes)
#  • Parameters panel is code-first + shows Resolved DEMCODE(s)
#  • Raw data (debug) tab shows unsuppressed, unpivoted rows
#  • DEMCODE scanner (PS-wide) with optional "Force CSV scan (PS-wide)"
#  • FIX: replace brittle multi-line strings with safe triple-quoted strings
#         and remove mojibake characters.

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
        """<div class='diag-box'><pre>{}</pre></div>""".format(
            json.dumps(obj, ensure_ascii=False, indent=2)
        ),
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
    # Kept for potential cross-checks (not displayed)
    code_to_display = dict(zip(qdf["code"], qdf["display"])) if "code" in qdf.columns and "display" in qdf.columns else {}

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
        """Lists actual DEMCODE values present for the selected question(s) and year(s) at PS-wide (LEVEL1ID==0).
Ignores the demographic filter. Use this to verify presence independent of the render pipeline."""
    )

    force_csv = st.checkbox(
        "Force CSV scan (PS-wide)",
        value=False,
        help="Bypass caches and scan the vetted CSV directly."
    )

    # Helper: detect code column in Demographics.xlsx
    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    # Normalize selection for scanner
    qcodes_norm = [str(q).strip().upper() for q in (sel_codes or [])]
    years_norm  = [int(y) for y in (years_selected or [])]

    if not qcodes_norm or not years_norm:
        st.info("Select at least one question and year to scan DEMCODEs.")
        return

    try:
        if not force_csv:
            # In-memory PS-wide scan
            df_all = preload_pswide_dataframe()
            if isinstance(df_all, pd.DataFrame) and not df_all.empty:
                mask = df_all["question_code"].isin(qcodes_norm) & df_all["year"].astype(int).isin(years_norm)
                df_slice = df_all.loc[mask, ["group_value"]].copy()
                if df_slice.empty:
                    st.warning("No PS-wide rows found for the selected question(s) and year(s).")
                else:
                    # value counts
                    vc = df_slice["group_value"].astype("string").str.strip().value_counts(dropna=False)
                    counts = vc.rename("rows").to_frame().reset_index().rename(columns={"index": "group_value"})
                    # Join to metadata
                    join_df = demo_df.copy()
                    if code_col:
                        join_df["__code_str__"] = demo_df[code_col].astype("string").str.strip()
                        counts["__code_str__"] = counts["group_value"].astype("string").str.strip()
                        counts = counts.merge(
                            join_df[["__code_str__", "DESCRIP_E", "DEMCODE Category"]],
                            how="left", on="__code_str__"
                        ).drop(columns="__code_str__", errors="ignore")
                    # Sort & show
                    counts = counts.sort_values("rows", ascending=False)
                    st.dataframe(
                        counts[["group_value", "DESCRIP_E", "DEMCODE Category", "rows"]],
                        use_container_width=True, hide_index=True
                    )
            else:
                st.warning("In-memory PS-wide DataFrame is empty or unavailable; use CSV scan.")
        else:
            if ensure_results2024_local is None:
                st.error("CSV scan unavailable: ensure_results2024_local() not importable.")
            else:
                csv_path = ensure_results2024_local()
                wanted_q = set(qcodes_norm)
                wanted_y = set(years_norm)
                counts = Counter()
                # Stream CSV in chunks (PS-wide only)
                usecols = ["LEVEL1ID", "SURVEYR", "QUESTION", "DEMCODE"]
                for chunk in pd.read_csv(csv_path, compression="gzip", usecols=usecols, chunksize=2_000_000, low_memory=True):
                    # PS-wide filter
                    lvl_mask = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(0).astype(int).eq(0) if "LEVEL1ID" in chunk.columns else pd.Series(True, index=chunk.index)
                    # Question/year filter
                    q_ser = chunk["QUESTION"].astype("string").str.strip().str.upper()
                    y_ser = pd.to_numeric(chunk["SURVEYR"], errors="coerce").astype("Int64")
                    mask = lvl_mask & q_ser.isin(list(wanted_q)) & y_ser.isin(list(wanted_y))
                    if not mask.any():
                        continue
                    gv = chunk.loc[mask, "DEMCODE"].astype("string").str.strip().fillna("All")
                    # empty DEMCODE → "All" (PS-wide overall)
                    gv = gv.mask(gv == "", "All")
                    # accumulate
                    counts.update(gv.tolist())

                if not counts:
                    st.warning("No PS-wide rows found in CSV for the selected question(s) and year(s).")
                else:
                    counts_df = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["group_value", "rows"])
                    if code_col:
                        demo_df["__code_str__"] = demo_df[code_col].astype("string").str.strip()
                        counts_df["__code_str__"] = counts_df["group_value"].astype("string").str.strip()
                        counts_df = counts_df.merge(
                            demo_df[["__code_str__", "DESCRIP_E", "DEMCODE Category"]],
                            how="left", on="__code_str__"
                        ).drop(columns="__code_str__", errors="ignore")
                    st.dataframe(
                        counts_df[["group_value", "DESCRIP_E", "DEMCODE Category", "rows"]],
                        use_container_width=True, hide_index=True
                    )
    except Exception as e:
    ...
