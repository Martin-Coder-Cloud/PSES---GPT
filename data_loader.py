# utils/data_loader.py
from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Config / defaults
# -----------------------------
# You can override this in Streamlit secrets: st.secrets["RESULTS_PATH"]
_DEFAULT_CANDIDATES = [
    "data/results2024.csv.gz",
    "results2024.csv.gz",
    "/mnt/data/results2024.csv.gz",
]

# Module-level “last used” info for diagnostics
_LAST_BACKEND: str | None = None
_LAST_SOURCE_PATH: str | None = None


# -----------------------------
# Small helpers
# -----------------------------
def _normalize_qcode(s: str) -> str:
    """Uppercase; keep only A-Z0-9 (e.g., 'Q 19-a' -> 'Q19A')."""
    if s is None:
        return ""
    s = str(s).upper()
    return re.sub(r"[^A-Z0-9]+", "", s)


def _norm_or_empty(x) -> str:
    return _normalize_qcode(x) if pd.notna(x) else ""


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def _canon_demcode(x: str | None) -> str:
    """Return a stripped demcode. If purely digits up to 4, zero-pad to 4."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.isdigit() and len(s) <= 4:
        return s.zfill(4)
    return s


def _listify_one_or_many(
    group_values: Optional[Iterable[str | None]] = None,
    group_value: Optional[str | None] = None,
) -> Optional[List[str | None]]:
    if group_values is not None:
        return list(group_values)
    if group_value is not None:
        return [group_value]
    return None


# -----------------------------
# Public helpers for the UI
# -----------------------------
def _resolve_results_path() -> str:
    """
    Resolve the results file path (CSV.GZ). Order:
    1) st.secrets["RESULTS_PATH"]
    2) Known defaults
    Raises FileNotFoundError if none exist.
    """
    cand = []
    try:
        v = (st.secrets.get("RESULTS_PATH") or "").strip()
        if v:
            cand.append(v)
    except Exception:
        pass
    cand.extend(_DEFAULT_CANDIDATES)
    for p in cand:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "PSES results file not found. Checked: " + ", ".join(cand)
    )


def get_backend_info() -> dict:
    """
    Return the backend & source path used by the last successful call to
    load_results2024_filtered().
    """
    return {
        "backend": _LAST_BACKEND,
        "source_path": _LAST_SOURCE_PATH,
    }


# -----------------------------
# Schema helpers (global previews)
# -----------------------------
@st.cache_data(show_spinner=False)
def get_results2024_schema() -> pd.DataFrame:
    """
    Show a quick schema with text/object dtypes (no inference).
    Reads a small sample with pandas dtype=str so everything is TEXT.
    """
    path = _resolve_results_path()
    df = pd.read_csv(
        path,
        compression="infer",
        dtype=str,
        nrows=50,
        keep_default_na=False,
        na_filter=False,
    )
    schema = pd.DataFrame(
        {"column": df.columns, "dtype": ["object"] * len(df.columns)}
    )
    return schema


@st.cache_data(show_spinner=False)
def get_results2024_schema_inferred() -> pd.DataFrame:
    """
    Preview schema with pandas' type inference (for debugging only).
    """
    path = _resolve_results_path()
    df = pd.read_csv(path, compression="infer", nrows=200)
    schema = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    return schema


# -----------------------------
# Core loader (DuckDB → pandas fallback)
# -----------------------------
def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False


def _duckdb_query(
    path: str,
    question_code: str,
    years: List[str],
    group_values: Optional[List[str | None]],
) -> pd.DataFrame:
    import duckdb

    qc_norm = _normalize_qcode(question_code)
    years_str = [str(y) for y in years]

    # Build DEMCODE predicate
    dem_pred = None
    params: list = [path]

    if group_values is not None:
        # Separate “overall/blank” and concrete codes
        raw = group_values
        want_blank = any(gv is None or str(gv).strip() == "" for gv in raw)
        codes = [gv for gv in raw if gv is not None and str(gv).strip() != ""]
        codes = [_canon_demcode(str(c)) for c in codes]

        parts = []
        if codes:
            parts.append("trim(DEMCODE) IN (" + ", ".join(["?"] * len(codes)) + ")")
            params.extend(codes)
        if want_blank:
            parts.append("(DEMCODE IS NULL OR trim(DEMCODE)='')")
        if parts:
            dem_pred = "(" + " OR ".join(parts) + ")"

    year_place = ", ".join(["?"] * len(years_str))
    params.extend(years_str)
    params.append(qc_norm)

    sql = f"""
    SELECT *
    FROM read_csv_auto(?, header=true)
    WHERE
      CAST(SURVEYR AS VARCHAR) IN ({year_place})
      AND regexp_replace(upper(COALESCE(QUESTION,'')), '[^A-Z0-9]', '') = ?
    """
    if dem_pred:
        sql += f" AND {dem_pred}"

    con = duckdb.connect()
    try:
        df = con.execute(sql, params).fetch_df()
    finally:
        con.close()

    # Ensure TEXT-like behavior
    for c in df.columns:
        df[c] = df[c].astype(str)
    return df


def _pandas_query(
    path: str,
    question_code: str,
    years: List[str],
    group_values: Optional[List[str | None]],
) -> pd.DataFrame:
    # Read everything as TEXT and filter in pandas
    df = pd.read_csv(
        path,
        compression="infer",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    # Normalize helper columns
    df["_Q_NORM"] = df["QUESTION"].map(_norm_or_empty) if "QUESTION" in df.columns else ""
    df["_Y_STR"] = df["SURVEYR"].astype(str) if "SURVEYR" in df.columns else ""

    mask = df["_Q_NORM"].eq(_normalize_qcode(question_code)) & df["_Y_STR"].isin([str(y) for y in years])

    if group_values is not None and "DEMCODE" in df.columns:
        want_blank = any(gv is None or str(gv).strip() == "" for gv in group_values)
        codes = [gv for gv in group_values if gv is not None and str(gv).strip() != ""]
        codes = [_canon_demcode(str(c)) for c in codes]

        dem = df["DEMCODE"].astype(str).str.strip()
        cond = pd.Series(False, index=df.index)
        if codes:
            cond |= dem.isin(codes)
        if want_blank:
            cond |= dem.eq("") | df["DEMCODE"].isna()
        mask &= cond

    out = df.loc[mask].copy()
    # Drop helper cols
    out.drop(columns=[c for c in ["_Q_NORM", "_Y_STR"] if c in out.columns], inplace=True, errors="ignore")

    # Ensure every column is text (object) to match app expectations
    for c in out.columns:
        out[c] = out[c].astype(str)
    return out


@st.cache_data(show_spinner=False)
def load_results2024_filtered(
    question_code: str,
    years: Iterable[str],
    group_values: Optional[Iterable[str | None]] = None,
    group_value: Optional[str | None] = None,
    _data_mtime_key: Optional[float] = None,  # internal: for cache busting on file change
) -> pd.DataFrame:
    """
    Return filtered rows for:
      - QUESTION == question_code (normalized)
      - SURVEYR IN years
      - DEMCODE in group_values (if provided). Use None to include overall/blank.
    Notes:
      * All columns are returned as TEXT (object).
      * Uses DuckDB if available; otherwise pandas.
      * Backwards compatible with legacy 'group_value='.
    """
    # Resolve file path first
    path = _resolve_results_path()
    # Cache invalidation: include data mtime in the cache key
    if _data_mtime_key is None:
        _data_mtime_key = _file_mtime(path)

    # Normalize parameters (cache needs deterministic types)
    years = [str(y) for y in years]
    gvals = _listify_one_or_many(group_values, group_value)

    global _LAST_BACKEND, _LAST_SOURCE_PATH
    _LAST_SOURCE_PATH = path

    # Prefer DuckDB if installed
    if _duckdb_available():
        df = _duckdb_query(path, question_code, years, gvals)
        _LAST_BACKEND = "duckdb"
        return df

    # Fallback: pandas
    df = _pandas_query(path, question_code, years, gvals)
    _LAST_BACKEND = "pandas"
    return df
