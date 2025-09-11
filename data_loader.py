# utils/data_loader.py
# Cached text-mode loader + one-pass filtering for Menu 1

from __future__ import annotations

import os
import gzip
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────
def ensure_results2024_local() -> str:
    """
    Returns the local path to Results2024.csv.gz.
    Override with env var RESULTS2024_PATH if needed.
    """
    env_path = os.environ.get("RESULTS2024_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    # Default location (adjust if your infra stores it elsewhere)
    return "/tmp/Results2024.csv.gz"


def _file_signature(path: str) -> Tuple[str, int, int]:
    """(path, mtime, size) to key the cache and auto-invalidate on file change."""
    try:
        stat = os.stat(path)
        return (path, int(stat.st_mtime), int(stat.st_size))
    except Exception:
        return (path, 0, 0)


# ─────────────────────────────────────────
# Cached read as TEXT
# ─────────────────────────────────────────
@st.cache_data(show_spinner=True)
def _read_results2024_text_mode(path: str, mtime: int, size: int) -> pd.DataFrame:
    """
    Read CSV.GZ once as TEXT (dtype=str).
    - No NA parsing (keep_default_na=False, na_filter=False)
    - Uppercase headers
    - DO NOT trim all columns (we’ll trim only filter cols during masking)
    """
    with gzip.open(path, mode="rt", newline="") as f:
        df = pd.read_csv(
            f,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=True,
        )
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def _trim(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


# ─────────────────────────────────────────
# One-pass filter (accepts list or legacy single)
# ─────────────────────────────────────────
def load_results2024_filtered(
    question_code: str,
    years: Iterable[str],
    group_values: Optional[List[Optional[str]]] = None,
    group_value: Optional[str] = None,  # legacy support
) -> pd.DataFrame:
    """
    Filter from the cached DF using a single boolean mask:

      - QUESTION == question_code    (string-compare, trimmed)
      - SURVEYR ∈ {years}           (strings, trimmed)
      - DEMCODE ∈ {group_values}    (strings, trimmed); None -> "" (All respondents)
      - PS-wide only if LEVEL1ID exists: LEVEL1ID in {"", "0"}

    Returns a slice; all columns remain TEXT.
    """
    path = ensure_results2024_local()
    sig = _file_signature(path)
    df = _read_results2024_text_mode(*sig)

    # Guard for required columns
    required = {"QUESTION", "SURVEYR", "DEMCODE"}
    if not required.issubset(df.columns):
        return df.head(0).copy()

    # Normalize filters (strings only)
    qmask = _trim(df["QUESTION"]) == str(question_code).strip()
    years_set = {str(y).strip() for y in years}
    ymask = _trim(df["SURVEYR"]).isin(years_set)

    # Accept list or legacy single
    if group_values is None:
        group_values = [group_value]  # may be [None]
    targets = set()
    for gv in (group_values or [None]):
        if gv is None:
            targets.add("")  # blank DEMCODE is All respondents
        else:
            targets.add(str(gv).strip())

    gmask = _trim(df["DEMCODE"]).isin(targets)

    mask = qmask & ymask & gmask
    out = df.loc[mask].copy()

    # PS-wide guard
    if "LEVEL1ID" in out.columns:
        lvl = _trim(out["LEVEL1ID"])
        out = out[(lvl == "") | (lvl == "0")].copy()

    return out


# ─────────────────────────────────────────
# Diagnostics helpers (optional)
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_results2024_schema() -> pd.DataFrame:
    """Simple schema table after loader (all text)."""
    path = ensure_results2024_local()
    sig = _file_signature(path)
    df = _read_results2024_text_mode(*sig)

    rows = []
    for c in df.columns:
        s = df[c].astype(str)
        ex = ""
        for v in s:
            # pick the first non-blank example (as string)
            if str(v) != "":
                ex = str(v)
                break
        blank_rate = (s == "").mean() if len(s) else 0.0
        rows.append({
            "column": str(c),
            "dtype_after_loader": str(s.dtype),
            "example_non_blank": str(ex),
            "blank_rate": float(round(blank_rate, 3)),
        })
    out = pd.DataFrame(rows)
    # Ensure Arrow-friendly dtypes
    out["column"] = out["column"].astype(str)
    out["dtype_after_loader"] = out["dtype_after_loader"].astype(str)
    out["example_non_blank"] = out["example_non_blank"].astype(str)
    out["blank_rate"] = out["blank_rate"].astype(float)
    return out


@st.cache_data(show_spinner=False)
def get_results2024_schema_inferred(sample_rows: int = 5000) -> pd.DataFrame:
    """What pandas would infer without text-forcing (for comparison only)."""
    path = ensure_results2024_local()
    with gzip.open(path, mode="rt", newline="") as f:
        peek = pd.read_csv(
            f,
            nrows=sample_rows,
            keep_default_na=True,
            na_filter=True,
            low_memory=True,
        )
    peek.columns = [str(c).strip().upper() for c in peek.columns]

    rows = []
    for c in peek.columns:
        s = peek[c]
        ex = ""
        for v in s:
            if pd.notna(v):
                ex = str(v)
                break
        rows.append({
            "column": str(c),
            "dtype_inferred_by_pandas": str(s.dtype),
            "example_non_blank": str(ex),
        })
    out = pd.DataFrame(rows)
    out["column"] = out["column"].astype(str)
    out["dtype_inferred_by_pandas"] = out["dtype_inferred_by_pandas"].astype(str)
    out["example_non_blank"] = out["example_non_blank"].astype(str)
    return out
