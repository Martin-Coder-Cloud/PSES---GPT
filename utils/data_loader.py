# utils/data_loader.py
# Fast, cached reader + one-pass filter for Menu 1
from __future__ import annotations

import os
import gzip
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────
# Resolve the local Results2024.csv.gz path
# ─────────────────────────────────────────
def ensure_results2024_local() -> str:
    """
    Returns the local path to Results2024.csv.gz.
    You can override by setting RESULTS2024_PATH env var.
    """
    env_path = os.environ.get("RESULTS2024_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    # Project default — adjust if your repo uses a different location
    default_path = "/tmp/Results2024.csv.gz"
    return env_path or default_path


# ─────────────────────────────────────────
# Cached read (TEXT mode)
# ─────────────────────────────────────────
def _file_signature(path: str) -> Tuple[str, int, int]:
    """Return (path, mtime, size) to key the cache and auto-invalidate when file changes."""
    try:
        stat = os.stat(path)
        return (path, int(stat.st_mtime), int(stat.st_size))
    except Exception:
        return (path, 0, 0)


@st.cache_data(show_spinner=True)
def _read_results2024_text_mode(path: str, mtime: int, size: int) -> pd.DataFrame:
    """
    Read the large CSV.GZ once as TEXT (dtype=str).
    - No NA parsing.
    - Uppercase headers only.
    - Do NOT trim all columns here; we only trim filter columns during masking.
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
# One-pass filtered read (from cached DF)
# ─────────────────────────────────────────
def load_results2024_filtered(
    question_code: str,
    years: Iterable[str],
    group_values: Optional[List[Optional[str]]] = None,
    group_value: Optional[str] = None,  # legacy compatibility
) -> pd.DataFrame:
    """
    Filter from the cached DataFrame using a single boolean mask:
      - QUESTION == question_code (exact, trimmed)
      - SURVEYR in {years} (strings, trimmed)
      - DEMCODE in {group_values} (trimmed); None -> "" (All respondents)
      - PS-wide only: LEVEL1ID == "" or "0" (if column exists)

    Returns a DataFrame slice. All columns remain TEXT.
    """
    path = ensure_results2024_local()
    sig = _file_signature(path)
    df = _read_results2024_text_mode(*sig)

    # Required columns present?
    for col in ("QUESTION", "SURVEYR", "DEMCODE"):
        if col not in df.columns:
            return df.head(0).copy()

    qmask = _trim(df["QUESTION"]) == str(question_code).strip()
    years_set = {str(y).strip() for y in years}
    ymask = _trim(df["SURVEYR"]).isin(years_set)

    # Accept either new list param or legacy single param
    if group_values is None:
        group_values = [group_value]  # may be [None]
    targets = set()
    for gv in (group_values or [None]):
        if gv is None:
            targets.add("")  # blank DEMCODE = All respondents
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
# (Optional) schema helpers for diagnostics
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_results2024_schema() -> pd.DataFrame:
    """Report column dtypes (post-loader), example non-blank, and blank rate."""
    path = ensure_results2024_local()
    sig = _file_signature(path)
    df = _read_results2024_text_mode(*sig)
    rows = []
    for c in df.columns:
        s = df[c].astype(str)
        ex = next((v for v in s if v != ""), "")
        blank_rate = (s == "").mean() if len(s) else 0.0
        rows.append({
            "column": c,
            "dtype_after_loader": str(s.dtype),
            "example_non_blank": ex,
            "blank_rate": round(float(blank_rate), 3),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_results2024_schema_inferred(sample_rows: int = 5000) -> pd.DataFrame:
    """Preview what pandas would infer if we *didn't* force text (for comparison only)."""
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
        ex = next((v for v in s if pd.notna(v)), "")
        rows.append({
            "column": c,
            "dtype_inferred_by_pandas": str(s.dtype),
            "example_non_blank": ex,
        })
    return pd.DataFrame(rows)
