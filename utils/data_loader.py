# utils/data_loader.py
# Fast, cached reader + one-pass filter for Menu 1
import os
import gzip
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

# If you already have this in your file, keep your version.
# It should return the local file path to Results2024.csv.gz.
try:
    ensure_results2024_local  # type: ignore
except NameError:
    def ensure_results2024_local() -> str:
        # Fallback; adjust if your project resolves the path differently.
        return "/tmp/Results2024.csv.gz"

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
    - No whole-frame trimming; we’ll trim only filter columns during masking.
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

def load_results2024_filtered(
    question_code: str,
    years: Iterable[str],
    group_values: Optional[List[Optional[str]]] = None,
) -> pd.DataFrame:
    """
    One-pass filter on cached DF.
    Filters:
      - QUESTION == question_code (exact, trimmed)
      - SURVEYR in {years} (strings)
      - DEMCODE in {group_values} (trimmed) ; None -> blank "" (All respondents)
      - PS-wide only: LEVEL1ID == "" or "0" (if column exists)
    Returns the filtered slice as TEXT (all columns str).
    """
    path = ensure_results2024_local()
    sig = _file_signature(path)
    df = _read_results2024_text_mode(*sig)

    # Build masks using only the needed columns (trimmed strings)
    if "QUESTION" not in df.columns or "SURVEYR" not in df.columns or "DEMCODE" not in df.columns:
        # Return empty frame with same columns to keep callers happy
        return df.head(0).copy()

    qmask = _trim(df["QUESTION"]) == str(question_code).strip()
    years_set = {str(y).strip() for y in years}
    ymask = _trim(df["SURVEYR"]).isin(years_set)

    # DEMCODE set
    dem_series = _trim(df["DEMCODE"])
    targets = set()
    if not group_values:
        # treat as All respondents by default
        targets.add("")
    else:
        for gv in group_values:
            if gv is None:
                targets.add("")   # All respondents (blank DEMCODE)
            else:
                targets.add(str(gv).strip())
    gmask = dem_series.isin(targets)

    mask = qmask & ymask & gmask
    out = df.loc[mask].copy()

    # PS-wide guard
    if "LEVEL1ID" in out.columns:
        lvl = _trim(out["LEVEL1ID"])
        out = out[(lvl == "") | (lvl == "0")].copy()

    return out
