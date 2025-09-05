# utils/data_loader.py
# RAW loader for PSES Results2024.csv.gz (NO normalization, NO renaming).
# Operates on original CSV columns:
#   SURVEYR, QUESTION, DEMCODE, answer1..answer7, POSITIVE, NEUTRAL, NEGATIVE, ANSCOUNT, ...
#
# Filtering is STRICT on the raw trio (QUESTION, SURVEYR, DEMCODE).
#   - All respondents -> group_value=None  (filters DEMCODE == "")
#   - Single subgroup -> group_value="0123"  (4-digit code from metadata)
#   - Multiple       -> group_value=["0123","0456", ...]
#
# Usage:
#   from utils.data_loader import load_results2024_filtered
#   df = load_results2024_filtered("Q01", [2024,2022,2020,2019], group_value=None)

from __future__ import annotations
from typing import Iterable, List, Optional, Union

import gzip
import io
import os
import pandas as pd

# ------------------------------------------------------------------------------
# Data source configuration
# ------------------------------------------------------------------------------
# Default path to your gzipped CSV inside the repo/container.
# Adjust if your app uses a different location.
DATA_PATH: str = "data/Results2024.csv.gz"
# ------------------------------------------------------------------------------


def set_data_source(local_path: Optional[str] = None) -> None:
    """
    (Optional) Override the default local path at runtime from your app setup.
    """
    global DATA_PATH
    if local_path:
        DATA_PATH = local_path


def _open_binary_stream() -> io.BufferedReader:
    """
    Opens the local .gz file in binary mode.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Results file not found at {DATA_PATH}")
    return open(DATA_PATH, "rb")


def _chunk_reader(stream: io.BufferedReader, chunksize: int = 200_000):
    """
    Yields DataFrame chunks from the gzipped CSV **as text**.
    - dtype=str to preserve values exactly as strings
    - keep_default_na=False and na_filter=False to keep blanks as ""
    """
    with gzip.open(stream, mode="rt", newline="") as f:
        for chunk in pd.read_csv(
            f,
            chunksize=chunksize,
            dtype=str,             # everything as text
            keep_default_na=False, # do NOT convert empty to NaN
            na_filter=False,       # do NOT detect NA strings
        ):
            yield chunk


def _apply_raw_filter(
    df: pd.DataFrame,
    question_code: str,
    years: Iterable[int],
    group_values: Optional[List[Optional[str]]],
) -> pd.DataFrame:
    """
    Apply strict filters using RAW columns:
      - QUESTION == question_code (case-insensitive; trimmed)
      - SURVEYR in years (string compare against "2024","2022",...)
      - DEMCODE:
          * All respondents => blank only ("")
          * Specific lists  => .isin(list) (values passed already 4-digit from metadata)
    Returns a copy (could be empty).
    """
    required = {"QUESTION", "SURVEYR", "DEMCODE"}
    if any(col not in df.columns for col in required):
        # Schema mismatch; return empty preserving columns for consistency
        return df.iloc[0:0].copy()

    # QUESTION
    q_norm = str(question_code).strip().upper()
    qmask = df["QUESTION"].astype(str).str.strip().str.upper() == q_norm

    # SURVEYR (compare as strings)
    year_strs = {str(int(y)) for y in years}
    ymask = df["SURVEYR"].astype(str).isin(year_strs)

    # DEMCODE
    if group_values is None or (len(group_values) == 1 and group_values[0] is None):
        # All respondents -> DEMCODE must be blank
        gmask = df["DEMCODE"].astype(str).str.strip() == ""
    else:
        # Accept exact codes as strings (metadata provides 4-digit codes)
        codes = [("" if gv is None else str(gv)) for gv in group_values]
        gmask = df["DEMCODE"].astype(str).isin(codes)

    return df[qmask & ymask & gmask].copy()


def load_results2024_filtered(
    question_code: str,
    years: Iterable[int],
    group_value: Optional[Union[str, List[Optional[str]], None]] = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Stream and filter the large CSV WITHOUT any normalization.

    Parameters
    ----------
    question_code : str
        Exact question code, e.g. "Q01" (validated/selected from metadata).
    years : Iterable[int]
        e.g. [2024, 2022, 2020, 2019]
    group_value : None | str | list[str|None]
        None                  -> All respondents (blank DEMCODE only).
        "0123"                -> one subgroup (4-digit DEMCODE).
        ["0123","0456",None]  -> multiple (None includes blank).
    chunksize : int
        Rows per chunk to process (tune for memory/performance).

    Returns
    -------
    pd.DataFrame
        Concatenated filtered rows with RAW column names.
        Empty DataFrame if no matches.
    """
    # Normalize group_value into a list or None
    if isinstance(group_value, list):
        gv_list = group_value
    elif group_value is None:
        gv_list = None
    else:
        gv_list = [group_value]

    results: List[pd.DataFrame] = []

    with _open_binary_stream() as stream:
        for chunk in _chunk_reader(stream, chunksize=chunksize):
            # Ensure all columns are string-typed (paranoia; read_csv already set dtype=str)
            for col in chunk.columns:
                if chunk[col].dtype != object:
                    chunk[col] = chunk[col].astype(str)

            filtered = _apply_raw_filter(
                df=chunk,
                question_code=question_code,
                years=years,
                group_values=gv_list,
            )
            if not filtered.empty:
                results.append(filtered)

    if not results:
        # Return a truly empty frame with no rows (and let the caller handle the message)
        return pd.DataFrame(columns=[])

    return pd.concat(results, ignore_index=True)
