# utils/data_loader.py
# Streamed loader for Results2024.csv.gz with minimal header normalization.
# - Reads gzipped CSV in chunks
# - Preserves data as text (dtype=str) so blanks stay ""
# - DOES NOT rename columns; only strips whitespace and uppercases headers
# - Exposes: load_results2024_filtered(question_code, years, group_value)

from __future__ import annotations

import io
import os
import gzip
from typing import Iterable, List, Optional

import pandas as pd
import requests


# You can change this to point to a local file if you prefer.
DEFAULT_GDRIVE_FILE_ID = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
DEFAULT_LOCAL_PATH = os.environ.get("PSES_RESULTS_PATH", "/tmp/Results2024.csv.gz")


def _download_from_gdrive(file_id: str, dest_path: str) -> None:
    """Download a file from Google Drive to dest_path (simple uc?id=... URL)."""
    url = f"https://drive.google.com/uc?id={file_id}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _ensure_gz_file(path: str = DEFAULT_LOCAL_PATH, gdrive_file_id: str = DEFAULT_GDRIVE_FILE_ID) -> str:
    """Ensure the gz file exists locally; if not, download from Google Drive."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _download_from_gdrive(gdrive_file_id, path)
    return path


def _chunk_reader(stream: io.BufferedReader, chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    """
    Yields DataFrame chunks from the gzipped CSV **as text**.
    - dtype=str to preserve values exactly as strings
    - keep_default_na=False and na_filter=False to keep blanks as ""
    - column headers are STRIPPED and UPPERCASED (ONLY normalization we do)
    """
    with gzip.open(stream, mode="rt", newline="") as f:
        for chunk in pd.read_csv(
            f,
            chunksize=chunksize,
            dtype=str,             # everything as text
            keep_default_na=False, # do NOT convert empty to NaN
            na_filter=False,       # do NOT detect NA strings
        ):
            # 🔧 Normalize headers: strip whitespace, uppercase
            chunk.columns = [str(c).strip().upper() for c in chunk.columns]
            yield chunk


def load_results2024_filtered(
    question_code: str,
    years: List[int],
    group_value: Optional[str]
) -> pd.DataFrame:
    """
    Load Results2024.csv.gz in chunks and filter by:
      - QUESTION == question_code (case-insensitive, stripped)
      - SURVEYR in selected years (compared as strings)
      - DEMCODE == "" if group_value is None (All respondents),
        otherwise DEMCODE == str(group_value)

    Returns a concatenated DataFrame (may be empty). Columns remain as in source,
    except headers are guaranteed to be UPPERCASE w/ no leading/trailing spaces.
    """
    # Ensure local gz file is present
    gz_path = _ensure_gz_file()

    str_years = {str(y) for y in years}
    q_target = str(question_code).strip().upper()

    parts: List[pd.DataFrame] = []

    with open(gz_path, "rb") as fh:
        for chunk in _chunk_reader(fh):
            # Defensive: ensure required columns exist
            required = {"QUESTION", "SURVEYR", "DEMCODE"}
            missing = required.difference(set(chunk.columns))
            if missing:
                # If a chunk doesn't have expected columns, skip it
                continue

            # Build masks using text comparisons
            qmask = chunk["QUESTION"].astype(str).str.strip().str.upper() == q_target
            ymask = chunk["SURVEYR"].astype(str).isin(str_years)

            if group_value is None:
                # All respondents => DEMCODE must be blank
                gmask = chunk["DEMCODE"].astype(str).str.strip() == ""
            else:
                gmask = chunk["DEMCODE"].astype(str) == str(group_value)

            sub = chunk[qmask & ymask & gmask]
            if not sub.empty:
                parts.append(sub)

    if parts:
        return pd.concat(parts, ignore_index=True)

    return pd.DataFrame(columns=[
        # Provide a stable (empty) frame with the common headers uppercased.
        # This avoids KeyError downstream when concatenation returns empty.
        "LEVEL1ID","LEVEL2ID","LEVEL3ID","LEVEL4ID","LEVEL5ID",
        "SURVEYR","BYCOND","DEMCODE","QUESTION",
        "ANSWER1","ANSWER2","ANSWER3","ANSWER4","ANSWER5","ANSWER6","ANSWER7",
        "POSITIVE","NEUTRAL","NEGATIVE","SCORE5","SCORE100","ANSCOUNT"
    ])
