# utils/data_loader.py
# -----------------------------------------------------------------------------
# Streamed loader for Results2024.csv.gz (RAW passthrough).
# - Filters on QUESTION (exact trimmed), SURVEYR (years as strings), DEMCODE (exact trimmed).
# - Enforces PS-wide: LEVEL1ID == "" OR "0" (strings) when present; if missing, keep rows.
# - No BYCOND, no dedup. Preserves original columns/values.
# -----------------------------------------------------------------------------

from typing import List, Optional
import os
import gzip

import gdown
import pandas as pd
import streamlit as st

# You can override this in .streamlit/secrets.toml with RESULTS2024_FILE_ID
GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
LOCAL_GZ_PATH = "/tmp/Results2024.csv.gz"


@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
    """Ensure the gzipped CSV is present locally and return its path."""
    file_id = file_id or st.secrets.get("RESULTS2024_FILE_ID", GDRIVE_FILE_ID_FALLBACK)
    if not file_id:
        raise RuntimeError("RESULTS2024_FILE_ID missing in .streamlit/secrets.toml")

    if not os.path.exists(LOCAL_GZ_PATH) or os.path.getsize(LOCAL_GZ_PATH) == 0:
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(LOCAL_GZ_PATH), exist_ok=True)
        gdown.download(url, LOCAL_GZ_PATH, quiet=False)
    return LOCAL_GZ_PATH


def _chunk_reader(path: str, chunksize: int = 200_000):
    """
    Yield DataFrame chunks from the gzipped CSV as **text**:
      - dtype=str to preserve values exactly
      - keep_default_na=False / na_filter=False so blanks stay ""
      - Header normalized to UPPERCASE (values are not changed)
    """
    with gzip.open(path, mode="rt", newline="") as f:
        for chunk in pd.read_csv(
            f,
            chunksize=chunksize,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=True,
        ):
            chunk.columns = [str(c).strip().upper() for c in chunk.columns]
            yield chunk


@st.cache_data(show_spinner="🔎 Filtering results…")
def load_results2024_filtered(
    question_code: str,
    years: List[str] | List[int],
    group_value: Optional[str] = None,     # None or "" => All respondents (blank DEMCODE)
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Filter rows where:
      - QUESTION equals `question_code` (exact trimmed string)
      - SURVEYR is in `years` (compared as trimmed strings)
      - DEMCODE equals the provided code after .strip()  ("" when group_value is None/"")
      - PS-wide: LEVEL1ID is "" or "0" (strings) when present; if absent, keep rows.

    Returns all matching rows with their original columns preserved.
    """
    path = ensure_results2024_local()

    q_target = str(question_code).strip()
    years_str = {str(y).strip() for y in years}

    want_blank_dem = (group_value is None) or (str(group_value).strip() == "")
    dem_target = "" if want_blank_dem else str(group_value).strip()

    parts: list[pd.DataFrame] = []

    for chunk in _chunk_reader(path, chunksize=chunksize):
        # Guard: chunk must have these columns
        if not {"QUESTION", "SURVEYR", "DEMCODE"}.issubset(set(chunk.columns)):
            continue

        qmask = chunk["QUESTION"].astype(str).str.strip() == q_target
        ymask = chunk["SURVEYR"].astype(str).str.strip().isin(years_str)

        dem_series = chunk["DEMCODE"].astype(str).str.strip()
        if want_blank_dem:
            gmask = dem_series == ""
        else:
            gmask = dem_series == dem_target  # exact trimmed string match

        # PS-wide: LEVEL1ID == "" or "0" (strings) if column exists
        if "LEVEL1ID" in chunk.columns:
            lvl = chunk["LEVEL1ID"].astype(str).str.strip()
            lmask = (lvl == "") | (lvl == "0")
        else:
            lmask = pd.Series(True, index=chunk.index)

        sub = chunk[qmask & ymask & gmask & lmask]
        if not sub.empty:
            parts.append(sub)

    if parts:
        return pd.concat(parts, ignore_index=True)

    # Stable empty frame with common headers (helps downstream UI)
    return pd.DataFrame(columns=[
        "LEVEL1ID","LEVEL2ID","LEVEL3ID","LEVEL4ID","LEVEL5ID",
        "SURVEYR","DEMCODE","QUESTION","ANSWER1","ANSWER2","ANSWER3","ANSWER4","ANSWER5","ANSWER6","ANSWER7",
        "POSITIVE","NEUTRAL","NEGATIVE","SCORE5","SCORE100","ANSCOUNT"
    ])
