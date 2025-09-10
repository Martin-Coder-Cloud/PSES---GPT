# utils/data_loader.py
# -----------------------------------------------------------------------------
# Streamed loader for Results2024.csv.gz (RAW passthrough).
# - Filters ONLY on QUESTION, SURVEYR (year), and DEMCODE (blank => All).
# - Does NOT reference BYCOND or LEVEL* columns.
# - Preserves the original columns/values (reads as text; blanks stay "").
# - Returns the full matching rows (no normalization, no de-dup).
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
      - Only header normalization: strip + UPPERCASE
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
    years: List[int],
    group_value: Optional[str] = None,     # None or "" => All respondents (blank DEMCODE)
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Filter rows where:
      - QUESTION equals `question_code` (case-insensitive, stripped)
      - SURVEYR is in `years`
      - DEMCODE is "" (when group_value is None/"") OR equals the provided code

    Returns all matching rows with their original columns preserved.
    """
    path = ensure_results2024_local()

    q_target = str(question_code).strip().upper()
    years_str = {str(y) for y in years}
    want_blank_dem = (group_value is None) or (str(group_value).strip() == "") or (str(group_value).strip().upper() == "ALL")

    parts: list[pd.DataFrame] = []

    for chunk in _chunk_reader(path, chunksize=chunksize):
        # Guard: chunk must have these columns
        if not {"QUESTION", "SURVEYR", "DEMCODE"}.issubset(set(chunk.columns)):
            continue

        qmask = chunk["QUESTION"].astype(str).str.strip().str.upper() == q_target
        ymask = chunk["SURVEYR"].astype(str).isin(years_str)

        if want_blank_dem:
            gmask = chunk["DEMCODE"].astype(str).str.strip() == ""
        else:
            gmask = chunk["DEMCODE"].astype(str) == str(group_value)

        sub = chunk[qmask & ymask & gmask]
        if not sub.empty:
            parts.append(sub)

    if parts:
        return pd.concat(parts, ignore_index=True)

    # Provide a stable empty frame with common headers (helps downstream UI)
    return pd.DataFrame(columns=[
        "LEVEL1ID","LEVEL2ID","LEVEL3ID","LEVEL4ID","LEVEL5ID",
        "SURVEYR","DEMCODE","QUESTION","ANSWER1","ANSWER2","ANSWER3","ANSWER4","ANSWER5","ANSWER6","ANSWER7",
        "POSITIVE","NEUTRAL","NEGATIVE","SCORE5","SCORE100","ANSCOUNT"
    ])
