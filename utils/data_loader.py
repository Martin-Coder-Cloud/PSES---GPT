# utils/data_loader.py
# -----------------------------------------------------------------------------
# Loads Results2024.csv.gz from Google Drive and returns filtered slices
# Schema mapping (native -> normalized):
#   SURVEYR -> year (int16)
#   QUESTION -> question_code (str like "Q01")
#   DEMCODE -> group_value (demographic code; if missing => "All")
#   POSITIVE/NEUTRAL/NEGATIVE -> *_pct (float32, 0–100)
#   ANSCOUNT -> n (int32)
#   answer1..answer7 -> answer1..answer7 (float32, 0–100)
#
# BYCOND is explicitly ignored in this app.
# -----------------------------------------------------------------------------

import os
from typing import List, Optional

import gdown
import pandas as pd
import streamlit as st

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Put this in .streamlit/secrets.toml if possible:
#   RESULTS2024_FILE_ID = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
LOCAL_GZ_PATH = "/tmp/Results2024.csv.gz"

# Read only what we need from the native file (BYCOND intentionally omitted)
NATIVE_USECOLS = [
    "SURVEYR", "QUESTION", "DEMCODE",
    "POSITIVE", "NEUTRAL", "NEGATIVE",
    "ANSCOUNT",
    "answer1","answer2","answer3","answer4","answer5","answer6","answer7",
]

# Normalized output columns
OUT_COLS = [
    "year", "question_code", "group_value", "n",
    "positive_pct", "neutral_pct", "negative_pct",
    "answer1","answer2","answer3","answer4","answer5","answer6","answer7",
]

# Target dtypes (memory-friendly)
DTYPES = {
    "year": "int16",
    "question_code": "string",
    "group_value": "string",
    "n": "int32",
    "positive_pct": "float32",
    "neutral_pct": "float32",
    "negative_pct": "float32",
    "answer1": "float32", "answer2": "float32", "answer3": "float32",
    "answer4": "float32", "answer5": "float32", "answer6": "float32", "answer7": "float32",
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _normalize_native_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Map native columns to normalized ones and coerce types. BYCOND is ignored."""
    # Guard for required native columns
    required_native = ["SURVEYR", "QUESTION", "DEMCODE", "POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    missing = [c for c in required_native if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing expected columns in chunk: {', '.join(missing)}")

    out = pd.DataFrame({
        "year":          pd.to_numeric(chunk["SURVEYR"], errors="coerce"),
        "question_code": chunk["QUESTION"].astype("string"),
        # DEMCODE is the demographic code; blank/NaN => overall PS-wide
        "group_value":   chunk["DEMCODE"].astype("string"),
        "n":             pd.to_numeric(chunk["ANSCOUNT"], errors="coerce"),
        "positive_pct":  pd.to_numeric(chunk["POSITIVE"], errors="coerce"),
        "neutral_pct":   pd.to_numeric(chunk["NEUTRAL"], errors="coerce"),
        "negative_pct":  pd.to_numeric(chunk["NEGATIVE"], errors="coerce"),
    })

    # Keep distribution across answers (if any missing, create the column with NA)
    for i in range(1, 8):
        col = f"answer{i}"
        if col in chunk.columns:
            out[col] = pd.to_numeric(chunk[col], errors="coerce")
        else:
            out[col] = pd.NA

    # Normalize blanks/NaNs in group_value to "All"
    out["group_value"] = out["group_value"].fillna("All")
    out.loc[out["group_value"].astype("string").str.strip() == "", "group_value"] = "All"

    # Enforce dtypes
    out["year"] = out["year"].fillna(0).astype("int16")
    out["n"] = out["n"].fillna(0).astype("int32")
    for c in ["positive_pct", "neutral_pct", "negative_pct",
              "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
        out[c] = out[c].astype("float32")

    # Final column order
    out = out[OUT_COLS]
    return out


# ── PUBLIC API ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz from Google Drive…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
    """Ensure the gzipped CSV exists locally at /tmp and return its path."""
    file_id = file_id or st.secrets.get("RESULTS2024_FILE_ID", GDRIVE_FILE_ID_FALLBACK)
    if not file_id:
        raise RuntimeError("RESULTS2024_FILE_ID missing in .streamlit/secrets.toml")

    if os.path.exists(LOCAL_GZ_PATH) and os.path.getsize(LOCAL_GZ_PATH) > 0:
        return LOCAL_GZ_PATH

    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(LOCAL_GZ_PATH), exist_ok=True)
    gdown.download(url, LOCAL_GZ_PATH, quiet=False)

    if not os.path.exists(LOCAL_GZ_PATH) or os.path.getsize(LOCAL_GZ_PATH) == 0:
        raise RuntimeError("Download failed or produced an empty file.")
    return LOCAL_GZ_PATH


@st.cache_data(show_spinner="🔎 Filtering 2024 results in chunks…")
def load_results2024_filtered(
    question_code: str,
    years: List[int],
    group_value: Optional[str] = None,   # DEMCODE; pass None or "All" for overall PS-wide
    chunksize: int = 750_000,
) -> pd.DataFrame:
    """
    Stream-filter rows by QUESTION (e.g., "Q01"), SURVEYR (e.g., [2024]),
    and optional DEMCODE (overall when None/"All"). Returns OUT_COLS.
    """
    path = ensure_results2024_local()

    # Determine which columns are actually present (fast header read)
    header_df = pd.read_csv(path, compression="gzip", nrows=0)
    usecols = [c for c in NATIVE_USECOLS if c in header_df.columns]

    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=True,
    ):
        norm = _normalize_native_chunk(chunk)

        mask = (norm["question_code"] == question_code) & (norm["year"].isin(years))
        # Only filter by DEMCODE when a specific code is provided (not "All"/None)
        if group_value and group_value != "All":
            mask &= (norm["group_value"] == group_value)

        if mask.any():
            frames.append(norm.loc[mask, OUT_COLS])

    if not frames:
        # Return an empty frame with the right schema/dtypes
        empty = pd.DataFrame(columns=OUT_COLS)
        for c, dt in DTYPES.items():
            if c in empty.columns:
                empty[c] = empty[c].astype(dt)
        return empty

    out = pd.concat(frames, ignore_index=True)
    # Defensive re-typing
    for c, dt in DTYPES.items():
        if c in out.columns:
            try:
                out[c] = out[c].astype(dt)
            except Exception:
                pass
    return out
