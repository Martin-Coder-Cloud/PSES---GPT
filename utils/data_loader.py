# utils/data_loader.py
import os
import pandas as pd
import streamlit as st
import gdown

# Local cache path
LOCAL_GZ = "/tmp/Results2024.csv.gz"

# Required columns for Menu 1
REQ = ["year","question_code","group","group_value","n","positive_pct"]

# Memory-friendly dtypes
DTYPES = {
    "year": "int16",
    "question_code": "string",
    "group": "string",
    "group_value": "string",
    "n": "int32",
    "positive_pct": "float32",
}

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    if "group" in df.columns:
        df["group"] = df["group"].fillna("All respondents")
    if "group_value" in df.columns:
        df["group_value"] = df["group_value"].fillna("All")
    # apply dtypes softly
    for c, dt in DTYPES.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                pass
    return df

@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz from Google Drive…")
def ensure_local() -> str:
    file_id = st.secrets.get("RESULTS2024_FILE_ID")
    if not file_id:
        raise RuntimeError("RESULTS2024_FILE_ID missing in .streamlit/secrets.toml")
    if os.path.exists(LOCAL_GZ) and os.path.getsize(LOCAL_GZ) > 0:
        return LOCAL_GZ
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, LOCAL_GZ, quiet=False)
    if not os.path.exists(LOCAL_GZ) or os.path.getsize(LOCAL_GZ) == 0:
        raise RuntimeError("Download failed or produced an empty file.")
    return LOCAL_GZ

@st.cache_data(show_spinner="🔎 Filtering 2024 results in chunks…")
def load_results2024_filtered(
    question_code: str,
    years: list[int],
    group: str | None = None,
    group_value: str | None = None,
    chunksize: int = 750_000,
) -> pd.DataFrame:
    """
    Stream/filter rows from Results2024.csv.gz by question/year/(optional) demographic.
    Returns only REQ columns with stable dtypes.
    """
    path = ensure_local()
    frames = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=REQ,        # widen later if needed
        dtype=DTYPES,
        chunksize=chunksize,
        low_memory=True,
    ):
        chunk = _normalize_cols(chunk)
        mask = (chunk["question_code"] == question_code) & (chunk["year"].isin(years))
        if group:
            mask &= (chunk["group"] == group)
        if group_value:
            mask &= (chunk["group_value"] == group_value)
        if mask.any():
            frames.append(chunk.loc[mask, REQ])

    if not frames:
        # return an empty frame with the right schema
        empty = pd.DataFrame(columns=REQ)
        for c, dt in DTYPES.items():
            empty[c] = empty.get(c, pd.Series(dtype=dt))
        return empty

    out = pd.concat(frames, ignore_index=True)
    # enforce dtypes again (cheap, defensive)
    for c, dt in DTYPES.items():
        if c in out.columns:
            try:
                out[c] = out[c].astype(dt)
            except Exception:
                pass
    return out

