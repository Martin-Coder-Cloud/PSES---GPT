import os
import pandas as pd
import streamlit as st
import gdown

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Set in .streamlit/secrets.toml (you already have this file id):
# RESULTS2024_FILE_ID = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
LOCAL_GZ_PATH = "/tmp/Results2024.csv.gz"

# Your file’s native columns (exact spellings from the CSV)
NATIVE_USECOLS = [
    "SURVEYR","QUESTION","BYCOND","DEMCODE",
    "POSITIVE","NEUTRAL","NEGATIVE","AGREE",
    "SCORE5","SCORE100","ANSCOUNT",
    # keep if you might need later:
    "LEVEL1ID","LEVEL2ID","LEVEL3ID","LEVEL4ID","LEVEL5ID",
    "answer1","answer2","answer3","answer4","answer5","answer6","answer7"
]

# What the app expects after normalization
REQ_COLS = ["year","question_code","group","group_value","n","positive_pct"]

# Memory-friendly dtypes for the normalized frame
DTYPES = {
    "year": "int16",
    "question_code": "string",
    "group": "string",
    "group_value": "string",
    "n": "int32",
    "positive_pct": "float32",
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def _normalize_native_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Take a chunk with native column names and create the normalized columns:
      year, question_code, group, group_value, n, positive_pct
    """
    # Ensure expected native columns exist (case-sensitive)
    missing = [c for c in ["SURVEYR","QUESTION","BYCOND","DEMCODE","POSITIVE","ANSCOUNT"] if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing expected columns in chunk: {', '.join(missing)}")

    out = pd.DataFrame({
        "year":         chunk["SURVEYR"],
        "question_code":chunk["QUESTION"],
        "group":        chunk["BYCOND"].fillna("All respondents"),
        "group_value":  chunk["DEMCODE"].fillna("All"),
        "n":            chunk["ANSCOUNT"],
        "positive_pct": chunk["POSITIVE"],
    })

    # Dtypes
    try: out["year"] = out["year"].astype("int16")
    except: out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype("int16")

    out["question_code"] = out["question_code"].astype("string")
    out["group"] = out["group"].astype("string")
    out["group_value"] = out["group_value"].astype("string")

    try: out["n"] = out["n"].fillna(0).astype("int32")
    except: out["n"] = pd.to_numeric(out["n"], errors="coerce").fillna(0).astype("int32")

    try: out["positive_pct"] = out["positive_pct"].astype("float32")
    except: out["positive_pct"] = pd.to_numeric(out["positive_pct"], errors="coerce").astype("float32")

    return out


# ── PUBLIC API ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz from Google Drive…")
def ensure_results2024_local() -> str:
    file_id = st.secrets.get("RESULTS2024_FILE_ID", GDRIVE_FILE_ID_FALLBACK)
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
    years: list[int],
    group: str | None = None,
    group_value: str | None = None,
    chunksize: int = 750_000,
) -> pd.DataFrame:
    """
    Stream-filter rows from Results2024.csv.gz using your native schema.
    Returns a tidy frame with columns: year, question_code, group, group_value, n, positive_pct
    """
    path = ensure_results2024_local()
    frames = []

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=NATIVE_USECOLS,  # read only what we need
        chunksize=chunksize,
        low_memory=True,
    ):
        # Normalize this native chunk to our standard schema
        norm = _normalize_native_chunk(chunk)

        # Build mask
        mask = (norm["question_code"] == question_code) & (norm["year"].isin(years))
        if group:
            mask &= (norm["group"] == group)
        if group_value:
            mask &= (norm["group_value"] == group_value)

        if mask.any():
            frames.append(norm.loc[mask, REQ_COLS])

    if not frames:
        # empty with correct schema/dtypes
        empty = pd.DataFrame(columns=REQ_COLS)
        for c, dt in DTYPES.items():
            empty[c] = empty.get(c, pd.Series(dtype=dt))
        return empty

    out = pd.concat(frames, ignore_index=True)
    # Defensive re-typing
    for c, dt in DTYPES.items():
        if c in out.columns:
            try: out[c] = out[c].astype(dt)
            except Exception: pass
    return out
