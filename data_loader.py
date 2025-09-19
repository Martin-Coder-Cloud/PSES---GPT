# utils/data_loader.py — Parquet-first loader with CSV fallback
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

# Google Drive CSV (you already use this)
GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
LOCAL_GZ_PATH = os.environ.get("PSES_RESULTS_GZ", "/tmp/Results2024.csv.gz")

# Parquet dataset location (directory). Prefer a persistent folder.
PARQUET_ROOTDIR = os.environ.get("PSES_PARQUET_DIR", "data/parquet/PSES_Results2024")
PARQUET_FLAG = os.path.join(PARQUET_ROOTDIR, "_BUILD_OK")

# Output schema (normalized)
OUT_COLS = [
    "year", "question_code", "group_value", "n",
    "positive_pct", "neutral_pct", "negative_pct",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]

DTYPES = {
    "year": "Int16",
    "question_code": "string",
    "group_value": "string",
    "n": "Int32",
    "positive_pct": "Float32",
    "neutral_pct": "Float32",
    "negative_pct": "Float32",
    "answer1": "Float32", "answer2": "Float32", "answer3": "Float32",
    "answer4": "Float32", "answer5": "Float32", "answer6": "Float32", "answer7": "Float32",
}

# Minimal column set to read from CSV
CSV_USECOLS = [
    "SURVEYR", "QUESTION", "DEMCODE",
    "ANSCOUNT", "POSITIVE", "NEUTRAL", "NEGATIVE",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]


# =============================================================================
# Small capability checks
# =============================================================================
def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False

def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.dataset as ds  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        return True
    except Exception:
        return False


# =============================================================================
# Download the CSV (cached)
# =============================================================================
@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
    import gdown
    file_id = file_id or st.secrets.get("RESULTS2024_FILE_ID", GDRIVE_FILE_ID_FALLBACK)
    if not file_id:
        raise RuntimeError("RESULTS2024_FILE_ID missing in .streamlit/secrets.toml")

    # If present, reuse
    if os.path.exists(LOCAL_GZ_PATH) and os.path.getsize(LOCAL_GZ_PATH) > 0:
        return LOCAL_GZ_PATH

    os.makedirs(os.path.dirname(LOCAL_GZ_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, LOCAL_GZ_PATH, quiet=False)
    if not os.path.exists(LOCAL_GZ_PATH) or os.path.getsize(LOCAL_GZ_PATH) == 0:
        raise RuntimeError("Download failed or produced an empty file.")
    return LOCAL_GZ_PATH


# =============================================================================
# Build Parquet dataset once (preferred fast path)
# =============================================================================
def _build_parquet_with_duckdb(csv_path: str) -> None:
    import duckdb
    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    con = duckdb.connect()
    # One pass over CSV, normalize schema, write partitioned Parquet
    con.execute("""
        CREATE OR REPLACE TABLE pses AS
        SELECT
          CAST(SURVEYR AS INT)                                 AS year,
          CAST(QUESTION AS VARCHAR)                            AS question_code,
          COALESCE(NULLIF(TRIM(CAST(DEMCODE AS VARCHAR)), ''),'All') AS group_value,
          CAST(ANSCOUNT AS INT)                                AS n,
          CAST(POSITIVE AS DOUBLE)                             AS positive_pct,
          CAST(NEUTRAL  AS DOUBLE)                             AS neutral_pct,
          CAST(NEGATIVE AS DOUBLE)                             AS negative_pct,
          CAST(answer1  AS DOUBLE) AS answer1,
          CAST(answer2  AS DOUBLE) AS answer2,
          CAST(answer3  AS DOUBLE) AS answer3,
          CAST(answer4  AS DOUBLE) AS answer4,
          CAST(answer5  AS DOUBLE) AS answer5,
          CAST(answer6  AS DOUBLE) AS answer6,
          CAST(answer7  AS DOUBLE) AS answer7
        FROM read_csv_auto(?, header=true)
    """, [csv_path])

    con.execute(f"""
        COPY pses TO '{PARQUET_ROOTDIR}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 1000000,
         PARTITION_BY (year, question_code));
    """)

def _build_parquet_with_pandas(csv_path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.read_csv(csv_path, compression="gzip", usecols=CSV_USECOLS, low_memory=False)

    # Normalize to OUT_COLS
    out = pd.DataFrame({
        "year":          pd.to_numeric(df["SURVEYR"], errors="coerce").astype("Int64"),
        "question_code": df["QUESTION"].astype("string"),
        "group_value":   df["DEMCODE"].astype("string"),
        "n":             pd.to_numeric(df["ANSCOUNT"], errors="coerce").astype("Int64"),
        "positive_pct":  pd.to_numeric(df["POSITIVE"], errors="coerce"),
        "neutral_pct":   pd.to_numeric(df["NEUTRAL"],  errors="coerce"),
        "negative_pct":  pd.to_numeric(df["NEGATIVE"], errors="coerce"),
        "answer1": pd.to_numeric(df.get("answer1"), errors="coerce"),
        "answer2": pd.to_numeric(df.get("answer2"), errors="coerce"),
        "answer3": pd.to_numeric(df.get("answer3"), errors="coerce"),
        "answer4": pd.to_numeric(df.get("answer4"), errors="coerce"),
        "answer5": pd.to_numeric(df.get("answer5"), errors="coerce"),
        "answer6": pd.to_numeric(df.get("answer6"), errors="coerce"),
        "answer7": pd.to_numeric(df.get("answer7"), errors="coerce"),
    })
    out["group_value"] = out["group_value"].fillna("All")
    out.loc[out["group_value"].astype("string").str.strip() == "", "group_value"] = "All"

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    table = pa.Table.from_pandas(out[OUT_COLS], preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=PARQUET_ROOTDIR,
        partition_cols=["year", "question_code"],
        compression="zstd",
    )

@st.cache_resource(show_spinner="🗂️ Preparing Parquet dataset (one-time)…")
def ensure_parquet_dataset() -> str:
    """
    Ensures a partitioned Parquet dataset exists and returns its root directory.
    """
    if not _pyarrow_available():
        raise RuntimeError("pyarrow is required for Parquet fast path.")
    csv_path = ensure_results2024_local()

    if os.path.isdir(PARQUET_ROOTDIR) and os.path.exists(PARQUET_FLAG):
        return PARQUET_ROOTDIR

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)

    # Build with DuckDB if possible (fastest), else Pandas+PyArrow
    if _duckdb_available():
        _build_parquet_with_duckdb(csv_path)
    else:
        _build_parquet_with_pandas(csv_path)

    # Mark as ready
    with open(PARQUET_FLAG, "w") as f:
        f.write("ok")
    return PARQUET_ROOTDIR


# =============================================================================
# Fast Parquet query
# =============================================================================
def _parquet_query(question_code: str, years: Iterable[int | str], group_value: Optional[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    root = ensure_parquet_dataset()
    dataset = ds.dataset(root, format="parquet")

    q = str(question_code).strip()
    years_int = [int(y) for y in years]
    overall = (group_value is None) or (str(group_value).strip() == "") or (str(group_value).strip().lower() == "all")

    filt = (pc.field("question_code") == q) & (pc.field("year").isin(years_int))
    if overall:
        filt = filt & (pc.field("group_value") == "All")
    else:
        filt = filt & (pc.field("group_value") == str(group_value).strip())

    cols = OUT_COLS
    tbl = dataset.to_table(columns=cols, filter=filt)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)

    # Cast to friendly dtypes
    df = df.reindex(columns=OUT_COLS)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
    df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
    for c in ["positive_pct","neutral_pct","negative_pct",
              "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])
    df["question_code"] = df["question_code"].astype(DTYPES["question_code"])
    df["group_value"]   = df["group_value"].astype(DTYPES["group_value"])
    return df


# =============================================================================
# Legacy CSV chunk scanner (fallback)
# =============================================================================
def _csv_stream_filter(
    question_code: str,
    years: Iterable[int | str],
    group_value: Optional[str],
    chunksize: int = 1_500_000,
) -> pd.DataFrame:
    path = ensure_results2024_local()
    years_int = [int(y) for y in years]
    overall = (group_value is None) or (str(group_value).strip() == "") or (str(group_value).strip().lower() == "all")

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", usecols=CSV_USECOLS, chunksize=chunksize, low_memory=True):
        mask = (chunk["QUESTION"].astype(str) == question_code) & \
               (pd.to_numeric(chunk["SURVEYR"], errors="coerce").isin(years_int))
        if overall:
            gv = chunk["DEMCODE"].astype(str).str.strip()
            mask &= (gv.eq("")) | (gv.isna())
        else:
            mask &= (chunk["DEMCODE"].astype(str).str.strip() == str(group_value).strip())

        if mask.any():
            sel = chunk.loc[mask, :]
            out = pd.DataFrame({
                "year":          pd.to_numeric(sel["SURVEYR"], errors="coerce"),
                "question_code": sel["QUESTION"].astype("string"),
                "group_value":   sel["DEMCODE"].astype("string").fillna("All"),
                "n":             pd.to_numeric(sel["ANSCOUNT"], errors="coerce"),
                "positive_pct":  pd.to_numeric(sel["POSITIVE"], errors="coerce"),
                "neutral_pct":   pd.to_numeric(sel["NEUTRAL"],  errors="coerce"),
                "negative_pct":  pd.to_numeric(sel["NEGATIVE"], errors="coerce"),
                "answer1": pd.to_numeric(sel.get("answer1"), errors="coerce"),
                "answer2": pd.to_numeric(sel.get("answer2"), errors="coerce"),
                "answer3": pd.to_numeric(sel.get("answer3"), errors="coerce"),
                "answer4": pd.to_numeric(sel.get("answer4"), errors="coerce"),
                "answer5": pd.to_numeric(sel.get("answer5"), errors="coerce"),
                "answer6": pd.to_numeric(sel.get("answer6"), errors="coerce"),
                "answer7": pd.to_numeric(sel.get("answer7"), errors="coerce"),
            })
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.concat(frames, ignore_index=True)
    # Soft cast to UI-friendly dtypes
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
    df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
    for c in ["positive_pct","neutral_pct","negative_pct",
              "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])
    df["question_code"] = df["question_code"].astype(DTYPES["question_code"])
    df["group_value"]   = df["group_value"].astype(DTYPES["group_value"])
    return df[OUT_COLS]


# =============================================================================
# Public API (unchanged signature)
# =============================================================================
@st.cache_data(show_spinner="🔎 Filtering results…")
def load_results2024_filtered(
    question_code: str,
    years: Iterable[int | str],
    group_value: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a filtered slice at (question_code, years, group_value) grain.
    Prefers Parquet pushdown; falls back to CSV chunk scan.
    """
    # Try Parquet fast path
    try:
        if _pyarrow_available():
            return _parquet_query(question_code, years, group_value)
    except Exception:
        # Silent fallback to CSV; you can add st.warning here if you prefer.
        pass

    # CSV fallback
    return _csv_stream_filter(question_code, years, group_value)


# =============================================================================
# Optional helpers (for diagnostics / prewarm)
# =============================================================================
def get_backend_info() -> dict:
    """Lightweight indicator for UI captions."""
    return {
        "parquet_dir_exists": os.path.isdir(PARQUET_ROOTDIR),
        "parquet_ready": os.path.exists(PARQUET_FLAG),
        "parquet_dir": PARQUET_ROOTDIR,
        "csv_path": LOCAL_GZ_PATH,
    }

@st.cache_resource(show_spinner="⚡ Warming up data backend…")
def prewarm_fastpath() -> str:
    """
    Ensure CSV is present and Parquet dataset is built (one-time).
    Call this from your main page to pre-build before a user opens Menu 1.
    """
    ensure_results2024_local()
    try:
        return ensure_parquet_dataset()
    except Exception:
        # If pyarrow or duckdb is missing at runtime, we still prefetch CSV for faster fallback.
        return "csv"
