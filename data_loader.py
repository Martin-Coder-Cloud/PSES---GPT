# utils/data_loader.py — Parquet-first loader with pushdown filters
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import streamlit as st

# Optional accelerators
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"  # your Drive id
LOCAL_GZ_PATH   = "/tmp/Results2024.csv.gz"
PARQUET_ROOTDIR = "/tmp/PSES_Results2024_parquet"   # directory dataset (partitioned)
PARQUET_FLAG    = os.path.join(PARQUET_ROOTDIR, "_BUILD_OK")

# We keep your normalized output schema
OUT_COLS = [
    "year", "question_code", "group_value", "n",
    "positive_pct", "neutral_pct", "negative_pct",
    "answer1","answer2","answer3","answer4","answer5","answer6","answer7",
]

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

# ── CSV download (unchanged behavior) ─────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz from Google Drive…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
    import gdown
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

# ── One-time CSV → Parquet dataset (partitioned) ─────────────────────────────
def _build_parquet_with_duckdb(csv_path: str) -> None:
    import duckdb
    con = duckdb.connect()
    # Fast, vectorized read; compute normalized columns once; write partitioned Parquet
    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    con.execute("""
        CREATE OR REPLACE TABLE pses AS
        SELECT
          CAST(SURVEYR AS INT)                                 AS year,
          CAST(QUESTION AS VARCHAR)                            AS question_code,
          COALESCE(NULLIF(TRIM(CAST(DEMCODE AS VARCHAR)), ''),'All') AS group_value,
          CAST(ANSCOUNT AS INT)                                AS n,
          CAST(POSITIVE AS DOUBLE)                              AS positive_pct,
          CAST(NEUTRAL  AS DOUBLE)                              AS neutral_pct,
          CAST(NEGATIVE AS DOUBLE)                              AS negative_pct,
          CAST(answer1  AS DOUBLE) AS answer1,
          CAST(answer2  AS DOUBLE) AS answer2,
          CAST(answer3  AS DOUBLE) AS answer3,
          CAST(answer4  AS DOUBLE) AS answer4,
          CAST(answer5  AS DOUBLE) AS answer5,
          CAST(answer6  AS DOUBLE) AS answer6,
          CAST(answer7  AS DOUBLE) AS answer7
        FROM read_csv_auto(?, header=true)
    """, [csv_path])
    # Partition by year & question for aggressive pruning
    con.execute(f"""
        COPY pses TO '{PARQUET_ROOTDIR}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 1000000,
         PARTITION_BY (year, question_code));
    """)
    open(PARQUET_FLAG, "w").close()

def _build_parquet_with_pandas(csv_path: str) -> None:
    # Fallback if duckdb isn't available. We read once and write a dataset.
    import pyarrow as pa
    import pyarrow.parquet as pq

    usecols = [
        "SURVEYR","QUESTION","DEMCODE",
        "ANSCOUNT","POSITIVE","NEUTRAL","NEGATIVE",
        "answer1","answer2","answer3","answer4","answer5","answer6","answer7",
    ]
    df = pd.read_csv(csv_path, compression="gzip", usecols=usecols, low_memory=False)

    # Normalize to your target schema
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
        compression="zstd"
    )
    open(PARQUET_FLAG, "w").close()

@st.cache_resource(show_spinner="🗂️ Preparing Parquet dataset…")
def ensure_parquet_dataset() -> str:
    """
    Returns the root directory of the Parquet dataset, building it once if needed.
    """
    if not _pyarrow_available():
        raise RuntimeError("pyarrow is required for Parquet. Add `pyarrow` to requirements.txt.")
    csv_path = ensure_results2024_local()
    # Build only once (or if Parquet dir missing)
    if os.path.isdir(PARQUET_ROOTDIR) and os.path.exists(PARQUET_FLAG):
        return PARQUET_ROOTDIR
    # (Re)build
    if _duckdb_available():
        _build_parquet_with_duckdb(csv_path)
    else:
        _build_parquet_with_pandas(csv_path)
    return PARQUET_ROOTDIR

# ── Public API (same signature) ───────────────────────────────────────────────
@st.cache_data(show_spinner="🔎 Filtering Parquet with pushdown…")
def load_results2024_filtered(
    question_code: str,
    years: List[int] | List[str],
    group_value: Optional[str] = None,   # None or "All" => overall PS-wide
) -> pd.DataFrame:
    """
    Fast path: Query the Parquet dataset with predicate pushdown and
    return the normalized OUT_COLS with friendly dtypes.
    Falls back to the old CSV scanner only if Parquet is unavailable.
    """
    # Prefer Parquet
    if _pyarrow_available():
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        root = ensure_parquet_dataset()
        dataset = ds.dataset(root, format="parquet")

        # Normalize inputs
        q = str(question_code).strip()
        years_int = [int(y) for y in years]
        overall = (group_value is None) or (str(group_value).strip() == "") or (str(group_value).strip().lower() == "all")

        # Build filters: question_code == q AND year IN years AND (group_value == code OR == "All")
        filt = (pc.field("question_code") == q) & (pc.field("year").isin(years_int))
        if overall:
            filt = filt & (pc.field("group_value") == "All")
        else:
            filt = filt & (pc.field("group_value") == str(group_value).strip())

        # Read only needed columns
        cols = OUT_COLS
        tbl = dataset.to_table(columns=cols, filter=filt)
        df = tbl.to_pandas(types_mapper=pd.ArrowDtype)

        # Cast to UI-friendly dtypes
        # (Keep everything as numeric where appropriate, strings for codes)
        df = df.reindex(columns=OUT_COLS)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
        df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype("Int32")
        for c in ["positive_pct","neutral_pct","negative_pct",
                  "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float32")
        df["question_code"] = df["question_code"].astype("string")
        df["group_value"]   = df["group_value"].astype("string")
        return df

    # Fallback (rare): original CSV scan in chunks — slower
    return _csv_stream_filter(question_code, years, group_value)

# ── Legacy CSV chunk scanner (fallback only) ──────────────────────────────────
def _csv_stream_filter(
    question_code: str,
    years: List[int] | List[str],
    group_value: Optional[str] = None,
    chunksize: int = 1_500_000,
) -> pd.DataFrame:
    path = ensure_results2024_local()
    usecols = [
        "SURVEYR","QUESTION","DEMCODE",
        "ANSCOUNT","POSITIVE","NEUTRAL","NEGATIVE",
        "answer1","answer2","answer3","answer4","answer5","answer6","answer7",
    ]
    frames: list[pd.DataFrame] = []
    years_int = [int(y) for y in years]
    overall = (group_value is None) or (str(group_value).strip() == "") or (str(group_value).strip().lower() == "all")

    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=chunksize, low_memory=True):
        # Fast boolean mask in pandas
        mask = (chunk["QUESTION"].astype(str) == question_code) & (pd.to_numeric(chunk["SURVEYR"], errors="coerce").isin(years_int))
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
                "neutral_pct":   pd.to_numeric(sel["NEUTRAL"], errors="coerce"),
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
    for c, dt in DTYPES.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                pass
    return df[OUT_COLS]
