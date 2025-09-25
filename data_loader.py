# utils/data_loader.py — unified loader (metadata + PS-wide data, preloaded)
from __future__ import annotations

import os
import csv
import time
from typing import Iterable, Optional, Dict, Any, List, Tuple

import pandas as pd
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

# ---- Memory budget for in-memory dataframe (GB) ------------------------------
# Heuristic: if estimated in-memory size of full PS-wide parquet > budget,
# we preload only the *latest year* into RAM.
MEMORY_BUDGET_GB = float(os.environ.get("PSES_INMEM_BUDGET_GB", "3.0"))

# ---- Data locations ----------------------------------------------------------
# Google Drive CSV.gz (set real ID in Streamlit secrets as RESULTS2024_FILE_ID)
GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_uPIGvc8"  # placeholder
LOCAL_GZ_PATH = os.environ.get("PSES_RESULTS_GZ", "/tmp/Results2024.csv.gz")

# Parquet dataset (PS-wide only) — persistent folder
PARQUET_ROOTDIR = os.environ.get("PSES_PARQUET_DIR", "data/parquet/PSES_Results2024_PSWIDE")
PARQUET_FLAG = os.path.join(PARQUET_ROOTDIR, "_BUILD_OK")

# Strict metadata paths (repo-only)
META_DIR = "metadata"
QUESTIONS_PATHS = [os.path.join(META_DIR, "Survey Questions.xlsx"),
                   os.path.join(META_DIR, "Survey Questions.xls")]
DEMOGRAPHICS_PATH = os.path.join(META_DIR, "Demographics.xlsx")
SCALES_PATH = os.path.join(META_DIR, "Survey Scales.xlsx")

# ---- Output schema (normalized) ----------------------------------------------
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

# Minimal CSV read for building/query fallback (include LEVEL1ID for PS-wide)
CSV_USECOLS = [
    "LEVEL1ID",
    "SURVEYR", "QUESTION", "DEMCODE",
    "ANSCOUNT", "POSITIVE", "NEUTRAL", "NEGATIVE",
    "answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7",
]

# =============================================================================
# Internal diagnostics
# =============================================================================
_LAST_DIAG: dict = {}
_LAST_ENGINE: str = "unknown"

def _set_diag(**kwargs):
    _LAST_DIAG.clear()
    _LAST_DIAG.update(kwargs)

def get_last_query_diag() -> dict:
    """Diagnostics for the most recent load_results2024_filtered call."""
    return dict(_LAST_DIAG)

# =============================================================================
# Capability checks
# =============================================================================
def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa
        return True
    except Exception:
        return False

def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa
        import pyarrow.dataset as ds  # noqa
        import pyarrow.parquet as pq  # noqa
        return True
    except Exception:
        return False

# =============================================================================
# Metadata loaders (repo-only; strict)
# =============================================================================
@st.cache_data(show_spinner=False)
def _load_questions_meta() -> pd.DataFrame:
    path = next((p for p in QUESTIONS_PATHS if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("Missing `metadata/Survey Questions.xlsx` (or `.xls`).")
    df = pd.read_excel(path, dtype=str, engine=None)
    cols = {c.lower().strip(): c for c in df.columns}
    if "question" not in cols or "english" not in cols:
        raise ValueError("`Survey Questions` must have columns 'Question' and 'English'.")
    df = df.rename(columns={cols["question"]: "code", cols["english"]: "text"})
    df["code"] = df["code"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["qnum"] = pd.to_numeric(df["code"].str.extract(r"(\d+)", expand=False), errors="coerce")
    df["display"] = df["code"] + " – " + df["text"]
    return df[["code", "text", "display", "qnum"]].sort_values(["qnum", "code"], na_position="last").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def _load_demographics_meta() -> pd.DataFrame:
    if not os.path.exists(DEMOGRAPHICS_PATH):
        raise FileNotFoundError("Missing `metadata/Demographics.xlsx`.")
    df = pd.read_excel(DEMOGRAPHICS_PATH, dtype=str, engine=None)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def _load_scales_meta() -> pd.DataFrame:
    if not os.path.exists(SCALES_PATH):
        raise FileNotFoundError("Missing `metadata/Survey Scales.xlsx`.")
    sdf = pd.read_excel(SCALES_PATH, dtype=str, engine=None)
    sdf.columns = sdf.columns.str.strip().str.lower()
    code_col = "code" if "code" in sdf.columns else ("question" if "question" in sdf.columns else None)
    if code_col is None:
        raise ValueError("`Survey Scales.xlsx` must include a 'code' or 'question' column.")
    def _norm(s: str) -> str:
        s = "" if s is None else str(s)
        return "".join(ch for ch in s.upper() if ch.isalnum())
    sdf["__code_norm__"] = sdf[code_col].astype(str).map(_norm)
    return sdf

# Public getters for metadata
def get_questions() -> pd.DataFrame:   return _load_questions_meta()
def get_demographics() -> pd.DataFrame: return _load_demographics_meta()
def get_scales() -> pd.DataFrame:      return _load_scales_meta()

# =============================================================================
# CSV presence (download)
# =============================================================================
@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
    import gdown
    file_id = file_id or st.secrets.get("RESULTS2024_FILE_ID", GDRIVE_FILE_ID_FALLBACK)
    if not file_id:
        raise RuntimeError("RESULTS2024_FILE_ID missing in .streamlit/secrets.toml")

    if os.path.exists(LOCAL_GZ_PATH) and os.path.getsize(LOCAL_GZ_PATH) > 0:
        return LOCAL_GZ_PATH

    os.makedirs(os.path.dirname(LOCAL_GZ_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, LOCAL_GZ_PATH, quiet=False)
    if not os.path.exists(LOCAL_GZ_PATH) or os.path.getsize(LOCAL_GZ_PATH) == 0:
        raise RuntimeError("Download failed or produced an empty file.")
    return LOCAL_GZ_PATH

# =============================================================================
# Build Parquet (PS-wide only: LEVEL1ID=0)
# =============================================================================
def _build_parquet_with_duckdb(csv_path: str) -> None:
    import duckdb
    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    con = duckdb.connect()
    con.execute("""
        CREATE OR REPLACE TABLE pses AS
        SELECT
          CAST(SURVEYR AS INT)                                  AS year,
          CAST(QUESTION AS VARCHAR)                             AS question_code,
          COALESCE(NULLIF(TRIM(CAST(DEMCODE AS VARCHAR)), ''),'All') AS group_value,
          CAST(ANSCOUNT AS INT)                                 AS n,
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
        WHERE CAST(LEVEL1ID AS BIGINT) = 0  -- PS-wide only
    """, [csv_path])

    con.execute(f"""
        COPY pses TO '{PARQUET_ROOTDIR}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 1000000,
         PARTITION_BY (year, question_code));
    """)
    con.close()

def _build_parquet_with_pandas(csv_path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.read_csv(csv_path, compression="gzip", usecols=CSV_USECOLS, low_memory=False)

    # PS-wide filter first
    lvl = pd.to_numeric(df["LEVEL1ID"], errors="coerce").fillna(1).astype("Int64")
    df = df.loc[lvl.eq(0)].copy()

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

@st.cache_resource(show_spinner="🗂️ Preparing PS-wide Parquet (one-time)…")
def ensure_parquet_dataset() -> str:
    """Ensures a partitioned Parquet dataset (PS-wide only) exists and returns its root directory."""
    if not _pyarrow_available():
        raise RuntimeError("pyarrow is required for Parquet fast path.")
    csv_path = ensure_results2024_local()

    if os.path.isdir(PARQUET_ROOTDIR) and os.path.exists(PARQUET_FLAG):
        return PARQUET_ROOTDIR

    os.makedirs(PARQUET_ROOTDIR, exist_ok=True)
    if _duckdb_available():
        _build_parquet_with_duckdb(csv_path)
    else:
        _build_parquet_with_pandas(csv_path)

    with open(PARQUET_FLAG, "w") as f:
        f.write("ok")
    return PARQUET_ROOTDIR

# =============================================================================
# In-memory preload (all years if budget allows; else latest year)
# =============================================================================
_INMEM_DF: Optional[pd.DataFrame] = None
_INMEM_MODE: str = "none"     # "all" | "latest_year" | "none"

def _estimate_parquet_size_bytes(root: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".parquet"):
                total += os.path.getsize(os.path.join(base, fn))
    return total

@st.cache_resource(show_spinner="💾 Loading filtered database into memory…")
def _load_inmemory_df() -> Tuple[pd.DataFrame, str]:
    """
    Returns (df, mode). Mode is 'all' if we loaded all years; 'latest_year' otherwise.
    """
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    root = ensure_parquet_dataset()
    dataset = ds.dataset(root, format="parquet")

    # Heuristic: if compressed Parquet size * 2.5 < budget → load ALL
    parquet_bytes = _estimate_parquet_size_bytes(root)
    est_inmem_bytes = int(parquet_bytes * 2.5)  # rough inflation factor
    budget_bytes = int(MEMORY_BUDGET_GB * (1024**3))

    cols = OUT_COLS

    if est_inmem_bytes <= budget_bytes:
        table = dataset.to_table(columns=cols)
        df = table.to_pandas().astype({k: v for k, v in DTYPES.items() if k in cols})
        return df, "all"

    # Else: only latest year
    years_tbl = dataset.to_table(columns=["year"])
    years = pd.to_numeric(pd.Series(years_tbl.column("year").to_pylist()), errors="coerce").dropna().astype(int)
    if years.empty:
        return pd.DataFrame(columns=OUT_COLS), "none"
    latest = int(years.max())

    filt = (pc.field("year") == latest)
    table = dataset.to_table(columns=cols, filter=filt)
    df = table.to_pandas().astype({k: v for k, v in DTYPES.items() if k in cols})
    return df, "latest_year"

def _ensure_inmemory_ready():
    global _INMEM_DF, _INMEM_MODE
    if _INMEM_DF is None:
        df, mode = _load_inmemory_df()
        _INMEM_DF, _INMEM_MODE = df, mode

# =============================================================================
# Query helpers (Parquet + CSV fallbacks)
# =============================================================================
def _parquet_query(
    question_code: str,
    years: Iterable[int | str],
    group_values: Optional[List[Optional[str]]],
) -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    root = ensure_parquet_dataset()
    dataset = ds.dataset(root, format="parquet")

    q = str(question_code).strip()
    years_int = [int(y) for y in years]

    # Build filter
    filt = (pc.field("question_code") == q) & (pc.field("year").isin(years_int))
    if group_values is not None:
        want_overall = any(g is None for g in group_values)
        gvals = [str(g).strip() for g in group_values if g is not None]
        clauses = []
        if gvals:
            clauses.append(pc.field("group_value").isin(gvals))
        if want_overall:
            clauses.append(pc.field("group_value") == "All")
        if clauses:
            sub = clauses[0]
            for c in clauses[1:]:
                sub = pc.or_(sub, c)
            filt = pc.and_(filt, sub)
    else:
        # Default to overall when nothing is provided (matches previous behavior)
        filt = pc.and_(filt, pc.field("group_value") == "All")

    tbl = dataset.to_table(columns=OUT_COLS, filter=filt)
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

def _csv_stream_filter(
    question_code: str,
    years: Iterable[int | str],
    group_values: Optional[List[Optional[str]]],
    chunksize: int = 1_500_000,
) -> pd.DataFrame:
    path = ensure_results2024_local()
    years_int = [int(y) for y in years]

    want_overall = False
    gvals: List[str] = []
    if group_values is not None:
        want_overall = any(g is None for g in group_values)
        gvals = [str(g).strip() for g in group_values if g is not None]
    else:
        want_overall = True  # default overall when nothing provided

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", usecols=CSV_USECOLS, chunksize=chunksize, low_memory=True):
        # PS-wide filter first
        lvl = pd.to_numeric(chunk["LEVEL1ID"], errors="coerce").fillna(1).astype("Int64")
        mask = lvl.eq(0)

        # Question + year filters
        mask &= (chunk["QUESTION"].astype(str) == question_code)
        mask &= (pd.to_numeric(chunk["SURVEYR"], errors="coerce").isin(years_int))

        # DEMCODE filter: gvals and/or overall (empty/NA)
        if gvals and want_overall:
            gv = chunk["DEMCODE"].astype(str).str.strip()
            mask &= (gv.isin(gvals) | gv.eq("") | chunk["DEMCODE"].isna())
        elif gvals:
            mask &= chunk["DEMCODE"].astype(str).str.strip().isin(gvals)
        elif want_overall:
            gv = chunk["DEMCODE"].astype(str).str.strip()
            mask &= (gv.eq("") | chunk["DEMCODE"].isna())

        if not mask.any():
            continue

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
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(DTYPES["year"])
    df["n"]    = pd.to_numeric(df["n"], errors="coerce").astype(DTYPES["n"])
    for c in ["positive_pct","neutral_pct","negative_pct",
              "answer1","answer2","answer3","answer4","answer5","answer6","answer7"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPES[c])
    df["question_code"] = df["question_code"].astype(DTYPES["question_code"])
    df["group_value"]   = df["group_value"].astype(DTYPES["group_value"])
    return df[OUT_COLS]

# =============================================================================
# Public API — in-memory → Parquet → CSV
# =============================================================================
def _query_inmemory(
    question_code: str,
    years: Iterable[int | str],
    group_values: Optional[List[Optional[str]]],
) -> Optional[pd.DataFrame]:
    if _INMEM_DF is None or _INMEM_MODE == "none":
        return None
    df = _INMEM_DF
    yrs = [int(y) for y in years]
    sel = (df["question_code"].astype(str) == str(question_code)) & (df["year"].astype(int).isin(yrs))

    if group_values is not None:
        want_overall = any(g is None for g in group_values)
        gvals = [str(g).strip() for g in group_values if g is not None]
        mask_g = pd.Series(False, index=df.index)
        if gvals:
            mask_g = mask_g | df["group_value"].astype(str).isin(gvals)
        if want_overall:
            mask_g = mask_g | (df["group_value"] == "All")
        sel = sel & mask_g
    else:
        # Default to overall
        sel = sel & (df["group_value"] == "All")

    out = df.loc[sel, OUT_COLS].copy()
    return out

@st.cache_data(show_spinner="🔎 Filtering results…")
def load_results2024_filtered(
    question_code: str,
    years: Iterable[int | str],
    group_value: Optional[str] = None,
    group_values: Optional[List[Optional[str]]] = None,
) -> pd.DataFrame:
    """
    Returns a filtered slice at (question_code, years, group(s)) grain.
    - Accepts *either* a single `group_value` or a list `group_values` (preferred).
    - Preference order: in-memory → Parquet pushdown → CSV chunk scan.
    - Records diagnostics for the UI.
    """
    global _LAST_ENGINE
    parquet_error = None
    t0 = time.perf_counter()

    # Normalize groups argument
    groups_norm: Optional[List[Optional[str]]]
    if group_values is not None:
        groups_norm = list(group_values)
    else:
        groups_norm = [group_value] if group_value is not None else None

    # In-memory
    try:
        _ensure_inmemory_ready()
        df = _query_inmemory(question_code, years, groups_norm)
        if df is not None:
            _LAST_ENGINE = f"inmem:{_INMEM_MODE}"
            rows = int(df.shape[0])
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _set_diag(
                engine=_LAST_ENGINE, elapsed_ms=elapsed_ms, rows=rows,
                question_code=str(question_code),
                years=",".join(str(y) for y in years),
                group_value=("multiple" if (groups_norm and len(groups_norm) > 1)
                             else ("All" if (groups_norm is None or groups_norm == [None]) else str(groups_norm[0]))),
                inmem_mode=_INMEM_MODE, parquet_dir=PARQUET_ROOTDIR, csv_path=LOCAL_GZ_PATH,
                parquet_error=None
            )
            return df
    except Exception:
        pass

    # Parquet
    if _pyarrow_available():
        try:
            df = _parquet_query(question_code, years, groups_norm)
            _LAST_ENGINE = "parquet"
            rows = int(df.shape[0])
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _set_diag(
                engine=_LAST_ENGINE, elapsed_ms=elapsed_ms, rows=rows,
                question_code=str(question_code),
                years=",".join(str(y) for y in years),
                group_value=("multiple" if (groups_norm and len(groups_norm) > 1)
                             else ("All" if (groups_norm is None or groups_norm == [None]) else str(groups_norm[0]))),
                inmem_mode=_INMEM_MODE, parquet_dir=PARQUET_ROOTDIR, csv_path=LOCAL_GZ_PATH,
                parquet_error=None
            )
            return df
        except Exception as e:
            parquet_error = str(e)

    # CSV fallback
    df = _csv_stream_filter(question_code, years, groups_norm)
    _LAST_ENGINE = "csv"
    rows = int(df.shape[0])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _set_diag(
        engine=_LAST_ENGINE, elapsed_ms=elapsed_ms, rows=rows,
        question_code=str(question_code),
        years=",".join(str(y) for y in years),
        group_value=("multiple" if (groups_norm and len(groups_norm) > 1)
                     else ("All" if (groups_norm is None or groups_norm == [None]) else str(groups_norm[0]))),
        inmem_mode=_INMEM_MODE, parquet_dir=PARQUET_ROOTDIR, csv_path=LOCAL_GZ_PATH,
        parquet_error=parquet_error
    )
    return df

# Convenience: explicit multi wrapper (kept for external callers if needed)
@st.cache_data(show_spinner="🔎 Filtering multiple groups…")
def load_results2024_filtered_multi(
    question_code: str,
    years: Iterable[int | str],
    group_values: Optional[List[Optional[str]]] = None,
) -> pd.DataFrame:
    return load_results2024_filtered(question_code, years, group_values=group_values)

# Optional schema helpers (compatibility with older imports)
def get_results2024_schema() -> dict: return {}
def get_results2024_schema_inferred() -> dict: return {}

# =============================================================================
# Backend info + prewarm (call once at app start)
# =============================================================================
def get_backend_info() -> dict:
    meta_counts = st.session_state.get("metadata_counts", {})
    inmem_rows = int(_INMEM_DF.shape[0]) if isinstance(_INMEM_DF, pd.DataFrame) else 0
    return {
        "parquet_dir_exists": os.path.isdir(PARQUET_ROOTDIR),
        "parquet_ready": os.path.exists(PARQUET_FLAG),
        "parquet_dir": PARQUET_ROOTDIR,
        "csv_path": LOCAL_GZ_PATH,
        "last_engine": _LAST_ENGINE,
        "inmem_mode": _INMEM_MODE,
        "inmem_rows": inmem_rows,
        "metadata_counts": meta_counts,
        "pswide_only": True,
        "level_filter": "LEVEL1ID=0",
        "memory_budget_gb": MEMORY_BUDGET_GB,
    }

@st.cache_resource(show_spinner="⚡ Preloading metadata…")
def _prewarm_metadata() -> Dict[str, int]:
    q = _load_questions_meta(); d = _load_demographics_meta(); s = _load_scales_meta()
    counts = {"questions": int(q.shape[0]), "demographics": int(d.shape[0]), "scales": int(s.shape[0])}
    st.session_state["metadata_counts"] = counts
    return counts

@st.cache_resource(show_spinner="⚡ Warming up data backend…")
def prewarm_fastpath() -> str:
    """Ensure CSV present and PS-wide Parquet dataset built (one-time)."""
    ensure_results2024_local()
    try:
        ensure_parquet_dataset()
        return "parquet"
    except Exception:
        return "csv"

def prewarm_all() -> None:
    """
    Call this once at app start:
      1) Metadata → cached
      2) Data backend (Parquet) → ready
      3) PS-wide filtered data → loaded into memory (all years or latest year)
    """
    _prewarm_metadata()
    st.session_state["data_engine"] = prewarm_fastpath()
    _ensure_inmemory_ready()
