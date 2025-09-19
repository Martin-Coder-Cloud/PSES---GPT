# utils/data_loader.py (drop-in)
from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st

# -----------------------------
# Config / defaults
# -----------------------------
_DEFAULT_CANDIDATES = [
    "data/results2024.csv.gz",
    "results2024.csv.gz",
    "/mnt/data/results2024.csv.gz",
]

_DB_PATH = "pses.duckdb"  # persistent so we don't rebuild per session

_LAST_BACKEND: str | None = None
_LAST_SOURCE_PATH: str | None = None


# -----------------------------
# Small helpers
# -----------------------------
def _normalize_qcode(s: str) -> str:
    if s is None:
        return ""
    s = str(s).upper()
    return re.sub(r"[^A-Z0-9]+", "", s)


def _norm_or_empty(x) -> str:
    return _normalize_qcode(x) if pd.notna(x) else ""


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def _canon_demcode(x: str | None) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.isdigit() and len(s) <= 4:
        return s.zfill(4)
    return s


def _listify_one_or_many(
    group_values: Optional[Iterable[str | None]] = None,
    group_value: Optional[str | None] = None,
) -> Optional[List[str | None]]:
    if group_values is not None:
        return list(group_values)
    if group_value is not None:
        return [group_value]
    return None


# -----------------------------
# Public helpers for the UI
# -----------------------------
def _resolve_results_path() -> str:
    cand = []
    try:
        v = (st.secrets.get("RESULTS_PATH") or "").strip()
        if v:
            cand.append(v)
    except Exception:
        pass
    cand.extend(_DEFAULT_CANDIDATES)
    for p in cand:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("PSES results file not found. Checked: " + ", ".join(cand))


def get_backend_info() -> dict:
    return {"backend": _LAST_BACKEND, "source_path": _LAST_SOURCE_PATH}


# -----------------------------
# Schema helpers (global previews)
# -----------------------------
@st.cache_data(show_spinner=False)
def get_results2024_schema() -> pd.DataFrame:
    path = _resolve_results_path()
    df = pd.read_csv(
        path,
        compression="infer",
        dtype=str,
        nrows=50,
        keep_default_na=False,
        na_filter=False,
    )
    schema = pd.DataFrame({"column": df.columns, "dtype": ["object"] * len(df.columns)})
    return schema


@st.cache_data(show_spinner=False)
def get_results2024_schema_inferred() -> pd.DataFrame:
    path = _resolve_results_path()
    df = pd.read_csv(path, compression="infer", nrows=200)
    schema = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    return schema


# -----------------------------
# DuckDB support
# -----------------------------
def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _get_duck() :
    """
    Persistent DuckDB connection; reused across runs.
    """
    import duckdb
    con = duckdb.connect(_DB_PATH, read_only=False)
    try:
        con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 4))}")
    except Exception:
        pass
    con.execute("PRAGMA enable_progress_bar=false")
    con.execute("PRAGMA enable_object_cache=true")
    # (Optional) allow 2–4 GB if host permits; tune as needed:
    # con.execute("PRAGMA memory_limit='2GB'")
    return con


def _meta_get(con, key: str) -> str | None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS __meta__(
            k VARCHAR PRIMARY KEY,
            v VARCHAR
        )
    """)
    row = con.execute("SELECT v FROM __meta__ WHERE k=?", [key]).fetchone()
    return row[0] if row else None


def _meta_set(con, key: str, val: str) -> None:
    con.execute("INSERT OR REPLACE INTO __meta__(k, v) VALUES (?,?)", [key, val])


def _ensure_materialized(con, csv_path: str) -> None:
    """
    Create or refresh the materialized 'results' table if the CSV has changed.
    We:
      - Precompute QUESTION_NORM once
      - Store SURVEYR as INT
      - Trim DEMCODE to support exact matches and NULL detection
      - Keep only columns the app uses
    """
    src_mtime = str(_file_mtime(csv_path))
    cur_mtime = _meta_get(con, "results_csv_mtime")

    need_build = False
    # Build if table missing
    try:
        con.execute("SELECT 1 FROM results LIMIT 1")
    except Exception:
        need_build = True

    # Or rebuild if source changed
    if not need_build and cur_mtime != src_mtime:
        need_build = True

    if not need_build:
        return

    # (Re)build table
    con.execute("DROP TABLE IF EXISTS results")

    # IMPORTANT: project only the columns used by the UI
    con.execute(
        """
        CREATE TABLE results AS
        SELECT
            CAST(SURVEYR AS INT)                            AS SURVEYR,
            NULLIF(trim(COALESCE(DEMCODE, '')), '')         AS DEMCODE,   -- empty->NULL
            UPPER(REGEXP_REPLACE(COALESCE(QUESTION,''), '[^A-Za-z0-9]', '')) AS QUESTION_
