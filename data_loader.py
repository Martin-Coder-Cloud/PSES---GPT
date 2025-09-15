# utils/data_loader.py
# Chunked, text-mode loader with one-pass filtering and Google Drive download.
# - Reads Results2024.csv.gz in chunks (dtype=str).
# - Filters: QUESTION, SURVEYR (set), DEMCODE (set), LEVEL1ID ∈ {"", "0"}.
# - Auto-downloads from Google Drive using RESULTS2024_FILE_ID or RESULTS2024_GDRIVE_URL
#   (provided via env vars or st.secrets), saving to ./data/Results2024.csv.gz.
# - Supports group_values (list) and legacy group_value (single).

from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st

# Tunable
CHUNKSIZE: int = 200_000
DEFAULT_LOCAL_DIR = os.path.join(os.getcwd(), "data")
DEFAULT_LOCAL_PATH = os.path.join(DEFAULT_LOCAL_DIR, "Results2024.csv.gz")


# ─────────────────────────────────────────
# Helpers: trim / id parsing
# ─────────────────────────────────────────
def _trim(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _extract_file_id_from_url(url: str) -> Optional[str]:
    """
    Accepts a standard Google Drive 'file/d/<id>/view' URL and extracts the file id.
    """
    if not url:
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)/", str(url))
    return m.group(1) if m else None

# NEW: normalize QUESTION codes for robust matching (minimal, surgical)
def _norm_q(x: str) -> str:
    """
    Normalize a question code:
      - uppercase
      - remove spaces, underscores, dashes, and periods
      - map known aliases (e.g., D57_1 -> Q57_1)
    Examples: 'q57_2' -> 'Q572', 'Q57-2' -> 'Q572', ' Q19a ' -> 'Q19A'
    """
    if x is None:
        return ""
    s = str(x).upper().strip()
    s = s.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
    # Known dataset exception(s): data has D57_1/D57_2 while metadata/UI use Q57_1/Q57_2
    aliases = {
        "D571": "Q571",
        "D572": "Q572",
    }
    return aliases.get(s, s)

# NEW: for DB-style exact pass-through when needed (Q57_1 -> D57_1, Q57_2 -> D57_2)
def _alias_exact_question_value(question_code: str) -> Optional[str]:
    n = _norm_q(question_code)
    if n == "Q571":
        return "D57_1"
    if n == "Q572":
        return "D57_2"
    return None


# ─────────────────────────────────────────
# Path resolution + (optional) download
# ─────────────────────────────────────────
def _candidate_paths() -> list[str]:
    """
    Search common places for Results2024.csv.gz.
    """
    cwd = os.getcwd()
    return [
        os.environ.get("RESULTS2024_PATH", ""),                               # 1) env override
        st.secrets.get("RESULTS2024_PATH", "") if hasattr(st, "secrets") else "",
        DEFAULT_LOCAL_PATH,                                                   # 2) ./data/Results2024.csv.gz
        os.path.join(cwd, "Results2024.csv.gz"),
        os.path.join(cwd, "datasets", "Results2024.csv.gz"),
        "/mount/src/pses---gpt/data/Results2024.csv.gz",
        "/mount/src/pses---gpt/Results2024.csv.gz",
        "/workspace/pses---gpt/data/Results2024.csv.gz",
        "/tmp/Results2024.csv.gz",  # legacy
    ]


def _try_gdown_download(output_path: str) -> bool:
    """
    Try to download Results2024.csv.gz from Google Drive using:
      - RESULTS2024_FILE_ID (env or st.secrets), or
      - RESULTS2024_GDRIVE_URL (env or st.secrets)
    Returns True if file exists at output_path after this call.
    """
    file_id = os.environ.get("RESULTS2024_FILE_ID") or (
        st.secrets.get("RESULTS2024_FILE_ID") if hasattr(st, "secrets") else None
    )
    if not file_id:
        url_secret = os.environ.get("RESULTS2024_GDRIVE_URL") or (
            st.secrets.get("RESULTS2024_GDRIVE_URL") if hasattr(st, "secrets") else None
        )
        if url_secret:
            file_id = _extract_file_id_from_url(url_secret)

    if not file_id:
        return False

    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception:
        pass

    try:
        import gdown  # type: ignore
    except Exception:
        st.error(
            "Google Drive file ID found, but `gdown` is not installed. "
            "Please add `gdown>=5` to requirements.txt."
        )
        return False

    try:
        # gdown will skip download if the exact file already exists unless fuzzy=False is set;
        # we force overwrite=False by checking existence first.
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            gdown.download(id=file_id, output=output_path, quiet=True)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        st.error(f"Failed to download Results2024.csv.gz from Google Drive: {e}")
        return False


def _resolve_results_path() -> str:
    """
    Resolve a local path to Results2024.csv.gz.
    1) Check env/secrets and common repo paths.
    2) If not found, try Google Drive download via gdown using secrets/env.
    3) Raise FileNotFoundError if still missing.
    """
    # 1) Existing local copies
    for p in _candidate_paths():
        if p and os.path.exists(p):
            return p

    # 2) Try to download to ./data/Results2024.csv.gz
    if _try_gdown_download(DEFAULT_LOCAL_PATH):
        return DEFAULT_LOCAL_PATH

    # 3) Still nothing — error with hints
    hints = "\n".join(f" - {p}" for p in _candidate_paths() if p)
    raise FileNotFoundError(
        "Results2024.csv.gz not found locally and download was not possible.\n\n"
        "Make sure one of these is set:\n"
        "  • st.secrets['RESULTS2024_FILE_ID'] (preferred)\n"
        "  • st.secrets['RESULTS2024_GDRIVE_URL']\n"
        "  • env RESULTS2024_FILE_ID or RESULTS2024_GDRIVE_URL\n"
        "  • env/secret RESULTS2024_PATH (absolute path to the file)\n\n"
        "Paths checked:\n" + hints
    )


# ─────────────────────────────────────────
# Main filtered loader (chunked)
# ─────────────────────────────────────────
def load_results2024_filtered(
    question_code: str,
    years: Iterable[str],
    group_values: Optional[List[Optional[str]]] = None,
    group_value: Optional[str] = None,  # legacy support
) -> pd.DataFrame:
    """
    Stream the gz CSV in chunks and return only matching rows.
    All columns are kept as TEXT (dtype=str).
    Filters (trimmed string compares):
      - QUESTION (normalized) == question_code (normalized)
      - SURVEYR in {years}
      - DEMCODE in {group_values}; None -> "" (All respondents)
      - If LEVEL1ID exists: keep rows where LEVEL1ID in {"", "0"}
    """
    try:
        path = _resolve_results_path()
    except FileNotFoundError as e:
        st.error(str(e))
        return pd.DataFrame()

    # Normalize inputs once
    qcode_raw = str(question_code).strip()
    qcode_norm = _norm_q(qcode_raw)
    years_set = {str(y).strip() for y in years}

    if group_values is None:
        group_values = [group_value]  # may be [None]
    dem_set = set("" if gv is None else str(gv).strip() for gv in (group_values or [None]))

    parts: List[pd.DataFrame] = []

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=True,
        chunksize=CHUNKSIZE,
    ):
        # Normalize headers once per chunk
        chunk.columns = [str(c).strip().upper() for c in chunk.columns]

        # Required columns
        if not {"QUESTION", "SURVEYR", "DEMCODE"}.issubset(chunk.columns):
            continue

        # If this particular question needs an exact pass-through alias, use it
        alias_exact = _alias_exact_question_value(qcode_raw)

        # Default normalized series (used in both branches)
        qnorm_series = _trim(chunk["QUESTION"]).map(_norm_q)

        if alias_exact is not None:
            # MINIMAL CHANGE: allow either exact D57_x OR normalized match to succeed
            qmask = (_trim(chunk["QUESTION"]) == alias_exact) | (qnorm_series == qcode_norm)
        else:
            # Default path: normalized comparison on both sides
            qmask = (qnorm_series == qcode_norm)

        ymask = _trim(chunk["SURVEYR"]).isin(years_set)
        gmask = _trim(chunk["DEMCODE"]).isin(dem_set)
        m = qmask & ymask & gmask
        if not m.any():
            continue

        part = chunk.loc[m].copy()

        # PS-wide only
        if "LEVEL1ID" in part.columns:
            lvl = _trim(part["LEVEL1ID"])
            part = part[(lvl == "") | (lvl == "0")].copy()
            if part.empty:
                continue

        parts.append(part)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────
# Diagnostics (chunk-aware, lightweight)
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_results2024_schema(max_rows: int = 25_000) -> pd.DataFrame:
    """
    Approximate schema from the first N rows (scanned in chunks).
    """
    try:
        path = _resolve_results_path()
    except FileNotFoundError as e:
        st.error(str(e))
        return pd.DataFrame(columns=["column", "dtype_after_loader", "example_non_blank", "blank_rate"])

    total = 0
    col_stats: dict[str, dict] = {}

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=True,
        chunksize=min(CHUNKSIZE, max_rows),
    ):
        chunk.columns = [str(c).strip().upper() for c in chunk.columns]
        for c in chunk.columns:
            s = chunk[c].astype(str)
            nb = (s != "").sum()
            bl = (s == "").sum()
            rec = col_stats.setdefault(c, {"nonblank": 0, "blank": 0, "example": ""})
            rec["nonblank"] += int(nb)
            rec["blank"] += int(bl)
            if rec["example"] == "":
                vals = s[s != ""]
                if not vals.empty:
                    rec["example"] = str(vals.iloc[0])
        total += len(chunk)
        if total >= max_rows:
            break

    rows = []
    for c, r in sorted(col_stats.items()):
        denom = (r["nonblank"] + r["blank"]) or 1
        rows.append({
            "column": str(c),
            "dtype_after_loader": "object",
            "example_non_blank": str(r["example"]),
            "blank_rate": float(round(r["blank"] / denom, 3)),
        })

    df = pd.DataFrame(rows, columns=["column", "dtype_after_loader", "example_non_blank", "blank_rate"])
    df["column"] = df["column"].astype(str)
    df["dtype_after_loader"] = df["dtype_after_loader"].astype(str)
    df["example_non_blank"] = df["example_non_blank"].astype(str)
    df["blank_rate"] = df["blank_rate"].astype(float)
    return df


@st.cache_data(show_spinner=False)
def get_results2024_schema_inferred(sample_rows: int = 5000) -> pd.DataFrame:
    """
    What pandas would infer if we didn't force text (small sample).
    """
    try:
        path = _resolve_results_path()
    except FileNotFoundError as e:
        st.error(str(e))
        return pd.DataFrame(columns=["column", "dtype_inferred_by_pandas", "example_non_blank"])

    peek = pd.read_csv(
        path,
        compression="gzip",
        nrows=sample_rows,
        keep_default_na=True,
        na_filter=True,
        low_memory=True,
    )
    peek.columns = [str(c).strip().upper() for c in peek.columns]

    rows = []
    for c in peek.columns:
        s = peek[c]
        ex = ""
        for v in s:
            if pd.notna(v):
                ex = str(v)
                break
        rows.append({
            "column": str(c),
            "dtype_inferred_by_pandas": str(s.dtype),
            "example_non_blank": str(ex),
        })

    out = pd.DataFrame(rows, columns=["column", "dtype_inferred_by_pandas", "example_non_blank"])
    out["column"] = out["column"].astype(str)
    out["dtype_inferred_by_pandas"] = out["dtype_inferred_by_pandas"].astype(str)
    out["example_non_blank"] = out["example_non_blank"].astype(str)
    return out
