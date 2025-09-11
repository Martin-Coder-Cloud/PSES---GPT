# utils/data_loader.py
# -----------------------------------------------------------------------------
# Results2024 loader (character-only) + schema introspection utilities
# - Reads gz CSV as pure text (dtype=str), blanks preserved (""), headers UPPER.
# - Filters on QUESTION (exact trimmed), SURVEYR (years as strings, canonicalized),
#   DEMCODE (exact trimmed, canonicalized to handle "1937.0" and leading-zero loss).
# - PS-wide: LEVEL1ID == "" or "0" (and other zero-ish strings) when present; else keep rows.
# - No BYCOND. No dedup.
# - Extras:
#     • get_results2024_schema(): shows dtypes/examples AFTER the loader read
#     • get_results2024_schema_inferred(): optional “what pandas would infer” preview
# -----------------------------------------------------------------------------

from typing import List, Optional
import os
import gzip
import re

import gdown
import pandas as pd
import streamlit as st

GDRIVE_FILE_ID_FALLBACK = "1VdMQQfEP-BNXle8GeD-Z_upt2pPIGvc8"
LOCAL_GZ_PATH = "/tmp/Results2024.csv.gz"


@st.cache_resource(show_spinner="📥 Downloading Results2024.csv.gz…")
def ensure_results2024_local(file_id: Optional[str] = None) -> str:
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
    Yield DataFrame chunks from the gz CSV as text:
      - dtype=str -> all values read as Python strings
      - keep_default_na=False / na_filter=False -> blanks stay ""
      - headers normalized to UPPERCASE (values unchanged)
      - final safety: trim whitespace on ALL cells so matches are exact
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
            # force every cell to a trimmed string; keep blanks as ""
            for c in chunk.columns:
                chunk[c] = chunk[c].astype(str).str.strip()
            yield chunk


def _as_trimmed_str(x: Optional[str]) -> str:
    return "" if x is None else str(x).strip()


# ─────────────────────────────────────────────────────────────
# Canonicalization helpers (for MATCHING ONLY; display stays raw)
# ─────────────────────────────────────────────────────────────
def _canon_year(s: Optional[str]) -> str:
    """
    Normalize SURVEYR-like tokens for comparison:
      - trim
      - if looks like digits optionally followed by .0..., strip the .0...
    """
    t = "" if s is None else str(s).strip()
    return re.sub(r"\.0+$", "", t) if re.fullmatch(r"\d+(?:\.0+)?", t) else t


def _canon_demcode(s: Optional[str]) -> str:
    """
    Normalize DEMCODE for comparison only:
      - convert to str, normalize NBSP to space, trim
      - remove all spaces (defensive)
      - if looks like digits optionally followed by .0..., strip .0...
      - if pure digits and length < 4, left-pad to 4
    """
    if s is None:
        return ""
    t = str(s).replace("\u00A0", " ").strip()
    t = t.replace(" ", "")
    if re.fullmatch(r"\d+(?:\.0+)?", t):
        t = re.sub(r"\.0+$", "", t)
    return t.zfill(4) if t.isdigit() and len(t) < 4 else t


@st.cache_data(show_spinner="🔎 Filtering results…")
def load_results2024_filtered(
    question_code: str,
    years: List[str] | List[int],
    group_value: Optional[str] = None,  # None/"" -> All respondents (blank DEMCODE)
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Filter rows where:
      - QUESTION == question_code (exact trimmed string)
      - SURVEYR in years (compared via canonicalized strings)
      - DEMCODE == group_value (canonicalized; "" for All respondents)
      - PS-wide: LEVEL1ID is "" or "0" (strings) when present; if absent, keep rows
    """
    path = ensure_results2024_local()

    q_target = _as_trimmed_str(question_code)
    years_canon = {_canon_year(y) for y in years}

    want_blank_dem = (group_value is None) or (_as_trimmed_str(group_value) == "")
    dem_target_canon = "" if want_blank_dem else _canon_demcode(group_value)

    parts: list[pd.DataFrame] = []

    for chunk in _chunk_reader(path, chunksize=chunksize):
        if not {"QUESTION", "SURVEYR", "DEMCODE"}.issubset(set(chunk.columns)):
            continue

        # QUESTION exact (already trimmed strings)
        qmask = chunk["QUESTION"] == q_target

        # SURVEYR canonicalized on both sides
        ymask = chunk["SURVEYR"].apply(_canon_year).isin(years_canon)

        # DEMCODE canonicalized on both sides
        dem_series_canon = chunk["DEMCODE"].apply(_canon_demcode)
        gmask = (dem_series_canon == "") if want_blank_dem else (dem_series_canon == dem_target_canon)

        # PS-wide: LEVEL1ID == "" or "0" or other zero-ish strings (e.g., "0.0", "0000")
        if "LEVEL1ID" in chunk.columns:
            lvl = chunk["LEVEL1ID"].astype(str).str.strip()
            zeroish = lvl.eq("") | lvl.eq("0") | lvl.str.fullmatch(r"0+(?:\.0+)?").fillna(False)
            lmask = zeroish
        else:
            lmask = pd.Series(True, index=chunk.index)

        sub = chunk[qmask & ymask & gmask & lmask]
        if not sub.empty:
            parts.append(sub)

    if parts:
        # guarantee character-only & trimmed on the merged frame as well
        out = pd.concat(parts, ignore_index=True)
        for c in out.columns:
            out[c] = out[c].astype(str).str.strip()
        return out

    # stable empty frame
    return pd.DataFrame(columns=[
        "LEVEL1ID","LEVEL2ID","LEVEL3ID","LEVEL4ID","LEVEL5ID",
        "SURVEYR","DEMCODE","QUESTION",
        "ANSWER1","ANSWER2","ANSWER3","ANSWER4","ANSWER5","ANSWER6","ANSWER7",
        "POSITIVE","NEUTRAL","NEGATIVE","SCORE5","SCORE100","ANSCOUNT"
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers you can call from the UI to SEE what the loader actually read
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_results2024_schema(sample_rows: int = 5000) -> pd.DataFrame:
    """
    Returns a small schema snapshot AFTER the loader's text read (i.e., as used by the app):
      - column
      - dtype_after_loader (should be 'object' for all)
      - example_non_blank (first non-empty example)
      - blank_rate (share of blanks "")
    """
    path = ensure_results2024_local()
    with gzip.open(path, mode="rt", newline="") as f:
        peek = pd.read_csv(
            f,
            nrows=sample_rows,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=True,
        )
    peek.columns = [str(c).strip().upper() for c in peek.columns]
    for c in peek.columns:
        peek[c] = peek[c].astype(str).str.strip()

    rows = []
    for c in peek.columns:
        s = peek[c]
        ex = next((v for v in s if v != ""), "")
        blank_rate = (s == "").mean() if len(s) else 0.0
        rows.append({
            "column": c,
            "dtype_after_loader": str(s.dtype),  # expect 'object'
            "example_non_blank": ex,
            "blank_rate": round(float(blank_rate), 3),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_results2024_schema_inferred(sample_rows: int = 5000) -> pd.DataFrame:
    """
    Optional: what pandas WOULD infer if we didn't force text (helps diagnose odd exports).
      - column
      - dtype_inferred_by_pandas
      - example_non_blank
    """
    path = ensure_results2024_local()
    with gzip.open(path, mode="rt", newline="") as f:
        peek = pd.read_csv(
            f,
            nrows=sample_rows,
            # NOTE: no dtype=str on purpose here
            keep_default_na=True,
            na_filter=True,
            low_memory=True,
        )
    peek.columns = [str(c).strip().upper() for c in peek.columns]

    rows = []
    for c in peek.columns:
        s = peek[c]
        ex = next((v for v in s if pd.notna(v)), "")
        rows.append({
            "column": c,
            "dtype_inferred_by_pandas": str(s.dtype),
            "example_non_blank": ex,
        })
    return pd.DataFrame(rows)
