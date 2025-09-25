# utils/metadata_loader.py — preload & serve metadata for the app
from __future__ import annotations

import os
from typing import Dict
import pandas as pd
import streamlit as st

META_DIR = "metadata"
Q_PATHS = [os.path.join(META_DIR, "Survey Questions.xlsx"),
           os.path.join(META_DIR, "Survey Questions.xls")]  # optional xls
DEM_PATH = os.path.join(META_DIR, "Demographics.xlsx")
SCALES_PATH = os.path.join(META_DIR, "Survey Scales.xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# Loaders (cached). We read ONLY from ./metadata — no other fallbacks.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_questions() -> pd.DataFrame:
    path = next((p for p in Q_PATHS if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("Missing `metadata/Survey Questions.xlsx` (or `.xls`).")

    try:
        df = pd.read_excel(path, dtype=str, engine=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read questions metadata at `{path}`: {e}") from e

    # Your schema: Question (number/code), English (text)
    cols = {c.lower().strip(): c for c in df.columns}
    if "question" not in cols or "english" not in cols:
        raise ValueError("`Survey Questions` must have columns 'Question' and 'English'.")

    df = df.rename(columns={cols["question"]: "code", cols["english"]: "text"})
    df["code"] = df["code"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    # Numeric part (for sort & downstream convenience)
    df["qnum"] = pd.to_numeric(df["code"].str.extract(r"(\d+)", expand=False), errors="coerce")

    # Display label for UI lists
    df["display"] = df["code"] + " – " + df["text"]

    df = df.sort_values(["qnum", "code"], na_position="last").reset_index(drop=True)
    return df[["code", "text", "display", "qnum"]]


@st.cache_data(show_spinner=False)
def load_demographics() -> pd.DataFrame:
    if not os.path.exists(DEM_PATH):
        raise FileNotFoundError("Missing `metadata/Demographics.xlsx`.")
    try:
        df = pd.read_excel(DEM_PATH, dtype=str, engine=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read demographics metadata at `{DEM_PATH}`: {e}") from e
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_scales() -> pd.DataFrame:
    if not os.path.exists(SCALES_PATH):
        raise FileNotFoundError("Missing `metadata/Survey Scales.xlsx`.")
    try:
        sdf = pd.read_excel(SCALES_PATH, dtype=str, engine=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read scales metadata at `{SCALES_PATH}`: {e}") from e

    sdf.columns = sdf.columns.str.strip().str.lower()
    code_col = "code" if "code" in sdf.columns else ("question" if "question" in sdf.columns else None)
    if code_col is None:
        raise ValueError("`Survey Scales.xlsx` must include a 'code' or 'question' column.")

    def _norm(s: str) -> str:
        s = "" if s is None else str(s)
        return "".join(ch for ch in s.upper() if ch.isalnum())

    sdf["__code_norm__"] = sdf[code_col].astype(str).map(_norm)
    return sdf


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_questions() -> pd.DataFrame:
    return load_questions()

def get_demographics() -> pd.DataFrame:
    return load_demographics()

def get_scales() -> pd.DataFrame:
    return load_scales()

def prewarm_metadata(show_spinner: bool = True) -> Dict[str, int]:
    """
    Trigger all cached loaders so data is in memory before the user reaches the UI.
    Returns row counts for quick diagnostics.
    """
    if show_spinner:
        with st.spinner("Loading metadata…"):
            q = load_questions(); d = load_demographics(); s = load_scales()
    else:
        q = load_questions(); d = load_demographics(); s = load_scales()

    counts = {"questions": int(q.shape[0]), "demographics": int(d.shape[0]), "scales": int(s.shape[0])}
    st.session_state["metadata_ready"] = True
    st.session_state["metadata_counts"] = counts
    return counts
