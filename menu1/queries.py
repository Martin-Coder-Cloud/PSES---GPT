# app/menu1/queries.py
"""
Query wrappers and normalization for Menu 1.

- Provides a stable, testable surface over utils.data_loader
- Keeps all column name normalization in one place
"""

from __future__ import annotations
from typing import Iterable, Optional
import pandas as pd

# ---- Optional imports from your existing utils layer ------------------------
try:
    from utils.data_loader import load_results2024_filtered  # main query
except Exception:
    load_results2024_filtered = None  # type: ignore

# Optional diagnostics / preload (best-effort)
try:
    from utils.data_loader import get_backend_info  # type: ignore
except Exception:
    def get_backend_info() -> dict:  # type: ignore
        return {"engine": "csv.gz", "in_memory": False}

try:
    from utils.data_loader import preload_pswide_dataframe  # type: ignore
except Exception:
    def preload_pswide_dataframe():  # type: ignore
        return None


# ---- Public: normalization ---------------------------------------------------
def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column names from loader output.

    Input columns (various possibilities):
      - question_code / QUESTION / question
      - year / SURVEYR
      - group_value / DEMCODE / demcode
      - POSITIVE/NEUTRAL/NEGATIVE -> positive_pct/neutral_pct/negative_pct
      - ANSCOUNT -> n
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # QUESTION -> question_code
    if "question_code" not in out.columns:
        if "QUESTION" in out.columns:
            out = out.rename(columns={"QUESTION": "question_code"})
        else:
            for c in out.columns:
                if c.strip().lower() == "question":
                    out = out.rename(columns={c: "question_code"})
                    break

    # SURVEYR -> year
    if "year" not in out.columns:
        if "SURVEYR" in out.columns:
            out = out.rename(columns={"SURVEYR": "year"})
        else:
            for c in out.columns:
                if c.strip().lower() in ("surveyr", "year"):
                    out = out.rename(columns={c: "year"})
                    break

    # DEMCODE -> group_value
    if "group_value" not in out.columns:
        if "DEMCODE" in out.columns:
            out = out.rename(columns={"DEMCODE": "group_value"})
        else:
            for c in out.columns:
                if c.strip().lower() == "demcode":
                    out = out.rename(columns={c: "group_value"})
                    break

    # POS/NEU/NEG rename
    if "positive_pct" not in out.columns and "POSITIVE" in out.columns:
        out = out.rename(columns={"POSITIVE": "positive_pct"})
    if "neutral_pct" not in out.columns and "NEUTRAL" in out.columns:
        out = out.rename(columns={"NEUTRAL": "neutral_pct"})
    if "negative_pct" not in out.columns and "NEGATIVE" in out.columns:
        out = out.rename(columns={"NEGATIVE": "negative_pct"})
    if "n" not in out.columns and "ANSCOUNT" in out.columns:
        out = out.rename(columns={"ANSCOUNT": "n"})

    return out


# ---- Public: main query wrapper ---------------------------------------------
def fetch_per_question(
    question_code: str,
    years: Iterable[int],
    demcodes: Iterable[Optional[str]],
) -> pd.DataFrame:
    """
    Fetch results for a single question across the selected years and demographic codes.
    - Calls the repo's load_results2024_filtered for each demcode
    - Concats parts and returns a single DataFrame (may be empty)

    Args:
        question_code: e.g., "Q01"
        years: e.g., [2019, 2020, 2022, 2024]
        demcodes: iterable of demographic codes; may include None for overall

    Returns:
        pd.DataFrame (possibly empty). Caller is responsible for:
          - normalize_results(df)
          - suppression handling
          - display formatting
    """
    if load_results2024_filtered is None:
        return pd.DataFrame()

    parts = []
    for code in demcodes:
        try:
            df_part = load_results2024_filtered(
                question_code=question_code,
                years=list(years),
                group_value=(None if code in (None, "", "All") else str(code)),
            )
            if df_part is not None and not df_part.empty:
                parts.append(df_part)
        except Exception:
            # Keep robust; skip failing slice rather than crashing the page
            continue

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)
