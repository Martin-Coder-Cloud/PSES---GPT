# utils/menu1_helpers.py
# -------------------------------------------------------------------------
# Helpers for Menu 1: demographic resolution, table formatting, and diagnostics.
# -------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd

# -----------------------------
# Tiny utilities
# -----------------------------
def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Case/whitespace-insensitive column resolver."""
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None

def _code_to_str(v) -> str:
    """Normalize a DEMCODE-like value to a plain string (no decimals/whitespace)."""
    if v is None:
        return ""
    s = str(v).strip()
    # Handle Excel numeric cells like 1001.0
    try:
        if "." in s:
            s_num = int(float(s))
        else:
            s_num = int(s)
        return str(s_num)
    except Exception:
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits or s

# -----------------------------
# Demographic resolver
# -----------------------------
def resolve_demographic_codes(
    demo_df: pd.DataFrame,
    category_label: Optional[str],
    subgroup_label: Optional[str],
) -> Tuple[List[Optional[str]], Dict[Optional[str], str], bool]:
    """
    Returns:
      - codes: list of DEMCODE strings to query. Always includes None for "All respondents".
      - disp_map: maps DEMCODE (or None) -> display label.
      - category_in_play: True if a demographic category (other than All respondents) is selected.
    """
    # Normalize columns
    cat_col  = _find_col(demo_df, "demcode_category", "DEMCODE Category", "category")
    code_col = _find_col(demo_df, "demcode", "DEMCODE", "code")
    lbl_col  = _find_col(demo_df, "descrip_e", "DESCRIP_E", "english")

    # Default: All respondents only
    if not category_label or str(category_label).strip().lower() in ("all respondents", "all", ""):
        return [None], {None: "All respondents"}, False

    # Filter rows for the chosen category
    df_cat = demo_df
    if cat_col:
        df_cat = df_cat[df_cat[cat_col].astype(str).str.strip().str.casefold() == str(category_label).strip().casefold()]
    if df_cat.empty:
        # Fallback to overall if mis-matched category
        return [None], {None: "All respondents"}, False

    disp_map: Dict[Optional[str], str] = {None: "All respondents"}

    if subgroup_label and str(subgroup_label).strip() != "":
        # Single subgroup selected: include overall + that code
        if lbl_col:
            sub = df_cat[df_cat[lbl_col].astype(str).str.strip().str.casefold() == str(subgroup_label).strip().casefold()]
        else:
            sub = pd.DataFrame()
        if not sub.empty and code_col in df_cat.columns:
            code = _code_to_str(sub.iloc[0][code_col])
            if lbl_col:
                disp_map[code] = str(sub.iloc[0][lbl_col]).strip()
            else:
                disp_map[code] = code
            return [None, code], disp_map, True
        # Fallback: treat provided subgroup_label as a code string
        code = _code_to_str(subgroup_label)
        disp_map[code] = subgroup_label
        return [None, code], disp_map, True

    # No subgroup => include overall + all sub-codes for category
    codes: List[str] = []
    if code_col and code_col in df_cat.columns:
        codes = [ _code_to_str(v) for v in df_cat[code_col].dropna().tolist() ]
        codes = [c for c in codes if c != ""]
    # Build display map
    if lbl_col and lbl_col in df_cat.columns:
        for _, r in df_cat.iterrows():
            code = _code_to_str(r[code_col]) if code_col in df_cat.columns else None
            if code:
                disp_map[code] = str(r[lbl_col]).strip()
    else:
        for c in codes:
            disp_map[c] = c
    # Always include overall (None) first, then the category codes
    final_codes: List[Optional[str]] = [None] + codes
    return final_codes, disp_map, True

# -----------------------------
# Scales metadata
# -----------------------------
def get_scale_labels(scales_df: pd.DataFrame, question_code: str) -> List[Tuple[str, str]]:
    """
    Returns list of (answer_column_key, human_label) pairs for answer1..answer7 if available.
    """
    if scales_df is None or scales_df.empty or not question_code:
        return []
    code_upper = str(question_code).strip().upper()
    # Find matching row by 'code' or 'question'
    low = {c.lower(): c for c in scales_df.columns}
    code_col = low.get("code") or low.get("question") or list(scales_df.columns)[0]
    sdf = scales_df[ scales_df[code_col].astype(str).str.upper() == code_upper ]
    if sdf.empty:
        return []
    row = sdf.iloc[0]
    pairs: List[Tuple[str,str]] = []
    for i in range(1, 8):
        k = f"answer{i}"
        if k in scales_df.columns:
            label = row.get(k)
            label = (str(label).strip() if pd.notna(label) and str(label).strip() != "" else f"Answer {i}")
            pairs.append((k, label))
    return pairs

# -----------------------------
# Data cleanup
# -----------------------------
def drop_na_999(df: pd.DataFrame) -> pd.DataFrame:
    """Turn 999/9999 into NA across known numeric columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in [f"answer{i}" for i in range(1,8)] + ["POSITIVE","NEUTRAL","NEGATIVE","AGREE",
                                                     "positive_pct","neutral_pct","negative_pct","n","ANSCOUNT"]:
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce")
            out.loc[v.isin([999, 9999]), col] = pd.NA
    return out

def normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename native columns to normalized names used by Menu 1."""
    if df is None or df.empty:
        return df
    out = df.copy()
    low = {c.lower(): c for c in out.columns}
    # year
    y = low.get("year") or low.get("surveyr")
    if y and y != "year":
        out = out.rename(columns={y: "year"})
    # question_code
    q = low.get("question_code") or low.get("question")
    if q and q != "question_code":
        out = out.rename(columns={q: "question_code"})
    # group_value (DEMCODE)
    g = low.get("group_value") or low.get("demcode")
    if g and g != "group_value":
        out = out.rename(columns={g: "group_value"})
    # POSITIVE/NEUTRAL/NEGATIVE rename to *_pct if needed
    for src, dst in [("POSITIVE","positive_pct"),("NEUTRAL","neutral_pct"),("NEGATIVE","negative_pct")]:
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    # ANSCOUNT to n
    if "ANSCOUNT" in out.columns and "n" not in out.columns:
        out = out.rename(columns={"ANSCOUNT":"n"})
    # Types
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    if "group_value" in out.columns:
        # Normalize overall/blank to the literal "All"
        out["group_value"] = out["group_value"].astype("string").fillna("All").replace({"": "All"})
    return out

# -----------------------------
# Display formatting
# -----------------------------
def format_table_for_display(
    df: pd.DataFrame,
    category_in_play: bool,
    dem_disp_map: Dict[Optional[str], str],
    scale_pairs: List[Tuple[str,str]],
) -> pd.DataFrame:
    """
    Build a display-ready table with Year (+ Demographic), answer distribution and positive/agree columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = normalize_results_columns(out)

    out["YearNum"] = pd.to_numeric(out["year"], errors="coerce")
    out["Year"] = out["YearNum"].astype("Int64").astype("string")

    # Demographic label column
    if category_in_play:
        def to_label(code):
            if code is None:
                return "All respondents"
            s = str(code)
            if s == "All":
                return "All respondents"
            return dem_disp_map.get(s, dem_disp_map.get(code, s))
        out["Demographic"] = out["group_value"].apply(to_label)

    # Keep and rename
    dist_cols = [k for k,_ in scale_pairs if k in out.columns]
    rename_map = {k:v for k,v in scale_pairs if k in out.columns}
    keep = ["YearNum","Year"] + (["Demographic"] if category_in_play else []) + dist_cols + \
           ["positive_pct","neutral_pct","negative_pct","AGREE","n"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].rename(columns=rename_map)
    out = out.rename(columns={"positive_pct":"Positive","neutral_pct":"Neutral","negative_pct":"Negative","AGREE":"Agree"})

    # Sort
    sort_cols = ["YearNum"] + (["Demographic"] if category_in_play else [])
    sort_asc  = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["YearNum"], errors="ignore")

    # Types/rounding
    for c in out.columns:
        if c not in ("Year","Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year","Demographic","n")]
    if pct_like:
        out[pct_like] = out[pct_like].round(1)
    if "n" in out.columns:
        out["n"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")

    return out

# -----------------------------
# Positive-only tiny narrative (used when AI disabled)
# -----------------------------
def build_positive_only_narrative(df_disp: pd.DataFrame, category_in_play: bool) -> str:
    """Very short, deterministic summary on Positive only; used when AI is off."""
    if df_disp is None or df_disp.empty or "Year" not in df_disp.columns:
        return "No summary available for this selection."

    t = df_disp.copy()
    # Prefer Positive; fallback to Agree; else first answer label
    metric_col = None
    low = {c.lower(): c for c in t.columns}
    if "positive" in low:
        metric_col = low["positive"]
    elif "agree" in low:
        metric_col = low["agree"]
    else:
        for c in t.columns:
            if c.lower().startswith("answer"):
                metric_col = c
                break
    if metric_col is None:
        return "No summary available for this selection."

    t["Y"] = pd.to_numeric(t["Year"], errors="coerce")
    latest_year = int(t["Y"].max()) if pd.notna(t["Y"].max()) else None
    if latest_year is None:
        return "No summary available for this selection."

    lines: List[str] = []
    df_latest = t[t["Y"] == latest_year]

    def prev_year(df: pd.DataFrame) -> Optional[int]:
        ys = sorted([int(y) for y in df["Y"].dropna().unique().tolist()])
        if len(ys) < 2: return None
        return ys[-2]

    if category_in_play and "Demographic" in t.columns:
        groups = [g for g in df_latest["Demographic"].dropna().unique().tolist()]
        for g in groups:
            gslice = t[t["Demographic"] == g]
            latest_pos = pd.to_numeric(gslice[gslice["Y"] == latest_year][metric_col], errors="coerce").dropna()
            prev = prev_year(gslice)
            if prev is not None:
                prev_pos = pd.to_numeric(gslice[gslice["Y"] == prev][metric_col], errors="coerce").dropna()
                if not latest_pos.empty and not prev_pos.empty:
                    delta = latest_pos.iloc[0] - prev_pos.iloc[0]
                    lines.append(f"{g}: {latest_year} {latest_pos.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")
    else:
        prev = prev_year(t)
        if prev is not None:
            latest_pos = pd.to_numeric(df_latest[metric_col], errors="coerce").dropna()
            prev_pos   = pd.to_numeric(t[t["Y"] == prev][metric_col], errors="coerce").dropna()
            if not latest_pos.empty and not prev_pos.empty:
                delta = latest_pos.iloc[0] - prev_pos.iloc[0]
                lines.append(f"Overall: {latest_year} {latest_pos.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")

    if not lines:
        return "No notable changes to report."
    return " ".join(lines)

# -----------------------------
# Diagnostics for "no data" cases
# -----------------------------
def build_no_data_diagnostic(
    question_code: str,
    years: List[int],
    requested_codes: List[Optional[str]],
    loader_fn,
) -> str:
    """
    Builds a meaningful diagnostic explaining which dimension didn't match.
    - Checks existence of the question+years without demographics.
    - Compares requested_codes against the codes actually present in the data slice.
    """
    # Normalize request
    req_codes = []
    for c in requested_codes or []:
        if c is None:
            req_codes.append("All")
        else:
            req_codes.append(str(c))

    # 1) Slice by question+years only
    try:
        base = loader_fn(question_code=question_code, years=years)
    except TypeError:
        base = loader_fn(question_code=question_code, years=years, group_value=None)

    if base is None or getattr(base, "empty", True):
        return (f"No rows were returned for question {question_code} in years {', '.join(map(str, years))}. "
                "This suggests the in-memory dataset is missing these year(s) for this question. "
                "Please verify the preload/ingest covers all years.")

    # 2) What codes exist for that slice?
    base = normalize_results_columns(base)
    present_codes = sorted([str(x) for x in base.get("group_value", pd.Series(dtype='string')).dropna().astype(str).unique().tolist()])

    # Ensure 'All' is visible if any overall rows exist
    present_codes = ["All" if x == "" else x for x in present_codes]

    # Intersection / difference
    missing = [c for c in req_codes if c not in present_codes]
    present = [c for c in req_codes if c in present_codes]

    if present and missing:
        return (f"The backend returned rows for {question_code} and the selected years, but not for all requested demographics. "
                f"Matched codes: {', '.join(present)}. Missing codes: {', '.join(missing)}. "
                "This usually means the requested codes don't exist for that question/year combination.")
    if not present:
        return (f"The backend returned rows for {question_code} and the selected years, but none matched the requested demographics. "
                f"Requested codes: {', '.join(req_codes) or '(none)'}; available codes include: "
                f"{', '.join(present_codes[:12])}{'…' if len(present_codes)>12 else ''}. "
                "This typically indicates the resolver produced codes that don't exist in the dataset for this slice.")
    # Fallback
    return "No matching rows after filtering; please double-check the selection."
