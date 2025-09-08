# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# RAW + METADATA-FIRST version
# Only modification: banner CSS capped at max-height:120px and updated image src

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.data_loader import load_results2024_filtered  # RAW loader (no normalization)

# ─────────────────────────────
# Cached metadata
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_demographics_metadata() -> pd.DataFrame:
    df = pd.read_excel("metadata/Demographics.xlsx")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_questions_metadata() -> pd.DataFrame:
    qdf = pd.read_excel("metadata/Survey Questions.xlsx")
    qdf.columns = [c.strip().lower() for c in qdf.columns]
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["code"] = qdf["code"].astype(str).str.strip().str.upper()
    qdf["qnum"] = qdf["code"].str.extract(r'Q?(\d+)', expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"] + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]

@st.cache_data(show_spinner=False)
def load_scales_metadata() -> pd.DataFrame:
    sdf = pd.read_excel("metadata/Survey Scales.xlsx")
    sdf.columns = sdf.columns.str.strip().str.lower()
    return sdf

# ─────────────────────────────
# Helpers (unchanged)
# ─────────────────────────────
def _find_demcode_col(demo_df: pd.DataFrame) -> str | None:
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            return c
    return None

def _four_digit(s: str) -> str:
    s = "".join(ch for ch in str(s) if ch.isdigit())
    return s.zfill(4) if s else ""

def resolve_demographic_codes_from_metadata(
    demo_df: pd.DataFrame,
    category_label: str | None,
    subgroup_label: str | None
):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    code_col = _find_demcode_col(demo_df)

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                raw_code = str(row.iloc[0][code_col])
                code4 = _four_digit(raw_code)
                code_final = code4 if code4 else raw_code
                return [code_final], {code_final: subgroup_label}, True
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    if code_col and LABEL_COL in df_cat.columns:
        pairs = []
        for _, r in df_cat.iterrows():
            raw_code = str(r[code_col]); label = str(r[LABEL_COL])
            code4 = _four_digit(raw_code)
            if code4:
                pairs.append((code4, label))
        if pairs:
            return [c for c, _ in pairs], {c: l for c, l in pairs}, True

    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        return labels, {l: l for l in labels}, True

    return [None], {None: "All respondents"}, False

def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    sdf = scales_df.copy()
    candidates = pd.DataFrame()
    for key in ["code", "question"]:
        if key in sdf.columns:
            candidates = sdf[sdf[key].astype(str).str.upper() == str(question_code).upper()]
            if not candidates.empty:
                break
    labels = []
    for i in range(1, 8):
        col = f"answer{i}"
        lbl = None
        if not candidates.empty and col in candidates.columns:
            vals = candidates[col].dropna().astype(str)
            if not vals.empty:
                lbl = vals.iloc[0].strip()
        if not lbl:
            lbl = f"Answer {i}"
        labels.append((col, lbl))
    return labels

def exclude_999_raw(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"answer{i}" for i in range(1, 8)] + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    out = df.copy()
    keep = pd.Series(True, index=out.index)
    for c in present:
        vals = pd.to_numeric(out[c], errors="coerce")
        keep &= (vals != 999)
    return out.loc[keep].copy()

def format_display_table_raw(df: pd.DataFrame, category_in_play: bool, dem_disp_map: dict, scale_pairs) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["__YearNum__"] = pd.to_numeric(out["SURVEYR"], errors="coerce").astype("Int64")
    out["Year"] = out["SURVEYR"].astype(str)

    if category_in_play:
        def to_label(code):
            code = "" if code is None else str(code)
            if code.strip() == "":
                return "All respondents"
            return dem_disp_map.get(code, dem_disp_map.get(str(code), str(code)))
        out["Demographic"] = out["DEMCODE"].apply(to_label)

    dist_cols = [f"answer{i}" for i in range(1, 8) if f"answer{i}" in out.columns]
    rename_map = {k: v for k, v in scale_pairs if k in out.columns}

    keep_cols = ["__YearNum__", "Year"] + (["Demographic"] if category_in_play else []) \
                + dist_cols + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    sort_cols = ["__YearNum__"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["__YearNum__"])

    for c in out.columns:
        if c not in ("Year", "Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year", "Demographic", "ANSCOUNT")]
    if pct_like:
        out[pct_like] = out[pct_like].round(1)
    if "ANSCOUNT" in out.columns:
        out["ANSCOUNT"] = pd.to_numeric(out["ANSCOUNT"], errors="coerce").astype("Int64")
    return out

def narrative_positive_only_raw(df_disp: pd.DataFrame, category_in_play: bool) -> str:
    pos_col = "POSITIVE" if "POSITIVE" in df_disp.columns else ("Positive" if "Positive" in df_disp.columns else None)
    if df_disp.empty or pos_col is None:
        return "No results available to summarize."
    t = df_disp.copy()
    t["_Y"] = pd.to_numeric(t["Year"], errors="coerce")
    if t["_Y"].isna().all():
        return "No results available to summarize."
    latest = int(t["_Y"].max())
    latest_rows = t[t["_Y"] == latest]
    lines = []
    if category_in_play and "Demographic" in t.columns:
        groups = latest_rows.dropna(subset=[pos_col]).sort_values(pos_col, ascending=False)
        if len(groups) >= 2:
            top = groups.iloc[0]; bot = groups.iloc[-1]
            lines.append(
                f"In {latest}, {top['Demographic']} is highest on Positive ({float(top[pos_col]):.1f}%), "
                f"while {bot['Demographic']} is lowest ({float(bot[pos_col]):.1f}%)."
            )
        elif len(groups) == 1:
            g = groups.iloc[0]
            lines.append(f"In {latest}, {g['Demographic']} has Positive at {float(g[pos_col]):.1f}%.")

        ys = sorted(pd.to_numeric(t["Year"], errors="coerce").dropna().unique().tolist())
        prev = int(ys[-2]) if len(ys) >= 2 else None
        if prev is not None:
            for gname in latest_rows.sort_values(pos_col, ascending=False).head(3)["Demographic"]:
                s = t[t["Demographic"] == gname]
                lp = s[s["Year"] == str(latest)][pos_col].dropna()
                pp = s[s["Year"] == str(prev)][pos_col].dropna()
                if not lp.empty and not pp.empty:
                    delta = float(lp.iloc[0]) - float(pp.iloc[0])
                    lines.append(f"{gname}: {latest} {float(lp.iloc[0]):.1f}% ({delta:+.1f} pts vs {prev}).")
    else:
        ys = sorted(pd.to_numeric(t["Year"], errors="coerce").dropna().unique().tolist())
        prev = int(ys[-2]) if len(ys) >= 2 else None
        if prev is not None:
            lp = latest_rows[pos_col].dropna()
            pp = t[t["Year"] == str(prev)][pos_col].dropna()
            if not lp.empty and not pp.empty:
                delta = float(lp.iloc[0]) - float(pp.iloc[0])
                lines.append(f"Overall: {latest} {float(lp.iloc[0]):.1f}% ({delta:+.1f} pts vs {prev}).")
    return " ".join(lines) if lines else "No notable changes to report based on Positive."

# ─────────────────────────────
# UI
# ─────────────────────────────
def run_menu1():
    st.markdown("""
        <style>
            body { background-image: none !important; background-color: white !important; }
            .block-container { padding-top: 1rem !important; }
            .menu-banner {
                display: block;
                margin: 0 auto 16px;
                max-width: 100%;
                height: auto;
                max-height: 120px;  /* shrink new banner */
                object-fit: contain;
            }
            .custom-header { font-size: 30px !important; font-weight: 700; margin-bottom: 10px; }
            .custom-instruction { font-size: 16px !important; line-height: 1.4; margin-bottom: 10px; color: #333; }
            .field-label { font-size: 18px !important; font-weight: 600 !important; margin-top: 12px !important; margin-bottom: 2px !important; color: #222 !important; }
            .big-button button { font-size: 18px !important; padding: 0.75em 2em !important; margin-top: 20px; }
        </style>
    """, unsafe_allow_html=True)

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    left, center, right = st.columns([1, 3, 1])
    with center:
        # 🔄 Banner updated here
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png' alt='PSES Banner'>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-instruction">
                Select a question, year(s), and (optionally) a demographic category and subgroup.<br>
                The query always uses <b>QUESTION</b>, <b>Year</b>, and <b>DEMCODE</b>.
            </div>
        """, unsafe_allow_html=True)

        # ... rest of your code unchanged ...
