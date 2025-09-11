# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# RAW + METADATA-FIRST (centered UI, banner)
# • Filters on QUESTION, SURVEYR (years as strings), DEMCODE (exact trimmed characters).
# • PS-wide only: keep LEVEL1ID == "" or "0" (strings) when present.
# • Shows Raw results + formatted table (Answer 1–7 + P/N/N + ANSCOUNT) + 2024-first narrative.
# • No BYCOND, no dedup. No st.stop().
# • Diagnostics expander includes schema previews + DEMCODE scan.
# • Scoping fix: inside run_menu1 we use PD instead of pd to avoid local shadowing.

import io
import gzip
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.data_loader import load_results2024_filtered  # RAW loader (character-only)

# Use a stable alias for pandas inside run_menu1 to avoid any local-shadowing issues
PD = pd

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
    # Expect "question" (code) and "english" (text)
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["code"] = qdf["code"].astype(str).str.strip()
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
# Helpers (metadata-first)
# ─────────────────────────────
def _find_demcode_col(demo_df: pd.DataFrame) -> str | None:
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            return c
    return None

def _four_digit(s: str) -> str:
    # Keep digits only and left-pad to 4 (metadata is authoritative)
    s = "".join(ch for ch in str(s) if ch.isdigit())
    return s.zfill(4) if s else ""

def resolve_demographic_codes_from_metadata(
    demo_df: pd.DataFrame,
    category_label: str | None,
    subgroup_label: str | None
):
    """
    Returns:
      demcodes: list[str|None]   -> 4-digit codes from metadata (or [None] for All respondents)
      disp_map: dict[key,label]  -> {code: human label}, includes {None: "All respondents"}
      category_in_play: bool
    """
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    code_col = _find_demcode_col(demo_df)

    # All respondents => blank DEMCODE
    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    # Subset metadata to chosen category
    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    # If a specific subgroup is selected, resolve its code from metadata
    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                raw_code = str(row.iloc[0][code_col])
                code4 = _four_digit(raw_code)
                code_final = code4 if code4 else raw_code
                return [code_final.strip()], {code_final.strip(): subgroup_label}, True
        # Fallback: if code column missing, use the label as identifier
        return [str(subgroup_label).strip()], {str(subgroup_label).strip(): subgroup_label}, True

    # No subgroup selected -> include all 4-digit codes defined for the category
    if code_col and LABEL_COL in df_cat.columns:
        pairs = []
        for _, r in df_cat.iterrows():
            raw_code = str(r[code_col])
            label = str(r[LABEL_COL])
            code4 = _four_digit(raw_code)
            if code4:
                pairs.append((code4.strip(), label))
        if pairs:
            demcodes = [c for c, _ in pairs]
            disp_map = {c: l for c, l in pairs}
            return demcodes, disp_map, True

    # Fallback when code column missing: use labels (last resort)
    if LABEL_COL in df_cat.columns:
        labels = [str(l).strip() for l in df_cat[LABEL_COL].tolist()]
        return labels, {l: l for l in labels}, True

    # If nothing resolvable, treat as overall
    return [None], {None: "All respondents"}, False

def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    """Return [(raw_col, display_label)] for answer1..answer7 using scales metadata."""
    sdf = scales_df.copy()
    candidates = pd.DataFrame()
    for key in ["code", "question"]:
        if key in sdf.columns:
            candidates = sdf[sdf[key].astype(str).str.strip() == str(question_code).strip()]
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
        labels.append((col, lbl or f"Answer {i}"))
    return labels

def exclude_999_raw(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"answer{i}" for i in range(1, 8)] + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    out = df.copy()
    keep = PD.Series(True, index=out.index)
    for c in present:
        vals = PD.to_numeric(out[c], errors="coerce")
        keep &= (vals != 999)
    return out.loc[keep].copy()

def format_display_table_raw(df: pd.DataFrame, category_in_play: bool, dem_disp_map: dict, scale_pairs) -> pd.DataFrame:
    """Case-robust display table for ANSWER1..7 + POS/NEU/NEG + ANSCOUNT."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out["__YearNum__"] = PD.to_numeric(out["SURVEYR"], errors="coerce").astype("Int64")
    out["Year"] = out["SURVEYR"].astype(str)

    if category_in_play:
        def to_label(code):
            key = "" if code is None else str(code).strip()
            if key == "":
                return "All respondents"
            return dem_disp_map.get(key, str(code))
        out["Demographic"] = out["DEMCODE"].apply(to_label)

    # Distribution columns (keep original case where available)
    dist_cols = []
    for i in range(1, 8):
        lc, uc = f"answer{i}", f"ANSWER{i}"
        if uc in out.columns: dist_cols.append(uc)
        elif lc in out.columns: dist_cols.append(lc)

    # Map to scale labels
    rename_map = {}
    for k, v in scale_pairs:  # k like "answer1"
        ku = k.upper()
        if ku in out.columns: rename_map[ku] = v
        if k  in out.columns: rename_map[k]  = v

    keep_cols = ["__YearNum__", "Year"] + (["Demographic"] if category_in_play else []) \
                + dist_cols + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    sort_cols = ["__YearNum__"] + (["Demographic"] if category_in_play else [])
    sort_asc  = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["__YearNum__"])

    # Numeric formatting only for display
    for c in out.columns:
        if c not in ("Year", "Demographic"):
            out[c] = PD.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year", "Demographic", "ANSCOUNT")]
    if pct_like:
        out[pct_like] = out[pct_like].round(1)
    if "ANSCOUNT" in out.columns:
        out["ANSCOUNT"] = PD.to_numeric(out["ANSCOUNT"], errors="coerce").astype("Int64")

    return out

# ─────────────────────────────
# Narrative: 2024-first, full sentences
# ─────────────────────────────
def build_narrative_2024_first_full(df_disp: pd.DataFrame, category_in_play: bool, question_code: str, question_text: str) -> str:
    if df_disp is None or df_disp.empty:
        return "No results are available to summarize."

    # Find Positive column case-robustly
    pos_col = None
    for c in df_disp.columns:
        if str(c).strip().lower() == "positive":
            pos_col = c; break
    if pos_col is None:
        return "No results are available to summarize."

    def pct(v):
        try: return f"{float(v):.1f}%"
        except Exception: return "n/a"

    def pts(v):
        try:
            v = float(v); sign = "+" if v >= 0 else "-"
            return f"{sign}{abs(v):.1f} pts"
        except Exception: return "±0.0 pts"

    t = df_disp.copy()
    t["_Y"] = PD.to_numeric(t["Year"], errors="coerce")
    t = t.dropna(subset=["_Y"]).copy()
    if t.empty: return "No results are available to summarize."

    years_sorted = sorted(t["_Y"].unique().tolist())
    target_year = 2024 if 2024 in years_sorted else int(max(years_sorted))
    prev_years = [y for y in years_sorted if y < target_year]
    prev_year = int(max(prev_years)) if prev_years else None
    earliest_year = int(min(years_sorted))
    latest_year = int(max(years_sorted))

    # Overall series: use explicit overall if present; else (weighted) mean across groups per year
    def overall_series(df: pd.DataFrame) -> pd.Series:
        if "Demographic" not in df.columns:
            return PD.to_numeric(df.set_index("_Y")[pos_col], errors="coerce").dropna()
        cand = df[df["Demographic"].astype(str).str.lower().isin(["all respondents", "overall"])]
        if not cand.empty:
            s = PD.to_numeric(cand.set_index("_Y")[pos_col], errors="coerce").dropna()
            if not s.empty: return s
        tmp = df.copy(); tmp[pos_col] = PD.to_numeric(tmp[pos_col], errors="coerce")
        series = {}
        for y, g in tmp.groupby("_Y"):
            vals = g[pos_col].dropna()
            if vals.empty: continue
            if "ANSCOUNT" in g.columns and g["ANSCOUNT"].notna().any():
                w = PD.to_numeric(g["ANSCOUNT"], errors="coerce").fillna(0)
                total = float(w.sum())
                series[y] = float((vals * w).sum() / total) if total > 0 else float(vals.mean())
            else:
                series[y] = float(vals.mean())
        return pd.Series(series)

    overall = overall_series(t)
    parts: list[str] = []

    # Overall 2024 (or latest)
    if target_year in overall.index:
        ty = overall.loc[target_year]
        parts.append(f"Overall, the results for {question_code} show that {pct(ty)} {question_text} in {target_year}.")
        if earliest_year in overall.index and earliest_year != target_year:
            ey = overall.loc[earliest_year]; change_total = ty - ey
            if abs(change_total) < 1.0:
                parts.append(f"Results have been stable since {earliest_year} (from {pct(ey)} to {pct(ty)}, {pts(change_total)}).")
            else:
                direction = 'increased' if change_total > 0 else 'decreased'
                parts.append(f"Since {earliest_year}, overall results have {direction} by {pts(change_total)} (from {pct(ey)} to {pct(ty)}).")
        if prev_year is not None and prev_year in overall.index:
            py = overall.loc[prev_year]; diff = ty - py; verb = "higher" if diff >= 0 else "lower"
            parts.append(f"Compared with {prev_year}, {target_year} is {pts(diff)} {verb} (from {pct(py)} to {pct(ty)}).")

    # Demographic snapshot + gap + trends
    if category_in_play and "Demographic" in t.columns:
        ty_slice = t[t["_Y"] == target_year].copy()
        ty_slice[pos_col] = PD.to_numeric(ty_slice[pos_col], errors="coerce")
        ty_slice = ty_slice.dropna(subset=[pos_col])

        if not ty_slice.empty:
            lines = []
            for g, gdf in t.groupby("Demographic"):
                gdf = gdf.set_index("_Y")
                if target_year in gdf.index:
                    tyv = PD.to_numeric(gdf.loc[target_year, pos_col], errors="coerce")
                    if pd.notna(tyv):
                        if prev_year is not None and prev_year in gdf.index:
                            pyv = PD.to_numeric(gdf.loc[prev_year, pos_col], errors="coerce")
                            if pd.notna(pyv):
                                delta_gp = tyv - pyv; dir_word = "higher" if delta_gp >= 0 else "lower"
                                lines.append(f"{g}: {pct(tyv)} in {target_year} ({pts(delta_gp)} {dir_word} than {prev_year}).")
                            else:
                                lines.append(f"{g}: {pct(tyv)} in {target_year}.")
                        else:
                            lines.append(f"{g}: {pct(tyv)} in {target_year}.")
            if lines:
                parts.append("By demographic in " + str(target_year) + ": " + " ".join(lines))

            ordered = ty_slice[["Demographic", pos_col]].sort_values(pos_col, ascending=False)
            if not ordered.empty:
                top_g, top_v = ordered.iloc[0]["Demographic"], float(ordered.iloc[0][pos_col])
                bot_g, bot_v = ordered.iloc[-1]["Demographic"], float(ordered.iloc[-1][pos_col])
                gap_now = top_v - bot_v
                if prev_year is not None:
                    prev_slice = t[t["_Y"] == prev_year][["Demographic", pos_col]].copy()
                    prev_slice[pos_col] = PD.to_numeric(prev_slice[pos_col], errors="coerce")
                    prev_map = prev_slice.set_index("Demographic")[pos_col].to_dict()
                    if top_g in prev_map and bot_g in prev_map and pd.notna(prev_map[top_g]) and pd.notna(prev_map[bot_g]):
                        gap_prev = float(prev_map[top_g]) - float(prev_map[bot_g])
                        gap_change = gap_now - gap_prev
                        widen_status = "widened" if gap_change > 0 else ("narrowed" if gap_change < 0 else "held steady")
                        parts.append(
                            f"The {target_year} gap between {top_g} ({pct(top_v)}) and {bot_g} ({pct(bot_v)}) is {pts(gap_now)}; it has {widen_status} by {pts(gap_change)} since {prev_year}."
                        )
                    else:
                        parts.append(
                            f"The {target_year} gap between {top_g} ({pct(top_v)}) and {bot_g} ({pct(bot_v)}) is {pts(gap_now)}."
                        )
                else:
                    parts.append(
                        f"The {target_year} gap between {top_g} ({pct(top_v)}) and {bot_g} ({pct(bot_v)}) is {pts(gap_now)}."
                    )

        trend_bits = []
        for g, gdf in t.groupby("Demographic"):
            gdf = gdf.dropna(subset=[pos_col]).copy()
            if gdf.empty: continue
            gdf["_Y"] = PD.to_numeric(gdf["Year"], errors="coerce")
            gdf = gdf.dropna(subset=["_Y"]).sort_values("_Y")
            if gdf.empty: continue
            fy, ly = int(gdf["_Y"].iloc[0]), int(gdf["_Y"].iloc[-1])
            fv = float(PD.to_numeric(gdf[pos_col].iloc[0], errors="coerce"))
            lv = float(PD.to_numeric(gdf[pos_col].iloc[-1], errors="coerce"))
            if fy == ly: trend_bits.append(f"{g}: {pct(lv)} in {ly}.")
            else:        trend_bits.append(f"{g}: {pct(fv)} in {fy} → {pct(lv)} in {ly} ({pts(lv - fv)}).")
        if trend_bits:
            parts.append("Trends over time by demographic: " + " ".join(trend_bits))
    else:
        if earliest_year in overall.index and latest_year in overall.index and earliest_year != latest_year:
            fv, lv = float(overall.loc[earliest_year]), float(overall.loc[latest_year])
            parts.append(f"Over time overall: {pct(fv)} in {earliest_year} → {pct(lv)} in {latest_year} ({pts(lv - fv)}).")

    return " ".join(parts) if parts else "No notable changes are evident."

# ─────────────────────────────
# UI
# ─────────────────────────────
def run_menu1():
    st.markdown("""
    <style>
      .custom-header{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }
      .custom-instruction{ font-size: 15px; line-height: 1.4; margin-bottom: 8px; color: #333; }
      .field-label{ font-size: 16px; font-weight: 600; margin: 10px 0 2px; color: #222; }
      .big-button button{ font-size: 16px; padding: 0.6em 1.6em; margin-top: 16px; }
    </style>
    """, unsafe_allow_html=True)

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    # Centered column with generous margins on both sides
    left, center, right = st.columns([1, 2, 1])
    with center:
        # Banner (85% width, max 940px, centered)
        st.markdown(
            "<img style='width:85%;max-width:940px;height:auto;display:block;margin:0 auto 16px;' "
            "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="custom-instruction">Select a question, year(s), and (optionally) a demographic category and subgroup.<br>'
            'The query always uses <b>QUESTION</b>, <b>Year</b>, and <b>DEMCODE</b>.</div>',
            unsafe_allow_html=True
        )

        # Question (from metadata)
        st.markdown('<div class="field-label">Select a survey question:</div>', unsafe_allow_html=True)
        question_options = qdf["display"].tolist()
        selected_label = st.selectbox("Question", question_options, key="question_dropdown", label_visibility="collapsed")
        question_code = qdf.loc[qdf["display"] == selected_label, "code"].values[0]
        question_text = qdf.loc[qdf["display"] == selected_label, "text"].values[0]

        # Years (strings)
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        all_years = ["2024", "2022", "2020", "2019"]
        select_all = st.checkbox("All years", value=True, key="select_all_years")
        selected_years: list[str] = []
        year_cols = st.columns(len(all_years))
        for idx, yr in enumerate(all_years):
            with year_cols[idx]:
                checked = True if select_all else False
                if st.checkbox(yr, value=checked, key=f"year_{yr}"):
                    selected_years.append(yr)
        selected_years = sorted(selected_years)
        if not selected_years:
            st.warning("⚠️ Please select at least one year.")
            return

        # Demographic category/subgroup (resolve codes from metadata)
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
        st.markdown('<div class="field-label">Select a demographic category (or All respondents):</div>', unsafe_allow_html=True)
        demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
        demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

        sub_selection = None
        if demo_selection != "All respondents":
            st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
            sub_items = demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL].dropna().astype(str).unique().tolist()
            sub_items = sorted(sub_items)
            sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=f"sub_{demo_selection.replace(' ', '_')}", label_visibility="collapsed")
            if sub_selection == "": sub_selection = None

        # Resolve DEMCODE(s) from metadata (4-digit) and show parameters BEFORE query
        demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(demo_df, demo_selection, sub_selection)
        dem_display = ["(blank)"] if demcodes == [None] else [str(c).strip() for c in demcodes]

        params_df = PD.DataFrame({
            "Parameter": ["QUESTION (from metadata)", "SURVEYR (years)", "DEMCODE(s) (from metadata)"],
            "Value": [question_code, ", ".join(selected_years), ", ".join(dem_display)]
        })
        st.markdown("##### Parameters that will be passed to the database")
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # ─────────────────────────────────────────
        # Diagnostics (self-contained fallback)
        # ─────────────────────────────────────────
        with st.expander("🛠 Diagnostics: file schema", expanded=False):
            def _fallback_after_loader(sample_rows: int = 5000) -> pd.DataFrame:
                """Read a small sample in text mode and report schema (object dtypes expected)."""
                try:
                    from utils.data_loader import ensure_results2024_local  # type: ignore
                    path = ensure_results2024_local()
                except Exception:
                    path = "/tmp/Results2024.csv.gz"
                with gzip.open(path, mode="rt", newline="") as f:
                    peek = PD.read_csv(
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
                        "dtype_after_loader": str(s.dtype),
                        "example_non_blank": ex,
                        "blank_rate": round(float(blank_rate), 3),
                    })
                return PD.DataFrame(rows)

            def _fallback_inferred(sample_rows: int = 5000) -> pd.DataFrame:
                """Show what pandas would infer (preview only; app still uses text mode)."""
                try:
                    from utils.data_loader import ensure_results2024_local  # type: ignore
                    path = ensure_results2024_local()
                except Exception:
                    path = "/tmp/Results2024.csv.gz"
                with gzip.open(path, mode="rt", newline="") as f:
                    peek = PD.read_csv(
                        f,
                        nrows=sample_rows,
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
                return PD.DataFrame(rows)

            colA, colB = st.columns(2)
            with colA:
                if st.button("Show dtypes after loader read (text mode)"):
                    try:
                        from utils.data_loader import get_results2024_schema  # optional helper
                        sch = get_results2024_schema()
                    except Exception:
                        sch = _fallback_after_loader()
                    st.write("All columns should be object (text).")
                    st.dataframe(sch, use_container_width=True, hide_index=True)
            with colB:
                if st.button("Show what pandas would infer (preview)"):
                    try:
                        from utils.data_loader import get_results2024_schema_inferred  # optional helper
                        sch2 = get_results2024_schema_inferred()
                    except Exception:
                        sch2 = _fallback_inferred()
                    st.write("Preview only — the app still reads as text.")
                    st.dataframe(sch2, use_container_width=True, hide_index=True)

            # --- DEMCODE scan (global + current QUESTION/years) ---
            st.markdown("---")
            if st.button("Scan DEMCODEs (global + for current QUESTION/years)"):
                import gzip as _gzip, re as _re
                from collections import Counter
                import pandas as _pd  # alias to avoid shadowing

                # Canonicalization identical to loader (matching only)
                def _canon_year(s):
                    t = "" if s is None else str(s).strip()
                    return _re.sub(r"\.0+$", "", t) if _re.fullmatch(r"\d+(?:\.0+)?", t) else t

                def _canon_demcode(s):
                    if s is None:
                        return ""
                    t = str(s).replace("\u00A0", " ").strip()
                    t = t.replace(" ", "")
                    if _re.fullmatch(r"\d+(?:\.0+)?", t):
                        t = _re.sub(r"\.0+$", "", t)
                    return t.zfill(4) if t.isdigit() and len(t) < 4 else t

                # Get source path via loader (same file the app uses)
                try:
                    from utils.data_loader import ensure_results2024_local as _ensure
                    path = _ensure()
                except Exception:
                    path = "/tmp/Results2024.csv.gz"

                total_rows = 0
                dem_blank = 0
                dem_nonblank = 0
                dem_counter_all = Counter()
                dem_counter_cur = Counter()

                years_canon = set(_canon_year(y) for y in selected_years)
                q_current = str(question_code).strip()

                # Stream the file in text mode, everything as strings (no inference)
                try:
                    with _gzip.open(path, mode="rt", newline="") as f:
                        for chunk in _pd.read_csv(
                            f,
                            chunksize=300_000,
                            dtype=str,
                            keep_default_na=False,
                            na_filter=False,
                            low_memory=True,
                        ):
                            # normalize headers and trim all cells
                            chunk.columns = [str(c).strip().upper() for c in chunk.columns]
                            for c in chunk.columns:
                                chunk[c] = chunk[c].astype(str).str.strip()

                            total_rows += len(chunk)

                            if "DEMCODE" not in chunk.columns:
                                continue

                            # Global DEMCODE presence
                            dem_norm = chunk["DEMCODE"].apply(_canon_demcode)
                            nonblank_mask = dem_norm != ""
                            dem_nonblank += int(nonblank_mask.sum())
                            dem_blank += int((~nonblank_mask).sum())

                            # Count global codes
                            dem_counter_all.update(dem_norm[nonblank_mask].tolist())

                            # Count codes for current QUESTION + selected years
                            if {"QUESTION", "SURVEYR"}.issubset(chunk.columns):
                                qmask = chunk["QUESTION"] == q_current
                                ymask = chunk["SURVEYR"].apply(_canon_year).isin(years_canon)
                                cur_mask = nonblank_mask & qmask & ymask
                                if cur_mask.any():
                                    dem_counter_cur.update(dem_norm[cur_mask].tolist())

                    # Summaries
                    st.write("**DEMCODE presence (entire file):**")
                    st.write({
                        "total_rows": int(total_rows),
                        "demcode_nonblank_rows": int(dem_nonblank),
                        "demcode_blank_rows": int(dem_blank),
                        "distinct_demcodes_overall": int(len(dem_counter_all)),
                    })

                    # Top overall DEMCODEs
                    if dem_counter_all:
                        df_all = _pd.DataFrame(dem_counter_all.most_common(30), columns=["DEMCODE (canon)", "count"])
                        st.markdown("**Top DEMCODEs overall (first 30):**")
                        st.dataframe(df_all, use_container_width=True, hide_index=True)
                    else:
                        st.info("No non-blank DEMCODEs found anywhere in the file.")

                    # Top DEMCODEs for current QUESTION + years
                    st.markdown(f"**Top DEMCODEs for {q_current} in years {', '.join(selected_years)} (first 30):**")
                    if dem_counter_cur:
                        df_cur = _pd.DataFrame(dem_counter_cur.most_common(30), columns=["DEMCODE (canon)", "count"])
                        st.dataframe(df_cur, use_container_width=True, hide_index=True)
                    else:
                        st.info("No DEMCODEs found for the current QUESTION + years selection.")

                except FileNotFoundError:
                    st.error(f"Results file not found at: {path}")
                except Exception as e:
                    st.exception(e)

        # Run query (RAW, character-only filtering in loader)
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Run query"):
                # Scale labels
                scale_pairs = get_scale_labels(sdf, question_code)

                # Pull results per DEMCODE via loader
                parts = []
                for code in demcodes:
                    df_part = load_results2024_filtered(
                        question_code=question_code,               # exact string
                        years=selected_years,                      # strings
                        group_value=(None if code is None else str(code).strip())
                    )
                    if df_part is not None and not df_part.empty:
                        parts.append(df_part)

                if not parts:
                    st.info("No data found for this selection.")
                    return

                df_raw = PD.concat(parts, ignore_index=True)

                # Case-robust column alignment (rename in-memory)
                def _find(df, target):
                    t = target.strip().lower()
                    for c in df.columns:
                        if c is None: continue
                        if str(c).strip().lower() == t:
                            return c
                    return None

                QCOL = _find(df_raw, "QUESTION")
                SCOL = _find(df_raw, "SURVEYR")
                DCOL = _find(df_raw, "DEMCODE")

                if not all([QCOL, SCOL, DCOL]):
                    st.error(
                        "Required columns not found in data for filtering.\n\n"
                        f"Expected at least: QUESTION, SURVEYR, DEMCODE\n"
                        f"Columns present: {list(df_raw.columns)}"
                    )
                    return  # no st.stop(); let root render Return button

                df_raw = df_raw.rename(columns={QCOL: "QUESTION", SCOL: "SURVEYR", DCOL: "DEMCODE"})

                # Second strict filter (exact, trimmed)
                qmask = df_raw["QUESTION"].astype(str).str.strip() == str(question_code).strip()
                ymask = df_raw["SURVEYR"].astype(str).str.strip().isin(set(selected_years))
                dem_series = df_raw["DEMCODE"].astype(str).str.strip()
                if demcodes == [None]:
                    gmask = dem_series == ""
                else:
                    target_codes = {str(gv).strip() for gv in demcodes if gv is not None}
                    gmask = dem_series.isin(target_codes)
                df_raw = df_raw[qmask & ymask & gmask].copy()

                if df_raw.empty:
                    st.info("No data found after applying filters (exact QUESTION, selected SURVEYR years, DEMCODE set).")
                    return

                # PS-wide: LEVEL1ID == "" or "0"
                if "LEVEL1ID" in df_raw.columns:
                    lvl = df_raw["LEVEL1ID"].astype(str).str.strip()
                    df_raw = df_raw[(lvl == "") | (lvl == "0")].copy()

                if df_raw.empty:
                    st.info("No PS-wide rows (LEVEL1ID == '' or '0') for this selection.")
                    return

                # Exclude 999 for display & narrative
                df_raw = exclude_999_raw(df_raw)
                if df_raw.empty:
                    st.info("Data exists, but all rows are not applicable (999).")
                    return

                # ===== Raw results (full rows) =====
                st.markdown("#### Raw results (full rows)")
                if "SURVEYR" in df_raw.columns:
                    df_raw = df_raw.sort_values(
                        by="SURVEYR",
                        ascending=False,
                        key=lambda s: PD.to_numeric(s, errors="coerce")
                    )
                st.dataframe(df_raw, use_container_width=True)

                with io.BytesIO() as buf:
                    with PD.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_raw.to_excel(writer, sheet_name="RawRows", index=False)
                    raw_bytes = buf.getvalue()

                st.download_button(
                    label="⬇️ Download raw rows (full columns)",
                    data=raw_bytes,
                    file_name=f"PSES_{question_code}_{'-'.join(selected_years)}_RAW.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                # ===== end Raw results =====

                # Title
                st.subheader(f"{question_code} — {question_text}")

                # Display (formatted) table
                dem_map_clean = {None: "All respondents"}
                try:
                    for k, v in (disp_map or {}).items():
                        dem_map_clean[(None if k is None else str(k).strip())] = v
                except Exception:
                    pass

                df_disp = format_display_table_raw(
                    df=df_raw,
                    category_in_play=category_in_play,
                    dem_disp_map=dem_map_clean,
                    scale_pairs=scale_pairs
                )
                st.dataframe(df_disp, use_container_width=True)

                # Narrative (2024-first)
                st.markdown("#### Summary")
                st.write(build_narrative_2024_first_full(df_disp, category_in_play, question_code, question_text))

                # Excel download (displayed table)
                with io.BytesIO() as buf:
                    with PD.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_disp.to_excel(writer, sheet_name="Results", index=False)
                        ctx = {
                            "QUESTION": question_code,
                            "SURVEYR (years)": ", ".join(selected_years),
                            "DEMCODE(s)": ", ".join(dem_display),
                            "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        PD.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Context", index=False)
                    data = buf.getvalue()

                st.download_button(
                    label="⬇️ Download Excel",
                    data=data,
                    file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.markdown('</div>', unsafe_allow_html=True)  # end .big-button


if __name__ == "__main__":
    run_menu1()
