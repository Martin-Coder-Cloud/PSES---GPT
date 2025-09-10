# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# RAW + METADATA-FIRST
# • No normalization of the results dataset.
# • Always resolve QUESTION and DEMCODE(s) from metadata first.
# • DEMCODE(s) sent to the loader are 4-digit codes from Demographics.xlsx (blank only for "All respondents").
# • Adds "Raw results (full rows)" display for validation (all columns), as requested.

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
    # Expect "question" (code) and "english" (text)
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
# Helpers (metadata-first)
# ─────────────────────────────
def _find_demcode_col(demo_df: pd.DataFrame) -> str | None:
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            return c
    return None

def _four_digit(s: str) -> str:
    # Keep digits only and left-pad to 4 (metadata is authoritative; this just standardizes)
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
        # Fall back to overall if category not found
        return [None], {None: "All respondents"}, False

    # If a specific subgroup is selected, resolve its code from metadata
    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                raw_code = str(row.iloc[0][code_col])
                code4 = _four_digit(raw_code)
                # Only accept 4-digit; if it ends empty, we still return original (just in case)
                code_final = code4 if code4 else raw_code
                return [code_final], {code_final: subgroup_label}, True
        # Fallback: if code column is not present, use label as the identifier (not ideal)
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # No subgroup selected -> include all 4-digit codes defined for the category
    if code_col and LABEL_COL in df_cat.columns:
        pairs = []
        for _, r in df_cat.iterrows():
            raw_code = str(r[code_col])
            label = str(r[LABEL_COL])
            code4 = _four_digit(raw_code)
            if code4:  # must be 4-digit to pass
                pairs.append((code4, label))
        if pairs:
            demcodes = [c for c, _ in pairs]
            disp_map = {c: l for c, l in pairs}
            return demcodes, disp_map, True

    # Fallback when code column missing: use labels (last resort)
    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        return labels, {l: l for l in labels}, True

    # If nothing resolvable, treat as overall
    return [None], {None: "All respondents"}, False

def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    """Return [(raw_col, display_label)] for answer1..answer7 using scales metadata."""
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
            .menu-banner { width: 100%; height: auto; display: block; margin-top: 0px; margin-bottom: 20px; }
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
        st.markdown(
            "<img class='menu-banner' "
            "style='max-width:600px; height:auto;' "
            "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-instruction">
                Select a question, year(s), and (optionally) a demographic category and subgroup.<br>
                The query always uses <b>QUESTION</b>, <b>Year</b>, and <b>DEMCODE</b>.
            </div>
        """, unsafe_allow_html=True)

        # Question (from metadata)
        st.markdown('<div class="field-label">Select a survey question:</div>', unsafe_allow_html=True)
        question_options = qdf["display"].tolist()
        selected_label = st.selectbox("Question", question_options, key="question_dropdown", label_visibility="collapsed")
        question_code = qdf.loc[qdf["display"] == selected_label, "code"].values[0]
        question_text = qdf.loc[qdf["display"] == selected_label, "text"].values[0]

        # Years
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        all_years = [2024, 2022, 2020, 2019]
        select_all = st.checkbox("All years", value=True, key="select_all_years")
        selected_years = []
        year_cols = st.columns(len(all_years))
        for idx, yr in enumerate(all_years):
            with year_cols[idx]:
                checked = True if select_all else False
                if st.checkbox(str(yr), value=checked, key=f"year_{yr}"):
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
            if sub_selection == "":
                sub_selection = None

        # Resolve DEMCODE(s) from metadata (4-digit) and show parameters BEFORE query
        demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(demo_df, demo_selection, sub_selection)
        dem_display = ["(blank)"] if demcodes == [None] else [str(c) for c in demcodes]

        params_df = pd.DataFrame({
            "Parameter": ["QUESTION (from metadata)", "SURVEYR (years)", "DEMCODE(s) (from metadata)"],
            "Value": [question_code, ", ".join(map(str, selected_years)), ", ".join(dem_display)]
        })
        st.markdown("##### Parameters that will be passed to the database")
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Run query (RAW, no normalization)
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Run query"):
                # Scale labels
                scale_pairs = get_scale_labels(sdf, question_code)

                # Pull results per DEMCODE via loader (RAW filter on trio + PS-wide guards when All respondents)
                parts = []
                for code in demcodes:
                    df_part = load_results2024_filtered(
                        question_code=question_code,
                        years=selected_years,
                        group_value=code  # None => blank DEMCODE (PS-wide guards applied in loader)
                    )
                    if df_part is not None and not df_part.empty:
                        parts.append(df_part)

                if not parts:
                    st.info("No data found for this selection.")
                    return

                df_raw = pd.concat(parts, ignore_index=True)

                # Robust to header casing/whitespace WITHOUT normalizing the file
                def _find(df, target):
                    """Return the actual column name that matches target (case/space-insensitive)."""
                    t = target.strip().lower()
                    for c in df.columns:
                        if c is None:
                            continue
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
                    st.stop()

                # Rename only in-memory for the strict guard below
                df_raw = df_raw.rename(columns={QCOL: "QUESTION", SCOL: "SURVEYR", DCOL: "DEMCODE"})

                # SECOND strict guard on RAW columns (defensive)
                qmask = df_raw["QUESTION"].astype(str).str.strip().str.upper() == str(question_code).strip().upper()
                ymask = df_raw["SURVEYR"].astype(str).isin([str(y) for y in selected_years])
                if demcodes == [None]:
                    gmask = df_raw["DEMCODE"].astype(str).str.strip() == ""
                else:
                    gmask = df_raw["DEMCODE"].astype(str).isin([("" if gv is None else str(gv)) for gv in demcodes])
                df_raw = df_raw[qmask & ymask & gmask].copy()

                if df_raw.empty:
                    st.info("No data found after applying filters (exact QUESTION, selected SURVEYR years, DEMCODE set).")
                    return

                # Exclude 999 for display & narrative
                df_raw = exclude_999_raw(df_raw)
                if df_raw.empty:
                    st.info("Data exists, but all rows are not applicable (999).")
                    return

                # ===== NEW: Raw results (full rows) for validation =====
                st.markdown("#### Raw results (full rows)")
                # Sort by year desc for readability if SURVEYR exists
                if "SURVEYR" in df_raw.columns:
                    try:
                        _yn = pd.to_numeric(df_raw["SURVEYR"], errors="coerce")
                        df_raw = df_raw.loc[_yn.sort_values(ascending=False).index] if not _yn.isna().all() else df_raw
                    except Exception:
                        pass
                st.dataframe(df_raw, use_container_width=True)

                # Optional: raw Excel download to inspect all fields
                with io.BytesIO() as buf:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_raw.to_excel(writer, sheet_name="RawRows", index=False)
                    raw_bytes = buf.getvalue()

                st.download_button(
                    label="⬇️ Download raw rows (full columns)",
                    data=raw_bytes,
                    file_name=f"PSES_{question_code}_{'-'.join(map(str, selected_years))}_RAW.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                # ===== end NEW =====

                # Title
                st.subheader(f"{question_code} — {question_text}")

                # Display (formatted) table
                df_disp = format_display_table_raw(
                    df=df_raw,
                    category_in_play=category_in_play,
                    dem_disp_map=({None: "All respondents"} | {str(k): v for k, v in disp_map.items()}),
                    scale_pairs=scale_pairs
                )
                st.dataframe(df_disp, use_container_width=True)

                # Narrative (Positive only)
                st.markdown("#### Summary (Positive only)")
                st.write(narrative_positive_only_raw(df_disp, category_in_play))

                # Excel download (exactly what is displayed)
                with io.BytesIO() as buf:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_disp.to_excel(writer, sheet_name="Results", index=False)
                        ctx = {
                            "QUESTION": question_code,
                            "SURVEYR (years)": ", ".join(map(str, selected_years)),
                            "DEMCODE(s)": ", ".join(dem_display),
                            "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        pd.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Context", index=False)
                    data = buf.getvalue()

                st.download_button(
                    label="⬇️ Download Excel",
                    data=data,
                    file_name=f"PSES_{question_code}_{'-'.join(map(str, selected_years))}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_menu1()
