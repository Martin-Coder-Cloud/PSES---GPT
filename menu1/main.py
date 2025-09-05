# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# Shows the EXACT query parameters BEFORE running the data extraction,
# then runs a strict filter on (QUESTION, SURVEYR, DEMCODE) via the loader.
#
# DEMCODE handling:
#   - All respondents  -> DEMCODE is blank ("")
#   - Category + Subgroup chosen -> single DEMCODE
#   - Category chosen, no Subgroup -> all DEMCODEs for that category
#
# Output: parameters summary (pre-query) -> title -> formatted table -> Positive-only narrative -> Excel download.

import io
from datetime import datetime
import pandas as pd
import streamlit as st

from utils.data_loader import load_results2024_filtered  # raw (no normalization) loader


# ─────────────────────────────
# Metadata loaders (cached)
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
    # Build display label (Q## – text)
    qdf["qnum"] = qdf["code"].astype(str).str.extract(r'Q?(\d+)', expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"].astype(str) + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]

@st.cache_data(show_spinner=False)
def load_scales_metadata() -> pd.DataFrame:
    sdf = pd.read_excel("metadata/Survey Scales.xlsx")
    sdf.columns = [c.strip().lower() for c in c in sdf.columns]  # ensure lower
    return sdf


# ─────────────────────────────
# Helpers
# ─────────────────────────────
def resolve_demographic_codes(demo_df: pd.DataFrame, category_label: str | None, subgroup_label: str | None):
    """
    Returns:
      demcodes: list[str|None]   (None means overall -> blank in query)
      disp_map: dict[key,label]  (for display mapping; includes None -> "All respondents")
      category_in_play: bool     (True if a category, not overall)
    """
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    # Find the column that carries the DEMCODE values in metadata
    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    # All respondents => blank DEMCODE
    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    # Filter rows for the chosen category
    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    # If a specific subgroup is selected, return that single code
    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                code = str(row.iloc[0][code_col])
                return [code], {code: subgroup_label}, True
        # Fallback: use the label as the identifier if code_col is absent
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # No subgroup selected -> return ALL codes for the category
    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    # Fallback when no explicit code column
    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        disp_map = {l: l for l in labels}
        return labels, disp_map, True

    return [None], {None: "All respondents"}, False


def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    # scales_df is lower-cased columns (loaded above)
    sdf = scales_df.copy()
    candidates = pd.DataFrame()
    for key in ["code", "question"]:
        if key in sdf.columns:
            candidates = sdf[sdf[key].astype(str).str.upper() == str(question_code).upper()]
            if not candidates.empty:
                break
    labels = []
    for i in range(1, 7 + 1):
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
    """Drop rows where any distribution or summary metric equals 999 (not applicable)."""
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
    """
    Build the display table directly from RAW columns:
      - Year (from SURVEYR)
      - Optional Demographic (mapped from DEMCODE to label; blank -> "All respondents")
      - Scale distribution (answer1..answer7 with scale labels)
      - POSITIVE / NEUTRAL / NEGATIVE / ANSCOUNT
      - Sorted by year desc (and Demographic asc if present)
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Year as numeric for sorting; keep string for display (avoid commas)
    out["__YearNum__"] = pd.to_numeric(out["SURVEYR"], errors="coerce").astype("Int64")
    out["Year"] = out["__YearNum__"].astype(str)

    if category_in_play:
        def to_label(code):
            code = "" if code is None else str(code)
            if code.strip() == "":
                return "All respondents"
            return dem_disp_map.get(code, dem_disp_map.get(str(code), str(code)))
        out["Demographic"] = out["DEMCODE"].apply(to_label)

    # Scale columns + rename with labels
    dist_cols = [f"answer{i}" for i in range(1, 8) if f"answer{i}" in out.columns]
    rename_map = {k: v for k, v in scale_pairs if k in out.columns}

    keep_cols = ["__YearNum__", "Year"] + (["Demographic"] if category_in_play else []) \
                + dist_cols + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    # Sort: newest year first, then Demographic asc (if present)
    sort_cols = ["__YearNum__"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["__YearNum__"])

    # Numeric formatting
    for c in out.columns:
        if c not in ("Year", "Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year", "Demographic", "ANSCOUNT")]
    out[pct_like] = out[pct_like].round(1)
    if "ANSCOUNT" in out.columns:
        out["ANSCOUNT"] = pd.to_numeric(out["ANSCOUNT"], errors="coerce").astype("Int64")

    return out


def narrative_positive_only_raw(df_disp: pd.DataFrame, category_in_play: bool) -> str:
    if df_disp.empty or "Positive" not in [c.title() for c in df_disp.columns]:
        # Our column is "POSITIVE" in raw; after rename we keep "POSITIVE" unless scale map renames it to "Positive"
        # To be consistent, check both.
        pos_col = "POSITIVE" if "POSITIVE" in df_disp.columns else "Positive" if "Positive" in df_disp.columns else None
        if not pos_col:
            return "No results available to summarize."
    else:
        pos_col = "Positive" if "Positive" in df_disp.columns else "POSITIVE"

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

        # Optional: change vs previous year for the top 3 groups
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
    # Styling (kept close to your current look)
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
        # Banner + header
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-instruction">
                Select a question, year(s), and (optionally) a demographic category and subgroup.<br>
                The query always uses <b>QUESTION</b>, <b>Year</b>, and <b>DEMCODE</b>.
            </div>
        """, unsafe_allow_html=True)

        # Question
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

        # Demographic Category + Subgroup
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
            sub_selection = st.selectbox(
                "(leave blank to include all subgroups in this category)",
                [""] + sub_items,
                key=f"sub_{demo_selection.replace(' ', '_')}",
                label_visibility="collapsed"
            )
            if sub_selection == "":
                sub_selection = None

        # ── NEW: Show parameters BEFORE running any query ─────────────────────
        demcodes, disp_map, category_in_play = resolve_demographic_codes(demo_df, demo_selection, sub_selection)
        dem_display = ["(blank)"] if demcodes == [None] else [str(c) for c in demcodes]
        params_df = pd.DataFrame({
            "Parameter": ["QUESTION", "SURVEYR (years)", "DEMCODE(s)"],
            "Value": [question_code, ", ".join(map(str, selected_years)), ", ".join(dem_display)]
        })
        st.markdown("##### Parameters that will be passed to the database")
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Search (run extraction only AFTER showing parameters)
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Run query"):
                # Scale labels for this question
                scale_pairs = get_scale_labels(sdf, question_code)

                # Pull & combine results (loader filters by raw columns)
                parts = []
                for code in demcodes:
                    df_part = load_results2024_filtered(
                        question_code=question_code,  # exact code
                        years=selected_years,         # exact years
                        group_value=code              # None => blank DEMCODE in loader
                    )
                    if df_part is not None and not df_part.empty:
                        parts.append(df_part)

                if not parts:
                    st.info("No data found for this selection.")
                    return

                df_raw = pd.concat(parts, ignore_index=True)

                # Exclude sentinel 999 rows
                df_raw = exclude_999_raw(df_raw)
                if df_raw.empty:
                    st.info("Data exists, but all rows are not applicable (999).")
                    return

                # Title
                st.subheader(f"{question_code} — {question_text}")

                # Display table (RAW columns)
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
