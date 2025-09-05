# Menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# Strict filtering on (QUESTION, SURVEYR, DEMCODE).
# DEMCODE handling:
#   - All respondents  -> blank ("")
#   - Category+Subgroup chosen -> single DEMCODE
#   - Category chosen, no Subgroup -> all DEMCODEs for that category
# Output: parameters summary -> title -> formatted table -> Positive-only narrative -> Excel download.

import io
from datetime import datetime

import pandas as pd
import streamlit as st

# Loader: reads Drive .csv.gz in chunks and (optionally) filters
from utils.data_loader import load_results2024_filtered


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
    # Build display label
    qdf["qnum"] = qdf["code"].astype(str).str.extract(r'Q?(\d+)', expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"].astype(str) + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]

@st.cache_data(show_spinner=False)
def load_scales_metadata() -> pd.DataFrame:
    sdf = pd.read_excel("metadata/Survey Scales.xlsx")
    # FIX: remove typo "for c in c in ..." and use a safe vectorized form
    sdf.columns = sdf.columns.str.strip().str.lower()
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

    # Try to locate the column that holds the DEMCODE values in metadata
    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    # All respondents => blank (represented here as None; handled later)
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
        # Fallback: use subgroup label as identifier if no code_col found
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # No subgroup selected -> return ALL codes for the category
    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        # keep non-blank codes
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    # Fallback: no explicit code column -> use labels
    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        disp_map = {l: l for l in labels}
        return labels, disp_map, True

    # If we get here, we couldn't resolve; treat as overall
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


def normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to canonical names used in this app:
      question_code, year, group_value, answer1..answer7, positive_pct, neutral_pct, negative_pct, n
    Supports raw PSES names (SURVEYR/DEMCODE/QUESTION/POSITIVE/NEUTRAL/NEGATIVE/ANSCOUNT).
    """
    out = df.copy()
    rename_map = {}
    if "SURVEYR" in out.columns and "year" not in out.columns: rename_map["SURVEYR"] = "year"
    if "QUESTION" in out.columns and "question_code" not in out.columns: rename_map["QUESTION"] = "question_code"
    if "DEMCODE" in out.columns and "group_value" not in out.columns: rename_map["DEMCODE"] = "group_value"
    if "POSITIVE" in out.columns and "positive_pct" not in out.columns: rename_map["POSITIVE"] = "positive_pct"
    if "NEUTRAL"  in out.columns and "neutral_pct"  not in out.columns: rename_map["NEUTRAL"]  = "neutral_pct"
    if "NEGATIVE" in out.columns and "negative_pct" not in out.columns: rename_map["NEGATIVE"] = "negative_pct"
    if "ANSCOUNT" in out.columns and "n"           not in out.columns: rename_map["ANSCOUNT"] = "n"
    for i in range(1, 8):
        up = f"ANSWER{i}"; lo = f"answer{i}"
        if up in out.columns and lo not in out.columns: rename_map[up] = lo
    if rename_map:
        out = out.rename(columns=rename_map)

    # Ensure presence of core columns
    for col in ["year", "question_code", "group_value", "positive_pct", "neutral_pct", "negative_pct", "n"]:
        if col not in out.columns:
            out[col] = pd.NA

    # Normalize types/values
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["question_code"] = out["question_code"].astype(str).str.strip().str.upper()
    if "group_value" in out.columns:
        gv = out["group_value"]
        out["group_value"] = gv.astype(str).where(~gv.isna(), "").str.strip()
        out.loc[out["group_value"].str.lower().isin(["nan", "none", "null"]), "group_value"] = ""

    return out


def drop_na_999(df: pd.DataFrame) -> pd.DataFrame:
    cols_check = [f"answer{i}" for i in range(1, 8)]
    cols_check += ["positive_pct", "neutral_pct", "negative_pct", "n"]
    present = [c for c in cols_check if c in df.columns]
    if not present:
        return df
    mask_keep = pd.Series(True, index=df.index)
    for c in present:
        vals = pd.to_numeric(df[c], errors="coerce")
        mask_keep &= (vals != 999)
    return df.loc[mask_keep].copy()


def format_table_for_display(df_slice: pd.DataFrame, scale_pairs, category_in_play: bool, dem_disp_map: dict) -> pd.DataFrame:
    if df_slice.empty:
        return df_slice.copy()
    out = df_slice.copy()

    out["YearNum"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["Year"] = out["YearNum"].astype(str)

    if category_in_play:
        def lbl(code):
            if code is None or str(code).strip() == "":
                return "All respondents"
            return dem_disp_map.get(code, dem_disp_map.get(str(code), str(code)))
        out["Demographic"] = out["group_value"].apply(lbl)

    dist_cols = [k for k, _ in scale_pairs if k in out.columns]
    rename_map = {k: v for k, v in scale_pairs if k in out.columns}

    keep_cols = ["YearNum", "Year"] + (["Demographic"] if category_in_play else []) \
                + dist_cols + ["positive_pct", "neutral_pct", "negative_pct", "n"]
    out = out[keep_cols].copy()
    out = out.rename(columns=rename_map)
    out = out.rename(columns={"positive_pct": "Positive", "neutral_pct": "Neutral", "negative_pct": "Negative", "n": "n"})

    sort_cols = ["YearNum"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["YearNum"])

    for c in out.columns:
        if c not in ("Year", "Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year", "Demographic", "n")]
    out[pct_like] = out[pct_like].round(1)
    if "n" in out.columns:
        out["n"] = out["n"].astype("Int64")

    return out


def build_positive_only_narrative(df_disp: pd.DataFrame, category_in_play: bool) -> str:
    if df_disp.empty or "Positive" not in df_disp.columns:
        return "No results available to summarize."
    t = df_disp.copy()
    t["_Y"] = pd.to_numeric(t["Year"], errors="coerce")
    latest = int(t["_Y"].max())
    latest_rows = t[t["_Y"] == latest]
    lines = []

    if category_in_play and "Demographic" in t.columns:
        groups = latest_rows.dropna(subset=["Positive"]).sort_values("Positive", ascending=False)
        if len(groups) >= 2:
            top = groups.iloc[0]; bot = groups.iloc[-1]
            lines.append(
                f"In {latest}, {top['Demographic']} is highest on Positive ({top['Positive']:.1f}%), "
                f"while {bot['Demographic']} is lowest ({bot['Positive']:.1f}%)."
            )
        elif len(groups) == 1:
            g = groups.iloc[0]
            lines.append(f"In {latest}, {g['Demographic']} has Positive at {g['Positive']:.1f}%.")

    def prev_year(subdf: pd.DataFrame):
        ys = sorted(pd.to_numeric(subdf["Year"], errors="coerce").dropna().unique().tolist())
        return int(ys[-2]) if len(ys) >= 2 else None

    if category_in_play and "Demographic" in t.columns:
        top3 = latest_rows.sort_values("Positive", ascending=False).head(3)["Demographic"].tolist()
        for g in top3:
            s = t[t["Demographic"] == g]
            prev = prev_year(s)
            if prev is not None:
                lp = s[s["Year"] == str(latest)]["Positive"].dropna()
                pp = s[s["Year"] == str(prev)]["Positive"].dropna()
                if not lp.empty and not pp.empty:
                    delta = lp.iloc[0] - pp.iloc[0]
                    lines.append(f"{g}: {latest} {lp.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")
    else:
        prev = prev_year(t)
        if prev is not None:
            lp = latest_rows["Positive"].dropna()
            pp = t[t["Year"] == str(prev)]["Positive"].dropna()
            if not lp.empty and not pp.empty:
                delta = lp.iloc[0] - pp.iloc[0]
                lines.append(f"Overall: {latest} {lp.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")

    return " ".join(lines) if lines else "No notable changes to report based on Positive."


# ─────────────────────────────
# UI (kept close to your current look)
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
        selected_label = st.selectbox(
            "Question",
            question_options,
            key="question_dropdown",
            label_visibility="collapsed"
        )
        question_code = qdf.loc[qdf["display"] == selected_label, "code"].values[0]
        question_text = qdf.loc[qdf["display"] == selected_label, "text"].values[0]

        # Years (fixed set you’re using right now)
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
        demo_selection = st.selectbox(
            "Demographic category",
            demo_categories,
            key="demo_main",
            label_visibility="collapsed"
        )

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

        # Search
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                # 1) Resolve DEMCODE(s) from metadata
                demcodes, disp_map, category_in_play = resolve_demographic_codes(demo_df, demo_selection, sub_selection)

                # Prepare DEMCODEs text for parameter summary
                dem_display = ["(blank)"] if demcodes == [None] else [str(c) for c in demcodes]

                # 2) Show PARAMETERS USED (for validation)
                params_df = pd.DataFrame({
                    "Parameter": ["QUESTION", "SURVEYR (years)", "DEMCODE(s)"],
                    "Value": [question_code, ", ".join(map(str, selected_years)), ", ".join(dem_display)]
                })
                st.markdown("##### Parameters used")
                st.dataframe(params_df, use_container_width=True, hide_index=True)

                # 3) Scale labels for this question
                scale_pairs = get_scale_labels(sdf, question_code)

                # 4) Pull and combine results for each DEMCODE
                parts = []
                for code in demcodes:
                    df_part = load_results2024_filtered(
                        question_code=question_code,   # exact code
                        years=selected_years,          # exact years
                        group_value=code               # None -> overall (blank) enforced after normalization
                    )
                    if df_part is not None and not df_part.empty:
                        parts.append(df_part)

                if not parts:
                    st.info("No data found for this selection.")
                    return

                df = pd.concat(parts, ignore_index=True)

                # 5) Normalize to canonical columns
                df = normalize_results_columns(df)

                # 6) STRICT guard on exact selections (QUESTION, SURVEYR, DEMCODE)
                qmask = df["question_code"] == str(question_code).strip().upper()
                ymask = df["year"].isin(pd.Series(selected_years, dtype="Int64"))
                if demcodes == [None]:  # All respondents -> DEMCODE blank only
                    gmask = (df["group_value"].astype(str).str.strip() == "")
                else:
                    demcode_strs = [str(c) for c in demcodes]
                    gmask = df["group_value"].astype(str).isin(demcode_strs)

                df = df[qmask & ymask & gmask].copy()

                if df.empty:
                    st.info("No data found after applying filters (exact QUESTION, selected SURVEYR years, DEMCODE set).")
                    return

                # 7) Exclude sentinel 999 rows
                df = drop_na_999(df)
                if df.empty:
                    st.info("Data exists, but all rows are not applicable (999).")
                    return

                # 8) Title
                st.subheader(f"{question_code} — {question_text}")

                # 9) Build display table
                df_disp = format_table_for_display(
                    df_slice=df,
                    scale_pairs=scale_pairs,
                    category_in_play=category_in_play,
                    dem_disp_map=({None: "All respondents"} | {str(k): v for k, v in disp_map.items()})
                )
                st.dataframe(df_disp, use_container_width=True)

                # 10) Narrative (Positive only)
                st.markdown("#### Summary (Positive only)")
                st.write(build_positive_only_narrative(df_disp, category_in_play))

                # 11) Excel download (exactly what is displayed)
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


if __name__ == "__main__":
    run_menu1()
