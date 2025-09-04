# Menu1/main.py  — PSES AI Explorer (Menu 1: Search by Question)

import io
from datetime import datetime

import pandas as pd
import streamlit as st

# Loader: reads Drive .csv.gz in chunks and filters on QUESTION/YEAR/DEMCODE
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
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["qnum"] = qdf["code"].astype(str).str.extract(r'Q?(\d+)', expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"].astype(str) + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]

@st.cache_data(show_spinner=False)
def load_scales_metadata() -> pd.DataFrame:
    sdf = pd.read_excel("metadata/Survey Scales.xlsx")
    sdf.columns = [c.strip().lower() for c in sdf.columns]
    return sdf


# ─────────────────────────────
# Helpers
# ─────────────────────────────
def resolve_demographic_codes(demo_df, category_label, subgroup_label):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                code = str(row.iloc[0][code_col])
                return [code], {code: subgroup_label}, True
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        disp_map = {l: l for l in labels}
        return labels, disp_map, True

    return [None], {None: "All respondents"}, False


def get_scale_labels(scales_df, question_code):
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


def drop_na_999(df):
    cols_check = [f"answer{i}" for i in range(1, 8)] + ["positive_pct", "neutral_pct", "negative_pct", "n"]
    present = [c for c in cols_check if c in df.columns]
    if not present:
        return df
    mask_keep = pd.Series(True, index=df.index)
    for c in present:
        vals = pd.to_numeric(df[c], errors="coerce")
        mask_keep &= (vals != 999)
    return df.loc[mask_keep].copy()


def format_table_for_display(df_slice, dem_disp_map, category_in_play, scale_pairs):
    if df_slice.empty:
        return df_slice
    out = df_slice.copy()

    # Keep a numeric year for sorting, then render a string year to avoid "2,024"
    out["YearNum"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["Year"] = out["YearNum"].astype(str)

    if category_in_play:
        def lbl(code):
            if code is None or (isinstance(code, float) and pd.isna(code)) or str(code).strip() == "":
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

    # Sort by YearNum desc, then Demographic asc (if applicable)
    sort_cols = ["YearNum"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

    # Drop helper after sorting
    out = out.drop(columns=["YearNum"])

    # Numeric formatting
    for c in out.columns:
        if c not in ("Year", "Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year", "Demographic", "n")]
    out[pct_like] = out[pct_like].round(1)
    if "n" in out.columns:
        out["n"] = out["n"].astype("Int64")

    return out


def build_positive_only_narrative(df_disp, category_in_play):
    if df_disp.empty or "Positive" not in df_disp.columns:
        return "No results available to summarize."

    # Work with numeric years for analysis
    df_tmp = df_disp.copy()
    df_tmp["YearNum"] = pd.to_numeric(df_tmp["Year"], errors="coerce")

    latest_year = int(df_tmp["YearNum"].max())
    df_latest = df_tmp[df_tmp["YearNum"] == latest_year]

    lines = []

    # Across demographics (if applicable)
    if category_in_play and "Demographic" in df_tmp.columns:
        groups = df_latest.dropna(subset=["Positive"]).sort_values("Positive", ascending=False)
        if len(groups) >= 2:
            top = groups.iloc[0]
            bot = groups.iloc[-1]
            lines.append(
                f"In {latest_year}, {top['Demographic']} is highest on Positive ({top['Positive']:.1f}%), "
                f"while {bot['Demographic']} is lowest ({bot['Positive']:.1f}%)."
            )
        elif len(groups) == 1:
            g = groups.iloc[0]
            lines.append(f"In {latest_year}, {g['Demographic']} has Positive at {g['Positive']:.1f}%.")

    # Over time (latest vs most recent earlier)
    def previous_year(subdf):
        ys = sorted(pd.to_numeric(subdf["Year"], errors="coerce").dropna().unique().tolist())
        return int(ys[-2]) if len(ys) >= 2 else None

    if category_in_play and "Demographic" in df_tmp.columns:
        top3 = df_latest.sort_values("Positive", ascending=False).head(3)["Demographic"].tolist()
        for g in top3:
            s = df_tmp[df_tmp["Demographic"] == g]
            prev = previous_year(s)
            if prev is not None:
                latest_pos = s[s["Year"] == str(latest_year)]["Positive"].dropna()
                prev_pos = s[s["Year"] == str(prev)]["Positive"].dropna()
                if not latest_pos.empty and not prev_pos.empty:
                    delta = latest_pos.iloc[0] - prev_pos.iloc[0]
                    lines.append(f"{g}: {latest_year} {latest_pos.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")
    else:
        prev = previous_year(df_tmp)
        if prev is not None:
            latest_pos = df_latest["Positive"].dropna()
            prev_pos = df_tmp[df_tmp["Year"] == str(prev)]["Positive"].dropna()
            if not latest_pos.empty and not prev_pos.empty:
                delta = latest_pos.iloc[0] - prev_pos.iloc[0]
                lines.append(f"Overall: {latest_year} {latest_pos.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")

    if not lines:
        return "No notable changes to report based on Positive."
    return " ".join(lines)


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
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-instruction">
                Use this menu to explore results for a specific survey question.<br>
                Select a question, year(s), and optionally a demographic category and subgroup.
                The data pull always uses <b>QUESTION</b>, <b>Year</b>, and <b>DEMCODE</b>.
            </div>
        """, unsafe_allow_html=True)

        # Question
        st.markdown('<div class="field-label">Select a survey question:</div>', unsafe_allow_html=True)
        question_options = qdf["display"].tolist()
        selected_label = st.selectbox("Choose from the official list (type Q# or keywords to filter):", question_options, key="question_dropdown")
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

        # Demographics
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
        demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
        demo_selection = st.selectbox("", demo_categories, key="demo_main")

        sub_selection = None
        if demo_selection != "All respondents":
            st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
            sub_items = demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL].dropna().astype(str).unique().tolist()
            sub_items = sorted(sub_items)
            sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=f"sub_{demo_selection.replace(' ', '_')}")
            if sub_selection == "":
                sub_selection = None

        # Search
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                # 1) Resolve DEMCODE(s) via metadata
                demcodes, disp_map, category_in_play = resolve_demographic_codes(demo_df, demo_selection, sub_selection)

                # 2) Scale labels for this question
                scale_pairs = get_scale_labels(sdf, question_code)

                # 3) Pull and combine results for each DEMCODE (None => blank DEMCODE for overall)
                parts = []
                for code in demcodes:
                    df_part = load_results2024_filtered(
                        question_code=question_code,
                        years=selected_years,
                        group_value=code
                    )
                    if not df_part.empty:
                        parts.append(df_part)

                if not parts:
                    st.info("No data found for this selection.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                df = pd.concat(parts, ignore_index=True)

                # 3b) Post-filter guard to strictly enforce (QUESTION, YEAR, DEMCODE)
                qmask = df["question_code"].astype(str).str.strip().str.upper() == str(question_code).strip().upper()
                ymask = pd.to_numeric(df["year"], errors="coerce").astype("Int64").isin(selected_years)
                if demo_selection == "All respondents":
                    gmask = df["group_value"].isna() | (df["group_value"].astype(str).str.strip() == "")
                else:
                    gmask = df["group_value"].astype(str).isin([str(c) for c in demcodes])
                df = df[qmask & ymask & gmask].copy()

                if df.empty:
                    st.info("No data found after applying filters.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                # 4) Exclude any rows containing sentinel 999
                df = drop_na_999(df)
                if df.empty:
                    st.info("Data exists, but all rows are not applicable (999).")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                # 5) Title (question text)
                st.subheader(f"{question_code} — {question_text}")

                # 6) Build standardized display table
                df_disp = format_table_for_display(
                    df_slice=df,
                    dem_disp_map=disp_map,
                    category_in_play=category_in_play,
                    scale_pairs=scale_pairs
                )

                # 7) Show table
                st.dataframe(df_disp, use_container_width=True)

                # 8) Narrative (Positive only)
                st.markdown("#### Summary (Positive only)")
                summary = build_positive_only_narrative(df_disp, category_in_play)
                st.write(summary)

                # 9) Excel download (exactly what is displayed)
                with io.BytesIO() as buf:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_disp.to_excel(writer, sheet_name="Results", index=False)
                        ctx = {
                            "Question code": question_code,
                            "Question text": question_text,
                            "Years": ", ".join(map(str, selected_years)),
                            "Category": demo_selection,
                            "Subgroup": sub_selection or "(all in category)" if demo_selection != "All respondents" else "All respondents",
                            "DEMCODEs used": ", ".join(["(blank)" if (c is None or str(c).strip() == "") else str(c) for c in demcodes]),
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
