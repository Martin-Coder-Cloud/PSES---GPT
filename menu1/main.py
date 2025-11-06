# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)

from __future__ import annotations
import io
from datetime import datetime
from typing import Dict, List, Optional

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
    # Expect "question" (code) and "english" (text)
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["qnum"] = qdf["code"].astype(str).str.extract(r"Q?(\d+)", expand=False)
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

    # find actual code column
    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    # Overall PS-wide
    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    # Filter to category
    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    # Specific subgroup
    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                code = str(row.iloc[0][code_col])
                return [code], {code: subgroup_label}, True
        # fallback: use label as identifier
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # No subgroup → all codes in the category
    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    # Fallback: use labels
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


def drop_na_999(df: pd.DataFrame) -> pd.DataFrame:
    pos_col = "positive_pct" if "positive_pct" in df.columns else "POSITIVE" if "POSITIVE" in df.columns else None
    neu_col = "neutral_pct" if "neutral_pct" in df.columns else "NEUTRAL" if "NEUTRAL" in df.columns else None
    neg_col = "negative_pct" if "negative_pct" in df.columns else "NEGATIVE" if "NEGATIVE" in df.columns else None
    n_col = "n" if "n" in df.columns else "ANSCOUNT" if "ANSCOUNT" in df.columns else None

    cols_check = [c for c in [f"answer{i}" for i in range(1, 8)] if c in df.columns]
    cols_check += [c for c in [pos_col, neu_col, neg_col, n_col] if c]

    if not cols_check:
        return df

    mask_keep = pd.Series(True, index=df.index)
    for c in cols_check:
        vals = pd.to_numeric(df[c], errors="coerce")
        mask_keep &= (vals != 999)
    return df.loc[mask_keep].copy()


def normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # QUESTION → question_code
    if "question_code" not in out.columns:
        if "QUESTION" in out.columns:
            out = out.rename(columns={"QUESTION": "question_code"})
        else:
            for c in out.columns:
                if c.strip().lower() == "question":
                    out = out.rename(columns={c: "question_code"})
                    break

    # SURVEYR → year
    if "year" not in out.columns:
        if "SURVEYR" in out.columns:
            out = out.rename(columns={"SURVEYR": "year"})
        else:
            for c in out.columns:
                if c.strip().lower() in ("surveyr", "year"):
                    out = out.rename(columns={c: "year"})
                    break

    # DEMCODE → group_value
    if "group_value" not in out.columns:
        if "DEMCODE" in out.columns:
            out = out.rename(columns={"DEMCODE": "group_value"})
        else:
            for c in out.columns:
                if c.strip().lower() == "demcode":
                    out = out.rename(columns={c: "group_value"})
                    break

    # POSITIVE/NEUTRAL/NEGATIVE
    if "positive_pct" not in out.columns and "POSITIVE" in out.columns:
        out = out.rename(columns={"POSITIVE": "positive_pct"})
    if "neutral_pct" not in out.columns and "NEUTRAL" in out.columns:
        out = out.rename(columns={"NEUTRAL": "neutral_pct"})
    if "negative_pct" not in out.columns and "NEGATIVE" in out.columns:
        out = out.rename(columns={"NEGATIVE": "negative_pct"})

    # ANSCOUNT → n
    if "n" not in out.columns and "ANSCOUNT" in out.columns:
        out = out.rename(columns={"ANSCOUNT": "n"})

    return out


def format_table_for_display(df_slice, dem_disp_map, category_in_play, scale_pairs):
    if df_slice.empty:
        return df_slice

    out = df_slice.copy()
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
    out = out.rename(
        columns={
            "positive_pct": "Positive",
            "neutral_pct": "Neutral",
            "negative_pct": "Negative",
            "n": "n",
        }
    )

    # sort
    sort_cols = ["YearNum"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

    # drop helper
    out = out.drop(columns=["YearNum"])

    # numeric formatting
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

    t = df_disp.copy()
    t["_Y"] = pd.to_numeric(t["Year"], errors="coerce")
    latest_year = int(t["_Y"].max())
    df_latest = t[t["_Y"] == latest_year]

    lines: List[str] = []

    # Across demographics
    if category_in_play and "Demographic" in t.columns:
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

    # Over time helper
    def prev_year(subdf):
        ys = sorted(pd.to_numeric(subdf["Year"], errors="coerce").dropna().unique().tolist())
        return int(ys[-2]) if len(ys) >= 2 else None

    if category_in_play and "Demographic" in t.columns:
        top3 = df_latest.sort_values("Positive", ascending=False).head(3)["Demographic"].tolist()
        for g in top3:
            s = t[t["Demographic"] == g]
            prev = prev_year(s)
            if prev is not None:
                latest_pos = s[s["Year"] == str(latest_year)]["Positive"].dropna()
                prev_pos = s[s["Year"] == str(prev)]["Positive"].dropna()
                if not latest_pos.empty and not prev_pos.empty:
                    delta = latest_pos.iloc[0] - prev_pos.iloc[0]
                    lines.append(f"{g}: {latest_year} {latest_pos.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")
    else:
        prev = prev_year(t)
        if prev is not None:
            latest_pos = df_latest["Positive"].dropna()
            prev_pos = t[t["Year"] == str(prev)]["Positive"].dropna()
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
    # styling
    st.markdown(
        """
        <style>
            body { background-image: none !important; background-color: white !important; }
            .block-container { padding-top: 1rem !important; }
            .menu-banner { width: 100%; height: auto; display: block; margin-top: 0px; margin-bottom: 20px; }
            .custom-header { font-size: 30px !important; font-weight: 700; margin-bottom: 10px; }
            .custom-instruction { font-size: 16px !important; line-height: 1.4; margin-bottom: 10px; color: #333; }
            .field-label { font-size: 18px !important; font-weight: 600 !important; margin-top: 12px !important; margin-bottom: 2px !important; color: #222 !important; }
            .big-button button { font-size: 18px !important; padding: 0.75em 2em !important; margin-top: 20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    left, center, right = st.columns([1, 3, 1])
    with center:
        # banner
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True,
        )
        # title
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        # UPDATED instruction line
        st.markdown(
            '<div class="custom-instruction">To conduct your search, please follow the 3 steps below to query and view the results of the Public Service Employee Survey:</div>',
            unsafe_allow_html=True,
        )

        # Step 1 — Question
        st.markdown('<div class="field-label">Step 1: Select a survey question:</div>', unsafe_allow_html=True)
        question_options = qdf["display"].tolist()
        selected_label = st.selectbox(
            "Choose from the official list (type Q# or keywords to filter):",
            question_options,
            key="question_dropdown",
            label_visibility="collapsed",
        )
        question_code = qdf.loc[qdf["display"] == selected_label, "code"].values[0]
        question_text = qdf.loc[qdf["display"] == selected_label, "text"].values[0]

        # Step 2 — Years
        st.markdown('<div class="field-label">Step 2: Select survey year(s):</div>', unsafe_allow_html=True)
        all_years = [2024, 2022, 2020, 2019]
        select_all = st.checkbox("All years", value=True, key="select_all_years")
        selected_years: List[int] = []
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

        # Step 3 — Demographics
        st.markdown('<div class="field-label">Step 3: Select a demographic category (optional):</div>', unsafe_allow_html=True)
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
        demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
        demo_selection = st.selectbox(
            "Demographic category",
            demo_categories,
            key="demo_main",
            label_visibility="collapsed",
        )

        sub_selection: Optional[str] = None
        if demo_selection != "All respondents":
            st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
            sub_items = (
                demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            sub_items = sorted(sub_items)
            sub_selection = st.selectbox(
                "(leave blank to include all subgroups in this category)",
                [""] + sub_items,
                key=f"sub_{demo_selection.replace(' ', '_')}",
                label_visibility="collapsed",
            )
            if sub_selection == "":
                sub_selection = None

        # ACTION BUTTON
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Query and View Results"):
                # 1) Resolve DEMCODE(s)
                demcodes, disp_map, category_in_play = resolve_demographic_codes(demo_df, demo_selection, sub_selection)

                # 2) Scale labels
                scale_pairs = get_scale_labels(sdf, question_code)

                # 3) Pull data for each DEMCODE
                parts: List[pd.DataFrame] = []
                for code in demcodes:
                    df_part = load_results2024_filtered(
                        question_code=question_code,
                        years=selected_years,
                        group_value=code,
                    )
                    if not df_part.empty:
                        parts.append(df_part)

                if not parts:
                    st.info("No data found for this selection.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                df = pd.concat(parts, ignore_index=True)

                # normalize
                df = normalize_results_columns(df)

                # strict post-filter
                qmask = df["question_code"].astype(str).str.strip().str.upper() == str(question_code).strip().upper()
                ymask = pd.to_numeric(df["year"], errors="coerce").astype("Int64").isin(selected_years)
                if demo_selection == "All respondents":
                    gmask = df["group_value"].isna() | (df["group_value"].astype(str).str.strip() == "")
                else:
                    gmask = df["group_value"].astype(str).isin([str(c) for c in demcodes])

                df = df[qmask & ymask & gmask].copy()

                if df.empty:
                    st.info("No data found after applying filters.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                # drop 999
                df = drop_na_999(df)
                if df.empty:
                    st.info("Data exists, but all rows are not applicable (999).")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                # display title
                st.subheader(f"{question_code} — {question_text}")

                # display table
                df_disp = format_table_for_display(
                    df_slice=df,
                    dem_disp_map=disp_map,
                    category_in_play=category_in_play,
                    scale_pairs=scale_pairs,
                )
                st.dataframe(df_disp, use_container_width=True)

                # narrative
                st.markdown("#### Summary (Positive only)")
                summary = build_positive_only_narrative(df_disp, category_in_play)
                st.write(summary)

                # Excel download
                with io.BytesIO() as buf:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_disp.to_excel(writer, sheet_name="Results", index=False)
                        ctx = {
                            "Question code": question_code,
                            "Question text": question_text,
                            "Years": ", ".join(map(str, selected_years)),
                            "Category": demo_selection,
                            "Subgroup": sub_selection or "(all in category)" if demo_selection != "All respondents" else "All respondents",
                            "DEMCODEs used": ", ".join(
                                ["(blank)" if (c is None or str(c).strip() == "") else str(c) for c in demcodes]
                            ),
                            "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        pd.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(
                            writer, sheet_name="Context", index=False
                        )
                    data = buf.getvalue()

                st.download_button(
                    label="⬇️ Download Excel",
                    data=data,
                    file_name=f"PSES_{question_code}_{'-'.join(map(str, selected_years))}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    run_menu1()
