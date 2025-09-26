# Menu1/main.py — PSES AI Explorer (Menu 1: Search by Question + Keyword/Theme)
# Multi-question support (max 5). Hybrid keyword search. Summary matrix + per-question tabs.
# NOTE: Keeps existing narrative prompt/wording logic untouched (no AI prompt changes here).

import io
from datetime import datetime
from typing import List, Dict

import pandas as pd
import streamlit as st

# Loader: reads Drive .csv.gz in chunks and filters on QUESTION/YEAR/DEMCODE
from utils.data_loader import load_results2024_filtered

# Helpers extracted to keep this file short (no logic changes)
from utils.menu1_helpers import (
    resolve_demographic_codes,
    get_scale_labels,
    drop_na_999,
    normalize_results_columns,
    format_table_for_display,
    build_positive_only_narrative,
)

# Hybrid search utility (exact + substring + token-overlap)
from utils.hybrid_search import hybrid_question_search


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
        # Banner + header (unchanged)
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-instruction">
                Use this menu to explore results for survey questions.<br>
                You may select up to <b>5 questions</b> from the list and/or find questions via the keyword/theme search.
                Select year(s) and optionally a demographic category and subgroup.
            </div>
        """, unsafe_allow_html=True)

        # =========================
        # Question selection area
        # =========================
        st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)

        # 1) Multi-select from official list
        multi_choices = st.multiselect(
            "Choose one or more from the official list",
            qdf["display"].tolist(),
            default=[],
            max_selections=5,
            label_visibility="collapsed",
            key="menu1_multi_questions"
        )
        codes_from_multi = qdf.loc[qdf["display"].isin(multi_choices), "code"].tolist()

        # 2) Keyword/theme search (hybrid)
        with st.expander("🔎 Search by keywords or theme (optional)"):
            search_query = st.text_input("Enter keywords (e.g., harassment, recognition, onboarding)", key="menu1_kw_query")
            hits = pd.DataFrame()
            if st.button("Find matching questions", key="menu1_find_hits"):
                hits = hybrid_question_search(qdf, search_query, top_k=50)
                if hits.empty:
                    st.info("No matches found for your keywords.")
                else:
                    st.write(f"Top {len(hits)} matches (select any, total across both pickers limited to 5):")
                    # Checkbox pick area
                    sel = []
                    for i, row in hits.iterrows():
                        label = f"{row['code']} – {row['text']}"
                        if st.checkbox(label, key=f"kwpick_{row['code']}"):
                            sel.append(row["code"])
                    st.session_state["menu1_kw_selected_codes"] = sel

        codes_from_kw = st.session_state.get("menu1_kw_selected_codes", []) or []

        # Combine & enforce max=5
        question_codes: List[str] = []
        for c in codes_from_multi + codes_from_kw:
            if c not in question_codes:
                question_codes.append(c)
        if len(question_codes) > 5:
            question_codes = question_codes[:5]
            st.warning("Limit is 5 questions; extra selections were ignored.", icon="⚠️")

        # Backward-compatibility: single-select fallback if none picked yet
        fallback_code = None
        if not question_codes:
            st.markdown('<div class="field-label">Or select a single question:</div>', unsafe_allow_html=True)
            question_options = qdf["display"].tolist()
            selected_label = st.selectbox(
                "Choose from the official list (type Q# or keywords to filter):",
                question_options,
                key="question_dropdown",
                label_visibility="collapsed"
            )
            fallback_code = qdf.loc[qdf["display"] == selected_label, "code"].values[0]

        if not question_codes:
            if fallback_code:
                question_codes = [fallback_code]
            else:
                st.info("Please choose at least one question (via multi-select, keyword hits, or the single select).")
                return

        # =========================
        # Years
        # =========================
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

        # =========================
        # Demographics
        # =========================
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
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

        # =========================
        # Search
        # =========================
        with st.container():
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("🔎 Search"):
                # 1) Resolve DEMCODE(s) via metadata
                demcodes, disp_map, category_in_play = resolve_demographic_codes(demo_df, demo_selection, sub_selection)

                # 2) Pull & prepare per-question results
                per_q_disp_tables: Dict[str, pd.DataFrame] = {}
                per_q_texts: Dict[str, str] = {}
                per_q_scale_labels: Dict[str, list[tuple[str, str]]] = {}

                for qcode in question_codes:
                    # Get question text for display
                    if (qdf["code"] == qcode).any():
                        qtext = qdf.loc[qdf["code"] == qcode, "text"].values[0]
                    else:
                        qtext = ""

                    # Scale labels
                    scale_pairs = get_scale_labels(sdf, qcode)
                    per_q_scale_labels[qcode] = scale_pairs
                    per_q_texts[qcode] = qtext

                    # Pull parts per demcode
                    parts = []
                    for code in demcodes:
                        df_part = load_results2024_filtered(
                            question_code=qcode,
                            years=selected_years,
                            group_value=code
                        )
                        if not df_part.empty:
                            parts.append(df_part)

                    if not parts:
                        continue

                    df_all = pd.concat(parts, ignore_index=True)
                    df_all = normalize_results_columns(df_all)

                    # Strict guard on QUESTION/YEAR/DEMCODE
                    qmask = df_all["question_code"].astype(str).str.strip().str.upper() == str(qcode).strip().upper()
                    ymask = pd.to_numeric(df_all["year"], errors="coerce").astype("Int64").isin(selected_years)
                    if demo_selection == "All respondents":
                        gmask = df_all["group_value"].isna() | (df_all["group_value"].astype(str).str.strip() == "")
                    else:
                        gmask = df_all["group_value"].astype(str).isin([str(c) for c in demcodes])
                    df_all = df_all[qmask & ymask & gmask].copy()

                    # Exclude sentinel 999/NA rows
                    df_all = drop_na_999(df_all)
                    if df_all.empty:
                        continue

                    # Build standardized display table
                    df_disp = format_table_for_display(
                        df_slice=df_all,
                        dem_disp_map=disp_map,
                        category_in_play=category_in_play,
                        scale_pairs=scale_pairs
                    )
                    per_q_disp_tables[qcode] = df_disp

                if not per_q_disp_tables:
                    st.info("No data found for your selection.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                # 3) Tabs: one per question
                tab_labels = [qc for qc in question_codes if qc in per_q_disp_tables]
                tabs = st.tabs(tab_labels)
                for tlabel, qcode, tab in zip(tab_labels, question_codes, tabs):
                    if qcode not in per_q_disp_tables:
                        continue
                    df_disp = per_q_disp_tables[qcode]
                    qtext = per_q_texts.get(qcode, "")
                    with tab:
                        st.subheader(f"{qcode} — {qtext}")
                        st.dataframe(df_disp, use_container_width=True)

                        # Summary (Positive only) — unchanged narrative style
                        st.markdown("#### Summary (Positive only)")
                        summary = build_positive_only_narrative(df_disp, category_in_play)
                        st.write(summary)

                # 4) Summary matrix: rows=Question, cols=Year, value=% Positive (avg across subgroups if applicable)
                summary_rows = []
                for qcode, df_disp in per_q_disp_tables.items():
                    t = df_disp.copy()
                    # Average across demographics per year if present
                    if "Demographic" in t.columns:
                        grp = t.groupby("Year", as_index=False)["Positive"].mean(numeric_only=True)
                    else:
                        grp = t[["Year", "Positive"]].copy()
                    grp["Question"] = qcode
                    # Ensure Year numeric for ordering, keep display as int
                    grp["Year"] = pd.to_numeric(grp["Year"], errors="coerce").astype("Int64")
                    summary_rows.append(grp[["Question", "Year", "Positive"]])

                if summary_rows:
                    summary_df = pd.concat(summary_rows, ignore_index=True)
                    pivot = summary_df.pivot_table(index="Question", columns="Year", values="Positive", aggfunc="mean")
                    # Keep user-selected question order
                    pivot = pivot.reindex(index=[qc for qc in question_codes if qc in per_q_disp_tables])
                    # Sort years ascending
                    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                    st.markdown("### Summary matrix (% Positive)")
                    st.dataframe(pivot.round(1).reset_index(), use_container_width=True)

                    # Descriptive across-questions roll-up
                    st.markdown("#### Across selected questions (descriptive)")
                    means = pivot.mean(numeric_only=True).round(1)
                    trend_txt = ", ".join([f"{int(y)}: {v:.1f}%" for y, v in zip(pivot.columns, means)])
                    st.write(f"Average % Positive across selected questions by year → {trend_txt}.")

                # 5) Excel: Summary + one sheet per question + Context
                with io.BytesIO() as buf:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        # Summary sheet
                        if summary_rows:
                            pivot_out = pivot.round(1).reset_index()
                            pivot_out.to_excel(writer, sheet_name="Summary_Matrix", index=False)

                        # Per-question sheets
                        for qcode, df_disp in per_q_disp_tables.items():
                            safe = qcode[:28]  # Excel sheet name <= 31 chars
                            df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)

                        # Context
                        ctx = {
                            "Questions": ", ".join(question_codes),
                            "Years": ", ".join(map(str, selected_years)),
                            "Category": demo_selection,
                            "Subgroup": sub_selection or "(all in category)" if demo_selection != "All respondents" else "All respondents",
                            "DEMCODEs used": ", ".join(["(blank)" if (c is None or str(c).strip() == "") else str(c) for c in demcodes]),
                            "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        pd.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Context", index=False)

                    data = buf.getvalue()

                st.download_button(
                    label="⬇️ Download Excel (Summary + all tabs)",
                    data=data,
                    file_name=f"PSES_multiQ_{'-'.join(map(str, selected_years))}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_menu1()
