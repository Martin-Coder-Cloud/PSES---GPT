# Menu1/main.py — PSES AI Explorer (Menu 1: PSES Explorer Search)
# Multi-question support (max 5). Hybrid keyword search (threshold & up to 120 hits).
# Persistent selections across UI elements; rock-solid updates.
# Reset behavior: button at bottom + auto-reset on entry when coming from another menu (via last_active_menu).
# AI analysis toggle restored (state key: menu1_ai_toggle). No changes to your AI prompt logic here.

import io
from datetime import datetime
from typing import List, Dict, Set

import pandas as pd
import streamlit as st

# Loader: reads Drive .csv.gz in chunks and filters on QUESTION/YEAR/DEMCODE
from utils.data_loader import load_results2024_filtered

# Helpers (no logic changes)
from utils.menu1_helpers import (
    resolve_demographic_codes,
    get_scale_labels,
    drop_na_999,
    normalize_results_columns,
    format_table_for_display,
    build_positive_only_narrative,
)

# Hybrid search utility
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
# Reset helpers
# ─────────────────────────────
def _delete_keys(prefixes: List[str], exact_keys: List[str] = None):
    exact_keys = exact_keys or []
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in prefixes) or (k in exact_keys):
            try:
                del st.session_state[k]
            except Exception:
                pass

def reset_menu1_state():
    # Clear all Menu 1 specific state, including dynamic checkboxes and widgets.
    year_keys = [f"year_{y}" for y in (2024, 2022, 2020, 2019)]
    exact = [
        "menu1_selected_codes",
        "menu1_hits",
        "menu1_kw_query",
        "menu1_multi_questions",
        "menu1_ai_toggle",
        "select_all_years",
        "demo_main",
        "menu1_clear_sel",
        "menu1_find_hits",
    ] + year_keys
    prefixes = ["kwhit_", "sel_", "sub_"]
    _delete_keys(prefixes, exact)
    # Ensure search box is empty after reset
    st.session_state["menu1_kw_query"] = ""
    st.session_state["menu1_hits"] = []
    st.session_state["menu1_selected_codes"] = []
    st.session_state["menu1_multi_questions"] = []


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
            .pill { display:inline-block; padding:4px 8px; margin:2px 6px 2px 0; background:#f1f3f5; border-radius:999px; font-size:13px; }
            .action-row { display:flex; gap:10px; align-items:center; }
        </style>
    """, unsafe_allow_html=True)

    # Auto-reset on entry if navigating from another menu (requires router to set last_active_menu)
    if st.session_state.get("last_active_menu") != "menu1":
        reset_menu1_state()
    st.session_state["last_active_menu"] = "menu1"

    # Ensure base containers exist after possible reset
    st.session_state.setdefault("menu1_selected_codes", [])
    st.session_state.setdefault("menu1_hits", [])
    st.session_state.setdefault("menu1_kw_query", "")
    st.session_state.setdefault("menu1_ai_toggle", False)

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    # Helper maps
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    left, center, right = st.columns([1, 3, 1])
    with center:
        # Banner — original
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )

        # Header + instructions
        st.markdown('<div class="custom-header">PSES Explorer Search</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-instruction">
                Please use this menu to explore the survey results by questions.<br>
                You may select from the drop down menu below up to five questions or find questions via the keyword/theme search.
                Select year(s) and optionally a demographic category and subgroup.
            </div>
        """, unsafe_allow_html=True)

        # =========================
        # Question selection area
        # =========================
        st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)

        # 1) Multi-select from official list (authoritative source for dropdown selection)
        # Use & update st.session_state["menu1_multi_questions"] so deselects stick.
        all_displays = qdf["display"].tolist()
        st.session_state.setdefault("menu1_multi_questions", [])
        multi_choices = st.multiselect(
            "Choose one or more from the official list",
            all_displays,
            default=st.session_state["menu1_multi_questions"],
            max_selections=5,
            label_visibility="collapsed",
            key="menu1_multi_questions"
        )
        selected_from_multi: Set[str] = set(display_to_code[d] for d in multi_choices if d in display_to_code)

        # 2) Keyword/theme search (hybrid) with persistent results
        with st.expander("Search by keywords or theme (optional)"):
            # Bind to session so resets truly clear the text
            search_query = st.text_input(
                "Enter keywords (e.g., harassment, recognition, onboarding)",
                key="menu1_kw_query"
            )
            # Button to compute hits and store persistently
            if st.button("Search questions", key="menu1_find_hits"):
                hits_df = hybrid_question_search(qdf, search_query, top_k=120, min_score=0.40)
                st.session_state["menu1_hits"] = hits_df[["code", "text"]].to_dict(orient="records") if not hits_df.empty else []

            # Always render stored hits with independent checkboxes
            selected_from_hits: Set[str] = set()
            if st.session_state["menu1_hits"]:
                st.write(f"Top {len(st.session_state['menu1_hits'])} matches meeting the quality threshold:")
                for rec in st.session_state["menu1_hits"]:
                    code = rec["code"]; text = rec["text"]
                    label = f"{code} – {text}"
                    key = f"kwhit_{code}"
                    # Checked if previously checked, or if code is in dropdown selection
                    default_checked = st.session_state.get(key, False) or (code in selected_from_multi)
                    checked = st.checkbox(label, value=default_checked, key=key)
                    if checked:
                        selected_from_hits.add(code)
            else:
                st.info("Enter keywords and click “Search questions” to see matches.")

        # Combine sources: dropdown (authoritative) + search hits
        combined_order = []
        # Start with the order of the dropdown list to keep UX intuitive
        for d in st.session_state["menu1_multi_questions"]:
            c = display_to_code.get(d)
            if c and c not in combined_order:
                combined_order.append(c)
        # Then add any hit selections (that aren’t already in dropdown)
        for c in selected_from_hits:
            if c not in combined_order:
                combined_order.append(c)

        # Enforce max 5
        if len(combined_order) > 5:
            combined_order = combined_order[:5]
            st.warning("Limit is 5 questions; extra selections were ignored.")

        # Persist the current combined selection
        st.session_state["menu1_selected_codes"] = combined_order

        # “Selected questions” checklist to allow quick removal
        if st.session_state["menu1_selected_codes"]:
            st.markdown('<div class="field-label">Selected questions:</div>', unsafe_allow_html=True)
            updated = list(st.session_state["menu1_selected_codes"])
            cols = st.columns(min(5, len(updated)))
            for idx, code in enumerate(list(updated)):
                with cols[idx % len(cols)]:
                    label = code_to_display.get(code, code)
                    keep = st.checkbox(label, value=True, key=f"sel_{code}")
                    if not keep:
                        # Remove from combined
                        updated = [c for c in updated if c != code]
                        # Also uncheck the hit checkbox if it exists
                        hit_key = f"kwhit_{code}"
                        if hit_key in st.session_state:
                            st.session_state[hit_key] = False
                        # Also remove from multiselect widget state
                        disp = code_to_display.get(code)
                        if disp and "menu1_multi_questions" in st.session_state:
                            st.session_state["menu1_multi_questions"] = [
                                d for d in st.session_state["menu1_multi_questions"] if d != disp
                            ]
            if updated != st.session_state["menu1_selected_codes"]:
                st.session_state["menu1_selected_codes"] = updated

        # Final question list to use
        question_codes: List[str] = st.session_state["menu1_selected_codes"]
        if not question_codes:
            st.info("Please choose at least one question (via multi-select and/or keyword hits).")
            # Continue rendering rest of page to allow year/demo pre-selection if desired

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
        # AI analysis toggle (restored)
        # =========================
        st.session_state["menu1_ai_toggle"] = st.checkbox("Enable AI analysis", value=st.session_state["menu1_ai_toggle"], key="menu1_ai_toggle")

        # =========================
        # Action row: Search + Reset side by side
        # =========================
        st.markdown("<div class='action-row'>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1])
        with colA:
            disable_search = (not question_codes) or (not selected_years)
            if st.button("Search", disabled=disable_search):
                # Resolve DEMCODE(s)
                demcodes, disp_map, category_in_play = resolve_demographic_codes(demo_df, demo_selection, sub_selection)

                # Pull & prepare per-question results
                per_q_disp_tables: Dict[str, pd.DataFrame] = {}
                per_q_texts: Dict[str, str] = {}
                per_q_scale_labels: Dict[str, list[tuple[str, str]]] = {}

                for qcode in question_codes:
                    qtext = qdf.loc[qdf["code"] == qcode, "text"].values[0] if (qdf["code"] == qcode).any() else ""
                    scale_pairs = get_scale_labels(sdf, qcode)
                    per_q_scale_labels[qcode] = scale_pairs
                    per_q_texts[qcode] = qtext

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

                    qmask = df_all["question_code"].astype(str).str.strip().str.upper() == str(qcode).strip().upper()
                    ymask = pd.to_numeric(df_all["year"], errors="coerce").astype("Int64").isin(selected_years)
                    if demo_selection == "All respondents":
                        gmask = df_all["group_value"].isna() | (df_all["group_value"].astype(str).str.strip() == "")
                    else:
                        gmask = df_all["group_value"].astype(str).isin([str(c) for c in demcodes])
                    df_all = df_all[qmask & ymask & gmask].copy()

                    df_all = drop_na_999(df_all)
                    if df_all.empty:
                        continue

                    df_disp = format_table_for_display(
                        df_slice=df_all,
                        dem_disp_map=disp_map,
                        category_in_play=category_in_play,
                        scale_pairs=scale_pairs
                    )
                    per_q_disp_tables[qcode] = df_disp

                if not per_q_disp_tables:
                    st.info("No data found for your selection.")
                else:
                    # Tabs: one per question
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

                            # NOTE: Your existing AI prompt/logic can hook into st.session_state["menu1_ai_toggle"]
                            # if st.session_state["menu1_ai_toggle"]:
                            #     ... run your unchanged AI narrative with df_disp / context ...

                    # Summary matrix
                    summary_rows = []
                    for qcode, df_disp in per_q_disp_tables.items():
                        t = df_disp.copy()
                        if "Demographic" in t.columns:
                            grp = t.groupby("Year", as_index=False)["Positive"].mean(numeric_only=True)
                        else:
                            grp = t[["Year", "Positive"]].copy()
                        grp["Question"] = qcode
                        grp["Year"] = pd.to_numeric(grp["Year"], errors="coerce").astype("Int64")
                        summary_rows.append(grp[["Question", "Year", "Positive"]])

                    if summary_rows:
                        summary_df = pd.concat(summary_rows, ignore_index=True)
                        pivot = summary_df.pivot_table(index="Question", columns="Year", values="Positive", aggfunc="mean")
                        pivot = pivot.reindex(index=[qc for qc in question_codes if qc in per_q_disp_tables])
                        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                        st.markdown("### Summary matrix (% Positive)")
                        st.dataframe(pivot.round(1).reset_index(), use_container_width=True)

                        st.markdown("#### Across selected questions (descriptive)")
                        means = pivot.mean(numeric_only=True).round(1)
                        trend_txt = ", ".join([f"{int(y)}: {v:.1f}%" for y, v in zip(pivot.columns, means)])
                        st.write(f"Average % Positive across selected questions by year → {trend_txt}.")

                        # Excel
                        with io.BytesIO() as buf:
                            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                                pivot.round(1).reset_index().to_excel(writer, sheet_name="Summary_Matrix", index=False)
                                for qcode, df_disp in per_q_disp_tables.items():
                                    safe = qcode[:28]
                                    df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)
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
                            label="Download Excel (Summary + all tabs)",
                            data=data,
                            file_name=f"PSES_multiQ_{'-'.join(map(str, selected_years))}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
        with colB:
            if st.button("Reset all parameters"):
                reset_menu1_state()
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    run_menu1()
