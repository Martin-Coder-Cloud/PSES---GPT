# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# Cached big-file, one-pass DEMCODE filtering, opt-in raw/Excel.
# All data as TEXT; trims only filter columns in the loader.

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_results2024_filtered,
    get_results2024_schema,
    get_results2024_schema_inferred,
)

# Stable alias to avoid any accidental local-shadowing of `pd` in functions
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
# Helpers
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
                return [code_final.strip()], {code_final.strip(): subgroup_label}, True
        return [str(subgroup_label).strip()], {str(subgroup_label).strip(): subgroup_label}, True

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

    if LABEL_COL in df_cat.columns:
        labels = [str(l).strip() for l in df_cat[LABEL_COL].tolist()]
        return labels, {l: l for l in labels}, True

    return [None], {None: "All respondents"}, False


def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
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
        s = out[c].astype(str).str.strip()
        keep &= (s != "999")
    return out.loc[keep].copy()


def format_display_table_raw(df: pd.DataFrame, category_in_play: bool, dem_disp_map: dict, scale_pairs) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["Year"] = out["SURVEYR"].astype(str)

    if category_in_play:
        def to_label(code):
            key = "" if code is None else str(code).strip()
            if key == "":
                return "All respondents"
            return dem_disp_map.get(key, str(code))
        out["Demographic"] = out["DEMCODE"].apply(to_label)

    dist_cols = []
    for i in range(1, 8):
        lc, uc = f"answer{i}", f"ANSWER{i}"
        if uc in out.columns:
            dist_cols.append(uc)
        elif lc in out.columns:
            dist_cols.append(lc)

    rename_map = {}
    for k, v in scale_pairs:
        ku = k.upper()
        if ku in out.columns:
            rename_map[ku] = v
        if k in out.columns:
            rename_map[k] = v

    keep_cols = ["Year"] + (["Demographic"] if category_in_play else []) \
                + dist_cols + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    sort_cols = ["Year"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

    # All values remain text for display
    return out


def build_narrative_2024_first_full(df_disp: pd.DataFrame, category_in_play: bool, question_code: str, question_text: str) -> str:
    if df_disp is None or df_disp.empty:
        return "No results are available to summarize."

    # Find Positive column (case-robust)
    pos_col = None
    for c in df_disp.columns:
        if str(c).strip().lower() == "positive":
            pos_col = c
            break
    if pos_col is None:
        return "No results are available to summarize."

    def f2(v):
        try:
            return float(str(v).strip())
        except Exception:
            return None

    def pct(v):
        x = f2(v)
        return f"{x:.1f}%" if x is not None else "n/a"

    def pts(a, b):
        xa, xb = f2(a), f2(b)
        if xa is None or xb is None:
            return None
        d = xa - xb
        sign = "+" if d >= 0 else "-"
        return f"{sign}{abs(d):.1f} pts"

    t = df_disp.copy()
    years = []
    for y in t["Year"].astype(str):
        if y not in years:
            years.append(y)
    if not years:
        return "No results are available to summarize."

    target_year = "2024" if "2024" in years else years[-1]
    prev_year = None
    for y in reversed(years):
        if y < target_year:
            prev_year = y
            break
    earliest_year, latest_year = years[0], years[-1]

    # Overall map (if explicit overall present)
    overall_map = {}
    if "Demographic" in t.columns:
        overall_rows = t[t["Demographic"].astype(str).str.lower().isin(["all respondents", "overall"])]
        for _, r in overall_rows.iterrows():
            overall_map[str(r["Year"])] = r[pos_col]

    parts = []

    # Overall statement
    if target_year in overall_map:
        ty = overall_map[target_year]
        parts.append(f"Overall, the results for {question_code} show that {pct(ty)} {question_text} in {target_year}.")
        if earliest_year in overall_map and earliest_year != target_year:
            ey = overall_map[earliest_year]
            d = pts(ty, ey)
            if d is not None and abs((f2(ty) or 0) - (f2(ey) or 0)) < 1.0:
                parts.append(f"Results have been stable since {earliest_year} (from {pct(ey)} to {pct(ty)}, {d}).")
            elif d is not None:
                direction = "increased" if (f2(ty) or 0) > (f2(ey) or 0) else "decreased"
                parts.append(f"Since {earliest_year}, overall results have {direction} by {d} (from {pct(ey)} to {pct(ty)}).")
        if prev_year and prev_year in overall_map:
            py = overall_map[prev_year]
            d = pts(ty, py)
            if d is not None:
                verb = "higher" if (f2(ty) or 0) >= (f2(py) or 0) else "lower"
                parts.append(f"Compared with {prev_year}, {target_year} is {d} {verb} (from {pct(py)} to {pct(ty)}).")

    # Demographic snapshot + gap + trends
    if category_in_play and "Demographic" in t.columns:
        ty_slice = t[t["Year"] == target_year].copy()
        if not ty_slice.empty:
            lines = []
            for g, gdf in t.groupby("Demographic"):
                row_ty = gdf[gdf["Year"] == target_year]
                if not row_ty.empty:
                    vty = row_ty.iloc[0][pos_col]
                    if prev_year:
                        row_py = gdf[gdf["Year"] == prev_year]
                        if not row_py.empty:
                            vpy = row_py.iloc[0][pos_col]
                            d = pts(vty, vpy)
                            if d is not None:
                                dir_word = "higher" if (f2(vty) or 0) >= (f2(vpy) or 0) else "lower"
                                lines.append(f"{g}: {pct(vty)} in {target_year} ({d} {dir_word} than {prev_year}).")
                                continue
                    lines.append(f"{g}: {pct(vty)} in {target_year}.")
            if lines:
                parts.append("By demographic in " + target_year + ": " + " ".join(lines))

            # Gap analysis in target year
            ordered = ty_slice[["Demographic", pos_col]].dropna()
            if not ordered.empty:
                ordered["_posf_"] = ordered[pos_col].apply(f2)
                ordered = ordered.dropna(subset=["_posf_"]).sort_values("_posf_", ascending=False)
                if not ordered.empty:
                    top_g, top_v = ordered.iloc[0]["Demographic"], ordered.iloc[0][pos_col]
                    bot_g, bot_v = ordered.iloc[-1]["Demographic"], ordered.iloc[-1][pos_col]
                    gap_now = pts(top_v, bot_v)
                    if prev_year:
                        prev_slice = t[t["Year"] == prev_year][["Demographic", pos_col]].copy()
                        prev_slice["_posf_"] = prev_slice[pos_col].apply(f2)
                        pm = {r["Demographic"]: r[pos_col] for _, r in prev_slice.dropna(subset=["_posf_"]).iterrows()}
                        if top_g in pm and bot_g in pm:
                            gap_prev_f = (f2(pm[top_g]) or 0) - (f2(pm[bot_g]) or 0)
                            gap_now_f = (f2(top_v) or 0) - (f2(bot_v) or 0)
                            delta = gap_now_f - gap_prev_f
                            widen_status = "widened" if delta > 0 else ("narrowed" if delta < 0 else "held steady")
                            sign = "+" if delta >= 0 else "-"
                            parts.append(
                                f"The {target_year} gap between {top_g} ({pct(top_v)}) and {bot_g} ({pct(bot_v)}) is {gap_now}; it has {widen_status} by {sign}{abs(delta):.1f} pts since {prev_year}."
                            )
                        else:
                            if gap_now:
                                parts.append(
                                    f"The {target_year} gap between {top_g} ({pct(top_v)}) and {bot_g} ({pct(bot_v)}) is {gap_now}."
                                )

        # Trend snippets per demographic
        trend_bits = []
        for g, gdf in t.groupby("Demographic"):
            gdf = gdf.dropna(subset=[pos_col])
            if gdf.empty:
                continue
            years_g = [str(y) for y in gdf["Year"].tolist()]
            fv = gdf.iloc[0][pos_col]; fy = years_g[0]
            lv = gdf.iloc[-1][pos_col]; ly = years_g[-1]
            if fy == ly:
                trend_bits.append(f"{g}: {pct(lv)} in {ly}.")
            else:
                d = pts(lv, fv)
                if d is not None:
                    trend_bits.append(f"{g}: {pct(fv)} in {fy} → {pct(lv)} in {ly} ({d}).")
        if trend_bits:
            parts.append("Trends over time by demographic: " + " ".join(trend_bits))

    else:
        # Overall trend if overall series present
        if overall_map and earliest_year in overall_map and latest_year in overall_map and earliest_year != latest_year:
            d = pts(overall_map[latest_year], overall_map[earliest_year])
            if d is not None:
                parts.append(f"Over time overall: {pct(overall_map[earliest_year])} in {earliest_year} → {pct(overall_map[latest_year])} in {latest_year} ({d}).")

    return " ".join(parts) if parts else "No notable changes are evident."


# ─────────────────────────────
# UI
# ─────────────────────────────
def run_menu1():
    # Clean, centered layout; moderate banner size
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

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            "<img style='width:65%;max-width:540px;height:auto;display:block;margin:0 auto 16px;' "
            "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="custom-instruction">Select a question, year(s), and (optionally) a demographic category and subgroup.<br>'
            'The query always uses <b>QUESTION</b>, <b>Year</b>, and <b>DEMCODE</b>.</div>',
            unsafe_allow_html=True
        )

        # Question
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

        # Demographic category/subgroup
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

        # Resolve DEMCODEs once (as text)
        demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(demo_df, demo_selection, sub_selection)
        dem_display = ["(blank)"] if demcodes == [None] else [str(c).strip() for c in demcodes]

        # Parameters preview
        params_df = PD.DataFrame({
            "Parameter": ["QUESTION (from metadata)", "SURVEYR (years)", "DEMCODE(s) (from metadata)"],
            "Value": [question_code, ", ".join(selected_years), ", ".join(dem_display)]
        })
        st.markdown("##### Parameters that will be passed to the database")
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Lightweight toggles
        show_raw = st.checkbox("Show raw rows (validation)", value=False)
        make_downloads = st.checkbox("Prepare downloads (Excel)", value=False)

        # Diagnostics (optional)
        with st.expander("🛠 Diagnostics: file schema", expanded=False):
            colA, colB = st.columns(2)
            with colA:
                if st.button("Show dtypes after loader read (text mode)"):
                    sch = get_results2024_schema()
                    st.write("All columns are read as text (object).")
                    st.dataframe(sch, use_container_width=True, hide_index=True)
            with colB:
                if st.button("Show what pandas would infer (preview)"):
                    sch2 = get_results2024_schema_inferred()
                    st.write("Preview only — the app forces text on read.")
                    st.dataframe(sch2, use_container_width=True, hide_index=True)

        # Run query (single pass, cached big file)
        if st.button("🔎 Run query"):
            scale_pairs = get_scale_labels(load_scales_metadata(), question_code)

            # Preferred fast path (new loader signature)
            try:
                df_raw = load_results2024_filtered(
                    question_code=question_code,
                    years=selected_years,
                    group_values=demcodes,  # list[str|None]
                )
            except TypeError:
                # Back-compat: old loader that only accepts group_value=...
                parts = []
                if demcodes == [None]:
                    try:
                        parts.append(
                            load_results2024_filtered(
                                question_code=question_code,
                                years=selected_years,
                                group_value=None,
                            )
                        )
                    except TypeError:
                        parts = []
                else:
                    for gv in demcodes:
                        try:
                            parts.append(
                                load_results2024_filtered(
                                    question_code=question_code,
                                    years=selected_years,
                                    group_value=(None if gv is None else str(gv).strip()),
                                )
                            )
                        except TypeError:
                            continue
                df_raw = PD.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else PD.DataFrame()

            if df_raw is None or df_raw.empty:
                st.info("No data found for this selection.")
                return

            # Exclude 999s for display/narrative only
            df_raw = exclude_999_raw(df_raw)
            if df_raw.empty:
                st.info("Data exists, but all rows are not applicable (999).")
                return

            # Optional raw rows (skip by default)
            if show_raw:
                st.markdown("#### Raw results (full rows)")
                if "SURVEYR" in df_raw.columns:
                    df_raw = df_raw.sort_values(by="SURVEYR", ascending=False)
                st.dataframe(df_raw, use_container_width=True)

            # Title + formatted table
            st.subheader(f"{question_code} — {question_text}")
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

            # Narrative (2024-first, full sentences)
            st.markdown("#### Summary")
            st.write(build_narrative_2024_first_full(df_disp, category_in_play, question_code, question_text))

            # Optional downloads
            if make_downloads:
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


if __name__ == "__main__":
    run_menu1()
