# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# Cached big-file, one-pass DEMCODE filtering, always-on Excel & PDF downloads.
# All data as TEXT; trims only filter columns in the loader.
from __future__ import annotations

import io
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_results2024_filtered,
    get_results2024_schema,
    get_results2024_schema_inferred,
)

# Ensure OpenAI key is available from Streamlit secrets (no hardcoding)
os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

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
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
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
    subgroup_label: str | None,
):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    code_col = _find_demcode_col(demo_df)

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    df_cat = (
        demo_df[demo_df[DEMO_CAT_COL] == category_label]
        if DEMO_CAT_COL in demo_df.columns
        else demo_df.copy()
    )
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
    for i in range(1, 7 + 1):
        col = f"answer{i}"
        lbl = None
        if not candidates.empty and col in candidates.columns:
            vals = candidates[col].dropna().astype(str)
            if not vals.empty:
                lbl = vals.iloc[0].strip()
        labels.append((col, lbl or f"Answer {i}"))
    return labels


def exclude_999_raw(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"answer{i}" for i in range(1, 7 + 1)] + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    out = df.copy()
    keep = PD.Series(True, index=out.index)
    for c in present:
        s = out[c].astype(str).str.strip()
        keep &= (s != "999")
    return out.loc[keep].copy()


def format_display_table_raw(
    df: pd.DataFrame, category_in_play: bool, dem_disp_map: dict, scale_pairs
) -> pd.DataFrame:
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
    for i in range(1, 7 + 1):
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

    keep_cols = (
        ["Year"]
        + (["Demographic"] if category_in_play else [])
        + dist_cols
        + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE", "YES"]
    )
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    sort_cols = ["Year"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

    # All values remain text for display
    return out


# ─────────────────────────────
# AI helpers (auto-run)
# ─────────────────────────────
def _detect_metric_column(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (metric_col_name, human_label).
    Preference: POSITIVE → AGREE → YES (case-insensitive).
    """
    cols = {c.lower(): c for c in df.columns}
    if "positive" in cols:
        return (cols["positive"], "positive answer")
    if "agree" in cols:
        return (cols["agree"], "positive answer")
    if "yes" in cols:
        return (cols["yes"], "Yes answer")
    return ("POSITIVE", "positive answer")


def _ai_build_payload(
    df_disp: pd.DataFrame,
    question_code: str,
    question_text: str,
    category_in_play: bool,
    metric_col: str,
    metric_label: str,
) -> dict:
    # Robust column detection
    def col(df, *cands):
        for c in cands:
            if c in df.columns:
                return c
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None

    year_col = col(df_disp, "Year") or "Year"
    demo_col = col(df_disp, "Demographic") or "Demographic"
    n_col    = col(df_disp, "ANSCOUNT", "AnsCount", "N")

    ys = PD.to_numeric(df_disp[year_col], errors="coerce").dropna().astype(int).unique().tolist()
    ys = sorted(ys)
    latest = ys[-1] if ys else None
    baseline = ys[0] if len(ys) >= 2 else (ys[-1] if ys else None)

    groups = []
    has_subgroups = False
    if category_in_play and demo_col in df_disp.columns:
        unique_groups = [g for g in df_disp[demo_col].dropna().unique().tolist()]
        has_subgroups = len(unique_groups) > 0
        for gname, gdf in df_disp.groupby(demo_col, dropna=False):
            series = []
            for _, r in gdf.sort_values(year_col).iterrows():
                yr = int(PD.to_numeric(r[year_col], errors="coerce"))
                val = float(PD.to_numeric(r.get(metric_col, None), errors="coerce"))
                n = None
                if n_col in gdf.columns:
                    n_val = PD.to_numeric(r.get(n_col, None), errors="coerce")
                    n = int(n_val) if PD.notna(n_val) else None
                series.append({"year": yr, "value": val, "n": n})
            groups.append({"name": (str(gname) if PD.notna(gname) else ""), "series": series})
    else:
        series = []
        for _, r in df_disp.sort_values(year_col).iterrows():
            yr = int(PD.to_numeric(r[year_col], errors="coerce"))
            val = float(PD.to_numeric(r.get(metric_col, None), errors="coerce"))
            n = None
            if n_col and n_col in df_disp.columns:
                n_val = PD.to_numeric(r.get(n_col, None), errors="coerce")
                n = int(n_val) if PD.notna(n_val) else None
            series.append({"year": yr, "value": val, "n": n})
        groups = [{"name": "All respondents", "series": series}]
        has_subgroups = False

    return {
        "question_code": str(question_code),
        "question_text": str(question_text),
        "years": ys,
        "latest_year": latest,
        "baseline_year": baseline,
        "groups": groups,
        "has_subgroups": has_subgroups,
        "metric_column": str(metric_col),
        "metric_label": str(metric_label),
        "context": "Public Service Employee Survey (PSES): workplace/workforce perceptions among federal public servants. Values represent the share selecting a positive option (e.g., Strongly agree/Agree) or 'Yes' where applicable.",
    }


def _ai_narrative_and_storytable(
    df_disp: pd.DataFrame,
    question_code: str,
    question_text: str,
    category_in_play: bool,
    metric_col: str,
    metric_label: str,
    temperature: float = 0.2,
) -> dict:
    """Calls OpenAI and returns {'narrative': str, 'table': list[dict]]}."""
    try:
        from openai import OpenAI
    except Exception:
        st.error(
            "AI summary requires the OpenAI SDK. Add `openai>=1.40.0` to requirements.txt and set `OPENAI_API_KEY` in Streamlit secrets."
        )
        return {"narrative": "", "table": []}

    client = OpenAI()  # uses OPENAI_API_KEY from env
    data = _ai_build_payload(df_disp, question_code, question_text, category_in_play, metric_col, metric_label)

    system = (
        "You are a survey insights writer for the Public Service Employee Survey (PSES). "
        "Write an executive-ready narrative for senior management about workplace/workforce perceptions. "
        "The values are shares of employees selecting a positive option (e.g., Strongly agree/Agree) "
        "or 'Yes' where that scale applies.\n\n"
        "STRICT RULES:\n"
        "• Start with the latest year (prefer 2024) overall point if available, phrased as employees' views on the question.\n"
        "• If there are NO groups, DO NOT mention groups at all.\n"
        "• If groups exist, NEVER use the words 'subgroup' or 'segment'. Refer to each group by its label.\n"
        "• When comparing groups, identify the highest and lowest group in the latest year, state the gap in points, and how that gap changed vs the baseline year if both are present.\n"
        "• Summarize the trend concisely (no long lists): typical change range in points, plus largest increase/decrease; name the group(s).\n"
        "• Treat |Δ| ≥ 5 pts as notable and ≥ 3 pts as mention-worthy.\n"
        "• Use whole percents and 'pts' for deltas. Keep to ~4–6 sentences, plain language.\n"
        "• Do not invent data; only use provided numbers."
    )
    user = (
        "DATA (JSON):\n"
        + json.dumps(data, ensure_ascii=False)
        + "\n\n"
        "Instructions:\n"
        f"- The metric column is '{metric_col}' and represents '{metric_label}'.\n"
        "- If some rows are missing the metric, skip them; do not guess.\n"
        "- Use group labels exactly as given (but never say 'subgroup').\n"
        "- Return JSON with keys: 'narrative' (string) and 'table' (array). The 'table' may be empty."
    )

    try:
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = comp.choices[0].message.content if comp.choices else "{}"
        out = json.loads(content)
        if not isinstance(out, dict):
            out = {"narrative": "", "table": []}
        out.setdefault("narrative", "")
        out.setdefault("table", [])
        return out
    except Exception as e:
        st.error(f"AI summary failed: {e}")
        return {"narrative": "", "table": []}


# ─────────────────────────────
# Summary trend table builder (years as columns)
# ─────────────────────────────
def build_trend_summary_table(df_disp: pd.DataFrame, category_in_play: bool, metric_col: str) -> pd.DataFrame:
    if df_disp is None or df_disp.empty or "Year" not in df_disp.columns:
        return PD.DataFrame()

    # Case-insensitive metric check
    if metric_col not in df_disp.columns:
        low = {c.lower(): c for c in df_disp.columns}
        if metric_col.lower() in low:
            metric_col = low[metric_col.lower()]
        else:
            return PD.DataFrame()

    df = df_disp.copy()
    if not category_in_play or "Demographic" not in df.columns:
        df["Demographic"] = "All respondents"

    df["__Y__"] = PD.to_numeric(df["Year"], errors="coerce")
    df = df.sort_values("__Y__")

    pivot = df.pivot_table(
        index="Demographic", columns="Year", values=metric_col, aggfunc="first"
    ).copy()
    pivot.index.name = "Segment"

    for c in pivot.columns:
        vals = PD.to_numeric(pivot[c], errors="coerce").round(0)
        mask = vals.notna()
        out = PD.Series("n/a", index=pivot.index, dtype="object")
        out.loc[mask] = vals.loc[mask].astype(int).astype(str) + "%"
        pivot[c] = out

    pivot = pivot.reset_index()
    year_cols = sorted([col for col in pivot.columns if col != "Segment"], key=lambda x: int(x))
    pivot = pivot[["Segment"] + year_cols]
    return pivot


# ─────────────────────────────
# Report (PDF) builder
# ─────────────────────────────
def build_pdf_report(
    question_code: str,
    question_text: str,
    selected_years: list[str],
    dem_display: list[str],
    narrative: str,
    df_summary: pd.DataFrame,
) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        st.error("PDF export requires `reportlab`. Please add `reportlab==3.6.13` to requirements.txt.")
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=36, bottomMargin=36, leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet()
    title = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, textColor="#555555")

    flow = []
    flow.append(Paragraph("PSES Analysis Report", title))
    flow.append(Paragraph(f"{question_code} — {question_text}", body))
    flow.append(Paragraph("(% of positive answer)", small))
    flow.append(Spacer(1, 10))

    # Context
    ctx = f"""
    <b>Years:</b> {', '.join(selected_years)}<br/>
    <b>Demographic selection:</b> {', '.join(dem_display)}
    """
    flow.append(Paragraph(ctx, body))
    flow.append(Spacer(1, 10))

    # Narrative
    flow.append(Paragraph("Analysis Summary", h2))
    for chunk in narrative.split("\n"):
        if chunk.strip():
            flow.append(Paragraph(chunk.strip(), body))
            flow.append(Spacer(1, 4))

    # Summary table
    if df_summary is not None and not df_summary.empty:
        flow.append(Spacer(1, 10))
        flow.append(Paragraph("Summary Table", h2))
        data = [df_summary.columns.tolist()] + df_summary.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                ]
            )
        )
        flow.append(tbl)

    doc.build(flow)
    return buf.getvalue()


# ─────────────────────────────
# UI
# ─────────────────────────────
def run_menu1():
    # Clean, centered layout; moderate banner size + small-note styling
    st.markdown(
        """
    <style>
      .custom-header{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }
      .custom-instruction{ font-size: 15px; line-height: 1.4; margin-bottom: 8px; color: #333; }
      .field-label{ font-size: 16px; font-weight: 600; margin: 10px 0 2px; color: #222; }
      .big-button button{ font-size: 16px; padding: 0.6em 1.6em; margin-top: 16px; }
      .tiny-note{ font-size: 12px; color: #666; margin-top: -4px; margin-bottom: 10px; }
      .q-sub{ font-size: 14px; color: #333; margin-top: -4px; margin-bottom: 2px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            "<img style='width:65%;max-width:540px;height:auto;display:block;margin:0 auto 16px;' "
            "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="custom-header">🔍 Search by Question</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="custom-instruction">Select a question, year(s), and (optionally) a demographic category and subgroup.<br>'
            "The query always uses <b>QUESTION</b>, <b>Year</b>, and <b>DEMCODE</b>.</div>",
            unsafe_allow_html=True,
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
        demo_categories = ["All respondents"] + sorted(
            demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist()
        )
        demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

        sub_selection = None
        if demo_selection != "All respondents":
            st.markdown(
                f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>',
                unsafe_allow_html=True,
            )
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

        # Resolve DEMCODEs once (as text)
        demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(
            demo_df, demo_selection, sub_selection
        )
        dem_display = ["(blank)"] if demcodes == [None] else [str(c).strip() for c in demcodes]

        # Parameters preview
        params_df = PD.DataFrame(
            {
                "Parameter": ["QUESTION (from metadata)", "SURVEYR (years)", "DEMCODE(s) (from metadata)"],
                "Value": [question_code, ", ".join(selected_years), ", ".join(dem_display)],
            }
        )
        st.markdown("##### Parameters that will be passed to the database")
        st.dataframe(params_df, use_container_width=True, hide_index=True)

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
            with st.spinner("Processing data..."):
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
                    df_raw = (
                        PD.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
                        if parts
                        else PD.DataFrame()
                    )

                if df_raw is None or df_raw.empty:
                    st.info("No data found for this selection.")
                    return

                # Exclude 999s for display/narrative only
                df_raw = exclude_999_raw(df_raw)
                if df_raw.empty:
                    st.info("Data exists, but all rows are not applicable (999).")
                    return

                # Optional raw rows (skip by default)
                if "SURVEYR" in df_raw.columns:
                    df_raw = df_raw.sort_values(by="SURVEYR", ascending=False)

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
                    scale_pairs=scale_pairs,
                )

            # Show the results table (always)
            st.dataframe(df_disp, use_container_width=True)

            # Decide which metric to use (POSITIVE → AGREE → YES)
            metric_col, metric_label = _detect_metric_column(df_disp)

            # === AUTO AI narrative ===
            st.markdown("### Analysis Summary")
            st.markdown(
                f"<div class='q-sub'>{question_code} — {question_text}</div>"
                f"<div class='tiny-note'>(% of positive answer)</div>",
                unsafe_allow_html=True,
            )
            with st.spinner("Contacting AI…"):
                ai_out = _ai_narrative_and_storytable(
                    df_disp=df_disp,
                    question_code=question_code,
                    question_text=question_text,
                    category_in_play=category_in_play,
                    metric_col=metric_col,
                    metric_label=metric_label,
                    temperature=0.2,
                )
            narrative = (ai_out.get("narrative") or "").strip()
            if narrative:
                st.write(narrative)
            else:
                st.info("No AI narrative was produced.")

            # === Trend Summary Table (years as columns) ===
            st.markdown("### Summary Table")
            st.markdown(
                f"<div class='q-sub'>{question_code} — {question_text}</div>"
                f"<div class='tiny-note'>(% of positive answer)</div>",
                unsafe_allow_html=True,
            )
            trend_df = build_trend_summary_table(df_disp, category_in_play, metric_col)
            if trend_df is not None and not trend_df.empty:
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
            else:
                st.info("No summary table could be generated for the current selection.")

            # === Always-on downloads: Excel (data) & PDF (report) ===
            # Excel with Results + Summary + Context
            with io.BytesIO() as xbuf:
                with PD.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
                    df_disp.to_excel(writer, sheet_name="Results", index=False)
                    if trend_df is not None and not trend_df.empty:
                        trend_df.to_excel(writer, sheet_name="Summary Table", index=False)
                    ctx = {
                        "QUESTION": question_code,
                        "SURVEYR (years)": ", ".join(selected_years),
                        "DEMCODE(s)": ", ".join(dem_display),
                        "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    PD.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(
                        writer, sheet_name="Context", index=False
                    )
                xdata = xbuf.getvalue()

            st.download_button(
                label="⬇️ Download data (Excel)",
                data=xdata,
                file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # PDF report (Analysis Summary + Summary Table)
            pdf_bytes = build_pdf_report(
                question_code=question_code,
                question_text=question_text,
                selected_years=selected_years,
                dem_display=dem_display,
                narrative=narrative,
                df_summary=trend_df if trend_df is not None else PD.DataFrame(),
            )
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download summary report (PDF)",
                    data=pdf_bytes,
                    file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("PDF export unavailable (install `reportlab` in requirements to enable).")


if __name__ == "__main__":
    run_menu1()
