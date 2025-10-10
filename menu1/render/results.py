# menu1/render/results.py
# ---------------------------------------------------------------------
# Menu 1 – Results: Tables (tabs) + Export + Start New + AI Summary + Validation.
#
# Back-compat shims:
# - Exports tabs_summary_and_per_q(...) for legacy callers.
# - Accepts either:
#     tabs_summary_and_per_q(
#         tables_by_q=..., qtext_by_q=..., category_in_play=..., ai_enabled=..., selection_key=...
#       )
#   OR:
#     tabs_summary_and_per_q(payload={ ... })  # payload may use varied key names; we normalize.
#
# Restores finalized UX:
#   • Tabs first (with a leading “Summary” tab), AI Summary below.
#   • “Export to Excel” + “Start a new search” aligned on one row.
#   • Titles above tables; minimal spacing preserved.
#   • Source links under each table.
#
# Implements data rules:
#   • No math in reporting; narrate pre-aggregated values.
#   • POLARITY -> reporting field with fallbacks:
#       POS -> POSITIVE -> AGREE -> ANSWER1
#       NEG -> NEGATIVE -> AGREE -> ANSWER1
#       NEU -> AGREE    -> ANSWER1
#   • Meaning indices from Survey Questions.xlsx -> labels from Survey Scales.xlsx
#     (supports code column named 'code', 'question', or 'questions').
#   • Per-question AI summaries (spinners) + Overall synthesis (when ≥2 comparable questions).
#   • All-years trend classification (AI payload covers ALL years; only gap math is ever done).
#   • D57_2 exception: list ALL options (latest year) exactly as provided; excluded from Overall.
#   • Validation line ✅/❌ and “Start a new search” (clears AI caches; keeps AI toggle).
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import math
import json
import hashlib
import io
import pandas as pd
import streamlit as st

# Try imports for ai helper in multiple locations to be robust
try:
    from menu1 import ai as _ai_mod
except Exception:
    try:
        from menu1.render import ai as _ai_mod  # if you colocated ai.py under render/
    except Exception:
        import ai as _ai_mod  # last-resort local import

ai = _ai_mod

SENTINEL = 9999

# Multi-response items narrated as full distributions (no aggregation)
EXCEPT_MULTI_DISTRIBUTION = {"D57_2"}
# Excluded from Overall synthesis (no single comparable metric)
EXCLUDE_FROM_OVERALL = set(EXCEPT_MULTI_DISTRIBUTION)

# ─────────────────────────────────────────────────────────────────────────────
# Metadata loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_survey_questions() -> pd.DataFrame:
    df = pd.read_excel("metadata/Survey Questions.xlsx")
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if "question" in df.columns and "code" not in df.columns:
        df = df.rename(columns={"question": "code"})
    if "english" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"english": "text"})

    df["code"] = df["code"].astype(str).str.strip()
    df["text"] = df.get("text", pd.Series([None]*len(df))).astype(str)

    pol = df.get("polarity", pd.Series(["POS"]*len(df))).astype(str).str.upper().str.strip()
    pol = pol.where(pol.isin(["POS", "NEG", "NEU"]), "POS")
    df["polarity"] = pol

    # These hold the meaning indices for each reporting field (e.g., "1,2")
    for col in ["positive", "negative", "agree", "neutral"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)

    return df[["code", "text", "polarity", "positive", "negative", "agree", "neutral"]]


@st.cache_data(show_spinner=False)
def _load_scales() -> pd.DataFrame:
    df = pd.read_excel("metadata/Survey Scales.xlsx")
    return df.rename(columns={c: c.strip().lower() for c in df.columns})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: parsing metadata & resolving labels
# ─────────────────────────────────────────────────────────────────────────────

def _parse_indices(meta_val: Any) -> Optional[List[int]]:
    if meta_val is None or (isinstance(meta_val, float) and math.isnan(meta_val)):
        return None
    s = str(meta_val).strip()
    if not s:
        return None
    toks = [t.strip() for t in s.replace(";", ",").replace("|", ",").split(",") if t.strip()]
    out: List[int] = []
    for t in toks:
        try:
            out.append(int(t))
        except Exception:
            continue
    return out or None

def _scales_code_col(scales_df: pd.DataFrame) -> Optional[str]:
    for name in ("code", "question", "questions"):
        if name in scales_df.columns:
            return name
    return None

def _labels_for_indices(scales_df: pd.DataFrame, code: str, indices: Optional[List[int]]) -> Optional[List[str]]:
    if not indices:
        return None
    code_col = _scales_code_col(scales_df)
    if code_col is None:
        return None

    code_u = str(code).strip().upper()
    df = scales_df
    sub = df[df[code_col].astype(str).str.upper() == code_u]

    # Wide format: answer1..answer7
    wide_cols = [c for c in df.columns if c.startswith("answer") and c[6:].isdigit()]
    if not sub.empty and wide_cols:
        r0 = sub.iloc[0]
        labels: List[str] = []
        for i in indices:
            col = f"answer{i}".lower()
            if col in df.columns:
                val = str(r0[col]).strip()
                if val and val.lower() != "nan":
                    labels.append(val)
        if labels:
            return labels

    # Long format: index + (label|english)
    if "index" in df.columns and ("label" in df.columns or "english" in df.columns):
        labcol = "label" if "label" in df.columns else "english"
        if not sub.empty:
            labels = []
            for i in indices:
                hit = sub[pd.to_numeric(sub["index"], errors="coerce") == i]
                if not hit.empty:
                    lab = str(hit.iloc[0][labcol]).strip()
                    if lab and lab.lower() != "nan":
                        labels.append(lab)
            if labels:
                return labels

    return None

def _label_for_single(scales_df: pd.DataFrame, code: str, idx: int) -> Optional[str]:
    labs = _labels_for_indices(scales_df, code, [idx])
    return labs[0] if labs else None

def _compose_metric_label(base_label: str, meaning_lbls: Optional[List[str]]) -> str:
    """
    Make metric label descriptive for the AI (e.g., "% selecting Strongly agree / Agree").
    """
    if meaning_lbls:
        return f"% selecting {' / '.join(meaning_lbls)}"
    return base_label or "% of respondents"


# ─────────────────────────────────────────────────────────────────────────────
# Column detection & availability
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c is not None:
            return c
    return None

def _has_valid_values(series: pd.Series) -> bool:
    if series is None or series.empty:
        return False
    s = pd.to_numeric(series, errors="coerce").replace({SENTINEL: pd.NA})
    return s.notna().any()

def _detect_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Year", "year", "SURVEYR", "survey_year"):
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c)
        if len(s) == 4 and s.isdigit() and 1900 <= int(s) <= 2100:
            return c
    return None

def _detect_group_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Demographic", "group", "Group", "DEMOLBL", "demographic"):
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Choose reporting field & column (POLARITY + fallbacks)
# ─────────────────────────────────────────────────────────────────────────────

def _choose_reporting(df: pd.DataFrame, qcode: str, qmeta: pd.Series) -> Tuple[str, str, str, Optional[List[int]]]:
    """
    Returns (metric_col, reporting_field, metric_label, meaning_indices).
    reporting_field in {'POSITIVE','NEGATIVE','AGREE','ANSWER1'}.
    """
    col_positive = _find_col(df, ["Positive", "POSITIVE"])
    col_negative = _find_col(df, ["Negative", "NEGATIVE"])
    col_agree    = _find_col(df, ["AGREE", "Agree"])
    col_answer1  = _find_col(df, ["Answer1", "ANSWER1", "Answer 1"])

    pol = (qmeta.get("polarity") or "POS")
    pol = str(pol).upper().strip()

    def available(colname: Optional[str]) -> bool:
        return bool(colname) and _has_valid_values(df[colname])  # type: ignore[index]

    target = None
    metric_label = ""
    meta_field = None

    if pol == "POS":
        if available(col_positive):
            target, meta_field, metric_label = col_positive, "positive", "% favourable"
        elif available(col_agree):
            target, meta_field, metric_label = col_agree, "agree", "% selected (AGREE)"
        elif available(col_answer1):
            target, meta_field, metric_label = col_answer1, None, "% selected (Answer1)"
    elif pol == "NEG":
        if available(col_negative):
            target, meta_field, metric_label = col_negative, "negative", "% problem"
        elif available(col_agree):
            target, meta_field, metric_label = col_agree, "agree", "% selected (AGREE)"
        elif available(col_answer1):
            target, meta_field, metric_label = col_answer1, None, "% selected (Answer1)"
    else:  # NEU
        if available(col_agree):
            target, meta_field, metric_label = col_agree, "agree", "% selected (AGREE)"
        elif available(col_answer1):
            target, meta_field, metric_label = col_answer1, None, "% selected (Answer1)"

    if not target:
        for col, label, mfield in [
            (col_positive, "% favourable", "positive"),
            (col_negative, "% problem", "negative"),
            (col_agree, "% selected (AGREE)", "agree"),
            (col_answer1, "% selected (Answer1)", None),
        ]:
            if available(col):
                target, metric_label, meta_field = col, label, mfield
                break

    if not target:
        return "", "", "", None

    if target == col_positive:
        rf = "POSITIVE"
    elif target == col_negative:
        rf = "NEGATIVE"
    elif target == col_agree:
        rf = "AGREE"
    else:
        rf = "ANSWER1"

    meaning_idx = [1] if rf == "ANSWER1" else (_parse_indices(qmeta.get(meta_field)) if meta_field else None)
    return target, rf, metric_label, meaning_idx


# ─────────────────────────────────────────────────────────────────────────────
# Validation (advisory only)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _validate_frame(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    pos_col = _find_col(df, ["Positive", "POSITIVE"])
    neg_col = _find_col(df, ["Negative", "NEGATIVE"])
    neu_col = _find_col(df, ["Neutral", "NEUTRAL"])

    all_ok = True
    for _, row in df.iterrows():
        yr = str(row.get("Year", row.get("year", "?")))
        demo = row.get("Demographic", row.get("group", None))
        who = f"Year {yr}" + (f", {demo}" if isinstance(demo, str) and demo else "")

        if pos_col and neg_col:
            p = _safe_float(row[pos_col])  # type: ignore[index]
            n = _safe_float(row[neg_col])  # type: ignore[index]
            if p is not None and n is not None:
                delta = abs((100.0 - p) - n)
                if delta > 1.0:
                    all_ok = False
                    issues.append(f"{who}: (100 − Positive) vs Negative differs by {delta:.1f} pts.")

        if pos_col and neu_col and neg_col:
            p = _safe_float(row[pos_col])   # type: ignore[index]
            u = _safe_float(row[neu_col])   # type: ignore[index]
            n = _safe_float(row[neg_col])   # type: ignore[index]
            if None not in (p, u, n):
                s = p + u + n  # type: ignore[operator]
                if abs(s - 100.0) > 1.0:
                    all_ok = False
                    issues.append(f"{who}: Positive+Neutral+Negative = {s:.1f} (≠ 100).")

    return all_ok, issues


# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_ai_summary(selection_key: str, qcode: str, payload_str: str, system_prompt: str) -> Tuple[str, Optional[str]]:
    json_text, err = ai.call_openai_json(system_prompt, payload_str)
    return json_text or "", err

def _clear_ai_caches():
    try:
        _cached_ai_summary.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        if str(k).startswith("menu1_ai_cache_"):
            del st.session_state[k]


# ─────────────────────────────────────────────────────────────────────────────
# Special payload for multi-response distribution items (e.g., D57_2)
# ─────────────────────────────────────────────────────────────────────────────

def _build_distribution_payload_for_d57_2(qcode: str, qtext: str, df_disp: pd.DataFrame, scales_df: pd.DataFrame) -> Optional[str]:
    if df_disp is None or df_disp.empty:
        return None

    ycol = _detect_year_col(df_disp)
    if not ycol:
        return None

    gcol = _detect_group_col(df_disp)
    df = df_disp.copy()

    years = pd.to_numeric(df[ycol], errors="coerce")
    df["_year_"] = years
    df = df[~df["_year_"].isna()]
    if df.empty:
        return None
    latest_year = int(df["_year_"].max())

    latest_rows = df[df["_year_"] == latest_year]
    if latest_rows.empty:
        return None
    if gcol:
        mask_all = latest_rows[gcol].astype(str).str.lower().str.contains("all respondents", na=False)
        chosen_row = latest_rows[mask_all].iloc[0] if mask_all.any() else latest_rows.iloc[0]
    else:
        chosen_row = latest_rows.iloc[0]

    answer_cols = [(c, int(c.lower().replace("answer", "").strip())) for c in df_disp.columns
                   if c.lower().startswith("answer") and c.lower().replace("answer", "").strip().isdigit()]
    if not answer_cols:
        return None

    options: List[Dict[str, Any]] = []
    for col, idx in sorted(answer_cols, key=lambda x: x[1]):
        raw = chosen_row[col]
        try:
            val = int(round(float(raw)))
        except Exception:
            continue
        if val == SENTINEL or pd.isna(val):
            continue
        label = _label_for_single(scales_df, qcode, idx) or f"Answer {idx}"
        options.append({"index": idx, "label": label, "value": val})

    if not options:
        return None

    group_name = str(chosen_row[gcol]) if gcol else "All respondents"

    payload = {
        "instruction": "Multi-response item. Report ALL options for the latest year exactly as provided; no aggregation.",
        "question": {
            "code": qcode,
            "text": qtext or qcode,
            "polarity_hint": "NEU",
            "reporting_field": "DISTRIBUTION"
        },
        "distribution": {
            "year": latest_year,
            "group": group_name,
            "options": options
        }
    }
    user_msg = "Multi-response item: list each option and its integer percent exactly as provided.\n\n" + json.dumps(payload, ensure_ascii=False)
    return user_msg


# ─────────────────────────────────────────────────────────────────────────────
# TABLES + TOOLBAR (Export + Start New)
# ─────────────────────────────────────────────────────────────────────────────

def _render_toolbar_and_tabs(tables_by_q: Dict[str, pd.DataFrame],
                             qtext_by_q: Dict[str, str],
                             sources_by_q: Optional[Dict[str, str]],
                             category_in_play: bool) -> None:
    """
    Renders:
      • Toolbar row: Export to Excel (left) + Start a new search (right), aligned
      • Tabs: Summary (computed) + one tab per question
    """
    # Toolbar row
    c_left, c_spacer, c_right = st.columns([2, 6, 2], gap="small")
    with c_left:
        _render_export_button(tables_by_q)
    with c_right:
        if st.button("🔁 Start a new search", key="menu1_ai_new_search_top"):
            _clear_ai_caches()
            st.toast("AI selections cleared. Adjust filters above and run again.", icon="✅")

    # Tabs
    if not tables_by_q:
        st.info("No questions selected.")
        return

    ordered_codes = [k for k in sorted(tables_by_q.keys()) if isinstance(tables_by_q.get(k), pd.DataFrame)]
    if not ordered_codes:
        st.info("No questions selected.")
        return

    # Build a computed "Summary" table (latest year + prev year + YoY + multi-year trend)
    summary_df = _build_overall_summary_table(tables_by_q, qtext_by_q, category_in_play)

    tab_labels = ["Summary"] + ordered_codes
    tabs = st.tabs(tab_labels)

    # Summary tab first
    with tabs[0]:
        st.markdown("##### Overall summary")
        st.dataframe(summary_df, use_container_width=True)

    # Then per-question tabs
    for i, qcode in enumerate(ordered_codes, start=1):
        with tabs[i]:
            title = qtext_by_q.get(qcode, qcode)
            st.markdown(f"##### {qcode} — {title}")
            st.dataframe(tables_by_q[qcode], use_container_width=True)
            if sources_by_q:
                src = sources_by_q.get(qcode)
                if isinstance(src, str) and src.strip():
                    st.caption(src)


def _render_export_button(tables_by_q: Dict[str, pd.DataFrame]) -> None:
    if not tables_by_q:
        st.button("⬇️ Export to Excel", disabled=True, help="No tables to export.")
        return

    # Build one workbook, one sheet per question (safe sheet names)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for qcode, df in tables_by_q.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            sheet_name = str(qcode)[:31] if str(qcode).strip() else "Sheet"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        writer.close()
    buffer.seek(0)

    st.download_button(
        label="⬇️ Export to Excel",
        data=buffer,
        file_name="PSES_Explorer_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="menu1_export_excel",
        help="Download selected tabulations as an Excel workbook (one sheet per question).",
    )


def _build_overall_summary_table(tables_by_q: Dict[str, pd.DataFrame],
                                 qtext_by_q: Dict[str, str],
                                 category_in_play: bool) -> pd.DataFrame:
    """
    Builds a compact table summarizing the selected metric per question:
      Question, Title, Latest Year, Value, Prev Year, YoY Δ, Trend (All Years)
    NOTE: The "selected metric" for each question is chosen by POLARITY+fallbacks, same as AI.
    """
    qmeta_df = _load_survey_questions()
    rows: List[Dict[str, Any]] = []

    for qcode, df in tables_by_q.items():
        if qcode in EXCLUDE_FROM_OVERALL:
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # Polarity + metric selection
        rowm = qmeta_df[qmeta_df["code"].str.upper() == str(qcode).upper()]
        qmeta = rowm.iloc[0] if not rowm.empty else pd.Series({"polarity": "POS", "positive": None, "negative": None, "agree": None})
        metric_col, reporting_field, _, _ = _choose_reporting(df, qcode, qmeta)
        if not metric_col:
            continue

        ycol = _detect_year_col(df)
        if not ycol:
            continue

        # Only the Year + metric are relevant for this summary
        tmp = df[[ycol, metric_col]].copy()
        tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce").replace({SENTINEL: pd.NA})
        tmp = tmp.dropna(subset=[ycol, metric_col]).sort_values(ycol)
        if tmp.empty:
            continue

        years = tmp[ycol].astype(int).tolist()
        vals = tmp[metric_col].astype(float).tolist()
        latest_y = int(years[-1])
        latest_v = vals[-1]

        prev_y, prev_v, yoy = None, None, None
        if len(vals) >= 2:
            prev_y = int(years[-2])
            prev_v = vals[-2]
            yoy = None if (prev_v is None or pd.isna(prev_v)) else round(latest_v - prev_v, 1)

        trend = _classify_trend(vals)

        rows.append({
            "Question": qcode,
            "Title": qtext_by_q.get(qcode, qcode),
            "Latest year": latest_y,
            "Value": round(latest_v, 1) if latest_v is not None and not pd.isna(latest_v) else None,
            "Prev year": prev_y,
            "YoY Δ (pts)": yoy,
            "Trend (all years)": trend,
        })

    if not rows:
        return pd.DataFrame({"Message": ["No data available for summary under current filters."]})

    out = pd.DataFrame(rows)
    # Keep a stable, readable order
    return out[["Question", "Title", "Latest year", "Value", "Prev year", "YoY Δ (pts)", "Trend (all years)"]]


def _classify_trend(series_vals: List[float]) -> str:
    """Very light-touch multi-year trend: increase/decrease/stable/insufficient."""
    vals = [v for v in series_vals if v is not None and not pd.isna(v)]
    if len(vals) < 2:
        return "insufficient"
    first, last = vals[0], vals[-1]
    net = last - first
    if abs(net) <= 1.0:
        return "stable"
    return "increase" if net > 0 else "decrease"


# ─────────────────────────────────────────────────────────────────────────────
# AI SUMMARY: public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_ai_summary_section(
    tables_by_q: Dict[str, pd.DataFrame],
    qtext_by_q: Dict[str, str],
    category_in_play: bool,
    ai_enabled: bool,
    selection_key: str,
) -> None:
    st.markdown("---")
    st.markdown("### AI Summary")

    if not tables_by_q:
        st.info("No questions selected.")
        return

    qmeta_df = _load_survey_questions()
    scales_df = _load_scales()

    overall_rows: List[Dict[str, Any]] = []
    q_to_metric: Dict[str, str] = {}
    code_to_text: Dict[str, str] = {}
    code_to_polhint: Dict[str, str] = {}
    code_to_rfield: Dict[str, str] = {}
    code_to_midx: Dict[str, List[int]] = {}
    code_to_mlbl: Dict[str, List[str]] = {}

    # Per-question
    for qcode, df in tables_by_q.items():
        qtext = qtext_by_q.get(qcode, qcode)
        code_to_text[qcode] = qtext

        # D57_2 special handling (distribution)
        if qcode in EXCEPT_MULTI_DISTRIBUTION:
            narrative = ""
            if ai_enabled:
                special_payload = _build_distribution_payload_for_d57_2(qcode, qtext, df, scales_df)
                if special_payload is not None:
                    with st.spinner(f"Generating summary…"):
                        json_text, err = _cached_ai_summary(selection_key, qcode, special_payload, ai.AI_SYSTEM_PROMPT)
                    narrative = ai.extract_narrative(json_text) or ""
            if narrative:
                # Show only the question TEXT (avoid code duplication)
                st.markdown(f"**{qtext}**")
                st.write(narrative)
            # Exclude from overall
            continue

        # Attach metadata (fallback if missing)
        row = qmeta_df[qmeta_df["code"].str.upper() == str(qcode).upper()]
        qmeta = row.iloc[0] if not row.empty else pd.Series({"polarity": "POS", "positive": None, "negative": None, "agree": None})

        # Choose reporting field
        metric_col, reporting_field, metric_label, meaning_idx = _choose_reporting(df, qcode, qmeta)
        if not metric_col:
            st.caption(f"⚠️ No data to summarize for {qcode} under current filters.")
            continue

        meaning_lbls = _labels_for_indices(scales_df, qcode, meaning_idx) if meaning_idx else None
        metric_label_verbose = _compose_metric_label(metric_label, meaning_lbls)

        q_to_metric[qcode] = metric_label_verbose
        code_to_polhint[qcode] = (qmeta.get("polarity") or "POS").upper()
        code_to_rfield[qcode] = reporting_field
        if meaning_idx:
            code_to_midx[qcode] = meaning_idx
        if meaning_lbls:
            code_to_mlbl[qcode] = meaning_lbls

        # Build per-q payload (include ALL years; trim to Year+metric when no subgroup to avoid bogus "gaps")
        ycol = _detect_year_col(df)
        if not ycol:
            st.caption(f"⚠️ No year column detected for {qcode}.")
            continue
        if category_in_play:
            df_for_ai = df
        else:
            # Only the Year + selected metric to prevent gap claims on All Respondents
            df_for_ai = df[[ycol, metric_col]].copy()

        payload_str = ai.build_per_q_prompt(
            question_code=qcode,
            question_text=qtext,
            df_disp=df_for_ai,
            metric_col=metric_col,
            metric_label=metric_label_verbose,
            category_in_play=bool(category_in_play),
            polarity_hint=code_to_polhint[qcode],
            reporting_field=reporting_field,
            meaning_indices=meaning_idx,
            meaning_labels=meaning_lbls,
        )

        narrative = ""
        if ai_enabled:
            with st.spinner(f"Generating summary…"):
                json_text, err = _cached_ai_summary(selection_key, qcode, payload_str, ai.AI_SYSTEM_PROMPT)
            narrative = ai.extract_narrative(json_text) or ""
        if narrative:
            # Show only the question TEXT (avoid code duplication)
            st.markdown(f"**{qtext}**")
            st.write(narrative)

        # Collect rows for Overall (for synthesis context, not shown as a table here)
        tmp = df[[ycol, metric_col]].copy()
        tmp = tmp.rename(columns={ycol: "Year", metric_col: "Value"})
        tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce").astype("Int64")
        tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").astype("Int64").replace({SENTINEL: pd.NA})
        tmp = tmp.dropna(subset=["Year", "Value"])
        for _, r in tmp.iterrows():
            overall_rows.append({"q": qcode, "y": int(r["Year"]), "v": int(r["Value"])})

    # Overall synthesis (exclude multi-response items)
    usable_rows = [r for r in overall_rows if r["q"] not in EXCLUDE_FROM_OVERALL]
    if len(usable_rows) >= 2:
        ov = pd.DataFrame(usable_rows)
        if not ov.empty:
            pivot = ov.pivot_table(index="q", columns="y", values="v", aggfunc="first").reset_index()
            pivot = pivot.rename(columns={"q": "question_code"})
            user_msg = ai.build_overall_prompt(
                tab_labels=[q for q in tables_by_q.keys() if q not in EXCLUDE_FROM_OVERALL],
                pivot_df=pivot,
                q_to_metric=q_to_metric,
                code_to_text=code_to_text,
                code_to_polarity_hint=code_to_polhint,
                code_to_reporting_field=code_to_rfield,
                code_to_meaning_indices=code_to_midx if code_to_midx else None,
                code_to_meaning_labels=code_to_mlbl if code_to_mlbl else None,
            )
            if user_msg != "__NO_DATA_OVERALL__":
                with st.spinner("Generating overall synthesis…"):
                    json_text, err = ai.call_openai_json(ai.AI_SYSTEM_PROMPT, user_msg)
                overall_narr = ai.extract_narrative(json_text)
                if overall_narr:
                    st.markdown("#### Overall")
                    st.write(overall_narr)

    # Validation line + (a second) Start New (kept below per your spec)
    _render_validation_line(tables_by_q)

    c_left, _, c_right = st.columns([2, 6, 2], gap="small")
    with c_right:
        if st.button("🔁 Start a new search", key="menu1_ai_new_search_bottom"):
            _clear_ai_caches()
            st.toast("AI selections cleared. Adjust filters above and run again.", icon="✅")


# ─────────────────────────────────────────────────────────────────────────────
# Payload normalizer and back-compat entry
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_bool(x: Any, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return default

def _pick_best_df_dict(d: Dict[str, Any]) -> Optional[Dict[str, pd.DataFrame]]:
    """From a dict of arbitrary values, pick the sub-dict that looks like {code: DataFrame} with max DF count."""
    candidates: List[Tuple[str, Dict[str, pd.DataFrame], int]] = []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            df_count = sum(1 for _k, _v in v.items() if isinstance(_v, pd.DataFrame))
        else:
            df_count = 0
        if df_count:
            candidates.append((k, v, df_count))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[0][1]

def _extract_tables_and_texts_from_payload(p: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    # Direct names first
    tables_by_q = p.get("tables_by_q") or p.get("tables") or p.get("results_by_q") or p.get("dfs_by_q") or {}
    if not isinstance(tables_by_q, dict) or not any(isinstance(v, pd.DataFrame) for v in tables_by_q.values()):
        # Try to auto-detect the best dict of DataFrames
        guess = _pick_best_df_dict(p)
        tables_by_q = guess or {}

    # Texts
    qtext_by_q = p.get("qtext_by_q") or p.get("question_texts") or p.get("labels_by_q") or {}
    if not isinstance(qtext_by_q, dict) or not qtext_by_q:
        # Try list-shaped payloads: tabs/panes/items/questions etc.
        for list_key in ("tabs", "panes", "items", "questions", "rows", "entries"):
            lst = p.get(list_key)
            if isinstance(lst, list) and lst:
                acc: Dict[str, str] = {}
                for it in lst:
                    if isinstance(it, dict):
                        code = it.get("code") or it.get("question") or it.get("id") or it.get("name")
                        text = it.get("text") or it.get("label") or it.get("title")
                        if code:
                            acc[str(code)] = str(text or code)
                if acc:
                    qtext_by_q = acc
                    break

    # Fallback: derive texts from tables_by_q keys
    if (not isinstance(qtext_by_q, dict)) or (not qtext_by_q):
        qtext_by_q = {str(k): str(k) for k in tables_by_q.keys()}

    # Ensure keys line up as strings
    tables_by_q = {str(k): v for k, v in tables_by_q.items() if isinstance(v, pd.DataFrame)}
    qtext_by_q = {str(k): str(v) for k, v in qtext_by_q.items()}

    return tables_by_q, qtext_by_q

def _extract_sources_from_payload(p: Dict[str, Any]) -> Optional[Dict[str, str]]:
    # Accept several common names for per-question source link text
    for key in ("sources_by_q", "source_links_by_q", "source_links", "links_by_q", "links"):
        v = p.get(key)
        if isinstance(v, dict) and v:
            return {str(k): str(vv) for k, vv in v.items() if isinstance(vv, (str, int, float))}
    return None


# Back-compat: existing callers expect this symbol in menu1.render.results
def tabs_summary_and_per_q(*args, **kwargs) -> None:
    """
    Backward-compatible entry point expected by some app code.

    Supports:
      1) tabs_summary_and_per_q(tables_by_q, qtext_by_q, category_in_play, ai_enabled, selection_key)
      2) tabs_summary_and_per_q(payload={ ... }) where payload keys may vary.
         If a payload is provided, this will render:
           - Toolbar (Export + Start New),
           - TABULATED TABLES (including a computed 'Summary' tab),
           - then the AI Summary section below.
    """
    # Style 2: payload kwarg
    if "payload" in kwargs and isinstance(kwargs["payload"], dict):
        p = dict(kwargs["payload"])  # shallow copy
        tables_by_q, qtext_by_q = _extract_tables_and_texts_from_payload(p)
        sources_by_q = _extract_sources_from_payload(p)

        # Toolbar + Tabs (restores table UX)
        category_in_play = _coerce_bool(
            p.get("category_in_play", p.get("demographic_active", p.get("has_subgroup", False))), False
        )
        _render_toolbar_and_tabs(tables_by_q, qtext_by_q, sources_by_q, category_in_play)

        # AI flags
        ai_enabled = _coerce_bool(p.get("ai_enabled", p.get("ai", p.get("use_ai", True))), True)
        selection_key = p.get("selection_key")
        if not selection_key:
            try:
                years_seen = set()
                for q, df in (tables_by_q or {}).items():
                    ycol = _detect_year_col(df) if isinstance(df, pd.DataFrame) else None
                    if ycol and isinstance(df, pd.DataFrame):
                        years_seen.update(pd.to_numeric(df[ycol], errors="coerce").dropna().astype(int).tolist())
                sig = {"qs": sorted(list(tables_by_q.keys())),
                       "years": sorted(list(years_seen)),
                       "demo": "ON" if category_in_play else "ALL"}
                selection_key = hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()
            except Exception:
                selection_key = "menu1_legacy_selection"

        # AI Summary below the tabs
        return render_ai_summary_section(
            tables_by_q=tables_by_q or {},
            qtext_by_q=qtext_by_q or {},
            category_in_play=category_in_play,
            ai_enabled=ai_enabled,
            selection_key=selection_key,
        )

    # Style 1: positional/kwargs (assume tables are rendered upstream)
    if args and len(args) >= 5:
        tables_by_q, qtext_by_q, category_in_play, ai_enabled, selection_key = args[:5]
        # Render toolbar + tabs so UX remains consistent even in direct calls
        _render_toolbar_and_tabs(tables_by_q, qtext_by_q, None, bool(category_in_play))
        return render_ai_summary_section(tables_by_q, qtext_by_q, category_in_play, ai_enabled, selection_key)

    # Mixed kwargs
    tables_by_q = kwargs.get("tables_by_q", args[0] if args else {})
    qtext_by_q = kwargs.get("qtext_by_q", {})
    category_in_play = _coerce_bool(kwargs.get("category_in_play", False), False)
    _render_toolbar_and_tabs(tables_by_q, qtext_by_q, None, category_in_play)
    return render_ai_summary_section(
        tables_by_q=tables_by_q,
        qtext_by_q=qtext_by_q,
        category_in_play=category_in_play,
        ai_enabled=_coerce_bool(kwargs.get("ai_enabled", True), True),
        selection_key=kwargs.get("selection_key", "menu1_default_selection"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation line UI
# ─────────────────────────────────────────────────────────────────────────────

def _render_validation_line(tables_by_q: Dict[str, pd.DataFrame]) -> None:
    all_ok = True
    problems: List[str] = []
    for code, df in tables_by_q.items():
        ok, issues = _validate_frame(df)
        if not ok:
            all_ok = False
            problems.extend([f"{code}: {msg}" for msg in issues])

    if all_ok:
        st.markdown("**AI Data Validation:** ✅ All summaries consistent.")
    else:
        st.markdown("**AI Data Validation:** ❌ Data mismatch detected.")
        with st.expander("Details"):
            for msg in problems:
                st.write("• " + msg)
