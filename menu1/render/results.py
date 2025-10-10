# menu1/results.py
# ---------------------------------------------------------------------
# Menu 1 – Results: AI Summary + AI Data Validation (metadata-driven).
#
# Implements:
# - No math in reporting: we only forward pre-aggregated values.
# - POLARITY in Survey Questions.xlsx selects the reporting field:
#       POS -> POSITIVE
#       NEG -> NEGATIVE
#       NEU -> AGREE
# - Fallbacks when the chosen field is missing or sentinel (9999):
#       POS/NEG -> AGREE -> ANSWER1
#       NEU     -> AGREE -> ANSWER1
# - NEUTRAL exists in metadata for completeness (not used in reporting).
# - Meaning indices for POSITIVE/NEGATIVE/AGREE are read from metadata (e.g., "1,2")
#   and mapped to labels via Survey Scales.xlsx; both are passed to the AI.
# - Per-question AI summaries render progressively with spinners.
# - Overall synthesis appears when >1 question is selected.
# - Caching reuses summaries on identical selections (no duplicate calls when toggling AI).
# - "AI Data Validation" shows ✅ or ❌ with a details expander (advisory only).
# - "Start a new search" preserves AI toggle state and clears AI caches/selections only.
# - Technical Notes and source links under tables remain unchanged.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import math
import json
import pandas as pd
import streamlit as st

# Import your AI helpers (supports addendum + all-years trend classification)
try:
    from menu1 import ai
except Exception:
    import ai  # fallback if your module path differs

SENTINEL = 9999

# ─────────────────────────────────────────────────────────────────────────────
# Metadata loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_survey_questions() -> pd.DataFrame:
    """
    Expected columns (case-insensitive; normalized to lower-case):
      - code (question code, e.g., Q44a)
      - text or english (optional; used when available)
      - polarity in {POS, NEG, NEU}
      - positive (metadata string like "1,2")
      - negative (metadata string like "4,5")
      - agree   (metadata string like "4,5")
      - neutral (optional; not used for reporting)
    """
    df = pd.read_excel("metadata/Survey Questions.xlsx")
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    # Normalize canonical columns
    if "question" in df.columns and "code" not in df.columns:
        df = df.rename(columns={"question": "code"})
    if "english" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"english": "text"})

    # Standardize essentials
    df["code"] = df["code"].astype(str).str.strip()
    df["text"] = df.get("text", pd.Series([None]*len(df))).astype(str)
    pol = df.get("polarity", pd.Series(["POS"]*len(df))).astype(str).str.upper().str.strip()
    pol = pol.where(pol.isin(["POS", "NEG", "NEU"]), "POS")
    df["polarity"] = pol

    # Keep raw strings for indices mapping
    for col in ["positive", "negative", "agree", "neutral"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)

    return df[["code", "text", "polarity", "positive", "negative", "agree", "neutral"]]


@st.cache_data(show_spinner=False)
def _load_scales() -> pd.DataFrame:
    """
    Loads Survey Scales.xlsx (wide or long).
    Wide example:
      code | answer1 | answer2 | answer3 | ...
    Long example:
      code | index | label (or english)
    """
    df = pd.read_excel("metadata/Survey Scales.xlsx")
    return df.rename(columns={c: c.strip().lower() for c in df.columns})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: parsing metadata & resolving labels
# ─────────────────────────────────────────────────────────────────────────────

def _parse_indices(meta_val: Any) -> Optional[List[int]]:
    """Parse a metadata string like '1,2' into [1,2]. Returns None if empty."""
    if meta_val is None or (isinstance(meta_val, float) and math.isnan(meta_val)):
        return None
    s = str(meta_val).strip()
    if not s:
        return None
    # Accept separators: comma, semicolon, pipe
    tokens = [t.strip() for t in s.replace(";", ",").replace("|", ",").split(",") if t.strip()]
    out: List[int] = []
    for t in tokens:
        try:
            out.append(int(t))
        except Exception:
            continue
    return out or None


def _labels_for_indices(scales_df: pd.DataFrame, code: str, indices: Optional[List[int]]) -> Optional[List[str]]:
    """Return labels for the given indices from Survey Scales; supports wide and long formats."""
    if not indices:
        return None
    code_u = str(code).strip().upper()
    df = scales_df

    # Try wide format
    if "code" in df.columns:
        row = df[df["code"].astype(str).str.upper() == code_u]
        wide_cols = [c for c in df.columns if c.startswith("answer") and c[6:].isdigit()]
        if not row.empty and wide_cols:
            labels: List[str] = []
            r0 = row.iloc[0]
            for i in indices:
                col = f"answer{i}".lower()
                if col in row.columns:
                    val = str(r0[col]).strip()
                    if val and val.lower() != "nan":
                        labels.append(val)
            if labels:
                return labels

    # Try long format
    long_ok = {"code", "index"} <= set(df.columns) and ("label" in df.columns or "english" in df.columns)
    if long_ok:
        labcol = "label" if "label" in df.columns else "english"
        sub = df[df["code"].astype(str).str.upper() == code_u]
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


# ─────────────────────────────────────────────────────────────────────────────
# Selecting the reporting field & column based on POLARITY + fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first column present (case-insensitive) among candidates, return original name."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c is not None:
            return c
    return None


def _has_valid_values(series: pd.Series) -> bool:
    """True if at least one non-sentinel, non-NA value exists."""
    if series is None or series.empty:
        return False
    s = pd.to_numeric(series, errors="coerce").replace({SENTINEL: pd.NA})
    return s.notna().any()


def _choose_reporting(df: pd.DataFrame, qcode: str, qmeta: pd.Series) -> Tuple[str, str, str, Optional[List[int]]]:
    """
    Decide (metric_col, reporting_field, metric_label, meaning_indices) for this question.
      metric_col: exact column name in df (case-preserving).
      reporting_field: 'POSITIVE' | 'NEGATIVE' | 'AGREE' | 'ANSWER1'
      metric_label: human label for AI (compat).
      meaning_indices: [1,2], [4,5], etc. Based on metadata for the chosen field,
                       or [1] for ANSWER1; can be None.

    NEUTRAL is intentionally not used for reporting.
    """
    # Common column name candidates
    col_positive = _find_col(df, ["Positive", "POSITIVE"])
    col_negative = _find_col(df, ["Negative", "NEGATIVE"])
    col_agree    = _find_col(df, ["AGREE", "Agree"])
    col_answer1  = _find_col(df, ["Answer1", "ANSWER1", "Answer 1"])

    pol = (qmeta.get("polarity") or "POS").upper().strip()

    def available(colname: Optional[str]) -> bool:
        return bool(colname) and _has_valid_values(df[colname])  # type: ignore[index]

    # Selection by polarity + fallback chain
    target = None
    metric_label = ""
    meta_field = None  # which metadata column we read indices from
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

    # As a last resort, try any present in a sensible order
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
        # No usable column under current filters
        return "", "", "", None

    # reporting_field string for the AI
    rf = "ANSWER1"
    if target == col_positive:
        rf = "POSITIVE"
    elif target == col_negative:
        rf = "NEGATIVE"
    elif target == col_agree:
        rf = "AGREE"

    # meaning indices
    if rf == "ANSWER1":
        meaning_idx = [1]
    else:
        meaning_idx = _parse_indices(qmeta.get(meta_field)) if meta_field else None

    return target, rf, metric_label, meaning_idx


# ─────────────────────────────────────────────────────────────────────────────
# Validation (advisory only; never affects reported numbers)
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
    """
    Advisory checks across displayed rows (year × demographic):
      (a) If Positive and Negative exist: |(100 - Positive) - Negative| <= 1.0
      (b) If Positive+Neutral+Negative exist: sum ≈ 100 (+/- 1.0)
    """
    issues: List[str] = []
    cols = set(df.columns)

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
def _cached_ai_summary(
    selection_key: str,
    qcode: str,
    payload_str: str,
    system_prompt: str,
) -> Tuple[str, Optional[str]]:
    """
    Cache the AI response for a specific (selection_key, qcode, payload).
    Returns raw JSON text and error hint from ai.call_openai_json.
    """
    json_text, err = ai.call_openai_json(system_prompt, payload_str)
    return json_text or "", err


def _clear_ai_caches():
    try:
        _cached_ai_summary.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Also clear any local session scratch keys used by this module
    for k in list(st.session_state.keys()):
        if str(k).startswith("menu1_ai_cache_"):
            del st.session_state[k]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_ai_summary_section(
    tables_by_q: Dict[str, pd.DataFrame],
    qtext_by_q: Dict[str, str],
    category_in_play: bool,
    ai_enabled: bool,
    selection_key: str,
) -> None:
    """
    Render the AI Summary and the AI Data Validation line directly below the tabs.

    Parameters
    ----------
    tables_by_q : Dict[qcode, DataFrame]
        Display tables already shown in tabs; columns may include:
        Year, optional Demographic/group, Positive, Neutral, Negative, AGREE, Answer1..7, n.
    qtext_by_q : Dict[qcode, text]
        Question text for each code.
    category_in_play : bool
        True if a demographic subgroup is active; used for phrasing by the AI.
    ai_enabled : bool
        Current AI toggle state. When False, we skip calls but still show validation.
    selection_key : str
        Stable identifier for the current selection (questions + years + demo), used for caching.
    """
    st.markdown("---")
    st.markdown("### AI Summary")

    if not tables_by_q:
        st.info("No questions selected.")
        return

    # Load metadata once
    qmeta_df = _load_survey_questions()
    scales_df = _load_scales()

    narratives: Dict[str, str] = {}
    overall_rows: List[Dict[str, Any]] = []
    q_to_metric: Dict[str, str] = {}
    code_to_text: Dict[str, str] = {}
    code_to_polhint: Dict[str, str] = {}
    code_to_rfield: Dict[str, str] = {}
    code_to_midx: Dict[str, List[int]] = {}
    code_to_mlbl: Dict[str, List[str]] = {}

    # 1) Per-question summaries (progressive)
    for qcode, df in tables_by_q.items():
        qtext = qtext_by_q.get(qcode, qcode)
        code_to_text[qcode] = qtext

        # Attach metadata row (fallback defaults if missing)
        row = qmeta_df[qmeta_df["code"].str.upper() == str(qcode).upper()]
        if row.empty:
            qmeta = pd.Series({"polarity": "POS", "positive": None, "negative": None, "agree": None})
        else:
            qmeta = row.iloc[0]

        # Choose reporting field + column by POLARITY with fallbacks
        metric_col, reporting_field, metric_label, meaning_idx = _choose_reporting(df, qcode, qmeta)
        if not metric_col:
            st.caption(f"⚠️ No data to summarize for {qcode} under current filters.")
            continue

        # Resolve labels (optional)
        meaning_lbls = _labels_for_indices(scales_df, qcode, meaning_idx) if meaning_idx else None

        # Save for overall payload
        q_to_metric[qcode] = metric_label or "% of respondents"
        code_to_polhint[qcode] = (qmeta.get("polarity") or "POS").upper()
        code_to_rfield[qcode] = reporting_field
        if meaning_idx:
            code_to_midx[qcode] = meaning_idx
        if meaning_lbls:
            code_to_mlbl[qcode] = meaning_lbls

        # Build per-question payload (include ALL years so AI can classify trend across all years)
        payload_str = ai.build_per_q_prompt(
            question_code=qcode,
            question_text=qtext,
            df_disp=df,
            metric_col=metric_col,
            metric_label=metric_label,
            category_in_play=bool(category_in_play),
            polarity_hint=code_to_polhint[qcode],
            reporting_field=reporting_field,
            meaning_indices=meaning_idx,
            meaning_labels=meaning_lbls,
        )

        # Progressive rendering (spinner only when AI is enabled)
        narrative = ""
        if ai_enabled:
            with st.spinner(f"Generating summary for {qcode}…"):
                json_text, err = _cached_ai_summary(selection_key, qcode, payload_str, ai.AI_SYSTEM_PROMPT)
            narrative = ai.extract_narrative(json_text) or ""
        narratives[qcode] = narrative

        # Render per-question block (minimal spacing)
        if narrative:
            st.write(f"**{qcode} — {qtext}**  \n{narrative}")

        # For overall payload: collect compact rows for chosen metric
        ycol = "Year" if "Year" in df.columns else ("year" if "year" in df.columns else None)
        if ycol is not None:
            tmp = df[[ycol, metric_col]].copy()
            tmp = tmp.rename(columns={ycol: "Year", metric_col: "Value"})
            tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce").astype("Int64")
            tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").astype("Int64").replace({SENTINEL: pd.NA})
            tmp = tmp.dropna(subset=["Year", "Value"])
            for _, r in tmp.iterrows():
                overall_rows.append({"q": qcode, "y": int(r["Year"]), "v": int(r["Value"])})

    # 2) Overall synthesis (appears when >1 question)
    if len(overall_rows) >= 2:
        ov = pd.DataFrame(overall_rows)
        if not ov.empty:
            pivot = ov.pivot_table(index="q", columns="y", values="v", aggfunc="first").reset_index()
            pivot = pivot.rename(columns={"q": "question_code"})
            user_msg = ai.build_overall_prompt(
                tab_labels=list(tables_by_q.keys()),
                pivot_df=pivot,
                q_to_metric=q_to_metric,
                code_to_text=code_to_text,
                code_to_polarity_hint=code_to_polhint,
                code_to_reporting_field=code_to_rfield,
                code_to_meaning_indices=code_to_midx if code_to_midx else None,
                code_to_meaning_labels=code_to_mlbl if code_to_mlbl else None,
            )
            if ai_enabled and user_msg != "__NO_DATA_OVERALL__":
                with st.spinner("Generating overall synthesis…"):
                    json_text, err = ai.call_openai_json(ai.AI_SYSTEM_PROMPT, user_msg)
                overall_narr = ai.extract_narrative(json_text)
                if overall_narr:
                    st.markdown("#### Overall")
                    st.write(overall_narr)

    # 3) AI Data Validation line (advisory only)
    _render_validation_line(tables_by_q)

    # 4) Start a new search (preserve AI toggle; clear AI caches only)
    _render_start_new_search()


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


# ─────────────────────────────────────────────────────────────────────────────
# Clear-only-AI-caches control
# ─────────────────────────────────────────────────────────────────────────────

def _render_start_new_search() -> None:
    c1, c2, _ = st.columns([1.6, 6, 2.4])
    with c1:
        if st.button("🔁 Start a new search", key="menu1_ai_new_search"):
            _clear_ai_caches()
            st.toast("AI selections cleared. Adjust filters above and run again.", icon="✅")
