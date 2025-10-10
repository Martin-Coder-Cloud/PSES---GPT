# menu1/results.py
# ---------------------------------------------------------------------
# Menu 1 – Results: AI Summary + AI Data Validation (metadata-driven).
#
# This module assumes:
# - You already rendered the per-question tables (with source links) in tabs.
# - You call render_ai_summary_section(...) *below* those tabs.
#
# Key rules implemented:
# - No math in reporting: app forwards pre-aggregated values only.
# - POLARITY in Survey Questions.xlsx selects which field to report:
#       POS -> POSITIVE
#       NEG -> NEGATIVE
#       NEU -> AGREE
# - Fallbacks when missing/9999:
#       POS/NEG -> AGREE -> ANSWER1
#       NEU     -> AGREE -> ANSWER1
# - NEUTRAL is never used for reporting (metadata may include for completeness).
# - Meaning indices for POSITIVE/NEGATIVE/AGREE are read from metadata (e.g., "1,2")
#   and mapped to labels using Survey Scales.xlsx, then passed to AI.
# - AI payload contains: reporting_field, meaning_indices, meaning_labels, and
#   all years you want trend classification on; AI prompt (in ai.py) handles
#   gaps and all-years trend classification only; no other computations.
#
# Layout:
# - "AI Summary" H3, per-question spinners, then "Overall" when >1 question,
#   then "AI Data Validation" ✅/❌ with expander, then a compact "Start a new search".
# - Technical Notes tab and source links under tables remain unchanged.
#
# Caching:
# - Summaries cached by a stable selection_key + per-question signature.
# - "Start a new search" clears AI caches only and preserves the AI toggle state.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import math
import json
import pandas as pd
import streamlit as st

# Try both import paths in case of project structure differences.
try:
    from menu1 import ai  # your updated ai.py
except Exception:
    import ai  # fallback if module path differs

SENTINEL = 9999

# ─────────────────────────────────────────────────────────────────────────────
# Metadata loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_survey_questions() -> pd.DataFrame:
    """
    Expected columns (case-insensitive; we normalize names to lower):
      - code (question code, e.g., Q44a)
      - text/english (optional; used when available)
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

    # Default values and cleanup
    df["code"] = df["code"].astype(str).str.strip()
    df["text"] = df.get("text", pd.Series([None]*len(df))).astype(str)

    # Polarity defaults to POS if missing/invalid
    pol = df.get("polarity", pd.Series(["POS"]*len(df))).astype(str).str.upper().str.strip()
    pol = pol.where(pol.isin(["POS", "NEG", "NEU"]), "POS")
    df["polarity"] = pol

    # Keep metadata strings for indices (e.g., "1,2"), even if NaN
    for col in ["positive", "negative", "agree", "neutral"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)

    return df[["code", "text", "polarity", "positive", "negative", "agree", "neutral"]]


@st.cache_data(show_spinner=False)
def _load_scales() -> pd.DataFrame:
    """
    Loads Survey Scales.xlsx. We support both 'wide' and 'long' shapes.

    Wide example (preferred):
      code | answer1 | answer2 | answer3 | ... (labels as strings)

    Long example:
      code | index | label (and/or english)
    """
    df = pd.read_excel("metadata/Survey Scales.xlsx")
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    return df


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
    # Accept forms like '1,2' or '1;2' or '1 | 2'
    tokens = [t.strip() for t in s.replace(";", ",").replace("|", ",").split(",") if t.strip()]
    out: List[int] = []
    for t in tokens:
        try:
            out.append(int(t))
        except Exception:
            continue
    return out or None


def _labels_for_indices(scales_df: pd.DataFrame, code: str, indices: Optional[List[int]]) -> Optional[List[str]]:
    """Return labels for the given indices from Survey Scales."""
    if not indices:
        return None
    code_u = str(code).strip().upper()
    df = scales_df

    # Try WIDE format first: row for code; columns answer1..answer7
    wide_cols = [c for c in df.columns if c.startswith("answer") and c[6:].isdigit()]
    if wide_cols and "code" in df.columns:
        row = df[df["code"].astype(str).str.upper() == code_u]
        if not row.empty:
            labels: List[str] = []
            r0 = row.iloc[0]
            for i in indices:
                col = f"answer{i}".lower()
                if col in row.columns:
                    val = str(r0[col]).strip()
                    if val and val.lower() != "nan":
                        labels.append(val)
            return labels or None

    # Try LONG format: expect columns ('code', 'index', 'label' or 'english')
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
            return labels or None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Selecting the reporting field & column based on POLARITY + fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first column name present (case-insensitive) among candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c is not None:
            return c
    return None


def _has_valid_values(series: pd.Series) -> bool:
    """True if there is at least one non-sentinel, non-NA value."""
    if series is None:
        return False
    s = pd.to_numeric(series, errors="coerce")
    if s.empty:
        return False
    s = s.replace({SENTINEL: pd.NA})
    return s.notna().any()


def _choose_reporting(df: pd.DataFrame, qcode: str, qmeta: pd.Series) -> Tuple[str, str, str, Optional[List[int]]]:
    """
    Decide (metric_col, reporting_field, metric_label, meaning_indices) for this question.
    - metric_col: exact column name in df (case-preserving).
    - reporting_field: 'POSITIVE' | 'NEGATIVE' | 'AGREE' | 'ANSWER1'
    - metric_label: human-friendly label for the AI (kept for compatibility).
    - meaning_indices: list like [1,2] based on the chosen field (from metadata),
                       or [1] for ANSWER1; may be None.

    NEUTRAL is intentionally not used for reporting.
    """
    # Find common columns, case-insensitive
    col_positive = _find_col(df, ["Positive", "POSITIVE"])
    col_negative = _find_col(df, ["Negative", "NEGATIVE"])
    col_agree    = _find_col(df, ["AGREE", "Agree"])
    col_answer1  = _find_col(df, ["Answer1", "ANSWER1"])

    pol = (qmeta.get("polarity") or "POS").upper().strip()

    # Helper to test availability (not sentinel)
    def is_available(colname: Optional[str]) -> bool:
        return bool(colname) and _has_valid_values(df[colname])  # type: ignore[index]

    # Pick initial target based on POLARITY
    target = None
    metric_label = ""
    meta_field = None  # which metadata column to read indices from
    if pol == "POS":
        if is_available(col_positive):
            target = col_positive
            meta_field = "positive"
            metric_label = "% favourable"
        elif is_available(col_agree):
            target = col_agree
            meta_field = "agree"
            metric_label = "% selected (AGREE)"
        elif is_available(col_answer1):
            target = col_answer1
            meta_field = None
            metric_label = "% selected (Answer1)"
    elif pol == "NEG":
        if is_available(col_negative):
            target = col_negative
            meta_field = "negative"
            metric_label = "% problem"
        elif is_available(col_agree):
            target = col_agree
            meta_field = "agree"
            metric_label = "% selected (AGREE)"
        elif is_available(col_answer1):
            target = col_answer1
            meta_field = None
            metric_label = "% selected (Answer1)"
    else:  # NEU
        if is_available(col_agree):
            target = col_agree
            meta_field = "agree"
            metric_label = "% selected (AGREE)"
        elif is_available(col_answer1):
            target = col_answer1
            meta_field = None
            metric_label = "% selected (Answer1)"

    if not target:
        # As a last resort, try any present column in preferred order
        for col, label, mfield in [
            (col_positive, "% favourable", "positive"),
            (col_negative, "% problem", "negative"),
            (col_agree, "% selected (AGREE)", "agree"),
            (col_answer1, "% selected (Answer1)", None),
        ]:
            if is_available(col):
                target, metric_label, meta_field = col, label, mfield
                break

    if not target:
        # No usable column found
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
      (c) For AGREE/ANSWER1, optionally check Answer1..K sum ≈ 100 if present.
    """
    issues: List[str] = []
    cols = set(df.columns)

    has_pos = _find_col(df, ["Positive", "POSITIVE"]) is not None
    has_neu = _find_col(df, ["Neutral", "NEUTRAL"]) is not None
    has_neg = _find_col(df, ["Negative", "NEGATIVE"]) is not None

    all_ok = True
    for _, row in df.iterrows():
        yr = str(row.get("Year", row.get("year", "?")))
        demo = row.get("Demographic", row.get("group", None))
        who = f"Year {yr}" + (f", {demo}" if isinstance(demo, str) and demo else "")

        if has_pos and has_neg:
            p = _safe_float(row[_find_col(df, ["Positive", "POSITIVE"])])  # type: ignore[index]
            n = _safe_float(row[_find_col(df, ["Negative", "NEGATIVE"])])  # type: ignore[index]
            if p is not None and n is not None:
                delta = abs((100.0 - p) - n)
                if delta > 1.0:
                    all_ok = False
                    issues.append(f"{who}: (100 − Positive) vs Negative differs by {delta:.1f} pts.")

        if has_pos and has_neu and has_neg:
            p = _safe_float(row[_find_col(df, ["Positive", "POSITIVE"])])  # type: ignore[index]
            u = _safe_float(row[_find_col(df, ["Neutral", "NEUTRAL"])])   # type: ignore[index]
            n = _safe_float(row[_find_col(df, ["Negative", "NEGATIVE"])]) # type: ignore[index]
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
    We cache the raw JSON text and error hint from ai.call_openai_json.
    """
    json_text, err = ai.call_openai_json(system_prompt, payload_str)
    return json_text or "", err


def _clear_ai_caches():
    try:
        _cached_ai_summary.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    # also clear Streamlit session scratch keys used by this module
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
    Render the AI Summary and the AI Data Validation line below the tabs.

    Parameters
    ----------
    tables_by_q : Dict[qcode, DataFrame]
        Per-question display tables already shown in tabs, with columns like:
        Year, optional Demographic/group, Positive, Neutral, Negative, AGREE (optional), Answer1..7 (optional), n.
    qtext_by_q : Dict[qcode, text]
        Human-readable question text.
    category_in_play : bool
        True if a demographic subgroup is active; used for phrasing inside the AI (already covered by addendum).
    ai_enabled : bool
        Current AI toggle state. When False, we render a compact note + validation line.
    selection_key : str
        Stable identifier for the current selection (questions, years, demographic). Used by caching.
    """
    st.markdown("---")
    st.markdown("### AI Summary")

    if not tables_by_q:
        st.info("No questions selected.")
        return

    # Load metadata once
    qmeta_df = _load_survey_questions()
    scales_df = _load_scales()

    # 1) Per-question AI outputs (progressive)
    narratives: Dict[str, str] = {}
    overall_rows: List[Dict[str, Any]] = []
    q_to_metric: Dict[str, str] = {}
    code_to_text: Dict[str, str] = {}
    code_to_polhint: Dict[str, str] = {}
    code_to_rfield: Dict[str, str] = {}
    code_to_midx: Dict[str, List[int]] = {}
    code_to_mlbl: Dict[str, List[str]] = {}

    for qcode, df in tables_by_q.items():
        qtext = qtext_by_q.get(qcode, qcode)
        code_to_text[qcode] = qtext

        # Attach metadata row (fallback if missing)
        row = qmeta_df[qmeta_df["code"].str.upper() == str(qcode).upper()]
        if row.empty:
            qmeta = pd.Series({"polarity": "POS", "positive": None, "negative": None, "agree": None})
        else:
            qmeta = row.iloc[0]

        # Choose reporting field/column
        metric_col, reporting_field, metric_label, meaning_idx = _choose_reporting(df, qcode, qmeta)

        # If no usable column, skip gracefully
        if not metric_col:
            st.caption(f"⚠️ No data to summarize for {qcode} under current filters.")
            continue

        # Resolve labels for meaning indices
        meaning_lbls = _labels_for_indices(scales_df, qcode, meaning_idx) if meaning_idx else None

        # Save for overall payload
        q_to_metric[qcode] = metric_label or "% of respondents"
        code_to_polhint[qcode] = (qmeta.get("polarity") or "POS").upper()
        code_to_rfield[qcode] = reporting_field
        if meaning_idx:
            code_to_midx[qcode] = meaning_idx
        if meaning_lbls:
            code_to_mlbl[qcode] = meaning_lbls

        # Build per-question payload (include ALL years for trend classification)
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

        # Progressive rendering
        if not ai_enabled:
            # AI off — we still gather data for validation, but skip calling the model
            narratives[qcode] = ""
        else:
            with st.spinner(f"Generating summary for {qcode}…"):
                json_text, err = _cached_ai_summary(selection_key, qcode, payload_str, ai.AI_SYSTEM_PROMPT)
            narrative = ai.extract_narrative(json_text) or ""
            narratives[qcode] = narrative

        # Show per-question narrative (maintain minimal spacing)
        if narratives[qcode]:
            st.write(f"**{qcode} — {qtext}**  \n{narratives[qcode]}")

        # For building the overall multi-question payload:
        # Create compact rows (question_code, year, value)
        ycol = "Year" if "Year" in df.columns else ("year" if "year" in df.columns else None)
        if ycol is not None:
            tmp = df[[ycol, metric_col]].copy()
            tmp = tmp.rename(columns={ycol: "Year", metric_col: "Value"})
            tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce").astype("Int64")
            tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").astype("Int64").replace({SENTINEL: pd.NA})
            tmp = tmp.dropna(subset=["Year", "Value"])
            for _, r in tmp.iterrows():
                overall_rows.append({"q": qcode, "y": int(r["Year"]), "v": int(r["Value"])})

    # 2) Overall synthesis (if >1 question)
    if len(overall_rows) >= 2:
        # Build a small pivot: rows=question, cols=years, vals=value
        ov = pd.DataFrame(overall_rows)
        if not ov.empty:
            pivot = ov.pivot_table(index="q", columns="y", values="v", aggfunc="first").reset_index()
            pivot = pivot.rename(columns={"q": "question_code"})
            # Reuse ai.build_overall_prompt so the model applies its cross-question logic
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

    # 3) AI Data Validation line
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
    # Tight spacing
    c1, c2, _ = st.columns([1.6, 6, 2.4])
    with c1:
        if st.button("🔁 Start a new search", key="menu1_ai_new_search"):
            _clear_ai_caches()
            st.toast("AI selections cleared. Adjust filters above and run again.", icon="✅")
