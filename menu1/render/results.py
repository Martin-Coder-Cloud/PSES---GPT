# menu1/render/results.py
# ---------------------------------------------------------------------
# Menu 1 – Results: AI Summary + AI Data Validation (metadata-driven).
#
# Back-compat shims:
# - Exports tabs_summary_and_per_q(...) to match legacy callers.
# - Accepts either:
#     tabs_summary_and_per_q(
#         tables_by_q=..., qtext_by_q=..., category_in_play=..., ai_enabled=..., selection_key=...
#       )
#   OR:
#     tabs_summary_and_per_q(payload={ ... })  # where payload bundles those keys.
#
# Implements:
# - No math in reporting (only narrates pre-aggregated values).
# - POLARITY selects reporting field with fallbacks:
#       POS -> POSITIVE -> AGREE -> ANSWER1
#       NEG -> NEGATIVE -> AGREE -> ANSWER1
#       NEU -> AGREE    -> ANSWER1
# - NEUTRAL exists in metadata for completeness (not used for reporting).
# - Meaning indices from Survey Questions.xlsx -> labels from Survey Scales.xlsx.
# - Per-question AI summaries (spinners) + Overall synthesis (when ≥2 questions).
# - All-years trend classification via ai.py addendum; gap math only.
# - D57_2 exception: list ALL options for latest year (no aggregation); excluded from Overall.
# - Validation line ✅/❌ and “Start a new search” (clears AI caches; keeps toggle).
# - Scales code column can be 'code', 'question', or 'questions'.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import math
import json
import hashlib
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
    """
    Survey Questions.xlsx (case-insensitive headers). Expected/handled columns:
      - code (or 'question')
      - text (or 'english')
      - polarity in {POS, NEG, NEU}
      - positive / negative / agree / neutral (strings like "1,2")
    """
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

    for col in ["positive", "negative", "agree", "neutral"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)

    return df[["code", "text", "polarity", "positive", "negative", "agree", "neutral"]]


@st.cache_data(show_spinner=False)
def _load_scales() -> pd.DataFrame:
    """
    Survey Scales.xlsx (wide or long; accepts code column named 'code', 'question', or 'questions').
    """
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

    pol = (qmeta.get("polarity") or "POS").upper().strip()

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
    chosen_row = None
    if gcol and not latest_rows.empty:
        mask_all = latest_rows[gcol].astype(str).str.lower().str.contains("all respondents")
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
# Public entry points
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

    narratives: Dict[str, str] = {}
    overall_rows: List[Dict[str, Any]] = {}
    overall_rows = []
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
                    with st.spinner(f"Generating summary for {qcode}…"):
                        json_text, err = _cached_ai_summary(selection_key, qcode, special_payload, ai.AI_SYSTEM_PROMPT)
                    narrative = ai.extract_narrative(json_text) or ""
            if narrative:
                st.write(f"**{qcode} — {qtext}**  \n{narrative}")
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

        q_to_metric[qcode] = metric_label or "% of respondents"
        code_to_polhint[qcode] = (qmeta.get("polarity") or "POS").upper()
        code_to_rfield[qcode] = reporting_field
        if meaning_idx:
            code_to_midx[qcode] = meaning_idx
        if meaning_lbls:
            code_to_mlbl[qcode] = meaning_lbls

        # Build per-q payload (include ALL years; AI handles all-years trend)
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

        narrative = ""
        if ai_enabled:
            with st.spinner(f"Generating summary for {qcode}…"):
                json_text, err = _cached_ai_summary(selection_key, qcode, payload_str, ai.AI_SYSTEM_PROMPT)
            narrative = ai.extract_narrative(json_text) or ""
        if narrative:
            st.write(f"**{qcode} — {qtext}**  \n{narrative}")

        # Collect rows for Overall
        ycol = _detect_year_col(df)
        if ycol is not None:
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

    # Validation line + start-new-search
    _render_validation_line(tables_by_q)
    _render_start_new_search()


# Back-compat: existing callers expect this symbol in menu1.render.results
def tabs_summary_and_per_q(*args, **kwargs) -> None:
    """
    Backward-compatible entry point expected by some app code.

    Supports two calling styles:
      1) tabs_summary_and_per_q(tables_by_q, qtext_by_q, category_in_play, ai_enabled, selection_key)
      2) tabs_summary_and_per_q(payload={ ... }) where payload keys include:
         - tables_by_q: Dict[str, DataFrame]
         - qtext_by_q: Dict[str, str]
         - category_in_play: bool
         - ai_enabled: bool
         - selection_key: str  (optional; we will derive if missing)
    """
    # Style 2: payload kwarg (preferred in legacy code)
    if "payload" in kwargs and isinstance(kwargs["payload"], dict):
        p = kwargs["payload"]
        tables_by_q = p.get("tables_by_q") or kwargs.get("tables_by_q") or (args[0] if args else {})
        qtext_by_q = p.get("qtext_by_q") or kwargs.get("qtext_by_q") or {}
        category_in_play = bool(p.get("category_in_play", kwargs.get("category_in_play", False)))
        ai_enabled = bool(p.get("ai_enabled", kwargs.get("ai_enabled", True)))
        selection_key = p.get("selection_key") or kwargs.get("selection_key")

        # Derive a stable selection_key if not provided
        if not selection_key:
            try:
                # Use question codes + any visible year values to create a deterministic hash
                qs = sorted(list(tables_by_q.keys()))
                years_seen = set()
                for q, df in (tables_by_q or {}).items():
                    ycol = _detect_year_col(df) if isinstance(df, pd.DataFrame) else None
                    if ycol and isinstance(df, pd.DataFrame):
                        years_seen.update(pd.to_numeric(df[ycol], errors="coerce").dropna().astype(int).tolist())
                sig = {"qs": qs, "years": sorted(list(years_seen)), "demo": "ON" if category_in_play else "ALL"}
                selection_key = hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()
            except Exception:
                selection_key = "menu1_legacy_selection"

        return render_ai_summary_section(
            tables_by_q=tables_by_q or {},
            qtext_by_q=qtext_by_q or {},
            category_in_play=category_in_play,
            ai_enabled=ai_enabled,
            selection_key=selection_key,
        )

    # Style 1: positional or explicit kwargs
    if args and len(args) >= 5:
        tables_by_q, qtext_by_q, category_in_play, ai_enabled, selection_key = args[:5]
        return render_ai_summary_section(tables_by_q, qtext_by_q, category_in_play, ai_enabled, selection_key)

    # Mixed kwargs
    return render_ai_summary_section(
        tables_by_q=kwargs.get("tables_by_q", args[0] if args else {}),
        qtext_by_q=kwargs.get("qtext_by_q", {}),
        category_in_play=bool(kwargs.get("category_in_play", False)),
        ai_enabled=bool(kwargs.get("ai_enabled", True)),
        selection_key=kwargs.get("selection_key", "menu1_default_selection"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation line UI + Clear caches control
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

def _render_start_new_search() -> None:
    c1, c2, _ = st.columns([1.6, 6, 2.4])
    with c1:
        if st.button("🔁 Start a new search", key="menu1_ai_new_search"):
            _clear_ai_caches()
            st.toast("AI selections cleared. Adjust filters above and run again.", icon="✅")
