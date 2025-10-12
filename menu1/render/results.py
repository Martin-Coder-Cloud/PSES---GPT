# menu1/render/results.py
from __future__ import annotations
from typing import Dict, Callable, Any, Tuple, List, Set, Optional
import io
import json
import hashlib
import re

import pandas as pd
import streamlit as st

from ..ai import AI_SYSTEM_PROMPT  # unchanged

# ----------------------------- small helpers -----------------------------

def _hash_key(obj: Any) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            payload = obj.to_csv(index=True, na_rep="")
        else:
            payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _ai_cache_get(key: str):
    cache = st.session_state.get("menu1_ai_cache", {})
    return cache.get(key)

def _ai_cache_put(key: str, value: dict):
    cache = st.session_state.get("menu1_ai_cache", {})
    cache[key] = value
    st.session_state["menu1_ai_cache"] = cache

def _source_link_line(source_title: str, source_url: str) -> None:
    st.markdown(
        f"<div style='margin-top:6px; font-size:0.9rem;'>Source: "
        f"<a href='{source_url}' target='_blank'>{source_title}</a></div>",
        unsafe_allow_html=True
    )

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return None
    m = {c.lower(): c for c in df.columns}
    for n in names:
        c = m.get(n.lower())
        if c is not None:
            return c
    return None

def _has_data(df: pd.DataFrame, col: Optional[str]) -> bool:
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return False
    if not col or col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    return s.notna().any()

def _safe_year_col(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ("Year", "year", "SURVEYR", "survey_year"):
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c)
        if len(s) == 4 and s.isdigit():
            return c
    return None

# ---------------------- metadata (for polarity) ----------------------

@st.cache_data(show_spinner=False)
def _load_survey_questions_meta() -> pd.DataFrame:
    """
    Expect metadata/Survey Questions.xlsx with columns: code, polarity (POS/NEG/NEU).
    Falls back to POS if missing.
    """
    try:
        df = pd.read_excel("metadata/Survey Questions.xlsx")
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if "question" in df.columns and "code" not in df.columns:
            df = df.rename(columns={"question": "code"})
        out = df.copy()
        out["code"] = out["code"].astype(str).str.strip().str.upper()
        if "polarity" not in out.columns:
            out["polarity"] = "POS"
        out["polarity"] = out["polarity"].astype(str).str.upper().str.strip()
        return out[["code", "polarity"]]
    except Exception:
        return pd.DataFrame(columns=["code", "polarity"])

# ---------------------- Summary pivot (polarity-aware) ----------------------

def _pick_metric_for_summary(dfq: pd.DataFrame, qcode: str, meta: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Choose the correct reporting metric column + a generic label
    based on POLARITY with your fallback rules: POS→Positive→Agree→Answer1→Negative;
    NEG→Negative→Agree→Answer1→Positive; NEU→Agree→Answer1→Positive→Negative.
    """
    pol = None
    if not meta.empty:
        row = meta[meta["code"] == str(qcode).strip().upper()]
        if not row.empty:
            pol = str(row.iloc[0]["polarity"] or "").upper().strip()
    pol = pol or "POS"

    col_pos = _find_col(dfq, ["Positive", "POSITIVE"])
    col_neg = _find_col(dfq, ["Negative", "NEGATIVE"])
    col_ag  = _find_col(dfq, ["AGREE"])
    col_a1  = _find_col(dfq, ["Answer1", "Answer 1", "ANSWER1"])

    def choose(seq: List[Tuple[Optional[str], str]]) -> Tuple[Optional[str], str]:
        for c, lbl in seq:
            if _has_data(dfq, c):
                return c, lbl
        return None, ""

    if pol == "NEG":
        return choose([(col_neg, "% negative"), (col_ag, "% agree"), (col_a1, "% selecting Answer1"), (col_pos, "% positive")])
    if pol == "NEU":
        return choose([(col_ag, "% agree"), (col_a1, "% selecting Answer1"), (col_pos, "% positive"), (col_neg, "% negative")])
    return choose([(col_pos, "% positive"), (col_ag, "% agree"), (col_a1, "% selecting Answer1"), (col_neg, "% negative")])

def _build_summary_pivot_from_disp(
    *,
    per_q_disp: Dict[str, pd.DataFrame],
    tab_labels: List[str],
    meta: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    If a per-question display table includes a 'Demographic' column with >1 unique values,
    we emit one row *per (Question, Demographic)* so the Summary tab shows subgroup breakdowns.
    Otherwise we emit one row per Question.
    """
    rows: List[Dict[str, Any]] = []
    labels_used: Dict[str, str] = {}

    for q in tab_labels:
        dfq = per_q_disp.get(q)
        if not isinstance(dfq, pd.DataFrame) or dfq.empty:
            continue

        ycol = _safe_year_col(dfq)
        if not ycol:
            continue

        metric_col, metric_label = _pick_metric_for_summary(dfq, q, meta)
        if not metric_col:
            continue

        # Demographic-aware: if multiple subgroups exist, include them
        has_demo = "Demographic" in dfq.columns and dfq["Demographic"].astype(str).nunique(dropna=True) > 1
        group_cols = [ycol, metric_col] + (["Demographic"] if has_demo else [])
        tmp = dfq[group_cols].copy()

        tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
        tmp = tmp.dropna(subset=[ycol, metric_col])

        if has_demo:
            tmp["Demographic"] = tmp["Demographic"].astype(str)
            for _, r in tmp.iterrows():
                rows.append({
                    "Question": q,
                    "Demographic": r["Demographic"],
                    "Year": int(r[ycol]),
                    "Value": float(r[metric_col]),
                })
        else:
            for _, r in tmp.iterrows():
                rows.append({
                    "Question": q,
                    "Demographic": "",  # empty marker for non-subgroup cases
                    "Year": int(r[ycol]),
                    "Value": float(r[metric_col]),
                })

        labels_used[q] = metric_label or "% value"

    if not rows:
        return pd.DataFrame(), labels_used

    df = pd.DataFrame(rows)

    # If any demographic breakdowns exist, index by (Question, Demographic); else by Question only.
    has_any_demo = df["Demographic"].astype(str).str.len().gt(0).any()
    index_cols = ["Question", "Demographic"] if has_any_demo else ["Question"]

    pivot = df.pivot_table(index=index_cols, columns="Year", values="Value", aggfunc="first").sort_index()
    pivot = pivot.applymap(lambda v: round(v, 1) if pd.notna(v) else v)
    return pivot, labels_used

# ---------------------- Validator helpers (per-question) ----------------------

def _is_year_like(n: int) -> bool:
    return 1900 <= n <= 2100

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip().lower() in ("", "na", "n/a", "none", "nan", "null"):
            return None
        if x == 9999:
            return None
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return int(round(float(v)))
    except Exception:
        return None

def _pick_display_metric(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in df.columns:
        return prefer
    for c in ("value_display", "AGREE", "SCORE100"):
        if c in df.columns:
            return c
    return None

def _allowed_numbers_from_disp(df: pd.DataFrame, metric_col: str) -> Tuple[Set[int], Set[int]]:
    if df is None or df.empty:
        return set(), set()
    work = df.copy()
    years_col = None
    for c in work.columns:
        if c.lower() == "year":
            years_col = c
            break
    if years_col is None:
        # try melt if columns are years
        ycols = [c for c in work.columns if str(c).isdigit() and len(str(c)) == 4]
        if not ycols or metric_col not in work.columns:
            return set(), set()
        id_cols = [c for c in work.columns if c not in ycols]
        work = work.melt(id_vars=id_cols, value_vars=ycols, var_name="Year", value_name=metric_col)
        years_col = "Year"

    work["__Y__"] = pd.to_numeric(work[years_col], errors="coerce")
    work["__V__"] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=["__Y__", "__V__"])
    work["__Y__"] = work["__Y__"].astype(int).astype("Int64")
    work["__V__"] = work["__V__"].round().astype(int).astype("Int64")

    years = set([int(y) for y in work["__Y__"].dropna().unique().tolist() if _is_year_like(int(y))])
    allowed: Set[int] = set([int(v) for v in work["__V__"].dropna().unique().tolist()])

    # within-series deltas by year (and by subgroup if present)
    if "Demographic" in work.columns:
        groups = list(work["Demographic"].astype(str).unique())
        for g in groups:
            sub = work[work["Demographic"].astype(str) == g].sort_values("__Y__")
            vals = sub["__V__"].dropna().astype(int).tolist()
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    allowed.add(abs(vals[j] - vals[i]))
    else:
        sub = work.sort_values("__Y__")
        vals = sub["__V__"].dropna().astype(int).tolist()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                allowed.add(abs(vals[j] - vals[i]))

    # latest-year between-groups gaps
    if years:
        latest = max(years)
        latest_rows = work[work["__Y__"] == latest]
        if "Demographic" in latest_rows.columns:
            latest_rows = latest_rows.sort_values("Demographic")
        for i in range(len(latest_rows)):
            for j in range(i + 1, len(latest_rows)):
                try:
                    vi = int(latest_rows.iloc[i]["__V__"])
                    vj = int(latest_rows.iloc[j]["__V__"])
                    allowed.add(abs(vi - vj))
                except Exception:
                    pass
    return allowed, years

def _extract_datapoint_integers_with_sentences(text: str) -> List[Tuple[int, str]]:
    if not text:
        return []
    sentences = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    found: List[Tuple[int, str]] = []
    patterns = [
        re.compile(r"\((\d+)\)"),
        re.compile(r"\b(\d+)\s*points?\b", re.IGNORECASE),
        re.compile(r"\b(\d+)\s*%\b"),
        re.compile(r"\b(?:is|was|were|at|reached|stood(?:\s+at)?)\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bfrom\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bto\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\bvs\.?\s+(\d+)\b", re.IGNORECASE),
    ]
    for s in sentences:
        nums: Set[int] = set()
        for pat in patterns:
            for m in pat.finditer(s):
                try:
                    n = int(m.group(1))
                    nums.add(n)
                except Exception:
                    continue
        for n in sorted(nums):
            found.append((n, s))
    return found

def _validate_narrative(narrative: str, allowed: Set[int], years: Set[int]) -> dict:
    if not narrative:
        return {"ok": True, "bad_numbers": set(), "problems": []}
    pairs = _extract_datapoint_integers_with_sentences(narrative)
    bad: Set[int] = set()
    problems: List[str] = []
    for n, sentence in pairs:
        if _is_year_like(n):
            continue
        if n not in allowed:
            bad.add(n)
            problems.append(f"{n} — {sentence}")
    return {"ok": len(bad) == 0, "bad_numbers": bad, "problems": problems[:5]}

# ------------------- overall validator against summary_pivot ------------------

def _allowed_numbers_from_summary_pivot(pivot: pd.DataFrame) -> Tuple[Set[int], Set[int]]:
    if pivot is None or pivot.empty:
        return set(), set()
    work = pivot.copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    years = [int(c) for c in work.columns if str(c).isdigit() and len(str(c)) == 4]
    years_set = set(years)
    allowed: Set[int] = set()

    # values
    for v in work.values.flatten():
        if pd.notna(v):
            allowed.add(int(round(float(v))))
    # within-question(-demographic) deltas
    for _, row in work.iterrows():
        vals = [row.get(y) for y in years]
        vals = [int(round(float(v))) for v in vals if pd.notna(v)]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                allowed.add(abs(vals[j] - vals[i]))
    # same-year cross-row gaps + change vs prior
    for y in years:
        col = work[y]
        pairs = []
        for i in range(len(col)):
            for j in range(i + 1, len(col)):
                vi = col.iat[i]; vj = col.iat[j]
                if pd.notna(vi) and pd.notna(vj):
                    gap = abs(int(round(float(vi))) - int(round(float(vj))))
                    allowed.add(gap)
                    pairs.append(((i, j), gap))
        prev = [yy for yy in years if yy < y]
        if not prev:
            continue
        py = max(prev)
        col_prev = work[py]
        for (i, j), gap_latest in pairs:
            vi = col_prev.iat[i] if i < len(col_prev) else None
            vj = col_prev.iat[j] if j < len(col_prev) else None
            if pd.notna(vi) and pd.notna(vj):
                gap_prev = abs(int(round(float(vi))) - int(round(float(vj))))
                allowed.add(abs(gap_latest - gap_prev))
    return allowed, years_set

# ==================== AI narrative computation (unchanged core) ====================

def _compute_ai_narratives(
    *,
    tab_labels: List[str],
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    per_q_metric_label: Dict[str, str],
    code_to_text: Dict[str, str],
    demo_selection: Optional[str],
    pivot: pd.DataFrame,
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
) -> Tuple[Dict[str, str], Optional[str]]:
    per_q_narratives: Dict[str, str] = {}
    overall_narrative: Optional[str] = None

    for q in tab_labels:
        df_disp = per_q_disp[q]
        metric_col = per_q_metric_col[q]
        metric_label = per_q_metric_label[q]
        qtext = code_to_text.get(q, "")
        content, _hint = call_openai_json(
            system=AI_SYSTEM_PROMPT,
            user=build_per_q_prompt(
                question_code=q,
                question_text=qtext,
                df_disp=df_disp,
                metric_col=metric_col,
                metric_label=metric_label,
                category_in_play=("Demographic" in df_disp.columns and df_disp["Demographic"].astype(str).nunique(dropna=True) > 1)
            )
        )
        try:
            j = json.loads(content) if content else {}
            per_q_narratives[q] = (j.get("narrative") or "").strip()
        except Exception:
            per_q_narratives[q] = ""

    if len(tab_labels) > 1:
        content, _hint = call_openai_json(
            system=AI_SYSTEM_PROMPT,
            user=build_overall_prompt(
                tab_labels=tab_labels,
                pivot_df=pivot,
                q_to_metric={q: per_q_metric_label[q] for q in tab_labels},
                code_to_text=code_to_text,
            )
        )
        try:
            j = json.loads(content) if content else {}
            overall_narrative = (j.get("narrative") or "").strip()
        except Exception:
            overall_narrative = None

    return per_q_narratives, overall_narrative

# ----- AI Data Validation (per-question; unchanged) --------------------------

def _render_data_validation_subsection(
    *,
    tab_labels: List[str],
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col: Dict[str, str],
    per_q_narratives: Dict[str, str],
) -> None:
    any_issue = False
    details: List[Tuple[str, str]] = []

    for q in tab_labels:
        try:
            df_disp = per_q_disp.get(q)
            if not isinstance(df_disp, pd.DataFrame) or df_disp.empty:
                details.append(("caption", f"{q}: validation skipped (no table available)."))
                continue

            metric_col = per_q_metric_col.get(q) or _pick_display_metric(df_disp)
            if not metric_col:
                details.append(("caption", f"{q}: validation skipped (no metric column)."))
                continue

            allowed, years = _allowed_numbers_from_disp(df_disp.copy(deep=True), metric_col)
            narrative = per_q_narratives.get(q, "") or ""
            res = _validate_narrative(narrative, allowed, years)

            if not res["ok"]:
                any_issue = True
                nums = ", ".join(str(x) for x in sorted(res["bad_numbers"]))
                details.append(("warning", f"{q}: potential mismatches detected ({nums})."))
            else:
                details.append(("caption", f"{q}: no numeric inconsistencies detected."))
        except Exception as e:
            details.append(("caption", f"{q}: validation skipped ({type(e).__name__})."))

    st.markdown(
        """
        <style>
          #ai_data_validation_title {
            font-size: 1rem;
            line-height: 1.25;
            font-weight: 600;
            margin: 0.25rem 0 0.25rem 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div id='ai_data_validation_title'>AI Data Validation</div>", unsafe_allow_html=True)
    st.markdown("✅ The data points in the summaries have been validated and correspond to the data provided." if not any_issue
                else "❌ Some AI statements may not match the tables. Review the details below.")
    with st.expander("View per-question validation details", expanded=False):
        for level, msg in details:
            st.warning(msg) if level == "warning" else st.caption(msg)

# ------------------------------ main renderer -------------------------------

def tabs_summary_and_per_q(
    *,
    payload: Dict[str, Any],
    ai_on: bool,
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
    source_url: str,
    source_title: str,
) -> None:
    per_q_disp: Dict[str, pd.DataFrame] = payload["per_q_disp"]
    per_q_metric_col: Dict[str, str]   = payload["per_q_metric_col"]   # used as fallback
    per_q_metric_label: Dict[str, str] = payload["per_q_metric_label"] # used as fallback
    pivot_from_payload: pd.DataFrame   = payload["pivot"]
    tab_labels                         = payload["tab_labels"]
    years                              = payload["years"]
    demo_selection                     = payload["demo_selection"]
    sub_selection                      = payload["sub_selection"]
    code_to_text                       = payload["code_to_text"]

    # Build polarity-aware (and demographic-aware) Summary pivot
    meta = _load_survey_questions_meta()
    summary_pivot, labels_used = _build_summary_pivot_from_disp(
        per_q_disp=per_q_disp,
        tab_labels=tab_labels,
        meta=meta
    )

    # cache key (stable; include summary signature so subgroup changes re-compute AI)
    ai_sig = {
        "tab_labels": tab_labels,
        "years": years,
        "demo_selection": demo_selection,
        "sub_selection": sub_selection,
        "metric_labels": {q: per_q_metric_label[q] for q in tab_labels},
        "pivot_sig": _hash_key(pivot_from_payload),
        "summary_sig": _hash_key(summary_pivot),
    }
    ai_key = "menu1_ai_" + _hash_key(ai_sig)

    # ------------------------ UX: header + tabs ------------------------
    st.header("Results")

    tab_titles = ["Summary table"] + tab_labels + ["Technical notes"]
    tabs = st.tabs(tab_titles)

    # Summary tab (shows polarity + demographic aware pivot)
    with tabs[0]:
        st.markdown("### Summary table")
        if tab_labels:
            st.markdown("<div style='font-size:0.9rem; color:#444; margin-bottom:4px;'>"
                        "Questions & metrics included:</div>", unsafe_allow_html=True)
            for q in tab_labels:
                qtext = code_to_text.get(q, "")
                mlabel = labels_used.get(q) or per_q_metric_label.get(q, "% value")
                st.markdown(
                    f"<div style='font-size:0.85rem; color:#555;'>"
                    f"<strong>{q}</strong>: {qtext} "
                    f"<span style='opacity:.85;'>[{mlabel}]</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        if summary_pivot is not None and not summary_pivot.empty:
            st.dataframe(summary_pivot.reset_index(), use_container_width=True)
        else:
            st.info("No data available for the summary under current filters.")
        _source_link_line(source_title, source_url)

    # Per-question tabs
    for idx, qcode in enumerate(tab_labels, start=1):
        with tabs[idx]:
            qtext = code_to_text.get(qcode, "")
            st.subheader(f"{qcode} — {qtext}")
            st.dataframe(per_q_disp[qcode], use_container_width=True)
            _source_link_line(source_title, source_url)

    # Technical notes
    tech_tab_index = len(tab_titles) - 1
    with tabs[tech_tab_index]:
        st.markdown("### Technical notes")
        st.markdown(
            """
1. **Summary results** are mainly shown as “positive answers,” reflecting the affirmative responses. Positive answers are calculated by removing the "Don't know" and "Not applicable" responses from the total responses.  
2. **Weights/adjustment:** Results have been adjusted for non-response to better represent the target population. Therefore, percentages should not be used to determine the number of respondents within a response category.  
3. **Rounding:** Due to rounding, percentages may not add to 100.  
4. **Suppression:** Results were suppressed for questions with low respondent counts (under 10) and for low response category counts.
            """
        )

    # ------------------------- AI section -------------------------
    if ai_on:
        st.markdown("---")
        st.markdown("## AI Summary")

        cached = _ai_cache_get(ai_key)
        if cached:
            per_q_narratives = cached.get("per_q", {}) or {}
            overall_narrative = cached.get("overall")
            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    st.markdown(f"**{q} — {code_to_text.get(q, '')}**")
                    st.write(txt)
            if overall_narrative and len(tab_labels) > 1:
                st.markdown("**Overall**")
                st.write(overall_narrative)
        else:
            # ---------- per-question AI (aligned with polarity + demo presence) ----------
            per_q_narratives: Dict[str, str] = {}
            for q in tab_labels:
                dfq = per_q_disp.get(q)
                qtext = code_to_text.get(q, "")

                # choose the same metric the Summary logic picked
                metric_col_ai, metric_label_ai = _pick_metric_for_summary(dfq, q, meta)
                if not metric_col_ai:
                    # fallback to caller-provided
                    metric_col_ai = per_q_metric_col.get(q)
                    metric_label_ai = per_q_metric_label.get(q, "% value")
                else:
                    # prefer the exact label shown next to question in Summary tab
                    metric_label_ai = (metric_label_ai or labels_used.get(q) or per_q_metric_label.get(q, "% value"))

                # detect subgroup presence from the ACTUAL table
                category_in_play = ("Demographic" in dfq.columns and dfq["Demographic"].astype(str).nunique(dropna=True) > 1)

                with st.spinner(f"AI — analyzing {q}…"):
                    try:
                        content, _hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=build_per_q_prompt(
                                question_code=q,
                                question_text=qtext,
                                df_disp=(dfq.copy(deep=True) if isinstance(dfq, pd.DataFrame) else dfq),
                                metric_col=metric_col_ai,
                                metric_label=metric_label_ai,
                                category_in_play=category_in_play
                            )
                        )
                        j = json.loads(content) if content else {}
                        txt = (j.get("narrative") or "").strip()
                    except Exception as e:
                        txt = ""
                        st.warning(f"AI skipped for {q} due to an internal error ({type(e).__name__}).")
                per_q_narratives[q] = txt
                if txt:
                    st.markdown(f"**{q} — {qtext}**")
                    st.write(txt)

            # ---------- OVERALL: use polarity + demo aware summary_pivot ----------
            overall_narrative = None
            if len(tab_labels) > 1 and summary_pivot is not None and not summary_pivot.empty:
                with st.spinner("AI — synthesizing overall pattern…"):
                    try:
                        content, _hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=build_overall_prompt(
                                tab_labels=tab_labels,
                                pivot_df=summary_pivot.copy(deep=True),  # aligned matrix
                                q_to_metric={q: (labels_used.get(q) or per_q_metric_label[q]) for q in tab_labels},
                                code_to_text=code_to_text,
                            )
                        )
                    except Exception as e:
                        content = None
                        st.warning(f"AI skipped for Overall due to an internal error ({type(e).__name__}).")
                    try:
                        j = json.loads(content) if content else {}
                        overall_narrative = (j.get("narrative") or "").strip()
                    except Exception:
                        overall_narrative = None

                if overall_narrative:
                    st.markdown("**Overall**")
                    st.write(overall_narrative)

            _ai_cache_put(ai_key, {"per_q": per_q_narratives, "overall": overall_narrative})
            st.session_state["menu1_ai_narr_per_q"] = per_q_narratives
            st.session_state["menu1_ai_narr_overall"] = overall_narrative

        # ---------- AI Data Validation ----------
        try:
            # Per-question validator: ensure it validates the SAME column used for AI
            _render_data_validation_subsection(
                tab_labels=tab_labels,
                per_q_disp=per_q_disp,
                per_q_metric_col={
                    q: (_pick_metric_for_summary(per_q_disp[q], q, meta)[0] or per_q_metric_col.get(q))
                    for q in tab_labels
                },
                per_q_narratives=st.session_state.get("menu1_ai_narr_per_q", {}) or {},
            )
            # Overall narrative validation against summary_pivot (demo-aware)
            if len(tab_labels) > 1 and summary_pivot is not None and not summary_pivot.empty:
                overall_txt = st.session_state.get("menu1_ai_narr_overall", "")
                if overall_txt:
                    allowed, years_set = _allowed_numbers_from_summary_pivot(summary_pivot.copy(deep=True))
                    res = _validate_narrative(overall_txt, allowed, years_set)
                    if not res["ok"]:
                        st.warning("Overall: potential mismatches detected in the AI summary.")
                        with st.expander("View overall validation details"):
                            for prob in res["problems"]:
                                st.caption(prob)
                    else:
                        st.caption("Overall: no numeric inconsistencies detected.")
        except Exception:
            st.caption("AI Data Validation is unavailable for this run.")

    # ----------------------- Footer: Export + Start new -----------------------
    st.markdown("---")
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
    col_dl, col_new = st.columns([1, 1])

    with col_dl:
        export_per_q = st.session_state.get("menu1_ai_narr_per_q")
        export_overall = st.session_state.get("menu1_ai_narr_overall")

        if ai_on and export_per_q is None:
            cached = _ai_cache_get(ai_key)
            if cached:
                export_per_q = cached.get("per_q", {})
                export_overall = cached.get("overall")
            else:
                try:
                    per_q_disp_ai2: Dict[str, pd.DataFrame] = {}
                    for q in tab_labels:
                        dfq = per_q_disp.get(q)
                        per_q_disp_ai2[q] = (dfq.copy(deep=True) if isinstance(dfq, pd.DataFrame) else dfq)
                    export_per_q, export_overall = _compute_ai_narratives(
                        tab_labels=tab_labels,
                        per_q_disp=per_q_disp_ai2,
                        per_q_metric_col={
                            q: (_pick_metric_for_summary(per_q_disp[q], q, meta)[0] or per_q_metric_col.get(q))
                            for q in tab_labels
                        },
                        per_q_metric_label={q: (labels_used.get(q) or per_q_metric_label.get(q, "% value")) for q in tab_labels},
                        code_to_text=code_to_text,
                        demo_selection=demo_selection,
                        pivot=summary_pivot.copy(deep=True) if (summary_pivot is not None and not summary_pivot.empty) else pivot_from_payload.copy(deep=True),
                        build_overall_prompt=build_overall_prompt,
                        build_per_q_prompt=build_per_q_prompt,
                        call_openai_json=call_openai_json,
                    )
                    _ai_cache_put(ai_key, {"per_q": export_per_q, "overall": export_overall})
                except Exception:
                    export_per_q, export_overall = {}, None

        _render_excel_download(
            summary_pivot=summary_pivot,  # corrected summary pivot (demo-aware)
            per_q_disp=per_q_disp,
            tab_labels=tab_labels,
            per_q_narratives=(export_per_q or {}),
            overall_narrative=(export_overall if (export_overall and len(tab_labels) > 1) else None),
        )

    with col_new:
        if st.button("Start a new search", key="menu1_new_search"):
            _prev_ai_toggle = st.session_state.get("menu1_ai_toggle")
            try:
                from .. import state
                state.reset_menu1_state()
            except Exception:
                for k in [
                    "menu1_selected_codes", "menu1_multi_questions",
                    "menu1_show_diag",
                    "select_all_years", "demo_main", "last_query_info",
                ]:
                    st.session_state.pop(k, None)
            for k in [
                "menu1_hits", "menu1_hit_codes_selected",
                "menu1_search_done", "menu1_last_search_query",
                "menu1_kw_query",
            ]:
                st.session_state.pop(k, None)
            for k in list(st.session_state.keys()):
                if k.startswith("kwhit_") or k.startswith("sel_"):
                    st.session_state.pop(k, None)
            st.session_state.pop("menu1_global_hits_selected", None)
            st.session_state.pop("menu1_ai_cache", None)
            st.session_state.pop("menu1_ai_narr_per_q", None)
            st.session_state.pop("menu1_ai_narr_overall", None)
            if _prev_ai_toggle is not None:
                st.session_state["menu1_ai_toggle"] = _prev_ai_toggle
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

# ------------------- Excel export (uses corrected Summary) --------------------

def _render_excel_download(
    *,
    summary_pivot: pd.DataFrame,
    per_q_disp: Dict[str, pd.DataFrame],
    tab_labels: List[str],
    per_q_narratives: Dict[str, str],
    overall_narrative: Optional[str],
) -> None:
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            if summary_pivot is not None and not summary_pivot.empty:
                summary_pivot.reset_index().to_excel(writer, sheet_name="Summary_Table", index=False)
            else:
                pd.DataFrame({"Message": ["No data available for the summary under current filters."]}).to_excel(
                    writer, sheet_name="Summary_Table", index=False
                )
            for q, df_disp in per_q_disp.items():
                safe = q[:28]
                df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)
            rows = []
            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    rows.append({"Section": q, "Narrative": txt})
            if overall_narrative and len(tab_labels) > 1:
                rows.append({"Section": "Overall", "Narrative": overall_narrative})
            ai_df = pd.DataFrame(rows, columns=["Section", "Narrative"])
            ai_df.to_excel(writer, sheet_name="AI Summary", index=False)
        data = buf.getvalue()

    st.download_button(
        label="Download data and AI summaries",
        data=data,
        file_name="PSES_results_with_AI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="menu1_excel_download",
    )
