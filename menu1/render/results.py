# menu1/render/results.py
from __future__ import annotations
from typing import Dict, Callable, Any, Tuple, List, Set, Optional
import io
import json
import hashlib
import re
import os

import pandas as pd
import streamlit as st

from ..ai import AI_SYSTEM_PROMPT  # unchanged

# ----------------------------- small helpers (added) -----------------------------

def _compress_labels_for_footnote(labels: List[str]) -> Optional[str]:
    """
    Return a compressed label string in parentheses, e.g.:
    ["To a small extent","To a moderate extent","To a large extent","To a very large extent"]
      -> "(To a small/moderate/large extent/very large extent)".
    Falls back to full join if safe compression not possible.
    """
    if not labels:
        return None
    full = "(" + "/".join(labels) + ")"
    try:
        from os.path import commonprefix
        prefix = commonprefix(labels)
        rev = [s[::-1] for s in labels]
        suffix = commonprefix(rev)[::-1]
        parts: List[str] = []
        for i, lab in enumerate(labels):
            core = lab
            if prefix and core.startswith(prefix):
                core = core[len(prefix):]
            if suffix and core.endswith(suffix):
                core = core[: -len(suffix)]
            if not core.strip():
                core = lab
            if i == 0 and lab.startswith(prefix):
                core = prefix + core
            if suffix and lab.endswith(suffix) and not core.endswith(suffix):
                core = core + suffix
            parts.append(core)
        compressed = "(" + "/".join(parts) + ")"
        if all(p.strip() for p in parts) and len(parts) == len(labels):
            return compressed
        return full
    except Exception:
        return full

import re as _re
_percent_pat = _re.compile(r"(\d{1,3})\s*%")

def _insert_first_percent_asterisk(text: str) -> str:
    """
    Insert a single asterisk immediately after the first percentage occurrence (e.g., '54%' or '54 %').
    If an asterisk already follows the first %, leave as-is.
    """
    if not text:
        return text
    m = _percent_pat.search(text)
    if not m:
        return text
    i = m.end()
    if i < len(text) and text[i] == "*":
        return text
    return text[:i] + "*" + text[i:]

# ------------------------------------------------------------------------------------
# Everything below is your existing implementation (unchanged except noted injections).
# ------------------------------------------------------------------------------------

def _md(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)

def _hash_key(obj: Any) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            s = obj.to_csv(index=False)
        else:
            s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _ai_cache_get(key: str) -> Optional[Dict[str, Any]]:
    return (st.session_state.get("menu1_ai_cache") or {}).get(key)

def _ai_cache_put(key: str, value: Dict[str, Any]) -> None:
    cache = st.session_state.get("menu1_ai_cache") or {}
    cache[key] = value
    st.session_state["menu1_ai_cache"] = cache

def _source_link_line(source_title: str, source_url: str) -> None:
    st.markdown(
        f"<div style='margin-top:6px; font-size:0.9rem;'>Source: "
        f"<a href='{source_url}' target='_blank'>{source_title}</a></div>",
        unsafe_allow_html=True
    )

def _first_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def _load_questions_metadata() -> pd.DataFrame:
    """
    Tries both metadata/ and project root for:
      - Survey Questions.xlsx
    Required: code, polarity (POS/NEG/NEU)
    Optional: positive/negative/agree (indices like "1,2")
              scale keys: scale / scale_id / scale_name
    """
    try:
        path = _first_existing_path([
            "metadata/Survey Questions.xlsx",
            "./Survey Questions.xlsx",
            "Survey Questions.xlsx",
        ])
        if not path:
            return pd.DataFrame(columns=["code", "polarity", "positive", "negative", "agree", "scale", "scale_id", "scale_name"])
        df = pd.read_excel(path)
        # normalize columns
        cols = {str(c).strip().lower(): c for c in df.columns}
        ren = {}
        for want in ["code", "polarity", "positive", "negative", "agree", "scale", "scale_id", "scale_name"]:
            c = cols.get(want)
            if c and c != want:
                ren[c] = want
        if ren:
            df = df.rename(columns=ren)
        # lower-case polarity
        if "polarity" in df.columns:
            df["polarity"] = df["polarity"].astype(str).str.upper()
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "polarity", "positive", "negative", "agree", "scale", "scale_id", "scale_name"])

def _load_scales_metadata() -> pd.DataFrame:
    """
    Tries both metadata/ and project root for:
      - Survey Scales.xlsx
    Required: code (question or scale key), value (numeric index), label (display text)
    """
    try:
        path = _first_existing_path([
            "metadata/Survey Scales.xlsx",
            "./Survey Scales.xlsx",
            "Survey Scales.xlsx",
        ])
        if not path:
            return pd.DataFrame(columns=["code", "value", "label"])
        df = pd.read_excel(path)
        cols = {str(c).strip().lower(): c for c in df.columns}
        ren = {}
        for want in ["code", "value", "label"]:
            c = cols.get(want)
            if c and c != want:
                ren[c] = want
        if ren:
            df = df.rename(columns=ren)
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "value", "label"])

def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        lc = str(c).strip()
        mapping[c] = lc
    return df.rename(columns=mapping)

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return None
    m = {str(c).lower(): c for c in df.columns}
    for n in names:
        c = m.get(str(n).lower())
        if c is not None:
            return c
    return None

def _is_d57_exception(qcode: str) -> bool:
    q = (qcode or "").strip().lower()
    return q in {"d57_a", "d57_b", "d57a", "d57b"}

def _norm_numeric(val: Any) -> Optional[float]:
    try:
        f = float(val)
        if f == 9999:
            return None
        return f
    except Exception:
        return None

def _find_year_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit() and len(s) in (2, 4):
            cols.append(c)
    return cols

def _infer_reporting_field(metric_col: Optional[str]) -> Optional[str]:
    if not metric_col:
        return None
    m = str(metric_col).strip().lower()
    if m in {"positive", "pos", "top2", "toptwo", "positive_answers"}:
        return "POSITIVE"
    if m in {"negative", "neg", "bottom2", "bottomtwo", "negative_answers"}:
        return "NEGATIVE"
    if m in {"agree", "agreed", "agreement"}:
        return "AGREE"
    if m.startswith("answer1"):
        return "ANSWER1"
    return None

def _parse_indices_list(s: Any) -> List[int]:
    try:
        if s is None:
            return []
        parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
        return out
    except Exception:
        return []

def _meaning_labels_for_question(
    *,
    qcode: str,
    question_text: str,
    reporting_field: Optional[str],
    metric_label: str,
    meta_q: pd.DataFrame,
    meta_scales: pd.DataFrame,
) -> List[str]:
    """
    Resolve meaning labels based on Survey Questions + Survey Scales rules
    """
    try:
        qrow = meta_q[meta_q["code"].astype(str).str.strip().str.lower() == str(qcode).strip().lower()]
        if qrow.empty:
            return []
        pol = (qrow["polarity"].iloc[0] or "").upper()
        # Which list to use based on polarity or reporting_field
        if reporting_field in {"POSITIVE", "NEGATIVE", "AGREE"}:
            field = reporting_field
        else:
            if pol == "POS":
                field = "POSITIVE"
            elif pol == "NEG":
                field = "NEGATIVE"
            else:
                field = "AGREE"
        idx_list = []
        if field == "POSITIVE":
            idx_list = _parse_indices_list(qrow["positive"].iloc[0] if "positive" in qrow.columns else None)
        elif field == "NEGATIVE":
            idx_list = _parse_indices_list(qrow["negative"].iloc[0] if "negative" in qrow.columns else None)
        elif field == "AGREE":
            idx_list = _parse_indices_list(qrow["agree"].iloc[0] if "agree" in qrow.columns else None)

        # If empty, fallbacks based on polarity
        if not idx_list:
            if pol == "POS":
                # Positive → Agree → Answer1 → Negative
                idx_try = [
                    _parse_indices_list(qrow["positive"].iloc[0] if "positive" in qrow.columns else None),
                    _parse_indices_list(qrow["agree"].iloc[0] if "agree" in qrow.columns else None),
                    [1],
                    _parse_indices_list(qrow["negative"].iloc[0] if "negative" in qrow.columns else None)
                ]
            elif pol == "NEG":
                # Negative → Agree → Answer1 → Positive
                idx_try = [
                    _parse_indices_list(qrow["negative"].iloc[0] if "negative" in qrow.columns else None),
                    _parse_indices_list(qrow["agree"].iloc[0] if "agree" in qrow.columns else None),
                    [1],
                    _parse_indices_list(qrow["positive"].iloc[0] if "positive" in qrow.columns else None)
                ]
            else:
                # NEU → Agree → Answer1 → Positive → Negative
                idx_try = [
                    _parse_indices_list(qrow["agree"].iloc[0] if "agree" in qrow.columns else None),
                    [1],
                    _parse_indices_list(qrow["positive"].iloc[0] if "positive" in qrow.columns else None),
                    _parse_indices_list(qrow["negative"].iloc[0] if "negative" in qrow.columns else None)
                ]
            for li in idx_try:
                if li:
                    idx_list = li
                    break

        # Now map indices to labels using Survey Scales
        if not idx_list:
            return []
        # Determine scale code to use: prefer scale/scale_id on question row, else question code
        scale_key = None
        for key in ["scale", "scale_id", "scale_name"]:
            if key in qrow.columns and pd.notna(qrow[key].iloc[0]) and str(qrow[key].iloc[0]).strip():
                scale_key = str(qrow[key].iloc[0]).strip()
                break
        if not scale_key:
            scale_key = str(qcode).strip()
        sub = meta_scales[meta_scales["code"].astype(str).str.strip().str.lower() == str(scale_key).strip().lower()]
        if sub.empty:
            return []
        # keep order by numeric "value"
        sub = sub.copy()
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.sort_values("value", ascending=True)
        # collect labels corresponding to idx_list
        labels = []
        for v in idx_list:
            row = sub[sub["value"] == v]
            if not row.empty:
                lab = str(row["label"].iloc[0]).strip()
                if lab and lab != "9999" and lab != "NA" and lab != "N/A":
                    labels.append(lab)
        return labels
    except Exception:
        return []

def _pick_metric_for_summary(df: pd.DataFrame, qcode: str, meta_q: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Decide which metric column to narrate based on polarity and fallbacks.
    Returns (metric_col_name, metric_label_text)
    """
    try:
        m = meta_q[meta_q["code"].astype(str).str.strip().str.lower() == str(qcode).strip().lower()]
        if m.empty:
            return None, None
        pol = (m["polarity"].iloc[0] or "").upper()
        choices = {
            "POS": ["Positive", "Agree", "Answer1", "Negative"],
            "NEG": ["Negative", "Agree", "Answer1", "Positive"],
            "NEU": ["Agree", "Answer1", "Positive", "Negative"],
        }
        for col in choices.get(pol, ["Agree", "Answer1", "Positive", "Negative"]):
            if col in df.columns:
                return col, f"% selecting {col}"
        return None, None
    except Exception:
        return None, None

def _sanitize_df_values(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for c in work.columns:
        if str(c).strip().lower() == "demographic":
            continue
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work.loc[work[c] == 9999, c] = None
    return work

def _build_summary_pivot(per_q_disp: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    try:
        rows = []
        for q, df in per_q_disp.items():
            if df is None or df.empty:
                continue
            year_cols = _find_year_cols(df)
            if not year_cols:
                continue
            # prefer Positive/Negative/Agree if present
            metric = None
            for cand in ["Positive", "Negative", "Agree", "Answer1"]:
                if cand in df.columns:
                    metric = cand
                    break
            if not metric:
                continue
            row = {"Question": q}
            for yc in year_cols:
                val = df[yc].iloc[-1] if yc in df.columns else None
                row[str(yc)] = _norm_numeric(val)
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        piv = pd.DataFrame(rows)
        piv = piv.set_index("Question")
        return piv
    except Exception:
        return pd.DataFrame()

# ------------------------------- main UI render --------------------------------

def render_menu1_results(
    *,
    tab_titles: List[str],
    tab_labels: List[str],
    per_q_disp: Dict[str, pd.DataFrame],
    per_q_metric_col_in: Dict[str, Optional[str]],
    per_q_metric_label_in: Dict[str, str],
    summary_pivot: Optional[pd.DataFrame],
    pivot_from_payload: pd.DataFrame,
    meta_q: pd.DataFrame,
    meta_scales: pd.DataFrame,
    code_to_text: Dict[str, str],
    labels_used: Dict[str, str],
    data_source_title: str,
    data_source_url: str,
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
) -> Tuple[Dict[str, str], Optional[str]]:

    st.markdown("### Results")
    st.caption("Public Service Employee Survey (PSES) — Menu 1")

    # standard tabs: Questions (per-q), Summary matrix, Technical notes
    tabs = st.tabs(tab_titles)

    # ------------------------- Questions tab -------------------------
    with tabs[0]:
        st.markdown("### Per-question results")
        for q in tab_labels:
            st.markdown(f"**{q} — {code_to_text.get(q, '')}**")
            df = per_q_disp.get(q)
            if df is None or df.empty:
                st.info("No data available for this question.")
                continue
            st.dataframe(df, use_container_width=True)

    # ------------------------- Summary tab -------------------------
    with tabs[1]:
        st.markdown("### Summary (selected questions)")
        if summary_pivot is not None and not summary_pivot.empty:
            st.dataframe(summary_pivot.reset_index(), use_container_width=True)
        else:
            st.info("No summary available for the current selection.")

        _source_link_line(data_source_title, data_source_url)

    # Technical notes
    tech_tab_index = len(tab_titles) - 1
    with tabs[tech_tab_index]:
        st.markdown("### Technical notes")
        st.markdown(
            """
1. **Summary results** are mainly shown as “positive answers,” removing “I don’t know" and "Not applicable" responses from the total responses.  
2. **Weights/adjustment:** Results have been adjusted for non-response and calibrated by demographic variables. Weights also adjust for the number of respondents to determine the number of respondents within a response category.  
3. **Rounding:** Due to rounding, percentages may not add to 100.  
4. **Suppression:** Results were suppressed for questions with low respondent counts (under 10) and for low response category counts.
            """
        )

    # ------------------------- AI section -------------------------
    ai_on = st.session_state.get("menu1_ai_toggle", True)
    if ai_on:
        st.markdown("---")
        st.markdown("## AI Summary")

        ai_key = _hash_key({
            "tab_labels": tab_labels,
            "per_q_metric_col_in": per_q_metric_col_in,
            "labels_used": labels_used,
        })

        cached = _ai_cache_get(ai_key)
        if cached:
            per_q_narratives = cached.get("per_q", {}) or {}
            overall_narrative = cached.get("overall")
            overall_foot_labels: List[str] = []
            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    st.markdown(f"**{q} — {code_to_text.get(q, '')}**")
                    txt_star = _insert_first_percent_asterisk(txt)
                    st.write(txt_star)
                    try:
                        dfq = per_q_disp.get(q)
                        labels = []
                        if not _is_d57_exception(q):
                            metric_col = per_q_metric_col_in.get(q)
                            reporting_field = _infer_reporting_field(metric_col)
                            qtext = code_to_text.get(q, "")
                            labels = _meaning_labels_for_question(
                                qcode=q,
                                question_text=qtext,
                                reporting_field=reporting_field,
                                metric_label=(labels_used.get(q) or per_q_metric_label_in.get(q, "% value")),
                                meta_q=meta_q,
                                meta_scales=meta_scales,
                            ) or []
                        if labels:
                            lab = _compress_labels_for_footnote(labels)
                            if lab:
                                st.caption(f"* Percentages represent respondents’ aggregate answers: {lab}.")
                                overall_foot_labels.append(f"{q} — {lab}")
                    except Exception:
                        pass
            if overall_narrative and len(tab_labels) > 1:
                st.markdown("**Overall**")
                st.write(overall_narrative)
                if overall_foot_labels:
                    st.caption(
                        "* In this section, percentages refer to the same aggregates used above: "
                        + "; ".join(overall_foot_labels) + "."
                    )
        else:
            # ---------- per-question AI ----------
            per_q_narratives: Dict[str, str] = {}
            q_to_meaning_labels: Dict[str, List[str]] = {}
            q_distribution_only: Dict[str, bool] = {}
            overall_foot_labels: List[str] = []

            for q in tab_labels:
                dfq = per_q_disp.get(q)
                qtext = code_to_text.get(q, "")

                if _is_d57_exception(q):
                    metric_col_ai = None
                    metric_label_ai = "Distribution (Answer1..Answer6)"
                    reporting_field_ai = None
                    category_in_play = ("Demographic" in dfq.columns and dfq["Demographic"].astype(str).nunique(dropna=True) > 1)
                    meaning_labels_ai: List[str] = []
                    q_distribution_only[q] = True
                else:
                    metric_col_ai, metric_label_ai = _pick_metric_for_summary(dfq, q, meta_q)
                    if not metric_col_ai:
                        metric_col_ai = per_q_metric_col_in.get(q)
                        metric_label_ai = per_q_metric_label_in.get(q, "% value")
                    else:
                        metric_label_ai = (metric_label_ai or labels_used.get(q) or per_q_metric_label_in.get(q, "% value"))
                    reporting_field_ai = _infer_reporting_field(metric_col_ai)
                    category_in_play = ("Demographic" in dfq.columns and dfq["Demographic"].astype(str).nunique(dropna=True) > 1)

                    meaning_labels_ai = _meaning_labels_for_question(
                        qcode=q,
                        question_text=qtext,
                        reporting_field=reporting_field_ai,
                        metric_label=metric_label_ai or "",
                        meta_q=meta_q,
                        meta_scales=meta_scales
                    )
                    q_distribution_only[q] = False

                q_to_meaning_labels[q] = meaning_labels_ai

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
                                category_in_play=category_in_play,
                                meaning_labels=meaning_labels_ai,
                                reporting_field=reporting_field_ai,
                                distribution_only=_is_d57_exception(q)
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
                    txt_star = _insert_first_percent_asterisk(txt)
                    st.write(txt_star)
                    if meaning_labels_ai:
                        lab = _compress_labels_for_footnote(meaning_labels_ai)
                        if lab:
                            st.caption(f"* Percentages represent respondents’ aggregate answers: {lab}.")
                            overall_foot_labels.append(f"{q} — {lab}")

            # ---------- OVERALL ----------
            overall_narrative = None
            if len(tab_labels) > 1 and summary_pivot is not None and not summary_pivot.empty:
                with st.spinner("AI — synthesizing overall…"):
                    try:
                        content, _hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=build_overall_prompt(
                                tab_labels=tab_labels,
                                pivot_df=summary_pivot.copy(deep=True),
                                q_to_metric={q: (labels_used.get(q) or per_q_metric_label_in[q]) for q in tab_labels},
                                code_to_text=code_to_text,
                                q_to_meaning_labels=q_to_meaning_labels,
                                q_distribution_only=q_distribution_only
                            )
                        )
                        j = json.loads(content) if content else {}
                        overall_narrative = (j.get("narrative") or "").strip()
                    except Exception as e:
                        st.warning(f"AI skipped overall synthesis due to an internal error ({type(e).__name__}).")

                if overall_narrative:
                    st.markdown("**Overall**")
                    st.write(overall_narrative)
                    if overall_foot_labels:
                        st.caption(
                            "* In this section, percentages refer to the same aggregates used above: "
                            + "; ".join(overall_foot_labels) + "."
                        )

            # Cache results for reuse on re-toggle or same selection
            _ai_cache_put(ai_key, {"per_q": per_q_narratives, "overall": overall_narrative})

    # ------------------------- Export & restart (unchanged) -------------------------
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
                    # run fresh (unchanged) ...
                    export_per_q, export_overall = _run_ai_and_collect(
                        tab_labels=tab_labels,
                        per_q_disp=per_q_disp,
                        per_q_metric_col=per_q_metric_col_in,
                        per_q_metric_label=per_q_metric_label_in,
                        code_to_text=code_to_text,
                        demo_selection=None,
                        pivot=(summary_pivot.copy(deep=True) if (summary_pivot is not None and not summary_pivot.empty) else pivot_from_payload.copy(deep=True)),
                        build_overall_prompt=build_overall_prompt,
                        build_per_q_prompt=build_per_q_prompt,
                        call_openai_json=call_openai_json,
                    )
                    _ai_cache_put(ai_key, {"per_q": export_per_q, "overall": export_overall})
                except Exception:
                    export_per_q, export_overall = {}, None

        _render_excel_download(
            summary_pivot=summary_pivot,
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
                pass
            finally:
                st.session_state["menu1_ai_toggle"] = _prev_ai_toggle or False

    return {}, None  # (return signature unchanged)

# ----------------------------- export helpers (unchanged) -----------------------------

def _render_excel_download(
    *,
    summary_pivot: Optional[pd.DataFrame],
    per_q_disp: Dict[str, pd.DataFrame],
    tab_labels: List[str],
    per_q_narratives: Dict[str, str],
    overall_narrative: Optional[str],
) -> None:
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            if summary_pivot is not None and not summary_pivot.empty:
                summary_pivot.reset_index().to_excel(writer, sheet_name="Summary", index=False)
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
