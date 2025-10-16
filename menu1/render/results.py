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

# ----------------------------- small helpers -----------------------------

def _compress_labels_for_footnote(labels: List[str]) -> Optional[str]:
    """
    Return a compressed label string in parentheses, e.g.
    ["To a small extent","To a moderate extent","To a large extent","To a very large extent"]
      -> "(To a small/moderate/large extent/very large extent)".
    Falls back to full join if safe compression not possible.
    """
    if not labels:
        return None
    # Full form
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
            # Keep prefix on first for readability; suffix on all if present
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

_percent_pat = re.compile(r"(\d{1,3})\s*%")

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
    # If already has *, skip
    if i < len(text) and text[i] == "*":
        return text
    return text[:i] + "*" + text[i:]

# ----------------------------- existing code continues -----------------------------
# (Everything below is unchanged except where noted in the AI rendering section.)

def _md(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)

def _hash_key(obj: Any) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

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

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return None
    m = {c.lower(): c for c in df.columns}
    for n in names:
        c = m.get(n.lower())
        if c is not None:
            return c
    return None

def _is_d57_exception(qcode: str) -> bool:
    q = (qcode or "").strip().lower()
    return q in {"d57_a", "d57_b", "d57a", "d57b"}

# NOTE: The following helpers are expected to exist elsewhere in your codebase:
#  - _infer_reporting_field(metric_col)
#  - _meaning_labels_for_question(...)
#  - _pick_metric_for_summary(...)
#  - _run_ai_and_collect(...)

# ----------------------------- main render function -----------------------------
def render_results_tab(
    *,
    # (your existing parameters)
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
    build_overall_prompt: Callable[..., str],
    build_per_q_prompt: Callable[..., str],
    call_openai_json: Callable[..., Tuple[Optional[str], Optional[str]]],
) -> Tuple[Dict[str, str], Optional[str]]:
    # ... existing set-up and non-AI rendering code ...

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

            # Collect label strings for the Overall caption
            overall_foot_labels: List[str] = []

            for q in tab_labels:
                txt = per_q_narratives.get(q, "")
                if txt:
                    st.markdown(f"**{q} — {code_to_text.get(q, '')}**")
                    # add asterisk after first %
                    txt_star = _insert_first_percent_asterisk(txt)
                    st.write(txt_star)
                    # per-question footnote using meaning labels (skip for distribution-only)
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
                # Overall caption listing each question's label set (once)
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

            # Collect label strings for the Overall caption
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

                    # meaning labels with robust fallbacks + special NEGATIVE/EXTENT default
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
                    # Overall caption listing each question's label set (once)
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

# ----------------------------- compatibility wrapper -----------------------------
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
    """
    Back-compat shim: adapt legacy entry point to the current render function.
    Expects payload to carry the same keys the app already produces.
    """
    tab_labels = payload.get("tab_labels", [])
    per_q_disp = payload.get("per_q_disp", {})
    per_q_metric_col_in = payload.get("per_q_metric_col_in", {})
    per_q_metric_label_in = payload.get("per_q_metric_label_in", {})
    summary_pivot = payload.get("summary_pivot")
    pivot_from_payload = payload.get("pivot_from_payload", pd.DataFrame())
    meta_q = payload.get("meta_q", pd.DataFrame())
    meta_scales = payload.get("meta_scales", pd.DataFrame())
    code_to_text = payload.get("code_to_text", {})
    labels_used = payload.get("labels_used", {})

    # Preserve the toggle passed by caller
    st.session_state["menu1_ai_toggle"] = bool(ai_on)

    # Render header or any outer elements if your app expects them (omitted here by design).
    # Delegate to the new renderer:
    render_results_tab(
        tab_labels=tab_labels,
        per_q_disp=per_q_disp,
        per_q_metric_col_in=per_q_metric_col_in,
        per_q_metric_label_in=per_q_metric_label_in,
        summary_pivot=summary_pivot,
        pivot_from_payload=pivot_from_payload,
        meta_q=meta_q,
        meta_scales=meta_scales,
        code_to_text=code_to_text,
        labels_used=labels_used,
        build_overall_prompt=build_overall_prompt,
        build_per_q_prompt=build_per_q_prompt,
        call_openai_json=call_openai_json,
    )
