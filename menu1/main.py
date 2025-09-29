# menu1/main.py — PSES Explorer Search (Menu 1)
# Minimal update:
#   • Always include the Overall slice (group_value=None) together with selected demographic codes
#   • Replace the generic "No data found" with a diagnostic explaining what dimension failed
#
# Dependencies:
#   utils/menu1_helpers.py  -> resolve_demographic_codes, get_scale_labels,
#                              drop_na_999, normalize_results_columns,
#                              format_table_for_display, build_positive_only_narrative,
#                              build_no_data_diagnostic
#   utils/data_loader.py    -> load_results2024_filtered
#   utils/hybrid_search.py  -> hybrid_question_search

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ----- Imports from your utilities -----
from utils.menu1_helpers import (
    resolve_demographic_codes,
    get_scale_labels,
    drop_na_999,
    normalize_results_columns,
    format_table_for_display,
    build_positive_only_narrative,
    build_no_data_diagnostic,
)

try:
    from utils.data_loader import load_results2024_filtered
except Exception:
    load_results2024_filtered = None  # type: ignore

try:
    from utils.hybrid_search import hybrid_question_search
except Exception:
    # Safe fallback: simple keyword filter if hybrid module not present
    def hybrid_question_search(qdf: pd.DataFrame, query: str, top_k: int = 120, min_score: float = 0.40) -> pd.DataFrame:
        if not query or not str(query).strip():
            return pd.DataFrame(columns=["code", "text", "display", "score"])
        q = str(query).strip().lower()
        tokens = {t for t in q.replace(",", " ").split() if t}
        scores = []
        for _, r in qdf.iterrows():
            text = f"{r['code']} {r['text']}".lower()
            base = 1.0 if q in text else 0.0
            overlap = sum(1 for t in tokens if t in text) / max(len(tokens), 1)
            score = 0.6 * overlap + 0.4 * base
            scores.append(score)
        out = qdf.copy()
        out["score"] = scores
        out = out.sort_values("score", ascending=False)
        out = out[out["score"] >= min_score]
        return out.head(top_k)

# ----- AI system prompt (exact, unchanged) -----
AI_SYSTEM_PROMPT = (
    "You are preparing insights for the Government of Canada’s Public Service Employee Survey (PSES).\n\n"
    "Context\n"
    "- The PSES provides information to improve people management practices in the federal public service.\n"
    "- Results help departments and agencies identify strengths and concerns in areas such as employee engagement, anti-racism, equity and inclusion, and workplace well-being.\n"
    "- The survey tracks progress over time to refine action plans. Employees’ voices guide improvements to workplace quality, which leads to better results for the public service and Canadians.\n"
    "- Each cycle includes recurring questions (for tracking trends) and new/modified questions reflecting evolving priorities (e.g., updated Employment Equity questions and streamlined hybrid-work items in 2024).\n"
    "- Statistics Canada administers the survey with the Treasury Board of Canada Secretariat. Confidentiality is guaranteed under the Statistics Act (grouped reporting; results for groups <10 are suppressed).\n\n"
    "Data-use rules (hard constraints)\n"
    "- Use ONLY the provided JSON payload/table. DO NOT invent, assume, extrapolate, infer, or generalize beyond the numbers present. No speculation or hypotheses.\n"
    "- Public Service–wide scope ONLY; do not reference specific departments unless present in the payload.\n"
    "- Express percentages as whole numbers (e.g., “75%”). Use “points” for differences/changes.\n\n"
    "Analysis rules\n"
    "- Begin with the 2024 result for the selected question (metric_label).\n"
    "- Describe trend over time: compare 2024 with the earliest year available, using thresholds:\n"
    "  • stable ≤1 point\n"
    "  • slight >1–2 points\n"
    "  • notable >2 points\n"
    "- Compare demographic groups in 2024:\n"
    "  • Focus on the most relevant comparisons (largest gap(s), or those crossing thresholds).\n"
    "  • Report gaps in points and classify them: minimal ≤2, notable >2–5, important >5.\n"
    "- If multiple groups are present, highlight only the most meaningful contrasts instead of exhaustively listing all.\n"
    "- Mention whether gaps observed in 2024 have widened, narrowed, or remained stable compared with earlier years.\n"
    "- Conclude with a concise overall statement (e.g., “Overall, results have remained steady and demographic gaps are unchanged”).\n\n"
    "Style & output\n"
    "- Professional, concise, neutral. Narrative style (1–3 short paragraphs, no lists).\n"
    "- Output VALID JSON with exactly one key: \"narrative\".\n"
)

# ----- Tiny local metadata loaders (non-invasive) -----
@st.cache_data(show_spinner=False)
def _load_demographics() -> pd.DataFrame:
    df = pd.read_excel("metadata/Demographics.xlsx")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def _load_questions() -> pd.DataFrame:
    qdf = pd.read_excel("metadata/Survey Questions.xlsx")
    cols = {c.strip().lower(): c for c in qdf.columns}
    code_col = cols.get("question") or cols.get("code") or list(qdf.columns)[0]
    text_col = cols.get("english") or cols.get("text") or list(qdf.columns)[1]
    qdf = qdf.rename(columns={code_col: "code", text_col: "text"})
    qdf["code"] = qdf["code"].astype(str)
    qdf["display"] = qdf["code"].astype(str) + " — " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]

@st.cache_data(show_spinner=False)
def _load_scales() -> pd.DataFrame:
    sdf = pd.read_excel("metadata/Survey Scales.xlsx")
    sdf.columns = [c.strip() for c in sdf.columns]
    return sdf

# ----- Simple OpenAI wrapper (unchanged behavior) -----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

def _call_openai_json(system: str, user: str, model: str = OPENAI_MODEL, temperature: float = 0.2, max_retries: int = 2):
    if not OPENAI_API_KEY:
        st.session_state["menu1_last_ai_status"] = {"time": datetime.now().isoformat(timespec="seconds"),
                                                    "ok": False, "hint": "no_api_key"}
        return "", "no_api_key"
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        hint = "unknown_error"
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                content = resp.choices[0].message.content or ""
                st.session_state["menu1_last_ai_status"] = {"time": datetime.now().isoformat(timespec="seconds"),
                                                            "ok": True, "hint": None}
                return content, None
            except Exception as e:
                hint = f"openai_err_{attempt+1}: {type(e).__name__}"
                time.sleep(0.7 * (attempt + 1))
        st.session_state["menu1_last_ai_status"] = {"time": datetime.now().isoformat(timespec="seconds"),
                                                    "ok": False, "hint": hint}
        return "", hint
    except Exception:
        try:
            import openai  # type: ignore
            openai.api_key = OPENAI_API_KEY
            hint = "unknown_error"
            for attempt in range(max_retries + 1):
                try:
                    resp = openai.ChatCompletion.create(
                        model=model,
                        temperature=temperature,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    )
                    content = resp["choices"][0]["message"]["content"] or ""
                    st.session_state["menu1_last_ai_status"] = {"time": datetime.now().isoformat(timespec="seconds"),
                                                                "ok": True, "hint": None}
                    return content, None
                except Exception as e:
                    hint = f"openai_legacy_err_{attempt+1}: {type(e).__name__}"
                    time.sleep(0.7 * (attempt + 1))
            st.session_state["menu1_last_ai_status"] = {"time": datetime.now().isoformat(timespec="seconds"),
                                                        "ok": False, "hint": hint}
            return "", hint
        except Exception:
            st.session_state["menu1_last_ai_status"] = {"time": datetime.now().isoformat(timespec="seconds"),
                                                        "ok": False, "hint": "no_openai_sdk"}
            return "", "no_openai_sdk"

# ----- Minimal UI scaffolding (consistent with prior iterations) -----
def _ensure_defaults():
    st.session_state.setdefault("menu1_ai_toggle", True)
    st.session_state.setdefault("menu1_show_diag", False)
    st.session_state.setdefault("menu1_multi_questions", [])
    st.session_state.setdefault("menu1_kw_query", "")
    st.session_state.setdefault("menu1_hits", [])
    st.session_state.setdefault("menu1_selected_codes", [])
    st.session_state.setdefault("menu1_selected_order", [])
    st.session_state.setdefault("select_all_years", True)
    for y in (2019, 2020, 2022, 2024):
        st.session_state.setdefault(f"year_{y}", True)
    st.session_state.setdefault("last_query_info", None)

def run_menu1():
    _ensure_defaults()

    if load_results2024_filtered is None:
        st.error("Data loader unavailable. Please verify utils.data_loader.load_results2024_filtered.")
        return

    demo_df = _load_demographics()
    qdf = _load_questions()
    sdf = _load_scales()

    # Banner + header
    st.markdown(
        "<img style='width:100%;height:auto;margin:0 0 12px 0;' "
        "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
        unsafe_allow_html=True
    )
    st.write("### PSES Explorer Search")
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.toggle("🧠 Enable AI analysis", key="menu1_ai_toggle")
    with top_col2:
        st.toggle("🔧 Show technical parameters & diagnostics", key="menu1_show_diag")

    # Instructions
    st.caption(
        "Please use this menu to explore the survey results by questions. "
        "You may select from the drop down menu below up to five questions or find questions via the keyword/theme search. "
        "Select year(s) and optionally a demographic category and subgroup."
    )

    # Diagnostics (optional)
    if st.session_state.get("menu1_show_diag", False):
        yrs = [y for y in (2019, 2020, 2022, 2024) if st.session_state.get(f"year_{y}", False)]
        params = {
            "Selected questions": [q for q in st.session_state.get("menu1_multi_questions", [])],
            "Years selected": ", ".join(map(str, yrs)) if yrs else "(none)",
            "AI enabled": st.session_state.get("menu1_ai_toggle", True),
        }
        st.table(pd.DataFrame([(k, v) for k, v in params.items()], columns=["Field", "Value"]))

    # Question picker: multiselect + keyword search
    st.subheader("Pick up to 5 survey questions:")
    all_displays = qdf["display"].tolist()
    multi_choices = st.multiselect(
        "Choose one or more from the official list",
        all_displays,
        default=st.session_state.get("menu1_multi_questions", []),
        max_selections=5,
        label_visibility="collapsed",
        key="menu1_multi_questions",
    )
    display_to_code = {d: c for c, d in zip(qdf["code"], qdf["display"])}
    selected_from_multi: List[str] = [display_to_code[d] for d in multi_choices if d in display_to_code]

    with st.expander("Search by keywords or theme (optional)"):
        search_query = st.text_input("Enter keywords (e.g., harassment, recognition, onboarding)", key="menu1_kw_query")
        if st.button("Search questions"):
            hits_df = hybrid_question_search(qdf, search_query, top_k=120, min_score=0.40)
            st.session_state["menu1_hits"] = hits_df[["code", "text"]].to_dict(orient="records") if not hits_df.empty else []
        if st.session_state["menu1_hits"]:
            st.write(f"Top {len(st.session_state['menu1_hits'])} matches meeting the quality threshold:")
            selected_from_hits: List[str] = []
            for rec in st.session_state["menu1_hits"]:
                code, text = rec["code"], rec["text"]
                label = f"{code} — {text}"
                key = f"kwhit_{code}"
                checked = st.checkbox(label, value=(code in selected_from_multi), key=key)
                if checked and code not in selected_from_hits:
                    selected_from_hits.append(code)
        else:
            selected_from_hits = []

    # Merge order: keep user's prior order where possible
    current_selected = selected_from_multi + [c for c in selected_from_hits if c not in selected_from_multi]
    prev_order: List[str] = st.session_state.get("menu1_selected_order", [])
    new_order = [c for c in prev_order if c in current_selected]
    for c in current_selected:
        if c not in new_order:
            new_order.append(c)
    if len(new_order) > 5:
        new_order = new_order[:5]
        st.warning("Limit is 5 questions; extra selections were ignored.")

    st.session_state["menu1_selected_codes"] = new_order
    st.session_state["menu1_selected_order"] = new_order

    if new_order:
        st.write("**Selected questions:**")
        cols = st.columns(min(5, len(new_order)))
        code_to_display = dict(zip(qdf["code"], qdf["display"]))
        kept = list(new_order)
        for idx, code in enumerate(list(new_order)):
            with cols[idx % len(cols)]:
                keep = st.checkbox(code_to_display.get(code, code), value=True, key=f"sel_{code}")
                if not keep:
                    kept = [c for c in kept if c != code]
                    hk = f"kwhit_{code}"
                    if hk in st.session_state: st.session_state[hk] = False
        if kept != new_order:
            st.session_state["menu1_selected_codes"] = kept
            st.session_state["menu1_selected_order"] = kept
            new_order = kept

    question_codes: List[str] = st.session_state.get("menu1_selected_codes", [])

    # Years
    st.subheader("Select survey year(s):")
    st.checkbox("All years", key="select_all_years")
    years: List[int] = []
    year_cols = st.columns(4)
    for i, yr in enumerate([2019, 2020, 2022, 2024]):
        with year_cols[i]:
            default_checked = True if st.session_state.get("select_all_years", True) else st.session_state.get(f"year_{yr}", False)
            if st.checkbox(str(yr), value=default_checked, key=f"year_{yr}"):
                years.append(yr)
    years = sorted(years)

    # Demographics
    st.subheader("Select a demographic category (optional):")
    # Build category list from metadata
    cat_col = next((c for c in _load_demographics().columns if c.strip().lower() == "demcode_category"), None)
    demo_categories = ["All respondents"]
    if cat_col:
        demo_categories += sorted(_load_demographics()[cat_col].dropna().astype(str).unique().tolist())
    demo_selection = st.selectbox("Demographic category", demo_categories, label_visibility="collapsed", key="menu1_demo_main")

    sub_selection = None
    if demo_selection != "All respondents":
        # Get subgroups for this category
        ddf = _load_demographics()
        cols = {c.strip().lower(): c for c in ddf.columns}
        lbl_col = cols.get("descrip_e") or cols.get("english") or list(ddf.columns)[0]
        cat_col = cols.get("demcode_category") or cols.get("category") or list(ddf.columns)[0]
        sub_items = ddf.loc[ddf[cat_col].astype(str).str.strip() == demo_selection, lbl_col].dropna().astype(str).unique().tolist()
        sub_items = sorted(sub_items)
        sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, label_visibility="collapsed", key=f"sub_{demo_selection.replace(' ', '_')}")
        if sub_selection == "":
            sub_selection = None

    # ---- Search (run) ----
    st.write("")
    run = st.button("Search")
    if not run:
        return

    if not question_codes:
        st.info("Please select at least one question.")
        return

    if not years:
        st.info("Please select at least one year.")
        return

    # Resolve demographics to codes — this helper now always prepends Overall (None)
    codes_for_query, dem_disp_map, category_in_play = resolve_demographic_codes(
        _load_demographics(),
        demo_selection,
        sub_selection,
    )
    # NOTE: codes_for_query looks like [None, '1904', '1905'] for FOL with (all subgroups)

    # ---- Run per-question queries ----
    st.write("---")
    per_q_disp: Dict[str, pd.DataFrame] = {}
    per_q_text: Dict[str, str] = {}
    per_q_metric_col: Dict[str, str] = {}
    per_q_metric_label: Dict[str, str] = {}

    for qcode in question_codes:
        qtext = qdf.loc[qdf["code"] == qcode, "text"].iloc[0] if (qdf["code"] == qcode).any() else ""
        per_q_text[qcode] = qtext

        # Call the loader ONCE PER CODE (this is the only change in call shape, plus including Overall)
        frames = []
        for gv in codes_for_query:
            try:
                dfp = load_results2024_filtered(question_code=qcode, years=years, group_value=gv)
            except TypeError:
                # older signature without group_value: only usable for Overall
                dfp = load_results2024_filtered(question_code=qcode, years=years) if gv is None else None
            if dfp is not None and not getattr(dfp, "empty", True):
                frames.append(dfp)

        df_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # If empty: build a meaningful diagnostic and stop here for this question
        if df_raw is None or df_raw.empty:
            diag = build_no_data_diagnostic(
                question_code=qcode,
                years=years,
                requested_codes=codes_for_query,
                loader_fn=load_results2024_filtered,
            )
            st.warning(f"⚠️ No results for this selection.\n\n{diag}")
            continue

        # Normalize/sanitize, format for display
        df_raw = normalize_results_columns(df_raw)
        df_raw = drop_na_999(df_raw)

        scale_pairs = get_scale_labels(sdf, qcode)
        df_disp = format_table_for_display(
            df=df_raw,
            category_in_play=category_in_play,
            dem_disp_map=dem_disp_map,
            scale_pairs=scale_pairs,
        )

        if df_disp is None or df_disp.empty:
            diag = build_no_data_diagnostic(
                question_code=qcode,
                years=years,
                requested_codes=codes_for_query,
                loader_fn=load_results2024_filtered,
            )
            st.warning(f"⚠️ No results for this selection after formatting.\n\n{diag}")
            continue

        # Store for later (AI + cross-question summary)
        per_q_disp[qcode] = df_disp

        # Render: summary + details tabs (as before)
        st.subheader(f"{qcode} — {qtext}")
        tab_summary, tab_detail = st.tabs(["Summary results", "Detailed results"])

        # Decide metric availability for this question for the summary table
        low = {c.lower(): c for c in df_disp.columns}
        metric_col = None
        metric_label = None
        if "positive" in low and pd.to_numeric(df_disp[low["positive"]], errors="coerce").notna().any():
            metric_col = low["positive"]; metric_label = "% positive"
        elif "agree" in low and pd.to_numeric(df_disp[low["agree"]], errors="coerce").notna().any():
            metric_col = low["agree"]; metric_label = "% agree"
        else:
            # try first answer label
            for c in df_disp.columns:
                if c.lower().startswith("answer"):
                    if pd.to_numeric(df_disp[c], errors="coerce").notna().any():
                        metric_col = c; metric_label = f"% {c}"
                        break

        per_q_metric_col[qcode] = metric_col or ""
        per_q_metric_label[qcode] = metric_label or ""

        with tab_summary:
            if metric_col:
                # Build simple trend (rows = Segment, cols = Year)
                df = df_disp.copy()
                if "Demographic" not in df.columns:
                    df["Demographic"] = "All respondents"
                pivot = df.pivot_table(index="Demographic", columns="Year", values=metric_col, aggfunc="first")
                years_str = sorted([str(y) for y in years], key=lambda x: int(x))
                for y in years_str:
                    if y not in pivot.columns: pivot[y] = pd.NA
                pivot = pivot.reset_index()
                # format as percents
                for y in years_str:
                    vals = pd.to_numeric(pivot[y], errors="coerce").round(0)
                    out = pd.Series("n/a", index=pivot.index, dtype="object")
                    mask = vals.notna()
                    out.loc[mask] = vals.loc[mask].astype(int).astype(str) + "%"
                    pivot[y] = out
                st.caption(metric_label or "")
                st.dataframe(pivot[["Demographic"] + years_str], use_container_width=True, hide_index=True)
                st.caption("Source: 2024 Public Service Employee Survey (Open Government Portal).")
            else:
                st.info("Summary table is unavailable for this selection, please see the detailed results.")

        with tab_detail:
            st.dataframe(df_disp, use_container_width=True)

    # ---- AI summary (per question + overall) ----
    if st.session_state.get("menu1_ai_toggle", True) and per_q_disp:
        st.write("---")
        st.subheader("Summary analysis")
        # Per-question narratives
        for qcode, df_disp in per_q_disp.items():
            metric_col = per_q_metric_col.get(qcode, "")
            metric_label = per_q_metric_label.get(qcode, "")
            if not metric_col:
                continue
            payload = {
                "question_code": qcode,
                "question_text": per_q_text.get(qcode, ""),
                "metric_label": metric_label,
                "series": [
                    {"year": (int(y) if str(y).isdigit() else y),
                     "value": float(v) if pd.notna(v) else None}
                    for y, v in df_disp.groupby("Year")[metric_col].mean(numeric_only=True).items()
                ],
            }
            user = json.dumps(payload, ensure_ascii=False)
            out, _ = _call_openai_json(AI_SYSTEM_PROMPT, user)
            text = ""
            if out:
                try:
                    text = json.loads(out).get("narrative", "")
                except Exception:
                    text = ""
            if not text:
                text = build_positive_only_narrative(df_disp, category_in_play=False)
            st.markdown(f"**{qcode} — {per_q_text.get(qcode,'')}**")
            st.write(text)

        # Overall synthesis when multiple questions selected
        if len(per_q_disp) > 1:
            st.write("")
            st.subheader("Overall summary analysis")
            # Build combined payload (simple, respects per-question metric labels)
            items = []
            for qcode, df_disp in per_q_disp.items():
                metric_col = per_q_metric_col.get(qcode, "")
                metric_label = per_q_metric_label.get(qcode, "")
                if not metric_col:
                    continue
                series = {}
                for y, v in df_disp.groupby("Year")[metric_col].mean(numeric_only=True).items():
                    try:
                        series[int(y)] = float(v) if pd.notna(v) else None
                    except Exception:
                        pass
                items.append({
                    "question_label": f"{qcode} — {per_q_text.get(qcode,'')}",
                    "metric_label": metric_label,
                    "values_by_year": series
                })
            user = json.dumps({"questions": items}, ensure_ascii=False)
            out, _ = _call_openai_json(AI_SYSTEM_PROMPT, user)
            text = ""
            if out:
                try:
                    text = json.loads(out).get("narrative", "")
                except Exception:
                    text = ""
            if not text:
                # Fallback deterministic note
                text = "No AI summary available."
            st.write(text)
    elif per_q_disp:
        st.write("---")
        st.subheader("Summary analysis")
        st.info("AI analysis disabled. No AI summary generated for this selection.")

# Entrypoint when this file is run as a page
if __name__ == "__main__":
    run_menu1()
