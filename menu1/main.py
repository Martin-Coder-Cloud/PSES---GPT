# menu1/main.py — PSES Explorer Search (Menu 1)
# Multi-question (max 5) + hybrid keyword search + summary matrix + per-question tabs.
# AI analysis toggle (red slide) under the title; per-question + overall narratives use your exact system prompt.
# Reset-on-entry via last_active_menu + "Reset all parameters" beside Search.
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional

import pandas as pd
import streamlit as st

# -----------------------------
# External utilities (your repo)
# -----------------------------
import utils.data_loader as _dl
try:
    from utils.data_loader import load_results2024_filtered, get_backend_info, prewarm_fastpath
except Exception:
    from utils.data_loader import load_results2024_filtered  # type: ignore
    def get_backend_info(): return {}
    def prewarm_fastpath(): return "csv"

from utils.menu1_helpers import (
    resolve_demographic_codes,
    get_scale_labels,
    drop_na_999,
    normalize_results_columns,
    format_table_for_display,
    build_positive_only_narrative,
)

from utils.hybrid_search import hybrid_question_search

# Ensure OpenAI key from secrets/env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

SHOW_DEBUG = False
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
    qdf["code"] = qdf["code"].astype(str)
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"].astype(str) + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]

@st.cache_data(show_spinner=False)
def load_scales_metadata() -> pd.DataFrame:
    sdf = pd.read_excel("metadata/Survey Scales.xlsx")
    sdf.columns = [c.strip().lower() for c in df.columns] if (df := sdf) is not None else []
    return sdf

# ─────────────────────────────
# Reset helpers
# ─────────────────────────────
def _delete_keys(prefixes: List[str], exact_keys: List[str] = None):
    exact_keys = exact_keys or []
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in prefixes) or (k in exact_keys):
            try:
                del st.session_state[k]
            except Exception:
                pass

def reset_menu1_state():
    year_keys = [f"year_{y}" for y in (2024, 2022, 2020, 2019)]
    exact = [
        "menu1_selected_codes",
        "menu1_hits",
        "menu1_kw_query",
        "menu1_multi_questions",
        "menu1_ai_toggle",
        "select_all_years",
        "demo_main",
        "menu1_clear_sel",
        "menu1_find_hits",
    ] + year_keys
    prefixes = ["kwhit_", "sel_", "sub_"]
    _delete_keys(prefixes, exact)
    st.session_state["menu1_kw_query"] = ""
    st.session_state["menu1_hits"] = []
    st.session_state["menu1_selected_codes"] = []
    st.session_state["menu1_multi_questions"] = []
    st.session_state["menu1_ai_toggle"] = False

# ─────────────────────────────
# AI helpers: your exact system prompt + call
# ─────────────────────────────
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

def _format_series_for_ai(year_pos: pd.DataFrame) -> List[Dict[str, float]]:
    rows = []
    for _, r in year_pos.iterrows():
        try:
            y = int(r["Year"])
        except Exception:
            y = r["Year"]
        rows.append({"year": y, "positive": float(r["Positive"]) if pd.notna(r["Positive"]) else None})
    return rows

def _build_user_prompt_per_question(qcode: str, qtext: str, df_disp: pd.DataFrame, category_in_play: bool) -> str:
    t = df_disp.copy()
    if "Demographic" in t.columns:
        series = t.groupby("Year", as_index=False)["Positive"].mean(numeric_only=True)
    else:
        series = t[["Year", "Positive"]].copy()
    series = series.dropna(subset=["Year"]).sort_values("Year")
    series_json = _format_series_for_ai(series)

    group_info = []
    if category_in_play and "Demographic" in t.columns:
        latest_year = pd.to_numeric(t["Year"], errors="coerce").max()
        g = t[pd.to_numeric(t["Year"], errors="coerce") == latest_year][["Demographic", "Positive"]].dropna()
        g = g.sort_values("Positive", ascending=False)
        if not g.empty:
            top = g.iloc[0].to_dict()
            bot = g.iloc[-1].to_dict()
            group_info = [
                {"demographic": str(top["Demographic"]), "positive": float(top["Positive"]) if pd.notna(top["Positive"]) else None},
                {"demographic": str(bot["Demographic"]), "positive": float(bot["Positive"]) if pd.notna(bot["Positive"]) else None},
            ]

    payload = {
        "question_code": qcode,
        "question_text": qtext,
        "series_positive_by_year": series_json,
        "latest_year_group_snapshot": group_info,
        "notes": "Summarize trends and gaps using the classification thresholds provided in the system prompt."
    }
    return json.dumps(payload, ensure_ascii=False)

def _build_user_prompt_overall(selected_codes: List[str], pivot: pd.DataFrame) -> str:
    items = []
    for q in pivot.index.tolist():
        row = {"question_code": str(q), "positive_by_year": {}}
        for y in pivot.columns.tolist():
            val = pivot.loc[q, y]
            if pd.notna(val):
                row["positive_by_year"][int(y)] = float(val)
        items.append(row)
    payload = {"questions": items, "notes": "Synthesize the overall pattern across questions using the classification thresholds."}
    return json.dumps(payload, ensure_ascii=False)

def _call_openai_json(system: str, user: str, model: str = OPENAI_MODEL, temperature: float = 0.3, max_retries: int = 2) -> Tuple[str, Optional[str]]:
    if not OPENAI_API_KEY:
        return "", "no_api_key"
    try:
        from openai import OpenAI  # new SDK
        client = OpenAI(api_key=OPENAI_API_KEY)
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                content = resp.choices[0].message.content or ""
                return content, None
            except Exception as e:
                hint = f"openai_err_{attempt+1}: {type(e).__name__}"
                time.sleep(0.8 * (attempt + 1))
        return "", hint
    except Exception:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            for attempt in range(max_retries + 1):
                try:
                    resp = openai.ChatCompletion.create(
                        model=model,
                        temperature=temperature,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    )
                    content = resp["choices"][0]["message"]["content"] or ""
                    return content, None
                except Exception as e:
                    hint = f"openai_legacy_err_{attempt+1}: {type(e).__name__}"
                    time.sleep(0.8 * (attempt + 1))
            return "", hint
        except Exception:
            return "", "no_openai_sdk"

# ─────────────────────────────
# Column resolution helper (robust to naming)
# ─────────────────────────────
def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ─────────────────────────────
# Try to get the in-memory PS-wide DF (fastpath)
# ─────────────────────────────
def _get_pswide_df() -> Optional[pd.DataFrame]:
    try:
        if hasattr(_dl, "preload_pswide_dataframe"):
            df = _dl.preload_pswide_dataframe()
            if isinstance(df, pd.DataFrame):
                return df
        if hasattr(_dl, "pswide_df"):
            df = getattr(_dl, "pswide_df")
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        pass
    return None

# ─────────────────────────────
# UI
# ─────────────────────────────
def run_menu1():
    st.markdown("""
        <style>
            body { background-image: none !important; background-color: white !important; }
            .block-container { padding-top: 1rem !important; }
            .menu-banner { width: 100%; height: auto; display: block; margin-top: 0px; margin-bottom: 20px; }
            .custom-header { font-size: 30px !important; font-weight: 700; margin-bottom: 6px; }
            .custom-instruction { font-size: 16px !important; line-height: 1.4; margin-bottom: 10px; color: #333; }
            .field-label { font-size: 18px !important; font-weight: 600 !important; margin-top: 12px !important; margin-bottom: 2px !important; color: #222 !important; }
            .big-button button { font-size: 18px !important; padding: 0.75em 2em !important; margin-top: 20px; }
            .pill { display:inline-block; padding:4px 8px; margin:2px 6px 2px 0; background:#f1f3f5; border-radius:999px; font-size:13px; }
            .action-row { display:flex; gap:10px; align-items:center; }
            [data-testid="stSwitch"] div[role="switch"][aria-checked="true"] { background-color: #e03131 !important; }
            [data-testid="stSwitch"] div[role="switch"] { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1); }
        </style>
    """, unsafe_allow_html=True)

    # Auto-reset on entry when coming from another page (router must set last_active_menu)
    if st.session_state.get("last_active_menu") != "menu1":
        reset_menu1_state()
    st.session_state["last_active_menu"] = "menu1"

    # Ensure base state exists (after possible reset)
    st.session_state.setdefault("menu1_selected_codes", [])
    st.session_state.setdefault("menu1_hits", [])
    st.session_state.setdefault("menu1_kw_query", "")
    st.session_state.setdefault("menu1_multi_questions", [])
    st.session_state.setdefault("menu1_ai_toggle", False)

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    left, center, right = st.columns([1, 3, 1])
    with center:
        # Banner
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )

        # Title + AI toggle (red switch)
        st.markdown('<div class="custom-header">PSES Explorer Search</div>', unsafe_allow_html=True)
        ai_enabled = st.toggle(
            "🧠 Enable AI analysis",
            value=st.session_state.get("menu1_ai_toggle", False),
            key="menu1_ai_toggle",
            help="Include the AI-generated analysis alongside the tables."
        )

        # Instructions
        st.markdown("""
            <div class="custom-instruction">
                Please use this menu to explore the survey results by questions.<br>
                You may select from the drop down menu below up to five questions or find questions via the keyword/theme search.
                Select year(s) and optionally a demographic category and subgroup.
            </div>
        """, unsafe_allow_html=True)

        # ---------- Question selection ----------
        st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)

        # 1) Multi-select from list (authoritative)
        all_displays = qdf["display"].tolist()
        multi_choices = st.multiselect(
            "Choose one or more from the official list",
            all_displays,
            default=st.session_state["menu1_multi_questions"],
            max_selections=5,
            label_visibility="collapsed",
            key="menu1_multi_questions",
        )
        selected_from_multi: Set[str] = set(display_to_code[d] for d in multi_choices if d in display_to_code)

        # 2) Keyword/theme search (persistent)
        with st.expander("Search by keywords or theme (optional)"):
            search_query = st.text_input("Enter keywords (e.g., harassment, recognition, onboarding)", key="menu1_kw_query")
            if st.button("Search questions", key="menu1_find_hits"):
                hits_df = hybrid_question_search(qdf, search_query, top_k=120, min_score=0.40)
                st.session_state["menu1_hits"] = hits_df[["code", "text"]].to_dict(orient="records") if not hits_df.empty else []

            selected_from_hits: Set[str] = set()
            if st.session_state["menu1_hits"]:
                st.write(f"Top {len(st.session_state['menu1_hits'])} matches meeting the quality threshold:")
                for rec in st.session_state["menu1_hits"]:
                    code = rec["code"]; text = rec["text"]
                    label = f"{code} – {text}"
                    key = f"kwhit_{code}"
                    default_checked = st.session_state.get(key, False) or (code in selected_from_multi)
                    checked = st.checkbox(label, value=default_checked, key=key)
                    if checked:
                        selected_from_hits.add(code)
            else:
                st.info('Enter keywords and click "Search questions" to see matches.')

        # Merge sources: dropdown (ordered) + search hits
        combined_order: List[str] = []
        for d in st.session_state["menu1_multi_questions"]:
            c = display_to_code.get(d)
            if c and c not in combined_order:
                combined_order.append(c)
        for c in selected_from_hits:
            if c not in combined_order:
                combined_order.append(c)

        # Cap at 5
        if len(combined_order) > 5:
            combined_order = combined_order[:5]
            st.warning("Limit is 5 questions; extra selections were ignored.")

        st.session_state["menu1_selected_codes"] = combined_order

        # Selected questions checklist (for quick removal)
        if st.session_state["menu1_selected_codes"]:
            st.markdown('<div class="field-label">Selected questions:</div>', unsafe_allow_html=True)
            updated = list(st.session_state["menu1_selected_codes"])
            cols = st.columns(min(5, len(updated)))
            for idx, code in enumerate(list(updated)):
                with cols[idx % len(cols)]:
                    label = code_to_display.get(code, code)
                    keep = st.checkbox(label, value=True, key=f"sel_{code}")
                    if not keep:
                        updated = [c for c in updated if c != code]
                        hk = f"kwhit_{code}"
                        if hk in st.session_state:
                            st.session_state[hk] = False
                        disp = code_to_display.get(code)
                        if disp:
                            st.session_state["menu1_multi_questions"] = [d for d in st.session_state["menu1_multi_questions"] if d != disp]
            if updated != st.session_state["menu1_selected_codes"]:
                st.session_state["menu1_selected_codes"] = updated

        # Final list
        question_codes: List[str] = st.session_state["menu1_selected_codes"]

        # ---------- Years ----------
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        all_years = [2024, 2022, 2020, 2019]
        select_all = st.checkbox("All years", value=True, key="select_all_years")
        selected_years: List[int] = []
        year_cols = st.columns(len(all_years))
        for idx, yr in enumerate(all_years):
            with year_cols[idx]:
                checked = True if select_all else False
                if st.checkbox(str(yr), value=checked, key=f"year_{yr}"):
                    selected_years.append(yr)
        selected_years = sorted(selected_years)

        # ---------- Demographics ----------
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
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

        # ---------- Action row ----------
        st.markdown("<div class='action-row'>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1])
        with colA:
            disable_search = (not question_codes) or (not selected_years)
            if st.button("Search", disabled=disable_search):
                # Use in-memory PS-wide DF if available; fallback to chunked loader otherwise
                df_ps = _get_pswide_df()

                # Resolve DEMCODE(s)
                demcodes, disp_map, category_in_play = resolve_demographic_codes(demo_df, demo_selection, sub_selection)

                per_q_disp_tables: Dict[str, pd.DataFrame] = {}
                per_q_texts: Dict[str, str] = {}

                for qcode in question_codes:
                    qtext = qdf.loc[qdf["code"] == qcode, "text"].values[0] if (qdf["code"] == qcode).any() else ""
                    per_q_texts[qcode] = qtext

                    if isinstance(df_ps, pd.DataFrame) and not df_ps.empty:
                        # -------- Fastpath: filter in-memory PS-wide DF ----------
                        df_all = df_ps.copy()

                        # Resolve columns robustly
                        qcol = _first_col(df_all, ["question_code", "QUESTION", "Question"])
                        ycol = _first_col(df_all, ["year", "Year", "SURVEYR"])
                        gcol = _first_col(df_all, ["group_value", "DEMCODE", "Group", "group"])

                        # Apply filters
                        if qcol is None or ycol is None:
                            continue
                        qmask = df_all[qcol].astype(str).str.strip().str.upper() == str(qcode).strip().upper()
                        ymask = pd.to_numeric(df_all[ycol], errors="coerce").astype("Int64").isin(selected_years)

                        if gcol is None:
                            gmask = True  # PS-wide only
                        else:
                            if demo_selection == "All respondents":
                                gv = df_all[gcol].astype(str).fillna("").str.strip()
                                gmask = gv.isin(["", "All", "ALL", "All respondents", "ALL RESPONDENTS"])
                            else:
                                gmask = df_all[gcol].astype(str).isin([str(c) for c in demcodes])

                        df_all = df_all[qmask & ymask & gmask].copy()
                        if df_all.empty:
                            continue

                        # Normalize columns to standard names expected downstream
                        df_all = normalize_results_columns(df_all)
                        df_all = drop_na_999(df_all)
                        if df_all.empty:
                            continue

                    else:
                        # -------- Fallback: chunked loader path (per-demcode) ----------
                        parts = []
                        for code in demcodes:
                            df_part = load_results2024_filtered(question_code=qcode, years=selected_years, group_value=code)
                            if not df_part.empty:
                                parts.append(df_part)
                        if not parts:
                            continue
                        df_all = pd.concat(parts, ignore_index=True)
                        df_all = normalize_results_columns(df_all)
                        # Guard after normalization
                        qmask = df_all["question_code"].astype(str).str.strip().str.upper() == str(qcode).strip().upper()
                        ymask = pd.to_numeric(df_all["year"], errors="coerce").astype("Int64").isin(selected_years)
                        if demo_selection == "All respondents":
                            gv = df_all["group_value"].astype(str).fillna("").str.strip()
                            gmask = gv.isin(["", "All", "ALL", "All respondents", "ALL RESPONDENTS"])
                        else:
                            gmask = df_all["group_value"].astype(str).isin([str(c) for c in demcodes])
                        df_all = df_all[qmask & ymask & gmask].copy()
                        df_all = drop_na_999(df_all)
                        if df_all.empty:
                            continue

                    # Scales → display labels
                    scale_pairs = get_scale_labels(sdf, qcode)
                    df_disp = format_table_for_display(
                        df_slice=df_all,
                        dem_disp_map=disp_map,
                        category_in_play=category_in_play,
                        scale_pairs=scale_pairs
                    )
                    per_q_disp_tables[qcode] = df_disp

                if not per_q_disp_tables:
                    st.info("No data found for your selection.")
                else:
                    # Tabs: one per question
                    tab_labels = [qc for qc in question_codes if qc in per_q_disp_tables]
                    tabs = st.tabs(tab_labels)
                    for tlabel, qcode, tab in zip(tab_labels, question_codes, tabs):
                        if qcode not in per_q_disp_tables:
                            continue
                        df_disp = per_q_disp_tables[qcode]
                        qtext = per_q_texts.get(qcode, "")
                        with tab:
                            st.subheader(f"{qcode} — {qtext}")
                            st.dataframe(df_disp, use_container_width=True)

                            # Non-AI summary
                            st.markdown("#### Summary (Positive only)")
                            summary = build_positive_only_narrative(df_disp, category_in_play)
                            st.write(summary)

                            # AI per-question narrative
                            if st.session_state.get("menu1_ai_toggle", False):
                                try:
                                    user_payload = _build_user_prompt_per_question(qcode, qtext, df_disp, category_in_play)
                                    content, hint = _call_openai_json(system=AI_SYSTEM_PROMPT, user=user_payload, model=OPENAI_MODEL, temperature=0.2)
                                    if content:
                                        try:
                                            j = json.loads(content)
                                            if isinstance(j, dict) and j.get("narrative"):
                                                st.markdown("#### AI analysis")
                                                st.write(j["narrative"])
                                            else:
                                                st.caption("AI returned no narrative.")
                                        except Exception:
                                            st.caption("AI returned non-JSON content.")
                                    else:
                                        st.caption(f"AI unavailable ({hint}).")
                                except Exception as e:
                                    st.caption(f"AI error: {type(e).__name__}")

                    # Summary matrix across questions
                    summary_rows = []
                    for qcode, df_disp in per_q_disp_tables.items():
                        t = df_disp.copy()
                        if "Demographic" in t.columns:
                            grp = t.groupby("Year", as_index=False)["Positive"].mean(numeric_only=True)
                        else:
                            grp = t[["Year", "Positive"]].copy()
                        grp["Question"] = qcode
                        grp["Year"] = pd.to_numeric(grp["Year"], errors="coerce").astype("Int64")
                        summary_rows.append(grp[["Question", "Year", "Positive"]])

                    if summary_rows:
                        summary_df = pd.concat(summary_rows, ignore_index=True)
                        pivot = summary_df.pivot_table(index="Question", columns="Year", values="Positive", aggfunc="mean")
                        pivot = pivot.reindex(index=[qc for qc in question_codes if qc in per_q_disp_tables])
                        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

                        st.markdown("### Summary matrix (% Positive)")
                        st.dataframe(pivot.round(1).reset_index(), use_container_width=True)

                        st.markdown("#### Across selected questions (descriptive)")
                        means = pivot.mean(numeric_only=True).round(1)
                        trend_txt = ", ".join([f"{int(y)}: {v:.1f}%" for y, v in zip(pivot.columns, means)])
                        st.write(f"Average % Positive across selected questions by year → {trend_txt}.")

                        # AI overall narrative
                        if st.session_state.get("menu1_ai_toggle", False):
                            try:
                                user_payload = _build_user_prompt_overall(question_codes, pivot)
                                content, hint = _call_openai_json(system=AI_SYSTEM_PROMPT, user=user_payload, model=OPENAI_MODEL, temperature=0.2)
                                if content:
                                    try:
                                        j = json.loads(content)
                                        if isinstance(j, dict) and j.get("narrative"):
                                            st.markdown("#### AI overall analysis")
                                            st.write(j["narrative"])
                                        else:
                                            st.caption("AI returned no narrative.")
                                    except Exception:
                                        st.caption("AI returned non-JSON content.")
                                else:
                                    st.caption(f"AI unavailable ({hint}).")
                            except Exception as e:
                                st.caption(f"AI error: {type(e).__name__}")

                        # Excel download
                        with io.BytesIO() as buf:
                            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                                pivot.round(1).reset_index().to_excel(writer, sheet_name="Summary_Matrix", index=False)
                                for qcode, df_disp in per_q_disp_tables.items():
                                    safe = qcode[:28]
                                    df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)
                                ctx = {
                                    "Questions": ", ".join(question_codes),
                                    "Years": ", ".join(map(str, selected_years)),
                                    "Category": demo_selection,
                                    "Subgroup": sub_selection or "(all in category)" if demo_selection != "All respondents" else "All respondents",
                                    "DEMCODEs used": ", ".join(["(blank)" if (c is None or str(c).strip() == "") else str(c) for c in demcodes]),
                                    "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                }
                                pd.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Context", index=False)
                            data = buf.getvalue()
                        st.download_button(
                            label="Download Excel (Summary + all tabs)",
                            data=data,
                            file_name=f"PSES_multiQ_{'-'.join(map(str, selected_years))}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
        with colB:
            if st.button("Reset all parameters"):
                reset_menu1_state()
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if SHOW_DEBUG:
            st.markdown("---")
            st.caption("Diagnostics")
            st.json({
                "selected_codes": st.session_state.get("menu1_selected_codes"),
                "kw_query": st.session_state.get("menu1_kw_query"),
                "hits_count": len(st.session_state.get("menu1_hits", [])),
                "ai_enabled": st.session_state.get("menu1_ai_toggle"),
                "in_memory_loaded": isinstance(_get_pswide_df(), pd.DataFrame),
            })


if __name__ == "__main__":
    run_menu1()
