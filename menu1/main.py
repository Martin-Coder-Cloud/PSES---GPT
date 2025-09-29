# menu1/main.py — PSES Explorer Search (Menu 1)
# - Hybrid keyword search + dropdown multi-select (max 5) with visible “Selected questions”
# - Diagnostics panel toggle (top, next to AI toggle): Parameters preview • App setup status • Last query
# - Tabs:
#     1) Summary table (rows = Question # only or Question×Demographic; cols = selected Years; values = per-question metric)
#     2+) One tab per question with detailed distribution
# - AI:
#     • Per-question: one-paragraph "Summary Analysis"
#     • Overall: "Overall Summary Analysis" (when >1 question)
# - Excel export (Summary + each Q)
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
# Data loader (repo-provided)
# -----------------------------
try:
    from utils.data_loader import load_results2024_filtered  # main query function
    HAVE_LOADER = True
except Exception:
    load_results2024_filtered = None  # type: ignore
    HAVE_LOADER = False

# Optional backend info / preload hooks (best-effort)
try:
    from utils.data_loader import get_backend_info  # type: ignore
except Exception:
    def get_backend_info() -> dict:
        # Minimum info when advanced telemetry isn't available
        return {"engine": "csv.gz", "in_memory": False}

try:
    from utils.data_loader import preload_pswide_dataframe  # type: ignore
except Exception:
    def preload_pswide_dataframe():
        return None

try:
    import utils.data_loader as _dl  # type: ignore
except Exception:
    _dl = None  # type: ignore

# Hybrid search (module you created earlier); fallback if missing
try:
    from utils.hybrid_search import hybrid_question_search  # type: ignore
except Exception:
    def hybrid_question_search(qdf: pd.DataFrame, query: str, top_k: int = 120, min_score: float = 0.40) -> pd.DataFrame:
        """Simple fallback: case-insensitive substring + token overlap scorer."""
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

# -----------------------------
# OpenAI (best-effort; wrapped)
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")  # override in secrets if needed

def _call_openai_json(system: str, user: str, model: str = OPENAI_MODEL, temperature: float = 0.2, max_retries: int = 2):
    """Return (json_text, error_hint). Never throws."""
    if not OPENAI_API_KEY:
        return "", "no_api_key"
    # Prefer new SDK if present
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
                return content, None
            except Exception as e:
                hint = f"openai_err_{attempt+1}: {type(e).__name__}"
                time.sleep(0.8 * (attempt + 1))
        return "", hint
    except Exception:
        # Legacy package fallback
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
                    return content, None
                except Exception as e:
                    hint = f"openai_legacy_err_{attempt+1}: {type(e).__name__}"
                    time.sleep(0.8 * (attempt + 1))
            return "", hint
        except Exception:
            return "", "no_openai_sdk"

# -----------------------------
# AI system prompt (base rules)
# -----------------------------
AI_SYSTEM_PROMPT = (
    "You are preparing insights for the Government of Canada’s Public Service Employee Survey (PSES).\n\n"
    "Context\n"
    "- The PSES provides information to improve people management practices in the federal public service.\n"
    "- Results help departments and agencies identify strengths and concerns in areas such as employee engagement, anti-racism, equity and inclusion, and workplace well-being.\n"
    "- The survey tracks progress over time to refine action plans. Employees’ voices guide improvements to workplace quality.\n"
    "- Public Service–wide scope only.\n\n"
    "Data-use rules (hard constraints)\n"
    "- Use ONLY the provided JSON payload/table. Do NOT invent or infer beyond it.\n"
    "- Express percentages as whole numbers (e.g., “75%”). Use “points” for differences/changes.\n\n"
    "Analysis rules\n"
    "- Start from the latest year provided (typically 2024) for context.\n"
    "- Trend: compare the latest with the earliest year in the series (stable ≤1 pt; slight >1–2 pts; notable >2 pts).\n"
    "- Demographic gaps (latest year): classify size (minimal ≤2; notable >2–5; important >5) and indicate whether gaps widened/narrowed vs. earlier.\n"
    "- Style: professional, concise, neutral.\n"
    "- Output VALID JSON with exactly one key: \"narrative\".\n"
)

# -----------------------------
# Cached metadata
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_demographics() -> pd.DataFrame:
    df = pd.read_excel("metadata/Demographics.xlsx")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def _load_questions() -> pd.DataFrame:
    qdf = pd.read_excel("metadata/Survey Questions.xlsx")
    qdf.columns = [c.strip().lower() for c in qdf.columns]
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["code"] = qdf["code"].astype(str)
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    # Keep existing en dash in display (no change to selection behaviour)
    qdf["display"] = qdf["code"].astype(str) + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]

@st.cache_data(show_spinner=False)
def _load_scales() -> pd.DataFrame:
    sdf = pd.read_excel("metadata/Survey Scales.xlsx")
    sdf.columns = [c.strip().lower() for c in sdf.columns]
    return sdf

# -----------------------------
# Helpers (demographics / display / summary)
# -----------------------------
def _resolve_demcodes(demo_df: pd.DataFrame, category_label: str, subgroup_label: Optional[str]):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    # overall
    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    # find code column
    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    # single subgroup chosen
    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            r = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not r.empty:
                code = str(r.iloc[0][code_col])
                return [code], {code: subgroup_label}, True
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # no subgroup -> take all codes in category
    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    # fallback
    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        return labels, {l: l for l in labels}, True

    return [None], {None: "All respondents"}, False

def _get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    sdf = scales_df.copy()
    candidates = pd.DataFrame()
    for key in ["code", "question"]:
        if key in sdf.columns:
            candidates = sdf[sdf[key].astype(str).str.upper() == str(question_code).upper()]
            if not candidates.empty:
                break
    pairs = []
    for i in range(1, 8):
        col = f"answer{i}"
        lbl = None
        if not candidates.empty and col in candidates.columns:
            vals = candidates[col].dropna().astype(str)
            if not vals.empty:
                lbl = vals.iloc[0].strip()
        pairs.append((col, lbl or f"Answer {i}"))
    return pairs

def _drop_999(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in [f"answer{i}" for i in range(1, 8)] + ["POSITIVE","NEUTRAL","NEGATIVE","ANSCOUNT","positive_pct","neutral_pct","negative_pct","n"]:
        if c in out.columns:
            v = pd.to_numeric(out[c], errors="coerce")
            out.loc[v.isin([999, 9999]), c] = pd.NA
    return out

def _normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # QUESTION -> question_code
    if "question_code" not in out.columns:
        if "QUESTION" in out.columns:
            out = out.rename(columns={"QUESTION": "question_code"})
        else:
            for c in out.columns:
                if c.strip().lower() == "question":
                    out = out.rename(columns={c: "question_code"}); break
    # SURVEYR -> year
    if "year" not in out.columns:
        if "SURVEYR" in out.columns:
            out = out.rename(columns={"SURVEYR": "year"})
        else:
            for c in out.columns:
                if c.strip().lower() in ("surveyr","year"):
                    out = out.rename(columns={c: "year"}); break
    # DEMCODE -> group_value
    if "group_value" not in out.columns:
        if "DEMCODE" in out.columns:
            out = out.rename(columns={"DEMCODE": "group_value"})
        else:
            for c in out.columns:
                if c.strip().lower() == "demcode":
                    out = out.rename(columns={c: "group_value"}); break
    # POS/NEU/NEG rename
    if "positive_pct" not in out.columns and "POSITIVE" in out.columns:
        out = out.rename(columns={"POSITIVE": "positive_pct"})
    if "neutral_pct" not in out.columns and "NEUTRAL" in out.columns:
        out = out.rename(columns={"NEUTRAL": "neutral_pct"})
    if "negative_pct" not in out.columns and "NEGATIVE" in out.columns:
        out = out.rename(columns={"NEGATIVE": "negative_pct"})
    if "n" not in out.columns and "ANSCOUNT" in out.columns:
        out = out.rename(columns={"ANSCOUNT": "n"})
    return out

def _format_display(df_slice: pd.DataFrame, dem_disp_map: Dict, category_in_play: bool, scale_pairs: List[Tuple[str,str]]) -> pd.DataFrame:
    if df_slice.empty:
        return df_slice.copy()
    out = df_slice.copy()
    out["YearNum"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["Year"] = out["YearNum"].astype(str)

    if category_in_play:
        def to_label(code):
            if code is None or (isinstance(code, float) and pd.isna(code)) or str(code).strip() == "":
                return "All respondents"
            return dem_disp_map.get(code, dem_disp_map.get(str(code), str(code)))
        out["Demographic"] = out["group_value"].apply(to_label)

    # map answer labels
    dist_cols = [k for k,_ in scale_pairs if k in out.columns]
    rename_map = {k: v for k, v in scale_pairs if k in out.columns}

    keep_cols = ["YearNum","Year"] + (["Demographic"] if category_in_play else []) \
                + dist_cols + ["positive_pct","neutral_pct","negative_pct","n"]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()
    out = out.rename(columns={"positive_pct":"Positive","neutral_pct":"Neutral","negative_pct":"Negative"})

    # sort: Year desc, Demographic asc
    sort_cols = ["YearNum"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["YearNum"])

    # numerics
    for c in out.columns:
        if c not in ("Year","Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year","Demographic","n")]
    if pct_like:
        out[pct_like] = out[pct_like].round(1)
    if "n" in out.columns:
        out["n"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")
    return out

def _detect_metric_mode(df_disp: pd.DataFrame, scale_pairs) -> dict:
    """Choose the metric to summarize in the summary table: Positive → Agree → first answer label."""
    cols_l = {c.lower(): c for c in df_disp.columns}
    if "positive" in cols_l and pd.to_numeric(df_disp[cols_l["positive"]], errors="coerce").notna().any():
        return {"metric_col": cols_l["positive"], "metric_label": "% positive"}
    if "agree" in cols_l and pd.to_numeric(df_disp[cols_l["agree"]], errors="coerce").notna().any():
        return {"metric_col": cols_l["agree"], "metric_label": "% agree"}
    if scale_pairs:
        for _, label in scale_pairs:
            if label and label in df_disp.columns and pd.to_numeric(df_disp[label], errors="coerce").notna().any():
                return {"metric_col": label, "metric_label": f"% {label}"}
    return {"metric_col": cols_l.get("positive", "Positive"), "metric_label": "% positive"}

# -----------------------------
# Constants: Source link
# -----------------------------
SOURCE_URL = "https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f"
SOURCE_TITLE = "Public Service Employee Survey Results – Open Government Portal"

def _render_source_caption():
    # Show URL and clickable title, placed directly under tabulations (before AI summaries)
    st.caption(f"Source: {SOURCE_URL} — [{SOURCE_TITLE}]({SOURCE_URL})")

# -----------------------------
# State reset
# -----------------------------
def _delete_keys(prefixes: List[str], exact_keys: List[str] = None):
    exact_keys = exact_keys or []
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in prefixes) or (k in exact_keys):
            try: del st.session_state[k]
            except Exception: pass

def _reset_menu1_state():
    # Important: only set defaults here; do not modify after widgets render
    year_keys = [f"year_{y}" for y in (2024, 2022, 2020, 2019)]
    exact = [
        "menu1_selected_codes","menu1_hits","menu1_kw_query","menu1_multi_questions",
        "menu1_ai_toggle","menu1_show_diag","select_all_years","demo_main","menu1_find_hits",
        "last_query_info",
        # results stash keys (for centered rendering)
        "m1_has_results","m1_tab_labels","m1_pivot","m1_per_q_disp",
        "m1_per_q_metric_col","m1_per_q_metric_label","m1_code_to_text",
        "m1_selected_years","m1_demo_selection","m1_sub_selection"
    ] + year_keys
    prefixes = ["kwhit_","sel_","sub_"]
    _delete_keys(prefixes, exact)
    st.session_state.setdefault("menu1_kw_query", "")
    st.session_state.setdefault("menu1_hits", [])
    st.session_state.setdefault("menu1_selected_codes", [])
    st.session_state.setdefault("menu1_multi_questions", [])
    st.session_state.setdefault("menu1_ai_toggle", True)     # default ON
    st.session_state.setdefault("menu1_show_diag", False)    # default OFF
    st.session_state.setdefault("last_query_info", None)
    # results flags
    st.session_state["m1_has_results"] = False

# -----------------------------
# AI payload builders
# -----------------------------
def _series_json(df_disp: pd.DataFrame, metric_col: str) -> List[Dict[str, float]]:
    rows = []
    s = df_disp.copy()
    if "Demographic" in s.columns:
        s = s.groupby("Year", as_index=False)[metric_col].mean(numeric_only=True)
        s = s.rename(columns={metric_col: "Metric"})
    else:
        s = s[["Year", metric_col]].rename(columns={metric_col: "Metric"})
    s = s.dropna(subset=["Year"]).sort_values("Year")
    for _, r in s.iterrows():
        try: y = int(r["Year"])
        except Exception: y = r["Year"]
        rows.append({"year": y, "value": float(r["Metric"]) if pd.notna(r["Metric"]) else None})
    return rows

def _user_prompt_per_q(qcode: str, qtext: str, df_disp: pd.DataFrame, metric_col: str, metric_label: str, category_in_play: bool) -> str:
    latest = pd.to_numeric(df_disp["Year"], errors="coerce").max()
    group_info = []
    if category_in_play and "Demographic" in df_disp.columns and pd.notna(latest):
        g = df_disp[pd.to_numeric(df_disp["Year"], errors="coerce") == latest][["Demographic", metric_col]].dropna()
        g = g.sort_values(metric_col, ascending=False)
        if not g.empty:
            top = g.iloc[0].to_dict(); bot = g.iloc[-1].to_dict()
            group_info = [
                {"demographic": str(top["Demographic"]), "value": float(top[metric_col])},
                {"demographic": str(bot["Demographic"]), "value": float(bot[metric_col])},
            ]
    payload = {
        "question_code": qcode,
        "question_text": qtext,
        "metric_label": metric_label,
        "series_positive_by_year": _series_json(df_disp, metric_col),  # keep field name for compatibility
        "latest_year_group_snapshot": group_info,
        "output_style_hint": "Return exactly one short paragraph.",
    }
    # Explicit instruction appended for one-paragraph output
    return json.dumps(payload, ensure_ascii=False) + "\n\nReturn valid JSON with key 'narrative' containing ONE short paragraph."

def _user_prompt_overall(q_labels: List[str], pivot: pd.DataFrame, q_to_metric: Dict[str, str]) -> str:
    items = []
    for q in pivot.index.tolist():
        row = {"question_label": str(q), "metric_label": q_to_metric.get(q, "% positive"), "values_by_year": {}}
        for y in pivot.columns.tolist():
            val = pivot.loc[q, y]
            if pd.notna(val):
                row["values_by_year"][int(y)] = float(val)
        items.append(row)
    notes = "Synthesize overall pattern across questions using each question's metric_label."
    return json.dumps({"questions": items, "notes": notes}, ensure_ascii=False)

# -----------------------------
# Small utils
# -----------------------------
def _get_pswide_df() -> Optional[pd.DataFrame]:
    try:
        df = preload_pswide_dataframe()
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    try:
        if _dl and hasattr(_dl, "pswide_df"):
            df = getattr(_dl, "pswide_df")
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        pass
    return None

# -----------------------------
# UI
# -----------------------------
def run_menu1():
    st.markdown("""
        <style>
            body { background-image: none !important; background-color: white !important; }
            .block-container { padding-top: 1rem !important; }
            .menu-banner { width: 100%; height: auto; display: block; margin-top: 0px; margin-bottom: 20px; }
            .custom-header { font-size: 30px !important; font-weight: 700; margin-bottom: 6px; }
            .custom-instruction { font-size: 16px !important; line-height: 1.4; margin-bottom: 10px; color: #333; }
            .field-label { font-size: 18px !important; font-weight: 600 !important; margin-top: 12px !important; margin-bottom: 2px !important; color: #222 !important; }
            .action-row { display:flex; gap:10px; align-items:center; }
            [data-testid="stSwitch"] div[role="switch"][aria-checked="true"] { background-color: #e03131 !important; }
            [data-testid="stSwitch"] div[role="switch"] { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1); }
            .tiny-note { font-size: 13px; color: #444; margin-bottom: 6px; }
            .diag-box { background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 10px 12px; }
        </style>
    """, unsafe_allow_html=True)

    # Auto-reset when arriving here from another menu
    if st.session_state.get("last_active_menu") != "menu1":
        _reset_menu1_state()
    st.session_state["last_active_menu"] = "menu1"

    # Defaults BEFORE creating widgets (avoids streamlit value/session_state conflict)
    st.session_state.setdefault("menu1_selected_codes", [])
    st.session_state.setdefault("menu1_hits", [])
    st.session_state.setdefault("menu1_kw_query", "")
    st.session_state.setdefault("menu1_multi_questions", [])
    st.session_state.setdefault("menu1_ai_toggle", True)
    st.session_state.setdefault("menu1_show_diag", False)
    st.session_state.setdefault("last_query_info", None)
    st.session_state.setdefault("m1_has_results", False)

    demo_df = _load_demographics()
    qdf = _load_questions()
    sdf = _load_scales()

    code_to_text = dict(zip(qdf["code"], qdf["text"]))
    code_to_display = dict(zip(qdf["code"], qdf["display"]))
    display_to_code = {v: k for k, v in code_to_display.items()}

    left, center, right = st.columns([1, 3, 1])
    with center:
        # Banner
        st.markdown(
            "<img class='menu-banner' src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/refs/heads/main/PSES%20email%20banner.png'>",
            unsafe_allow_html=True
        )

        # Title
        st.markdown('<div class="custom-header">PSES Explorer Search</div>', unsafe_allow_html=True)

        # Row of toggles (AI + Diagnostics)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.toggle("🧠 Enable AI analysis", key="menu1_ai_toggle", help="Include the AI-generated analysis alongside the tables.")
        with c2:
            st.toggle("🔧 Show technical parameters & diagnostics", key="menu1_show_diag", help="Show current parameters, app setup status, and last query timings.")

        # Instructions
        st.markdown("""
            <div class="custom-instruction">
                Please use this menu to explore the survey results by questions.<br>
                You may select from the drop down menu below up to five questions or find questions via the keyword/theme search.
                Select year(s) and optionally a demographic category and subgroup.
            </div>
        """, unsafe_allow_html=True)

        # ---------- Diagnostics (optional) ----------
        if st.session_state.get("menu1_show_diag", False):
            with st.container(border=False):
                st.markdown("#### Parameters preview")
                preview = {
                    "Selected questions": [code_to_display.get(c, c) for c in st.session_state.get("menu1_selected_codes", [])],
                    "Years (selected)": [y for y in [2019, 2020, 2022, 2024] if st.session_state.get(f"year_{y}", True if st.session_state.get("select_all_years", True) else False)],
                    "Demographic category": st.session_state.get("demo_main", "All respondents"),
                }
                demo_cat = preview["Demographic category"]
                if demo_cat and demo_cat != "All respondents":
                    subkey = f"sub_{str(demo_cat).replace(' ', '_')}"
                    preview["Subgroup"] = st.session_state.get(subkey, "")
                else:
                    preview["Subgroup"] = "All respondents"

                st.markdown(f"<div class='diag-box'><pre>{json.dumps(preview, ensure_ascii=False, indent=2)}</pre></div>", unsafe_allow_html=True)

                st.markdown("#### App setup status")
                info = {}
                try:
                    info = get_backend_info() or {}
                except Exception:
                    info = {"engine": "csv.gz"}
                try:
                    df_ps = _get_pswide_df()
                    if isinstance(df_ps, pd.DataFrame) and not df_ps.empty:
                        ycol = "year" if "year" in df_ps.columns else ("SURVEYR" if "SURVEYR" in df_ps.columns else None)
                        qcol = "question_code" if "question_code" in df_ps.columns else ("QUESTION" if "QUESTION" in df_ps.columns else None)
                        yr_min = int(pd.to_numeric(df_ps[ycol], errors="coerce").min()) if ycol else None
                        yr_max = int(pd.to_numeric(df_ps[ycol], errors="coerce").max()) if ycol else None
                        info.update({
                            "in_memory": True,
                            "pswide_rows": int(len(df_ps)),
                            "unique_questions": int(df_ps[qcol].astype(str).nunique()) if qcol else None,
                            "year_range": f"{yr_min}–{yr_max}" if yr_min and yr_max else None,
                        })
                    else:
                        info.update({"in_memory": False})
                except Exception:
                    pass
                try: info["metadata_questions"] = int(len(qdf))
                except Exception: pass
                try: info["metadata_scales"] = int(len(sdf))
                except Exception: pass
                try: info["metadata_demographics"] = int(len(demo_df))
                except Exception: pass

                st.markdown(f"<div class='diag-box'><pre>{json.dumps(info, ensure_ascii=False, indent=2)}</pre></div>", unsafe_allow_html=True)

                st.markdown("#### Last query")
                last = st.session_state.get("last_query_info") or {"status": "No query yet"}
                st.markdown(f"<div class='diag-box'><pre>{json.dumps(last, ensure_ascii=False, indent=2)}</pre></div>", unsafe_allow_html=True)

        # ---------- Question selection ----------
        st.markdown('<div class="field-label">Pick up to 5 survey questions:</div>', unsafe_allow_html=True)

        # 1) Dropdown multi-select (authoritative)
        all_displays = qdf["display"].tolist()
        multi_choices = st.multiselect(
            "Choose one or more from the official list",
            all_displays,
            default=st.session_state.get("menu1_multi_questions", []),
            max_selections=5,
            label_visibility="collapsed",
            key="menu1_multi_questions",
        )
        selected_from_multi: Set[str] = set(display_to_code[d] for d in multi_choices if d in display_to_code)

        # 2) Hybrid search (expander)
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

        # Merge ordered selections (dropdown first, then search hits), cap at 5
        combined_order: List[str] = []
        for d in st.session_state.get("menu1_multi_questions", []):
            c = display_to_code.get(d)
            if c and c not in combined_order:
                combined_order.append(c)
        for c in selected_from_hits:
            if c not in combined_order:
                combined_order.append(c)
        if len(combined_order) > 5:
            combined_order = combined_order[:5]
            st.warning("Limit is 5 questions; extra selections were ignored.")
        st.session_state["menu1_selected_codes"] = combined_order

        # Selected list with checkboxes (quick unselect)
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
                        if hk in st.session_state: st.session_state[hk] = False
                        disp = code_to_display.get(code)
                        if disp:
                            st.session_state["menu1_multi_questions"] = [d for d in st.session_state["menu1_multi_questions"] if d != disp]
            if updated != st.session_state["menu1_selected_codes"]:
                st.session_state["menu1_selected_codes"] = updated

        question_codes: List[str] = st.session_state["menu1_selected_codes"]

        # ---------- Years ----------
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        all_years = [2024, 2022, 2020, 2019]
        st.session_state.setdefault("select_all_years", True)
        select_all = st.checkbox("All years", key="select_all_years")
        selected_years: List[int] = []
        year_cols = st.columns(len(all_years))
        for idx, yr in enumerate(all_years):
            with year_cols[idx]:
                default_checked = True if select_all else st.session_state.get(f"year_{yr}", False)
                if st.checkbox(str(yr), value=default_checked, key=f"year_{yr}"):
                    selected_years.append(yr)
        selected_years = sorted(selected_years)

        # ---------- Demographics ----------
        st.markdown('<div class="field-label">Select a demographic category (optional):</div>', unsafe_allow_html=True)
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
        demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
        st.session_state.setdefault("demo_main", "All respondents")
        demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

        sub_selection = None
        if demo_selection != "All respondents":
            st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
            sub_items = demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL].dropna().astype(str).unique().tolist()
            sub_items = sorted(sub_items)
            sub_key = f"sub_{demo_selection.replace(' ', '_')}"
            sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=sub_key, label_visibility="collapsed")
            if sub_selection == "":
                sub_selection = None

        # ---------- Action row ----------
        st.markdown("<div class='action-row'>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1])

        # Only the SEARCH/RESET buttons remain in the split row; results are rendered below, full center width.
        with colA:
            disable_search = (not question_codes) or (not selected_years)
            if st.button("Search", disabled=disable_search):
                t0 = time.time()
                with st.spinner(f"Running query… {datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')}"):
                    # Resolve DEMCODEs
                    demcodes, disp_map, category_in_play = _resolve_demcodes(demo_df, demo_selection, sub_selection)

                    per_q_disp: Dict[str, pd.DataFrame] = {}
                    per_q_text: Dict[str, str] = {}
                    per_q_metric_col: Dict[str, str] = {}
                    per_q_metric_label: Dict[str, str] = {}

                    for qcode in question_codes:
                        qtext = code_to_text.get(qcode, "")
                        per_q_text[qcode] = qtext

                        # Pull parts per DEMCODE (None/"All" means overall)
                        parts = []
                        if load_results2024_filtered is None:
                            continue
                        for code in demcodes:
                            df_part = load_results2024_filtered(
                                question_code=qcode,
                                years=selected_years,
                                group_value=(None if code in (None, "", "All") else str(code))
                            )
                            if df_part is not None and not df_part.empty:
                                parts.append(df_part)
                        if not parts:
                            continue
                        df_all = pd.concat(parts, ignore_index=True)

                        # Normalize + clean
                        df_all = _normalize_results(df_all)
                        # Guard filters
                        qmask = df_all["question_code"].astype(str).str.strip().str.upper() == str(qcode).strip().upper()
                        ymask = pd.to_numeric(df_all["year"], errors="coerce").astype("Int64").isin(selected_years)
                        if demo_selection == "All respondents":
                            gv = df_all["group_value"].astype(str).fillna("").str.strip()
                            gmask = gv.isin(["", "All", "ALL", "All respondents", "ALL RESPONDENTS"])
                        else:
                            gmask = df_all["group_value"].astype(str).isin([str(c) for c in demcodes])
                        df_all = df_all[qmask & ymask & gmask].copy()
                        df_all = _drop_999(df_all)
                        if df_all.empty:
                            continue

                        # Build display table with scale labels
                        scale_pairs = _get_scale_labels(sdf, qcode)
                        df_disp = _format_display(
                            df_slice=df_all,
                            dem_disp_map=disp_map,
                            category_in_play=category_in_play,
                            scale_pairs=scale_pairs
                        )
                        if df_disp.empty:
                            continue

                        # Detect metric per question (Positive → Agree → first answer label)
                        det = _detect_metric_mode(df_disp, scale_pairs)
                        per_q_metric_col[qcode] = det["metric_col"]
                        per_q_metric_label[qcode] = det["metric_label"]

                        per_q_disp[qcode] = df_disp

                t1 = time.time()
                st.session_state["last_query_info"] = {
                    "started": datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
                    "finished": datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": round(t1 - t0, 2),
                    "engine": (get_backend_info() or {}).get("engine", "unknown"),
                }

                # --------- STASH RESULTS FOR CENTERED RENDERING ---------
                if per_q_disp:
                    long_rows = []
                    tab_labels = [qc for qc in question_codes if qc in per_q_disp]
                    for qcode in tab_labels:
                        df_disp = per_q_disp[qcode]
                        metric_col = per_q_metric_col[qcode]
                        # Summary table row label must be the Question # only (no text)
                        qlabel = f"{qcode}"

                        t = df_disp.copy()
                        t["QuestionLabel"] = qlabel
                        t["Year"] = pd.to_numeric(t["Year"], errors="coerce").astype("Int64")
                        if "Demographic" not in t.columns:
                            t["Demographic"] = None

                        if metric_col not in t.columns:
                            continue
                        t = t.rename(columns={metric_col: "Value"})
                        long_rows.append(t[["QuestionLabel","Demographic","Year","Value"]])

                    if long_rows:
                        long_df = pd.concat(long_rows, ignore_index=True)
                        # Index: Question×Demographic when category selected (no single subgroup); else Question
                        if (demo_selection != "All respondents") and (sub_selection is None) and long_df["Demographic"].notna().any():
                            idx_cols = ["QuestionLabel","Demographic"]
                        else:
                            idx_cols = ["QuestionLabel"]
                        pivot = long_df.pivot_table(index=idx_cols, columns="Year", values="Value", aggfunc="mean")
                        pivot = pivot.reindex(selected_years, axis=1)

                        # Save everything needed to render results
                        st.session_state["m1_has_results"] = True
                        st.session_state["m1_tab_labels"] = tab_labels
                        st.session_state["m1_pivot"] = pivot
                        st.session_state["m1_per_q_disp"] = per_q_disp
                        st.session_state["m1_per_q_metric_col"] = per_q_metric_col
                        st.session_state["m1_per_q_metric_label"] = per_q_metric_label
                        st.session_state["m1_code_to_text"] = code_to_text
                        st.session_state["m1_selected_years"] = selected_years
                        st.session_state["m1_demo_selection"] = demo_selection
                        st.session_state["m1_sub_selection"] = sub_selection
                    else:
                        st.session_state["m1_has_results"] = False
                else:
                    st.session_state["m1_has_results"] = False

        with colB:
            if st.button("Reset all parameters"):
                _reset_menu1_state()
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # CENTERED RESULTS RENDERING
        # =========================
        if st.session_state.get("m1_has_results", False):
            tab_labels = st.session_state["m1_tab_labels"]
            pivot = st.session_state["m1_pivot"]
            per_q_disp = st.session_state["m1_per_q_disp"]
            per_q_metric_col = st.session_state["m1_per_q_metric_col"]
            per_q_metric_label = st.session_state["m1_per_q_metric_label"]
            code_to_text_local = st.session_state["m1_code_to_text"]
            selected_years = st.session_state["m1_selected_years"]
            demo_selection = st.session_state["m1_demo_selection"]
            sub_selection = st.session_state["m1_sub_selection"]

            # Tabs: Summary first, then per-Q
            tabs = st.tabs(["Summary table"] + tab_labels)

            # Summary tab (Question # only)
            with tabs[0]:
                st.markdown("### Summary table")
                st.dataframe(pivot.round(1).reset_index(), use_container_width=True)
                # Source directly under the table (before AI)
                _render_source_caption()

                # Overall Summary Analysis (when multi-question)
                if len(tab_labels) > 1 and st.session_state.get("menu1_ai_toggle", True):
                    with st.spinner("Generating Overall Summary Analysis…"):
                        # Mapping from Question # -> metric label (for context)
                        q_to_metric = {q: per_q_metric_label[q] for q in tab_labels}
                        content, hint = _call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=_user_prompt_overall(tab_labels, pivot, q_to_metric),
                            model=OPENAI_MODEL,
                            temperature=0.2
                        )
                    if content:
                        try:
                            j = json.loads(content)
                            if isinstance(j, dict) and j.get("narrative"):
                                st.markdown("### Overall Summary Analysis")
                                st.write(j["narrative"])
                                st.caption(f"Generated by OpenAI • model: {OPENAI_MODEL}")
                            else:
                                st.caption("AI returned no narrative.")
                        except Exception:
                            st.caption("AI returned non-JSON content.")
                    else:
                        st.caption(f"AI unavailable ({hint}).")
                elif len(tab_labels) <= 1:
                    st.info("Select more than one question to see an Overall Summary Analysis.")

            # Per-question tabs
            for idx, qcode in enumerate(tab_labels, start=1):
                with tabs[idx]:
                    qtext = code_to_text_local.get(qcode, "")
                    st.subheader(f"{qcode} — {qtext}")
                    st.dataframe(per_q_disp[qcode], use_container_width=True)
                    # Source directly under the table (before AI)
                    _render_source_caption()

                    metric_col = per_q_metric_col[qcode]
                    metric_label = per_q_metric_label[qcode]

                    # Per-question AI Summary Analysis (one short paragraph)
                    if st.session_state.get("menu1_ai_toggle", True):
                        with st.spinner("Generating Summary Analysis…"):
                            content, hint = _call_openai_json(
                                system=AI_SYSTEM_PROMPT,
                                user=_user_prompt_per_q(qcode, qtext, per_q_disp[qcode], metric_col, metric_label, (demo_selection != "All respondents")),
                                model=OPENAI_MODEL,
                                temperature=0.2
                            )
                        if content:
                            try:
                                j = json.loads(content)
                                if isinstance(j, dict) and j.get("narrative"):
                                    st.markdown("### Summary Analysis")
                                    st.write(j["narrative"])  # expected single paragraph
                                    st.caption(f"Generated by OpenAI • model: {OPENAI_MODEL}")
                                else:
                                    st.caption("AI returned no narrative.")
                            except Exception:
                                st.caption("AI returned non-JSON content.")
                        else:
                            st.caption(f"AI unavailable ({hint}).")
                    else:
                        st.info("No AI summary generated.")

            # -----------------------------------
            # Excel export: Summary + each Q
            # -----------------------------------
            with io.BytesIO() as buf:
                with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                    pivot.round(1).reset_index().to_excel(writer, sheet_name="Summary_Table", index=False)
                    for q, df_disp in per_q_disp.items():
                        safe = q[:28]
                        df_disp.to_excel(writer, sheet_name=f"{safe}", index=False)
                    ctx = {
                        "Questions": ", ".join(tab_labels),
                        "Years": ", ".join(map(str, selected_years)),
                        "Category": demo_selection,
                        "Subgroup": sub_selection or "(all in category)" if demo_selection != "All respondents" else "All respondents",
                        "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    pd.DataFrame(list(ctx.items()), columns=["Field","Value"]).to_excel(writer, sheet_name="Context", index=False)
                data = buf.getvalue()
            st.download_button(
                label="Download Excel (Summary + all tabs)",
                data=data,
                file_name=f"PSES_multiQ_{'-'.join(map(str, selected_years))}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

if __name__ == "__main__":
    run_menu1()
