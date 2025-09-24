# menu2/main.py — PSES AI Explorer (Menu 2: Theme/Keyword search → checkbox multi-select)
# Key features:
#   • Strict thresholded semantic/hybrid search (no top-K caps).
#   • Query expansion (synonyms) + hybrid gate for precision.
#   • Gated flow: Search → Pick questions → Pick Years & Demographics → Busy → Ready.
#   • Checkbox multi-select (no dropdown).
#   • AI analysis ON by default; per-question + overall theme.
#   • Updated summary-analysis system prompt (as provided).
#   • Cross-question summary helper included.

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# Loader
import utils.data_loader as _dl
try:
    from utils.data_loader import load_results2024_filtered
except Exception:
    from utils.data_loader import load_results2024_filtered  # type: ignore

try:
    from utils.data_loader import get_backend_info, prewarm_fastpath
except Exception:
    def get_backend_info(): return {}
    def prewarm_fastpath(): return "csv"

# OpenAI key from Streamlit secrets
os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

SHOW_DEBUG = False
PD = pd


# ─────────────────────────────
# Metadata loaders
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_demographics_metadata() -> pd.DataFrame:
    df = pd.read_excel("metadata/Demographics.xlsx")
    df.columns = [c.strip() for c in df.columns]
    return df

def _tokenize(text: str) -> List[str]:
    if text is None:
        return []
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9\s\-_/&']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split(" ")

@st.cache_data(show_spinner=False)
def load_questions_metadata() -> pd.DataFrame:
    qdf = pd.read_excel("metadata/Survey Questions.xlsx")
    qdf.columns = qdf.columns.str.strip().str.lower()
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["code"] = qdf["code"].astype(str).str.strip()
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"] + " – " + qdf["text"].astype(str)
    # Search helpers
    qdf["__norm__"] = (qdf["code"].astype(str) + " " + qdf["text"].astype(str)).str.lower()
    qdf["__tokens__"] = qdf["__norm__"].apply(_tokenize)
    return qdf[["code", "text", "display", "__norm__", "__tokens__"]]

def _normalize_qcode(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.upper()
    return "".join(ch for ch in s if ch.isalnum())

@st.cache_data(show_spinner=False)
def load_scales_metadata() -> pd.DataFrame:
    primary = "metadata/Survey Scales.xlsx"
    fallback = "/mnt/data/Survey Scales.xlsx"
    path = primary if os.path.exists(primary) else fallback
    sdf = pd.read_excel(path)
    sdf.columns = sdf.columns.str.strip().str.lower()
    code_col = None
    for c in ("code", "question"):
        if c in sdf.columns:
            code_col = c
            break
    if code_col is None:
        return sdf
    sdf["__code_norm__"] = sdf[code_col].astype(str).map(_normalize_qcode)
    return sdf


# ─────────────────────────────
# Helpers from Menu 1 (unchanged behavior)
# ─────────────────────────────
def _find_demcode_col(demo_df: pd.DataFrame) -> str | None:
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            return c
    return None

def _four_digit(s: str) -> str:
    s = "".join(ch for ch in str(s) if s is not None and s.isdigit())
    return s.zfill(4) if s else ""

def resolve_demographic_codes_from_metadata(demo_df, category_label, subgroup_label):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    code_col = _find_demcode_col(demo_df)

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                raw_code = str(row.iloc[0][code_col])
                code4 = _four_digit(raw_code)
                code_final = code4 if code4 else raw_code
                return [code_final.strip()], {code_final.strip(): subgroup_label}, True
        return [str(subgroup_label).strip()], {str(subgroup_label).strip(): subgroup_label}, True

    if code_col and LABEL_COL in df_cat.columns:
        pairs = []
        for _, r in df_cat.iterrows():
            raw_code = str(r[code_col]); label = str(r[LABEL_COL])
            code4 = _four_digit(raw_code)
            if code4:
                pairs.append((code4.strip(), label))
        if pairs:
            demcodes = [c for c, _ in pairs]
            disp_map = {c: l for c, l in pairs}
            return demcodes, disp_map, True

    if LABEL_COL in df_cat.columns:
        labels = [str(l).strip() for l in df_cat[LABEL_COL].tolist()]
        return labels, {l: l for l in labels}, True

    return [None], {None: "All respondents"}, False

def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    if scales_df is None or scales_df.empty:
        return None
    qnorm = _normalize_qcode(question_code)
    if "__code_norm__" not in scales_df.columns:
        return None
    match = scales_df[sdf := (scales_df["__code_norm__"] == qnorm)]
    if match.empty:
        return None
    row = match.iloc[0]
    pairs = []
    for i in range(1, 7 + 1):
        col = f"answer{i}"
        if col in scales_df.columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                pairs.append((col, str(val).strip()))
    return pairs if pairs else None

def exclude_999_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    candidates = [f"answer{i}" for i in range(1, 7 + 1)] + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE", "YES",
                                                             "positive_pct", "neutral_pct", "negative_pct", "n"]
    present = [c for c in candidates if c in out.columns]
    for c in present:
        s = out[c].astype(str).str.strip()
        mask = (s == "999") | (s == "9999")
        mask |= pd.to_numeric(out[c], errors="coerce").isin([999, 9999])
        out.loc[mask, c] = pd.NA
    return out

def format_display_table_raw(df, category_in_play, dem_disp_map, scale_pairs) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["SURVEYR"] = pd.to_numeric(out.get("SURVEYR", out.get("year")), errors="coerce").astype("Int64")
    out["Year"] = out["SURVEYR"].astype(str)

    if category_in_play:
        def to_label(code):
            key = "" if code is None else str(code).strip()
            if key == "":
                return "All respondents"
            return dem_disp_map.get(key, str(code))
        dem_src = "DEMCODE" if "DEMCODE" in out.columns else "group_value"
        out["Demographic"] = out[dem_src].apply(to_label)

    dist_cols_raw, rename_map = [], {}
    if scale_pairs:
        for k, v in scale_pairs:
            for kcand in (k.upper(), k):
                if kcand in out.columns:
                    dist_cols_raw.append(kcand); rename_map[kcand] = v; break

    keep_cols = (["Year"] + (["Demographic"] if category_in_play else []) + dist_cols_raw +
                 ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE",
                  "positive_pct","neutral_pct","negative_pct"])
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    # drop answer columns entirely NA
    answer_label_cols = [v for v in rename_map.values() if v in out.columns]
    drop_all_na = [c for c in answer_label_cols if PD.to_numeric(out[c], errors="coerce").isna().all()]
    if drop_all_na:
        out = out.drop(columns=drop_all_na)

    # filter rows where ALL core metrics are NA
    core_candidates = []
    core_candidates += ["POSITIVE", "AGREE"]
    core_candidates += [c for c in answer_label_cols if c in out.columns]
    core_candidates = [c for c in core_candidates if c in out.columns]
    if core_candidates:
        mask_any = PD.Series(False, index=out.index)
        for c in core_candidates:
            mask_any = mask_any | PD.to_numeric(out[c], errors="coerce").notna()
        out = out.loc[mask_any].copy()

    sort_cols = ["Year"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    return out

def detect_metric_mode(df_disp: pd.DataFrame, scale_pairs) -> dict:
    cols_l = {c.lower(): c for c in df_disp.columns}
    if "positive" in cols_l:
        col = cols_l["positive"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {"mode":"positive","metric_col":col,"ui_label":"(% positive answers)","metric_label":"% positive","summary_allowed":True}
    if "agree" in cols_l:
        col = cols_l["agree"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {"mode":"agree","metric_col":col,"ui_label":"(% agree)","metric_label":"% agree","summary_allowed":True}
    if scale_pairs:
        for k, v in scale_pairs:
            label = v
            if label in df_disp.columns and PD.to_numeric(df_disp[label], errors="coerce").notna().any():
                return {"mode":k.lower(),"metric_col":label,"ui_label":f"(% {label})","metric_label":f"% {label}","summary_allowed":False}
    return {"mode":"none","metric_col":cols_l.get("positive","POSITIVE"),"ui_label":"(% positive answers)","metric_label":"% positive","summary_allowed":False}

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out


# ─────────────────────────────
# AI helpers (per-question + overall), default ON
# ─────────────────────────────
def _class_name(e: Exception) -> str: return type(e).__name__

def _call_openai_with_retry(client, **kwargs) -> tuple[str, str]:
    try:
        comp = client.chat.completions.create(timeout=60.0, **kwargs)
        content = comp.choices[0].message.content if comp.choices else ""
        return (content or "", "" if content else "empty response")
    except Exception:
        try:
            kwargs2 = {k: v for k, v in kwargs.items() if k != "response_format"}
            comp = client.chat.completions.create(timeout=60.0, **kwargs2)
            content = comp.choices[0].message.content if comp.choices else ""
            return (content or "", "" if content else "empty response")
        except Exception as e2:
            name2 = _class_name(e2).lower()
            if "authentication" in name2 or "auth" in name2: return "", "invalid_api_key"
            if "timeout" in name2 or "timedout" in name2:   return "", "timeout"
            if "rate" in name2 and "limit" in name2:        return "", "rate_limit"
            if "connection" in name2 or "network" in name2: return "", "network_error"
            if "badrequest" in name2 or "invalidrequest" in name2: return "", "invalid_request"
            if "typeerror" in name2: return "", "type_error"
            return "", name2 or "unknown_error"

def _ai_build_payload_single_metric(df_disp, question_code, question_text, category_in_play, metric_col):
    def col(df, *cands):
        for c in cands:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None
    year_col = col(df_disp, "Year") or "Year"
    demo_col = col(df_disp, "Demographic") or "Demographic"
    n_col    = col(df_disp, "ANSCOUNT", "AnsCount", "N")

    ys = PD.to_numeric(df_disp[year_col], errors="coerce").dropna().astype(int).unique().tolist()
    ys = sorted(ys); latest = ys[-1] if ys else None; baseline = ys[0] if ys else None
    overall_label = "All respondents"
    base = df_disp[df_disp[demo_col] == overall_label].copy() if (category_in_play and demo_col in df_disp.columns) else df_disp.copy()

    overall_series = []
    for _, r in base.sort_values(year_col).iterrows():
        yr = PD.to_numeric(r[year_col], errors="coerce")
        if PD.isna(yr): continue
        val = PD.to_numeric(r.get(metric_col, None), errors="coerce")
        n = PD.to_numeric(r.get(n_col, None), errors="coerce") if n_col in base.columns else None
        overall_series.append({"year": int(yr), "value": (float(val) if PD.notna(val) else None), "n": (int(n) if PD.notna(n) else None) if n is not None else None})

    groups = []
    if category_in_play and demo_col in df_disp.columns:
        for gname, gdf in df_disp.groupby(demo_col, dropna=False):
            if str(gname) == overall_label: continue
            series = []
            for _, r in gdf.sort_values(year_col).iterrows():
                yr = PD.to_numeric(r[year_col], errors="coerce")
                if PD.isna(yr): continue
                val = PD.to_numeric(r.get(metric_col, None), errors="coerce")
                n = PD.to_numeric(r.get(n_col, None), errors="coerce") if n_col in gdf.columns else None
                series.append({"year": int(yr), "value": (float(val) if PD.notna(val) else None), "n": (int(n) if PD.notna(n) else None) if n is not None else None})
            groups.append({"name": (str(gname) if PD.notna(gname) else ""), "series": series})

    return {"question_code": str(question_code), "question_text": str(question_text), "years": ys, "latest_year": latest, "baseline_year": baseline, "overall_label": "All respondents", "overall_series": overall_series, "groups": groups, "has_groups": bool(groups)}

def _ai_narrative_single(df_disp, question_code, question_text, category_in_play, metric_col, metric_label, temperature: float = 0.2) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return ""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return ""
    client = OpenAI()
    data = _ai_build_payload_single_metric(df_disp, question_code, question_text, category_in_play, metric_col)
    model_name = (st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

    # Prompt change: "minimal ≤2" (was "normal ≤2")
    system = (
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

    user_payload = {"metric_label": metric_label, "payload": data}
    user = json.dumps(user_payload, ensure_ascii=False)
    kwargs = dict(model=model_name, temperature=0.2, response_format={"type":"json_object"},
                  messages=[{"role":"system","content":system},{"role":"user","content":user}])
    content, _ = _call_openai_with_retry(client, **kwargs)
    if not content:
        return ""
    try:
        out = json.loads(content)
        n = out.get("narrative", "")
        if isinstance(n, (dict, list)):
            n = json.dumps(n, ensure_ascii=False)
        return str(n).strip()
    except Exception:
        return ""

def _ai_overall_theme_summary(per_question_paragraphs: List[str], temperature: float = 0.2) -> str:
    if not per_question_paragraphs:
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return ""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return ""
    client = OpenAI()
    model_name = (st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    system = (
        "You are an analyst summarizing PSES findings.\n"
        "Given several short per-question findings, write ONE concise concluding paragraph capturing the cross-cutting theme.\n"
        "No speculation; keep it neutral and specific.\n"
        "Return VALID JSON with exactly one key: \"narrative\"."
    )
    user = json.dumps({"notes": per_question_paragraphs}, ensure_ascii=False)
    kwargs = dict(model=model_name, temperature=temperature, response_format={"type":"json_object"},
                  messages=[{"role":"system","content":system},{"role":"user","content":user}])
    content, _ = _call_openai_with_retry(client, **kwargs)
    if not content:
        return ""
    try:
        out = json.loads(content)
        n = out.get("narrative", "")
        if isinstance(n, (dict, list)):
            n = json.dumps(n, ensure_ascii=False)
        return str(n).strip()
    except Exception:
        return ""


# ─────────────────────────────
# Profiler
# ─────────────────────────────
class Profiler:
    def __init__(self):
        self.steps: list[tuple[str, float]] = []
    from contextlib import contextmanager
    @contextmanager
    def step(self, name: str, live=None, engine: str = "", t0_global: float | None = None):
        t0 = time.perf_counter()
        if live is not None and t0_global is not None:
            live.caption(f"Processing… {name} • engine: {engine} • {time.perf_counter() - t0_global:.1f}s")
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.steps.append((name, dt))


# ─────────────────────────────
# Embeddings + Hybrid strict search (no top-K)
# ─────────────────────────────
def _embeddings_available() -> bool:
    try:
        from openai import OpenAI  # noqa
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())
    except Exception:
        return False

def _get_embed_model() -> str:
    return (st.secrets.get("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()

def _get_embed_threshold() -> float:
    try:
        return float(st.secrets.get("OPENAI_EMBED_THRESHOLD", 0.40))
    except Exception:
        return 0.40

def _get_top_k() -> int:
    # Not used anymore (strict gate returns all passing items)
    try:
        int(st.secrets.get("OPENAI_EMBED_TOPK", 0))
    except Exception:
        pass
    return 0

@st.cache_resource(show_spinner=False)
def _build_question_embeddings(qdf: pd.DataFrame) -> Dict[str, List[float]]:
    from openai import OpenAI
    client = OpenAI()
    model = _get_embed_model()
    texts = (qdf["code"].astype(str) + " " + qdf["text"].astype(str)).tolist()
    BATCH = 96
    vectors: List[List[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        resp = client.embeddings.create(model=model, input=chunk)
        vectors.extend([d.embedding for d in resp.data])
    codes = qdf["code"].astype(str).tolist()
    return {c: v for c, v in zip(codes, vectors)}

def _keyword_overlap_score(qdf_row_tokens: List[str], query_tokens: set[str]) -> int:
    toks = set(qdf_row_tokens)
    return len(toks & query_tokens)

def _semantic_rank_basic(qdf: pd.DataFrame, user_query: str, _dummy: int) -> pd.DataFrame:
    if not user_query or qdf.empty:
        return qdf.head(0)
    uq = user_query.strip().lower()
    q_tokens = set(_tokenize(uq))
    # Require at least 2 token overlaps to be considered "relevant"
    def _score_row(tokens: List[str], norm: str) -> float:
        toks = set(tokens)
        exact = len(q_tokens & toks)
        subs = 0
        for t in q_tokens:
            if len(t) >= 3 and t in norm:
                subs += 1
        return exact + 0.5 * subs
    scores = qdf.apply(lambda r: _score_row(r["__tokens__"], r["__norm__"]), axis=1)
    out = qdf.assign(__score__=scores)
    out = out[out["__score__"] >= 2]  # strict keyword gate
    out = out.sort_values(["__score__", "code"], ascending=[False, True])
    return out[["code", "text", "display", "__score__"]]

def _expand_query_terms(uq: str) -> set[str]:
    base = set([t for t in _tokenize(uq) if len(t) >= 3])
    synonyms = {
        "pay": {"salary","compensation","wage","wages","remuneration","pay"},
        "salary": {"pay","compensation","remuneration","wage","wages"},
        "compensation": {"salary","pay","remuneration","wage","wages"},
        "recognition": {"recognize","recognized","appreciation","reward","recognition"},
        "harassment": {"harass","harassment","violence","bullying"},
        "inclusion": {"inclusion","inclusive","belonging","equity"},
        "psychological": {"psychological","mental","health","safety"},
    }
    for t in list(base):
        if t in synonyms:
            base |= synonyms[t]
    return base

def _semantic_search(qdf: pd.DataFrame, user_query: str, use_hybrid: bool) -> tuple[pd.DataFrame, dict]:
    """
    Strict relevance:
      - Embeddings-only: keep items with cosine >= thr_strict (default 0.40)
      - Hybrid: keep items that satisfy EITHER:
          A) cosine >= thr_strong (thr+0.06, e.g., 0.46), OR
          B) cosine >= thr_strict AND keyword_overlap >= 2
      - Return ALL that pass; if none, return empty with diag.
    """
    thr_strict = _get_embed_threshold()    # e.g., 0.40
    thr_strong = thr_strict + 0.06         # e.g., 0.46
    KW_MIN = 2

    if not user_query.strip():
        return qdf.head(0), {'path':'none','model':None,'hybrid':use_hybrid,'threshold':thr_strict,'rule':'strict'}

    if not _embeddings_available():
        out = _semantic_rank_basic(qdf, user_query, 0)
        return out, {'path':'fallback','model':None,'hybrid':use_hybrid,'threshold':None,'rule':f'kw_overlap>={KW_MIN}'}

    try:
        import numpy as np  # noqa
        from openai import OpenAI  # noqa
    except Exception:
        out = _semantic_rank_basic(qdf, user_query, 0)
        return out, {'path':'fallback','model':None,'hybrid':use_hybrid,'threshold':None,'rule':f'kw_overlap>={KW_MIN}'}

    model = _get_embed_model()
    uq = user_query.strip().lower()
    q_tokens = _expand_query_terms(uq)

    # Get embeddings (with visible spinner)
    with st.spinner(f"🔎 Searching… contacting OpenAI and computing embeddings (model: {model})…"):
        try:
            emb_map = _build_question_embeddings(qdf)
            client = OpenAI()
            qresp = client.embeddings.create(model=model, input=user_query.strip())
            qvec = qresp.data[0].embedding
        except Exception:
            out = _semantic_rank_basic(qdf, user_query, 0)
            return out, {'path':'fallback','model':None,'hybrid':use_hybrid,'threshold':None,'rule':f'kw_overlap>={KW_MIN}'}

    # Cosine
    import numpy as np
    def cos(a, b):
        a = np.array(a, dtype=float); b = np.array(b, dtype=float)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    # Score all; APPLY ONLY THE GATE; do not cap or prune
    rows = []
    kw_max = 1
    for _, r in qdf.iterrows():
        code = str(r["code"]); text = str(r["text"])
        vec = emb_map.get(code)
        if vec is None:
            continue
        coss = cos(qvec, vec)
        kw = _keyword_overlap_score(r["__tokens__"], q_tokens) if use_hybrid else 0
        kw_max = max(kw_max, kw)
        rows.append((code, text, f"{code} – {text}", coss, kw))

    kept = []
    if use_hybrid:
        for code, text, disp, coss, kw in rows:
            if (coss >= thr_strong) or (coss >= thr_strict and kw >= KW_MIN):
                score = 0.8 * coss + 0.2 * (kw / kw_max if kw_max > 0 else 0.0)
                kept.append((code, text, disp, score))
    else:
        for code, text, disp, coss, kw in rows:
            if coss >= thr_strict:
                kept.append((code, text, disp, coss))

    if not kept:
        return qdf.head(0), {'path':'embeddings','model':model,'hybrid':use_hybrid,'threshold':thr_strict,'rule':f'A:cos≥{thr_strong:.2f} OR B:cos≥{thr_strict:.2f}+KW≥{KW_MIN}','kept':0}

    out = PD.DataFrame(kept, columns=["code","text","display","__score__"]).sort_values(["__score__","code"], ascending=[False, True])
    return out[["code","text","display","__score__"]], {'path':'embeddings','model':model,'hybrid':use_hybrid,'threshold':thr_strict,'rule':f'A:cos≥{thr_strong:.2f} OR B:cos≥{thr_strict:.2f}+KW≥{KW_MIN}','kept':len(kept)}


# ─────────────────────────────
# Cross-question summary builder
# ─────────────────────────────
def build_cross_question_summary(
    per_q_disp: dict[str, pd.DataFrame],
    category_in_play: bool,
    selected_years: list[str]
) -> pd.DataFrame:
    """
    Overview table:
      - Rows = Question (and Demographic when selected)
      - Columns = selected years
      - Cells   = POSITIVE (or AGREE else first answer) as whole-number percents
    """
    rows: list[pd.DataFrame] = []
    years = sorted([str(y) for y in selected_years], key=lambda x: int(x))

    for qcode, df_disp in per_q_disp.items():
        if df_disp is None or df_disp.empty:
            continue

        df = df_disp.copy()
        if "Demographic" not in df.columns:
            df["Demographic"] = "All respondents"

        decision = detect_metric_mode(df, scale_pairs=None)
        metric_col = decision["metric_col"]

        keep = ["Demographic", "Year", metric_col]
        keep = [c for c in keep if c in df.columns]
        if "Year" not in keep:
            continue

        df = df[keep].copy()
        pivot = df.pivot_table(index="Demographic", columns="Year", values=metric_col, aggfunc="first")

        for y in years:
            if y not in pivot.columns:
                pivot[y] = pd.NA

        pivot = pivot.reset_index()
        pivot.insert(0, "Question", qcode)
        rows.append(pivot)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    for y in years:
        if y in combined.columns:
            vals = pd.to_numeric(combined[y], errors="coerce").round(0)
            out = pd.Series("n/a", index=combined.index, dtype="object")
            mask = vals.notna()
            out.loc[mask] = vals.loc[mask].astype(int).astype(str) + "%"
            combined[y] = out

    if not category_in_play:
        if "Demographic" in combined.columns:
            combined = combined[combined["Demographic"] == "All respondents"].copy()
            combined = combined.drop(columns=["Demographic"])

    sort_cols = ["Question"] + (["Demographic"] if category_in_play else [])
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    display_cols = ["Question"] + (["Demographic"] if category_in_play else []) + years
    return combined[display_cols]


# ─────────────────────────────
# Backend detect (soft)
# ─────────────────────────────
def _detect_backend():
    try:
        if hasattr(_dl, "LAST_BACKEND"):
            return getattr(_dl, "LAST_BACKEND")
        if hasattr(_dl, "get_last_backend") and callable(_dl.get_last_backend):
            return _dl.get_last_backend()
        if hasattr(_dl, "BACKEND_IN_USE"):
            return getattr(_dl, "BACKEND_IN_USE")
    except Exception:
        pass
    try:
        import pyarrow  # noqa
        if get_backend_info().get("parquet_ready"):
            return "parquet"
    except Exception:
        pass
    return "csv"


# ─────────────────────────────
# UI — Gated flow with CHECKBOX selection + Busy phase
# ─────────────────────────────
def run_menu2():
    st.markdown(
        """
        <style>
          .custom-header{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }
          .custom-instruction{ font-size: 15px; line-height: 1.4; margin-bottom: 8px; color: #333; }
          .field-label{ font-size: 16px; font-weight: 600; margin: 10px 0 2px; color: #222; }
          .tiny-note{ font-size: 12px; color: #666; margin-top: -4px; margin-bottom: 10px; }
          .q-sub{ font-size: 14px; color: #333; margin-top: -4px; margin-bottom: 2px; }
          .full-width textarea { min-height: 120px !important; }
          .pill { display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; margin-left:6px; background:#eef; color:#224; }
          .pill.red { background:#fee; color:#822; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # One-time prewarm
    if not st.session_state.get("prewarmed_once"):
        try:
            with st.spinner("⚡ Preparing fast path (one-time)…"):
                prewarm_fastpath()
            st.session_state["prewarmed_once"] = True
        except Exception:
            st.session_state["prewarmed_once"] = True

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    _ = load_scales_metadata()  # cached; called later per question

    # Persistent UI state
    st.session_state.setdefault("menu2_phase", "search")          # search | pick | pick_filters | busy | ready
    st.session_state.setdefault("menu2_matches", PD.DataFrame())
    st.session_state.setdefault("menu2_selected_codes", [])       # list[str]

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            "<img style='width:75%;max-width:740px;height:auto;display:block;margin:0 auto 16px;' "
            "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="custom-header">🎯 Search by Theme / Keywords</div>', unsafe_allow_html=True)

        # Toggles
        show_debug = st.toggle("🔧 Show technical parameters & diagnostics", value=st.session_state.get("show_debug", SHOW_DEBUG))
        st.session_state["show_debug"] = show_debug
        ai_enabled = st.toggle("🧠 Enable AI analysis (OpenAI)", value=st.session_state.get("ai_enabled", True))
        st.session_state["ai_enabled"] = ai_enabled

        # Optional: allow disabling hybrid when diagnosing relevance
        adv_col1, _ = st.columns([1,1])
        with adv_col1:
            use_hybrid_ui = st.toggle("Use hybrid re-ranker (recommended)", value=st.session_state.get("menu2_use_hybrid", True))
            st.session_state["menu2_use_hybrid"] = use_hybrid_ui

        # Phase: SEARCH
        if st.session_state["menu2_phase"] == "search":
            st.markdown('<div class="custom-instruction">Step 1 — Enter a theme or keywords (e.g., “compensation”, “recognition”, “psychological safety”).</div>', unsafe_allow_html=True)
            st.markdown('<div class="field-label">Enter keywords or a theme:</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="full-width">', unsafe_allow_html=True)
                user_query = st.text_area("Theme/keywords", key="menu2_theme", label_visibility="collapsed",
                                          placeholder="Type here… e.g., compensation, inclusion, harassment, recognition")
                st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔎 Search related questions"):
                if not (user_query or "").strip():
                    st.warning("Please enter a theme or keywords.")
                else:
                    matches_df, diag = _semantic_search(qdf, user_query.strip(), use_hybrid=st.session_state.get("menu2_use_hybrid", True))
                    st.session_state["menu2_diag"] = diag
                    if matches_df.empty:
                        st.info("No matching questions met the relevance threshold. Try different or broader keywords.")
                    else:
                        st.session_state["menu2_matches"] = matches_df
                        st.session_state["menu2_phase"] = "pick"

        # Phase: PICK QUESTIONS
        if st.session_state["menu2_phase"] == "pick":
            matches_df = st.session_state["menu2_matches"]
            diag = st.session_state.get("menu2_diag", {})
            if diag.get("path") == "embeddings":
                hybrid_note = "hybrid" if diag.get("hybrid") else "embeddings-only"
                rule = diag.get("rule","strict"); kept = diag.get("kept", None)
                kept_txt = f", kept={kept}" if kept is not None else ""
                st.markdown(f"<span class='pill'>Search: Embeddings ({diag.get('model')}, {hybrid_note}, {rule}{kept_txt})</span>", unsafe_allow_html=True)
            elif diag.get("path") == "fallback":
                st.markdown("<span class='pill red'>Search: Fallback (strict keyword match: ≥2 overlaps)</span>", unsafe_allow_html=True)

            st.markdown('<div class="custom-instruction">Step 2 — Tick one or more related questions to analyze.</div>', unsafe_allow_html=True)

            selected_codes: List[str] = []
            for idx, r in matches_df.reset_index(drop=True).iterrows():
                label = f"{r['code']} — {r['text']}"
                key = f"chk_{r['code']}"
                default_val = (idx == 0)  # preselect top 1 to give user a starting point
                val = st.checkbox(label, value=st.session_state.get(key, default_val), key=key)
                if val:
                    selected_codes.append(str(r["code"]))

            col_a, col_b = st.columns([1,1])
            with col_a:
                if st.button("✅ Confirm selection"):
                    if not selected_codes:
                        st.warning("Please select at least one question.")
                    else:
                        st.session_state["menu2_selected_codes"] = selected_codes
                        st.session_state["menu2_phase"] = "pick_filters"
            with col_b:
                if st.button("↩️ Back to search"):
                    st.session_state["menu2_phase"] = "search"
                    for _, r in matches_df.iterrows():
                        st.session_state.pop(f"chk_{r['code']}", None)

        # Phase: PICK FILTERS (Years & Demographics)
        if st.session_state["menu2_phase"] == "pick_filters":
            st.markdown('<div class="custom-instruction">Step 3 — Select survey year(s) and a demographic breakdown (or All respondents).</div>', unsafe_allow_html=True)

            # Years
            st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
            all_years = ["2024", "2022", "2020", "2019"]
            select_all = st.checkbox("All years", value=True, key="menu2_select_all_years")
            selected_years: list[str] = []
            year_cols = st.columns(len(all_years))
            for idx, yr in enumerate(all_years):
                with year_cols[idx]:
                    checked = True if select_all else False
                    if st.checkbox(yr, value=checked, key=f"menu2_year_{yr}"):
                        selected_years.append(yr)
            selected_years = sorted(selected_years)

            # Demographics
            DEMO_CAT_COL = "DEMCODE Category"; LABEL_COL = "DESCRIP_E"
            st.markdown('<div class="field-label">Select a demographic category (or All respondents):</div>', unsafe_allow_html=True)
            demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
            demo_selection = st.selectbox("Demographic category", demo_categories, key="menu2_demo_main", label_visibility="collapsed")

            sub_selection = None
            if demo_selection != "All respondents":
                st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
                sub_items = sorted(
                    demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL]
                    .dropna().astype(str).unique().tolist()
                )
                sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=f"menu2_sub_{demo_selection.replace(' ', '_')}", label_visibility="collapsed")
                if sub_selection == "":
                    sub_selection = None

            ready = True
            if not st.session_state.get("menu2_selected_codes"):
                ready = False; st.info("Please select related questions first.")
            if not selected_years:
                ready = False; st.warning("Please select at least one year.")
            if demo_selection is None:
                ready = False; st.warning("Please select a demographic category (or All respondents).")

            if ready:
                demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(demo_df, demo_selection, sub_selection)
                if category_in_play and (None not in demcodes):
                    demcodes = [None] + demcodes
                st.session_state["menu2_filters"] = {
                    "years": selected_years,
                    "demcodes": demcodes,
                    "disp_map": disp_map,
                    "category_in_play": category_in_play,
                    "demo_selection": demo_selection,
                    "sub_selection": sub_selection,
                }
                if st.button("🔎 Run query"):
                    st.session_state["menu2_phase"] = "busy"
                    st.rerun()

            if st.button("↩️ Back to question selection"):
                st.session_state["menu2_phase"] = "pick"

        # Phase: BUSY (compute; no greyed-out duplicates)
        if st.session_state["menu2_phase"] == "busy":
            selected_codes = st.session_state.get("menu2_selected_codes", [])
            filt = st.session_state.get("menu2_filters", {})
            selected_years = filt.get("years", [])
            demcodes = filt.get("demcodes", [None])
            disp_map = filt.get("disp_map", {None: "All respondents"})
            category_in_play = bool(filt.get("category_in_play", False))

            engine_guess = _detect_backend()
            status_line = st.empty()
            prof = Profiler()
            t0_global = time.perf_counter()

            per_q_disp: Dict[str, pd.DataFrame] = {}
            per_q_text: Dict[str, str] = {}
            per_q_metric: Dict[str, Tuple[str, str, bool]] = {}

            with st.spinner("Processing data…"):
                status_line.caption(f"Processing… engine: {engine_guess} • 0.0s")
                qmeta = load_questions_metadata()

                for qcode in selected_codes:
                    qrow = qmeta[qmeta["code"] == qcode].head(1)
                    qtext = (qrow["text"].iloc[0] if not qrow.empty else "")
                    per_q_text[qcode] = str(qtext)

                    # 1) scales
                    with prof.step(f"[{qcode}] Match scales", live=status_line, engine=engine_guess, t0_global=t0_global):
                        scale_pairs = get_scale_labels(load_scales_metadata(), qcode)
                        if not scale_pairs:
                            continue

                    # 2) load
                    with prof.step(f"[{qcode}] Load data", live=status_line, engine=engine_guess, t0_global=t0_global):
                        try:
                            df_raw = load_results2024_filtered(question_code=qcode, years=selected_years, group_values=demcodes)  # type: ignore[arg-type]
                        except TypeError:
                            parts = []
                            for gv in demcodes:
                                try:
                                    parts.append(load_results2024_filtered(question_code=qcode, years=selected_years, group_value=(None if gv is None else str(gv).strip())))
                                except TypeError:
                                    continue
                            df_raw = (PD.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else PD.DataFrame())

                    if df_raw is None or df_raw.empty:
                        continue

                    # 3) clean
                    with prof.step(f"[{qcode}] 999/9999 → NA", live=status_line, engine=engine_guess, t0_global=t0_global):
                        df_raw = exclude_999_raw(df_raw)

                    # 4) format
                    with prof.step(f"[{qcode}] Sort & format table", live=status_line, engine=engine_guess, t0_global=t0_global):
                        dem_map_clean = {None: "All respondents"}
                        try:
                            for k, v in (disp_map or {}).items():
                                dem_map_clean[(None if k is None else str(k).strip())] = v
                        except Exception:
                            pass
                        df_disp = format_display_table_raw(df=df_raw, category_in_play=category_in_play, dem_disp_map=dem_map_clean, scale_pairs=scale_pairs)

                    if df_disp is None or df_disp.empty:
                        continue

                    per_q_disp[qcode] = df_disp
                    decision = detect_metric_mode(df_disp, scale_pairs)
                    per_q_metric[qcode] = (decision["metric_col"], decision["ui_label"], bool(decision.get("summary_allowed", False)))

                # Cross-question summary
                with prof.step("Build cross-question summary", live=status_line, engine=engine_guess, t0_global=t0_global):
                    cross_summary = build_cross_question_summary(per_q_disp, category_in_play, selected_years)

            # Stash payload for rendering
            st.session_state["menu2_ready_payload"] = dict(
                per_q_disp=per_q_disp,
                per_q_text=per_q_text,
                per_q_metric=per_q_metric,
                cross_summary=cross_summary,
                selected_years=selected_years,
                category_in_play=category_in_play
            )

            st.session_state["menu2_phase"] = "ready"
            st.rerun()

        # Phase: READY (render results)
        if st.session_state["menu2_phase"] == "ready":
            payload = st.session_state.get("menu2_ready_payload", {})
            per_q_disp = payload.get("per_q_disp", {})
            per_q_text = payload.get("per_q_text", {})
            per_q_metric = payload.get("per_q_metric", {})
            cross_summary = payload.get("cross_summary", PD.DataFrame())
            selected_years = payload.get("selected_years", [])
            category_in_play = bool(payload.get("category_in_play", False))

            tabs = st.tabs(["Overview"] + [f"{q} — Details" for q in per_q_disp.keys()])

            # Overview
            with tabs[0]:
                st.subheader("Cross-question summary")
                if cross_summary is not None and not cross_summary.empty:
                    st.dataframe(make_arrow_safe(cross_summary), use_container_width=True, hide_index=True)
                else:
                    st.info("No summary available for the current selection.")

                # Analyses
                per_question_paras: List[str] = []
                if st.session_state.get("ai_enabled", True):
                    st.markdown("### Analysis — Per question (≤1 paragraph each)")
                    for qcode, df_disp in per_q_disp.items():
                        metric_col, ui_label, _summary_allowed = per_q_metric.get(qcode, ("POSITIVE", "(% positive answers)", True))
                        qtext = per_q_text.get(qcode, "")
                        with st.spinner(f"🤖 Analyzing {qcode}…"):
                            para = _ai_narrative_single(
                                df_disp=df_disp,
                                question_code=qcode,
                                question_text=qtext,
                                category_in_play=category_in_play,
                                metric_col=metric_col,
                                metric_label=("% positive" if metric_col.lower()=="positive" else ("% agree" if metric_col.lower()=="agree" else metric_col)),
                                temperature=0.2
                            )
                        if para:
                            per_question_paras.append(f"**{qcode}.** {para}")
                            st.write(f"**{qcode}.** {para}")
                        else:
                            st.write(f"**{qcode}.** (AI analysis unavailable for this question.)")

                    st.markdown("### Overall theme summary")
                    with st.spinner("🤖 Synthesizing overall theme…"):
                        overall = _ai_overall_theme_summary(per_question_paras, temperature=0.2)
                    if overall:
                        model_used = (st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
                        st.write(overall + f"\n\n_Powered by OpenAI model {model_used}_")
                    else:
                        st.info("Overall synthesis unavailable.")

                st.markdown(
                    "Source: [2024 Public Service Employee Survey Results - Open Government Portal]"
                    "(https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f)"
                )

            # Per-question tabs
            idx = 1
            for qcode, df_disp in per_q_disp.items():
                with tabs[idx]:
                    qtext = per_q_text.get(qcode, "")
                    st.subheader(f"{qcode} — {qtext}")
                    metric_col, ui_label, summary_allowed = per_q_metric.get(qcode, ("POSITIVE", "(% positive answers)", True))
                    trend_df = PD.DataFrame()
                    if summary_allowed:
                        trend_df = build_trend_summary_table(df_disp=df_disp, category_in_play=category_in_play, metric_col=metric_col, selected_years=selected_years)

                    st.markdown("#### Summary results")
                    if summary_allowed and trend_df is not None and not trend_df.empty:
                        st.markdown(f"<div class='q-sub'>{qcode} — {qtext}</div><div class='tiny-note'>{ui_label}</div>", unsafe_allow_html=True)
                        st.dataframe(make_arrow_safe(trend_df), use_container_width=True, hide_index=True)
                    else:
                        st.info("Summary table is unavailable for this selection, please see detailed results below.")

                    st.markdown("#### Detailed results")
                    st.dataframe(make_arrow_safe(df_disp), use_container_width=True)

                    st.markdown(
                        "Source: [2024 Public Service Employee Survey Results - Open Government Portal]"
                        "(https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f)"
                    )
                idx += 1

            # Back controls
            st.divider()
            cols = st.columns([1,1,6])
            with cols[0]:
                if st.button("↩️ Back to filters"):
                    st.session_state["menu2_phase"] = "pick_filters"
            with cols[1]:
                if st.button("🔁 New search"):
                    st.session_state["menu2_phase"] = "search"
                    st.session_state["menu2_matches"] = PD.DataFrame()
                    st.session_state["menu2_selected_codes"] = []


# Per-question trend table (reuse pattern from Menu 1)
def build_trend_summary_table(df_disp, category_in_play, metric_col, selected_years: list[str] | None = None) -> pd.DataFrame:
    if df_disp is None or df_disp.empty or "Year" not in df_disp.columns:
        return PD.DataFrame()
    if metric_col not in df_disp.columns:
        low = {c.lower(): c for c in df_disp.columns}
        if metric_col.lower() in low:
            metric_col = low[metric_col.lower()]
        else:
            return PD.DataFrame()
    df = df_disp.copy()
    if "Demographic" not in df.columns:
        df["Demographic"] = "All respondents"
    years = sorted([str(y) for y in (selected_years or df["Year"].astype(str).unique().tolist())], key=lambda x: int(x))
    pivot = df.pivot_table(index="Demographic", columns="Year", values=metric_col, aggfunc="first").copy()
    pivot.index.name = "Segment"
    for y in years:
        if y not in pivot.columns:
            pivot[y] = PD.NA
    for c in pivot.columns:
        vals = PD.to_numeric(pivot[c], errors="coerce").round(0)
        out = PD.Series("n/a", index=pivot.index, dtype="object")
        mask = vals.notna()
        out.loc[mask] = vals.loc[mask].astype(int).astype(str) + "%"
        pivot[c] = out
    pivot = pivot.reset_index()
    return pivot[["Segment"] + years]


if __name__ == "__main__":
    run_menu2()
