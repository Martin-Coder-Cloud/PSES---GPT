# wizard/main.py — Consolidated 3-step Search Wizard (Menu 1 + Menu 2)
# PS-wide only; supports single or multiple questions.
# Key behaviors preserved:
#  - Data read as TEXT; 999/9999 suppressed (not dropped)
#  - Metric selection: POSITIVE → AGREE → first answer label
#  - Summary table only when POSITIVE/AGREE is available
#  - Per-question AI narrative (optional) + overall theme synthesis
#  - Theme/keyword search with tiered retrieval (exact → embeddings/hybrid → fallback)
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# Loader (PS-wide results)
import utils.data_loader as _dl
try:
    from utils.data_loader import load_results2024_filtered, get_backend_info, prewarm_fastpath
except Exception:
    from utils.data_loader import load_results2024_filtered  # type: ignore
    def get_backend_info(): return {}
    def prewarm_fastpath(): return "csv"

# OpenAI key from Streamlit secrets (if present)
os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

PD = pd
SHOW_DEBUG_DEFAULT = False

# ─────────────────────────────
# Wizard session key
# ─────────────────────────────
WKEY = "wiz3"

def _init_state():
    if WKEY not in st.session_state:
        st.session_state[WKEY] = {
            "step": 1,                         # 1 → 2 → 3
            "mode": "select",                  # "select" | "search"
            "keywords": "",
            "matches": PD.DataFrame(),
            "diag": {},
            "selected_questions": [],          # list[str]
            "years": ["2024"],
            "demographic": {"category": "All respondents", "subgroup": None},
            "filters": {},
            "results": {},
            "profile": [],
            "ai_enabled": True,
            "use_hybrid": True,
        }

# ─────────────────────────────
# Metadata loaders (from Excel)
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_demographics_metadata() -> pd.DataFrame:
    df = pd.read_excel("metadata/Demographics.xlsx")
    df.columns = [c.strip() for c in df.columns]
    return df

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
    # Search helpers (Menu 2)
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
# Shared helpers (from Menus)
# ─────────────────────────────
def _find_demcode_col(demo_df: pd.DataFrame) -> str | None:
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            return c
    return None

def _four_digit(s: str) -> str:
    s = "".join(ch for ch in str(s) if s is not None and ch.isdigit())
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
    match = scales_df[(scales_df["__code_norm__"] == qnorm)]
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
    candidates = [f"answer{i}" for i in range(1, 7 + 1)] + \
                 ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE", "YES",
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

    # Drop answer columns that are entirely NA
    answer_label_cols = [v for v in rename_map.values() if v in out.columns]
    drop_all_na = [c for c in answer_label_cols if PD.to_numeric(out[c], errors="coerce").isna().all()]
    if drop_all_na:
        out = out.drop(columns=drop_all_na)

    # Filter rows where ALL core metrics are NA
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

def _drop_all_na_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    keep_always = {"Year", "Demographic"}
    cols = []
    for c in df.columns:
        if c in keep_always:
            cols.append(c); continue
        s_num = pd.to_numeric(df[c], errors="coerce")
        if s_num.notna().any():
            cols.append(c)
    if not cols:
        cols = [c for c in df.columns if c in keep_always]
        for c in df.columns:
            if c in keep_always: continue
            sc = df[c].astype(str).str.strip()
            if (sc != "").any():
                cols.append(c)
    return df[cols]

# ─────────────────────────────
# Trend summary helpers
# ─────────────────────────────
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

def build_cross_question_summary(
    per_q_disp: dict[str, pd.DataFrame],
    category_in_play: bool,
    selected_years: list[str]
) -> pd.DataFrame:
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

    if not category_in_play and "Demographic" in combined.columns:
        combined = combined[combined["Demographic"] == "All respondents"].copy()
        combined = combined.drop(columns=["Demographic"])

    sort_cols = ["Question"] + (["Demographic"] if category_in_play else [])
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    display_cols = ["Question"] + (["Demographic"] if category_in_play else []) + years
    return combined[display_cols]

# ─────────────────────────────
# Keyword/Theme search (Menu 2)
# ─────────────────────────────
def _tokenize(text: str) -> List[str]:
    if text is None:
        return []
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9\s\-_/&']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split(" ")

_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","so","to","of","for","in","on","at",
    "is","are","was","were","be","being","been","i","we","you","they","he","she","it",
    "with","about","regarding","re","re:", "re-", "re–","this","that","these","those",
    "how","what","why","which","when","where","please","find","interested","into"
}

def _extract_keywords(query: str) -> list[str]:
    if not query or not str(query).strip():
        return []
    q = str(query).strip().lower()
    phrases = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', q)
    phrases = [" ".join([p for p in tup if p]) for tup in phrases if any(tup)]
    q_wo_quotes = re.sub(r'"[^"]+"|“[^”]+”|\'[^\']+\'', " ", q)
    tokens = re.findall(r"[a-z0-9][a-z0-9\-\/_']*[a-z0-9]", q_wo_quotes)
    def keep(t: str) -> bool:
        if len(t) < 3: return False
        return t not in _STOPWORDS
    toks = [t for t in tokens if keep(t)]
    raw = phrases + toks
    seen = set(); out = []
    for t in raw:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _compile_word_regexes(query_terms: list[str]) -> list[re.Pattern]:
    regs = []
    for t in query_terms:
        if not t: continue
        if " " in t:
            pat = r"\b" + re.escape(t) + r"\b"
        else:
            pat = r"\b" + re.escape(t) + r"(?:s)?\b"
        regs.append(re.compile(pat, flags=re.IGNORECASE))
    return regs

def _exact_keyword_filter(qdf: pd.DataFrame, user_query: str) -> pd.DataFrame:
    if qdf.empty or not user_query.strip():
        return qdf.head(0)
    query_terms = _extract_keywords(user_query)
    if not query_terms:
        return qdf.head(0)
    regs = _compile_word_regexes(query_terms)
    def _hitcount(code: str, text: str) -> int:
        s = f"{code} {text}"
        return sum(1 for rg in regs if rg.search(s) is not None)
    hits = qdf.apply(lambda r: _hitcount(str(r["code"]), str(r["text"])), axis=1)
    out = qdf.assign(__score__=hits)
    out = out[out["__score__"] > 0]
    if out.empty:
        return out
    return out.sort_values(["__score__", "code"], ascending=[False, True])[["code","text","display","__score__"]]

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
        # career-related
        "career": {"career","advancement","promotion","promotions","progression","mobility",
                   "development","professional","professional development","growth",
                   "learning","training","upskilling","reskilling","mentorship","mentoring",
                   "opportunity","opportunities","talent","talent management","succession"},
        "development": {"development","professional development","learning","training","growth",
                        "upskilling","reskilling","mentorship"},
        "promotion": {"promotion","promotions","advancement","progression","career"},
        "mobility": {"mobility","internal mobility","career mobility","lateral move","rotation"},
        "mentorship": {"mentorship","mentor","mentoring","coaching"},
        "training": {"training","learning","courses","learning opportunities","professional development"},
    }
    for t in list(base):
        if t in synonyms:
            base |= synonyms[t]
    return base

def _keyword_overlap_score(qdf_row_tokens: List[str], query_tokens: set[str]) -> int:
    toks = set(qdf_row_tokens)
    return len(toks & query_tokens)

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

def _semantic_rank_basic(qdf: pd.DataFrame, user_query: str) -> pd.DataFrame:
    if not user_query or qdf.empty:
        return qdf.head(0)
    q_terms = _extract_keywords(user_query)
    if not q_terms:
        return qdf.head(0)
    q_set = set(q_terms)
    def _score_row(tokens: List[str], norm: str) -> float:
        toks = set(tokens)
        exact = len(q_set & toks)
        subs = sum(1 for t in q_set if len(t) >= 3 and t in norm)
        return exact + 0.5 * subs
    scores = qdf.apply(lambda r: _score_row(r["__tokens__"], r["__norm__"]), axis=1)
    out = qdf.assign(__score__=scores)
    out = out[out["__score__"] >= 1]
    return out.sort_values(["__score__", "code"], ascending=[False, True])[["code","text","display","__score__"]]

def _semantic_search(qdf: pd.DataFrame, user_query: str, use_hybrid: bool) -> tuple[pd.DataFrame, dict]:
    thr_strict = _get_embed_threshold()
    thr_strong = thr_strict + 0.06
    KW_MIN = 1

    if not user_query.strip():
        return qdf.head(0), {'path':'none','model':None,'hybrid':use_hybrid,'threshold':thr_strict,'rule':'strict','terms':[],'expanded':[]}

    base_terms = _extract_keywords(user_query)

    # Tier 0: exact whole-word
    exact_df = _exact_keyword_filter(qdf, user_query)
    if not exact_df.empty:
        return exact_df, {'path':'exact','model':None,'hybrid':False,'threshold':None,'rule':'whole-word','kept': int(exact_df.shape[0]), 'terms': base_terms, 'expanded': base_terms}

    # Tier 1: embeddings (hybrid gate) if available
    if _embeddings_available():
        try:
            import numpy as np  # noqa
            from openai import OpenAI  # noqa
        except Exception:
            pass
        else:
            model = _get_embed_model()
            expanded_terms = sorted(_expand_query_terms(" ".join(base_terms))) if base_terms else []
            q_tokens = set(expanded_terms) if expanded_terms else set()

            with st.spinner(f"🔎 Searching… contacting OpenAI and computing embeddings (model: {model})…"):
                try:
                    emb_map = _build_question_embeddings(qdf)
                    client = OpenAI()
                    qresp = client.embeddings.create(model=model, input=user_query.strip())
                    qvec = qresp.data[0].embedding
                except Exception:
                    # fall through to Tier 2
                    emb_map = {}

            if emb_map:
                import numpy as np
                def cos(a, b):
                    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
                    na = np.linalg.norm(a); nb = np.linalg.norm(b)
                    if na == 0 or nb == 0: return 0.0
                    return float(np.dot(a, b) / (na * nb))

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

                out = PD.DataFrame(kept, columns=["code","text","display","__score__"]).sort_values(["__score__","code"], ascending=[False, True])
                return out[["code","text","display","__score__"]], {
                    'path':'embeddings',
                    'model':model,
                    'hybrid':use_hybrid,
                    'threshold':thr_strict,
                    'rule':f'A:cos≥{thr_strong:.2f} OR B:cos≥{thr_strict:.2f}+KW≥{KW_MIN}',
                    'kept': int(out.shape[0]),
                    'terms': base_terms,
                    'expanded': expanded_terms or base_terms
                }

    # Tier 2: fallback keyword ranking
    out = _semantic_rank_basic(qdf, user_query)
    return out, {'path':'fallback','model':None,'hybrid':use_hybrid,'threshold':None,'rule':'keyword≥1','kept': int(out.shape[0]), 'terms': base_terms, 'expanded': base_terms}

# ─────────────────────────────
# AI (per-question + overall synthesis)
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
    ys = sorted(ys)
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

    return {
        "question_code": str(question_code),
        "question_text": str(question_text),
        "years": ys,
        "overall_label": "All respondents",
        "overall_series": overall_series,
        "groups": groups,
        "has_groups": bool(groups)
    }

def _ai_narrative_single(df_disp, question_code, question_text, category_in_play, metric_col, metric_label, temperature: float = 0.2) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return ""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return ""
    client = OpenAI()
    model_name = (st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

    # Prompt (aligned with Menu 1/2; "minimal ≤2")
    system = (
        "You are preparing insights for the Government of Canada’s Public Service Employee Survey (PSES).\n\n"
        "Data-use rules (hard constraints)\n"
        "- Use ONLY the provided JSON payload/table. No speculation.\n"
        "- Public Service–wide scope only.\n"
        "- Express percentages as whole numbers and differences as points.\n\n"
        "Analysis rules\n"
        "- Start with the most recent year in the payload for the selected metric.\n"
        "- Compare with the earliest year; classify trend:\n"
        "  • stable ≤1 point; • slight >1–2; • notable >2.\n"
        "- Compare demographic groups in the latest year:\n"
        "  • classify gaps: minimal ≤2, notable >2–5, important >5.\n"
        "- Indicate whether gaps are widening/narrowing/stable vs earlier years when visible.\n"
        "- Conclude with a concise overall statement.\n\n"
        "Style & output\n"
        "- Professional, concise prose (1–3 short paragraphs). No bullet lists.\n"
        "- Return VALID JSON with exactly one key: \"narrative\"."
    )

    payload = {"metric_label": metric_label, "payload": _ai_build_payload_single_metric(df_disp, question_code, question_text, category_in_play, metric_col)}
    kwargs = dict(
        model=model_name, temperature=temperature, response_format={"type":"json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content": json.dumps(payload, ensure_ascii=False)}]
    )
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
        "No speculation; keep it neutral and specific. Return VALID JSON with exactly one key: \"narrative\"."
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
# Profiler (simple)
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
# UI — 3-step wizard
# ─────────────────────────────
def _step1_question_selection():
    st.markdown("### Step 1 · Select question(s) or search by keywords/theme")

    # Global toggles (AI + Search hybrid)
    topcol1, topcol2 = st.columns([1,1])
    with topcol1:
        ai_enabled = st.toggle("🧠 Enable AI analysis (OpenAI)", value=st.session_state[WKEY].get("ai_enabled", True))
        st.session_state[WKEY]["ai_enabled"] = ai_enabled
    with topcol2:
        use_hybrid = st.toggle("Use hybrid re-ranker (recommended)", value=st.session_state[WKEY].get("use_hybrid", True))
        st.session_state[WKEY]["use_hybrid"] = use_hybrid

    # Mode switch
    mode = st.radio(
        "Choose input method",
        options=["Select from list", "Search by keywords/theme"],
        index=0 if st.session_state[WKEY]["mode"] == "select" else 1,
        horizontal=False
    )
    st.session_state[WKEY]["mode"] = "select" if mode.startswith("Select") else "search"

    qdf = load_questions_metadata()

    if st.session_state[WKEY]["mode"] == "select":
        # Multiselect from catalog (Menu 1-style input)
        options = qdf["display"].tolist()
        default = [opt for opt in options[:1]]  # preselect first to help users
        chosen_disp = st.multiselect(
            "Choose one or more survey questions",
            options=options,
            default=default,
            help="Tip: start typing a code like Q16 or a word to filter."
        )
        # Map display -> code
        sel_codes = qdf[qdf["display"].isin(chosen_disp)]["code"].astype(str).tolist()
        st.session_state[WKEY]["selected_questions"] = sel_codes

    else:
        # Theme/keyword search (Menu 2 behavior)
        uq = st.text_input(
            "Enter keywords or a theme (e.g., psychological safety, recognition, “career progression”)",
            value=st.session_state[WKEY].get("keywords", "")
        )
        st.session_state[WKEY]["keywords"] = uq

        if st.button("🔎 Search related questions"):
            matches_df, diag = _semantic_search(qdf, uq.strip(), use_hybrid=st.session_state[WKEY]["use_hybrid"])
            st.session_state[WKEY]["matches"] = matches_df
            st.session_state[WKEY]["diag"] = diag

        matches_df = st.session_state[WKEY].get("matches", PD.DataFrame())
        diag = st.session_state[WKEY].get("diag", {})

        # Diagnostics (visibility like Menu 2)
        if not matches_df.empty or diag.get("path") in {"exact","embeddings","fallback"}:
            if diag.get("path") == "exact":
                st.caption("Search mode: Exact whole-word match")
            elif diag.get("path") == "embeddings":
                hybrid_note = "hybrid" if diag.get("hybrid") else "embeddings-only"
                rule = diag.get("rule","strict"); kept = diag.get("kept", None)
                kept_txt = f", kept={kept}" if kept is not None else ""
                st.caption(f"Search mode: Embeddings ({diag.get('model')}, {hybrid_note}, {rule}{kept_txt})")
            elif diag.get("path") == "fallback":
                st.caption("Search mode: Fallback (keyword ≥1)")

            terms = diag.get("terms", []) or []
            expanded = diag.get("expanded", []) or terms
            if terms:
                chips = " ".join(f"`{t}`" for t in terms)
                st.markdown(f"**Extracted keywords:** {chips}")
            if diag.get("path") == "embeddings" and expanded and (set(expanded) - set(terms)):
                exp_only = [t for t in expanded if t not in terms]
                if exp_only:
                    chips2 = " ".join(f"`{t}`" for t in exp_only)
                    st.markdown(f"**Semantic expansion used:** {chips2}")

        # Checkbox multi-select of matches (top preselected)
        selected_codes: List[str] = []
        if not matches_df.empty:
            st.markdown("**Select one or more related questions:**")
            for idx, r in matches_df.reset_index(drop=True).iterrows():
                label = f"{r['code']} — {r['text']}"
                key = f"wiz_chk_{r['code']}"
                default_val = (idx == 0)
                val = st.checkbox(label, value=st.session_state.get(key, default_val), key=key)
                if val:
                    selected_codes.append(str(r["code"]))
        st.session_state[WKEY]["selected_questions"] = selected_codes

    # Navigation
    left, right = st.columns([1,1])
    can_next = bool(st.session_state[WKEY]["selected_questions"])
    if left.button("Next ▶", disabled=not can_next):
        st.session_state[WKEY]["step"] = 2
    if right.button("Reset"):
        keep_ai = st.session_state[WKEY]["ai_enabled"]
        keep_hybrid = st.session_state[WKEY]["use_hybrid"]
        st.session_state.pop(WKEY, None)
        _init_state()
        st.session_state[WKEY]["ai_enabled"] = keep_ai
        st.session_state[WKEY]["use_hybrid"] = keep_hybrid

def _step2_years():
    st.markdown("### Step 2 · Select survey years (time series)")
    all_years = ["2019","2020","2022","2024"]
    current = st.session_state[WKEY].get("years", ["2024"])
    chosen = st.multiselect("Survey years", options=all_years, default=current)
    if not chosen:
        st.warning("Please select at least one year.")
        return
    st.session_state[WKEY]["years"] = sorted(chosen)

    c1, c2 = st.columns([1,1])
    if c1.button("◀ Back"):
        st.session_state[WKEY]["step"] = 1
    if c2.button("Next ▶"):
        st.session_state[WKEY]["step"] = 3

def _step3_demographic_and_run():
    st.markdown("### Step 3 · Select a demographic (optional) and run")
    demo_df = load_demographics_metadata()
    DEMO_CAT_COL = "DEMCODE Category"; LABEL_COL = "DESCRIP_E"

    categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
    cur_cat = st.session_state[WKEY]["demographic"].get("category", "All respondents")
    cat = st.selectbox("Demographic category", categories, index=categories.index(cur_cat) if cur_cat in categories else 0)

    sub = None
    if cat != "All respondents":
        sub_items = sorted(demo_df.loc[demo_df[DEMO_CAT_COL] == cat, LABEL_COL].dropna().astype(str).unique().tolist())
        cur_sub = st.session_state[WKEY]["demographic"].get("subgroup", None) or ""
        sub = st.selectbox("(optional) Subgroup", [""] + sub_items, index=([""]+sub_items).index(cur_sub) if cur_sub in ([""]+sub_items) else 0)
        if sub == "": sub = None

    st.session_state[WKEY]["demographic"] = {"category": cat, "subgroup": sub}

    # Parameter summary
    qdf = load_questions_metadata()
    qtxt_map = dict(zip(qdf["code"].astype(str), qdf["text"].astype(str)))
    params = {
        "questions": [{ "code": q, "text": qtxt_map.get(q, "") } for q in st.session_state[WKEY]["selected_questions"]],
        "years": st.session_state[WKEY]["years"],
        "demographic": {"category": cat, "subgroup": (sub or "(none)")}
    }
    st.markdown("#### Your selection")
    st.json(params)

    # Run
    if st.button("Run search ▶"):
        _run_pipeline()

    c1, _ = st.columns([1,1])
    if c1.button("◀ Back"):
        st.session_state[WKEY]["step"] = 2

def _run_pipeline():
    # Resolve demographic codes
    demo_df = load_demographics_metadata()
    cat = st.session_state[WKEY]["demographic"]["category"]
    sub = st.session_state[WKEY]["demographic"]["subgroup"]
    years = st.session_state[WKEY]["years"]
    demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(demo_df, cat, sub)
    if category_in_play and (None not in demcodes):
        demcodes = [None] + demcodes

    engine_guess = _detect_backend()
    status_line = st.empty()
    prof = Profiler()
    t0_global = time.perf_counter()

    per_q_disp: Dict[str, pd.DataFrame] = {}
    per_q_text: Dict[str, str] = {}
    per_q_metric: Dict[str, Tuple[str, str, bool]] = {}

    qmeta = load_questions_metadata()
    st.info("🔄 Stage 1/4: Matching scales and loading data for each question…")
    with st.spinner("Processing data…"):
        status_line.caption(f"Processing… engine: {engine_guess} • 0.0s")
        for qcode in st.session_state[WKEY]["selected_questions"]:
            qrow = qmeta[qmeta["code"] == qcode].head(1)
            qtext = (qrow["text"].iloc[0] if not qrow.empty else "")
            per_q_text[qcode] = str(qtext)

            # 1) Match scales
            with prof.step(f"[{qcode}] Match scales", live=status_line, engine=engine_guess, t0_global=t0_global):
                scale_pairs = get_scale_labels(load_scales_metadata(), qcode)
                if not scale_pairs:
                    st.warning(f"Scales not found for {qcode}. Skipping this question.")
                    continue

            # 2) Load data (supports both group_values and group_value signatures)
            with prof.step(f"[{qcode}] Load data", live=status_line, engine=engine_guess, t0_global=t0_global):
                try:
                    df_raw = load_results2024_filtered(question_code=qcode, years=years, group_values=demcodes)  # type: ignore[arg-type]
                except TypeError:
                    parts = []
                    for gv in demcodes:
                        try:
                            parts.append(load_results2024_filtered(question_code=qcode, years=years, group_value=(None if gv is None else str(gv).strip())))
                        except TypeError:
                            continue
                    df_raw = (PD.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else PD.DataFrame())

            if df_raw is None or df_raw.empty:
                continue

            # 3) 999/9999 → NA
            with prof.step(f"[{qcode}] 999/9999 → NA", live=status_line, engine=engine_guess, t0_global=t0_global):
                df_raw = exclude_999_raw(df_raw)

            # 4) Sort & format
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
            cross_summary = build_cross_question_summary(per_q_disp, category_in_play, years)

    total_s = time.perf_counter() - t0_global
    status_line.caption(f"Processing complete • engine: {engine_guess} • {total_s:.1f}s")

    # Save results into session
    st.session_state[WKEY]["results"] = dict(
        per_q_disp=per_q_disp,
        per_q_text=per_q_text,
        per_q_metric=per_q_metric,
        cross_summary=cross_summary,
        years=years,
        category_in_play=category_in_play,
        profile=list(prof.steps)
    )

def _render_results():
    payload = st.session_state[WKEY].get("results", {})
    if not payload:
        return
    per_q_disp = payload.get("per_q_disp", {})
    per_q_text = payload.get("per_q_text", {})
    per_q_metric = payload.get("per_q_metric", {})
    cross_summary = payload.get("cross_summary", PD.DataFrame())
    years = payload.get("years", [])
    category_in_play = bool(payload.get("category_in_play", False))

    tabs = st.tabs(["Overview"] + [f"{q} — Details" for q in per_q_disp.keys()])

    # Overview
    with tabs[0]:
        st.subheader("Overview")
        if per_q_text:
            st.markdown("**Selected questions**")
            for qcode in sorted(per_q_text.keys()):
                qtext = per_q_text.get(qcode, "")
                st.markdown(f"- **{qcode}** — {qtext}")

        if cross_summary is not None and not cross_summary.empty:
            st.dataframe(make_arrow_safe(cross_summary), use_container_width=True, hide_index=True)
        else:
            st.info("No summary available for the current selection.")

        # AI analysis (per-question + overall)
        per_question_paras: List[str] = []
        if st.session_state[WKEY].get("ai_enabled", True):
            st.markdown("### Analysis — Per question")
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
        else:
            st.caption("AI analysis is disabled (toggle in Step 1).")

        st.markdown(
            "Source: [2024 Public Service Employee Survey Results - Open Government Portal]"
            "(https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f)"
        )

        # Diagnostics (timings)
        with st.expander("Diagnostics (timings)"):
            prof = payload.get("profile", [])
            if prof:
                df_prof = PD.DataFrame(prof, columns=["Stage","Time (s)"])
                df_prof["Time (ms)"] = (df_prof["Time (s)"] * 1000).round(0).astype(int)
                st.table(df_prof[["Stage","Time (ms)"]])
            else:
                st.caption("No profiling data recorded.")

    # Per-question tabs
    idx = 1
    for qcode, df_disp in per_q_disp.items():
        with tabs[idx]:
            qtext = per_q_text.get(qcode, "")
            st.subheader(f"{qcode} — {qtext}")
            metric_col, ui_label, summary_allowed = per_q_metric.get(qcode, ("POSITIVE", "(% positive answers)", True))
            trend_df = PD.DataFrame()
            if summary_allowed:
                trend_df = build_trend_summary_table(df_disp=df_disp, category_in_play=category_in_play, metric_col=metric_col, selected_years=years)

            st.markdown("#### Summary results")
            if summary_allowed and trend_df is not None and not trend_df.empty:
                st.markdown(f"<div style='font-size:14px;color:#333;margin-top:-4px;margin-bottom:2px;'>{qcode} — {qtext}</div><div style='font-size:12px;color:#666;margin-top:-4px;margin-bottom:10px;'>{ui_label}</div>", unsafe_allow_html=True)
                st.dataframe(make_arrow_safe(trend_df), use_container_width=True, hide_index=True)
            else:
                st.info("Summary table is unavailable for this selection, please see detailed results below.")

            st.markdown("#### Detailed results")
            df_pruned = _drop_all_na_columns_for_display(df_disp)
            st.dataframe(make_arrow_safe(df_pruned), use_container_width=True)

            st.markdown(
                "Source: [2024 Public Service Employee Survey Results - Open Government Portal]"
                "(https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f)"
            )
        idx += 1

# ─────────────────────────────
# Entry point
# ─────────────────────────────
def run_wizard():
    _init_state()

    # Optional: backend caption (mirrors main page style)
    try:
        info = get_backend_info() or {}
        store = info.get("store", "")
        if store == "in_memory_csv":
            st.caption("🧠 In-memory data store — fast lookups.")
        elif store == "csv":
            st.caption(f"⚠️ CSV fallback in use. Path: {info.get('csv_path')}")
        elif info.get("parquet_dir"):
            st.caption(f"✅ Parquet directory: {info['parquet_dir']}")
    except Exception:
        pass

    step = st.session_state[WKEY]["step"]
    if step == 1:
        _step1_question_selection()
    elif step == 2:
        _step2_years()
    elif step == 3:
        _step3_demographic_and_run()
        # If results from a recent run exist, render them below
        res = st.session_state[WKEY].get("results", {})
        if res and (res.get("per_q_disp") or res.get("cross_summary") is not None):
            st.markdown("---")
            _render_results()

if __name__ == "__main__":
    run_wizard()
