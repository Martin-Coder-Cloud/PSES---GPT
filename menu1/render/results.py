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

def _is_d57_exception(q: str) -> bool:
    qn = str(q).strip().upper().replace("-", "_")
    return qn in {"D57_A", "D57_B", "D57A", "D57B"}

def _sanitize_9999(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    num_cols = []
    for c in out.columns:
        lc = str(c).lower().replace(" ", "")
        if lc in {"positive", "negative", "agree",
                  "answer1", "answer2", "answer3", "answer4", "answer5", "answer6"}:
            num_cols.append(c)
        elif re.fullmatch(r"answer\s*[1-6]", str(c), flags=re.IGNORECASE):
            num_cols.append(c)
    for c in set(num_cols):
        out[c] = pd.to_numeric(out[c], errors="coerce").replace(9999, pd.NA)
    return out

# ---------------------- metadata (polarity + scales) ----------------------

def _first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def _load_survey_questions_meta() -> pd.DataFrame:
    try:
        path = _first_existing_path([
            "metadata/Survey Questions.xlsx",
            "./Survey Questions.xlsx",
            "Survey Questions.xlsx",
        ])
        if not path:
            return pd.DataFrame(columns=["code", "polarity", "positive", "negative", "agree"])
        df = pd.read_excel(path)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if "question" in df.columns and "code" not in df.columns:
            df = df.rename(columns={"question": "code"})
        df["code"] = df["code"].astype(str).str.strip().str.upper()
        if "polarity" not in df.columns:
            df["polarity"] = "POS"
        df["polarity"] = df["polarity"].astype(str).str.upper().str.strip()
        for c in ("positive", "negative", "agree"):
            if c not in df.columns:
                df[c] = None
            else:
                df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) else None)
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "polarity", "positive", "negative", "agree"])

@st.cache_data(show_spinner=False)
def _load_survey_scales_meta() -> pd.DataFrame:
    try:
        path = _first_existing_path([
            "metadata/Survey Scales.xlsx",
            "./Survey Scales.xlsx",
            "Survey Scales.xlsx",
        ])
        if not path:
            return pd.DataFrame(columns=["code","value","label"])
        df = pd.read_excel(path)
        df.columns = [c.strip().lower() for c in df.columns]
        def pick(opts):
            for n in opts:
                if n in df.columns:
                    return n
            return None
        c_code  = pick(["code","question","qcode","scale","item"])
        c_val   = pick(["value","option","index","answer","order","position"])
        c_label = pick(["label","answer_label","option_label","text","desc","description"])
        if not (c_code and c_val and c_label):
            return pd.DataFrame(columns=["code","value","label"])
        out = df[[c_code, c_val, c_label]].copy()
        out.columns = ["code", "value", "label"]
        out["code"] = out["code"].astype(str).str.strip().str.upper()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["code","value","label"])
        out["value"] = out["value"].astype(int)
        out["label"] = out["label"].astype(str).str.strip()
        return out
    except Exception:
        return pd.DataFrame(columns=["code","value","label"])

def _parse_index_list(s: Optional[str]) -> List[int]:
    if not s or not isinstance(s, str):
        return []
    toks = re.split(r"[,\;\|\s]+", s.strip())
    out: List[int] = []
    for t in toks:
        if not t:
            continue
        try:
            out.append(int(float(t)))
        except Exception:
            continue
    return out

# ---------------------- Literal meaning label logic ----------------------

def _meaning_labels_for_question(
    *,
    qcode: str,
    question_text: Optional[str],
    reporting_field: Optional[str],
    metric_label: str,
    meta_q: pd.DataFrame,
    meta_scales: pd.DataFrame
) -> List[str]:
    """Return exact labels from metadata (Survey Questions + Survey Scales)."""
    try:
        qU = str(qcode).strip().upper()
        row = meta_q[meta_q["code"] == qU]
        if row.empty:
            return []

        colname = None
        if reporting_field:
            colname = reporting_field.lower()
        if colname not in ("positive", "negative", "agree"):
            colname = "positive"

        idxs: List[int] = _parse_index_list(row.iloc[0].get(colname, None))
        if not idxs:
            # Try metric label parsing as backup
            return _derive_labels_from_metric_label(metric_label or "")

        # map indices to labels in Survey Scales
        def _map_by_code(key: str) -> List[str]:
            sc = meta_scales[meta_scales["code"] == str(key).strip().upper()]
            if sc.empty:
                return []
            m = {int(v): str(l) for v, l in zip(sc["value"], sc["label"])}
            return [m[i] for i in idxs if i in m]

        labels = _map_by_code(qU)
        if not labels and "scale" in row.columns:
            alt = str(row.iloc[0]["scale"] or "").strip()
            if alt:
                labels = _map_by_code(alt)
        return labels
    except Exception:
        return []

def _meaning_labels_for_build(q: str, qtext: str, metric_col: Optional[str], metric_label: str,
                              meta_q: pd.DataFrame, meta_scales: pd.DataFrame) -> List[str]:
    """Select column by Polarity (POS→POSITIVE, NEG→NEGATIVE, NEU→AGREE)."""
    qU = str(q).strip().upper()
    row = meta_q[meta_q["code"] == qU]
    field = None
    if not row.empty:
        pol = str(row.iloc[0].get("polarity", "POS")).upper().strip()
        if pol == "NEG":
            field = "negative"
        elif pol == "NEU":
            field = "agree"
        else:
            field = "positive"
    else:
        field = "positive"

    return _meaning_labels_for_question(
        qcode=q,
        question_text=qtext,
        reporting_field=field,
        metric_label=metric_label or "",
        meta_q=meta_q,
        meta_scales=meta_scales
    )

def _compress_labels_for_footnote(labels: List[str]) -> Optional[str]:
    """Literal join only—no prefix/suffix compression."""
    if not labels:
        return None
    if len(labels) == 1:
        return f"({labels[0]})"
    return "(" + "/".join(labels) + ")"

def _derive_labels_from_metric_label(metric_label: str) -> List[str]:
    if not isinstance(metric_label, str):
        return []
    m = re.search(r"%\s*selecting\s*(.+)$", metric_label.strip(), flags=re.IGNORECASE)
    if not m:
        return []
    tail = m.group(1).strip()
    parts = [p.strip(" []") for p in re.split(r"\s*/\s*", tail) if p.strip()]
    return parts

def _resolve_footnote_labels(
    *, q: str, qtext: str, metric_col: Optional[str], metric_label: str,
    meta_q: pd.DataFrame, meta_scales: pd.DataFrame
) -> List[str]:
    """Literal metadata lookup only; fallback to metric label if available."""
    labels = _meaning_labels_for_build(q, qtext, metric_col, metric_label, meta_q, meta_scales)
    if labels:
        return labels
    labels = _derive_labels_from_metric_label(metric_label or "")
    return labels

# --------------------------------------------------------------------------
# (rest of file unchanged — summary building, AI rendering, etc.)
# --------------------------------------------------------------------------
# The rest of your provided file remains exactly as it was.
