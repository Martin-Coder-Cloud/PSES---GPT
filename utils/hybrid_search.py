# utils/hybrid_search.py
# -------------------------------------------------------------------------
# Hybrid search for survey questions (LOCAL, API-free by default)
# - Lexical: phrase→AND→OR(coverage)→fuzzy cascade + normalized coverage score
# - Semantic (optional): sentence-transformer embeddings if available
# - Final score: blend(semantic, lexical); strict threshold (> min_score)
# - Dedupe by code; returns [code, text, display, score]
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import hashlib
import math
import os
import re

import pandas as pd

# Optional semantic support (gracefully degrades if unavailable)
_ST_OK = False
_ST_MODEL = None  # lazy-initialized
_ST_NAME = os.environ.get("MENU1_EMBED_MODEL", "all-MiniLM-L6-v2")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    _ST_OK = True
except Exception:
    _ST_OK = False

# -----------------------------
# Normalization / token helpers
# -----------------------------
_word_re = re.compile(r"[a-z0-9']+")
_stop = {
    "the","and","of","to","in","for","with","on","at","by","from",
    "a","an","is","are","was","were","be","been","being","or","as",
    "it","that","this","these","those","i","you","we","they","he","she",
}

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def _tokens(s: str) -> List[str]:
    return [t for t in _word_re.findall(_norm(s)) if t and t not in _stop]

def _uniq_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -----------------------------
# Light synonym bridge (only if semantic missing)
# -----------------------------
_SYNONYMS: Dict[str, List[str]] = {
    "harassment": ["harassment","respect","civility","bullying","discrimination","racism","microaggressions"],
    "recognition": ["recognition","appreciation","acknowledgement","reward","praise"],
    "onboarding": ["onboarding","orientation","induction","new hire","integration"],
    "career": ["career","advancement","promotion","development","progression","mobility"],
    "workload": ["workload","overtime","burnout","capacity","demand","pressure","work-life"],
    "psychological": ["psychological","mental health","well-being","stress","burnout","respect"],
}

def _expand_with_synonyms(qtoks: List[str]) -> List[str]:
    expanded = list(qtoks)
    for t in qtoks:
        if t in _SYNONYMS:
            expanded.extend(_SYNONYMS[t])
    return _uniq_preserve(expanded)

# -----------------------------
# Indexing & (optional) embeddings cache
# -----------------------------
_EMBED_CACHE: Dict[str, "np.ndarray"] = {}  # key: md5 of concatenated texts
_TXT_CACHE: Dict[str, List[str]] = {}       # mirror to validate cache key

def _index_key(texts: List[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update((_norm(t) + "\n").encode("utf-8"))
    return h.hexdigest()

def _get_semantic_matrix(texts: List[str]) -> Optional["np.ndarray"]:
    """Build/retrieve embeddings for texts if ST is available."""
    if not _ST_OK:
        return None
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            _ST_MODEL = SentenceTransformer(_ST_NAME)
        except Exception:
            return None
    key = _index_key(texts)
    if key in _EMBED_CACHE and _TXT_CACHE.get(key) == texts:
        return _EMBED_CACHE[key]
    try:
        mat = _ST_MODEL.encode(texts, normalize_embeddings=True)
    except Exception:
        return None
    _EMBED_CACHE[key] = mat
    _TXT_CACHE[key] = texts
    return mat

def _cosine_sim(vecA: "np.ndarray", matB: "np.ndarray") -> "np.ndarray":
    # Both already normalized
    return (matB @ vecA)

# -----------------------------
# Lexical scoring cascade
# -----------------------------
def _phrase_hits(q: str, items: List[str]) -> List[bool]:
    if not q:
        return [False]*len(items)
    # quoted phrases first
    phrases = re.findall(r'"([^"]+)"', q)
    marks = [False]*len(items)
    if not phrases:
        return marks
    for i, txt in enumerate(items):
        ntxt = _norm(txt)
        for ph in phrases:
            if _norm(ph) and _norm(ph) in ntxt:
                marks[i] = True
                break
    return marks

def _coverage_and(tokens: List[str], items: List[str]) -> Tuple[List[int], List[int]]:
    # require all tokens
    cov = [0]*len(items)
    tot = [len(tokens)]*len(items)
    tset = set(tokens)
    for i, txt in enumerate(items):
        toks = set(_tokens(txt))
        if tset.issubset(toks):
            cov[i] = len(tset)
    return cov, tot

def _coverage_or(tokens: List[str], items: List[str]) -> Tuple[List[int], List[int]]:
    cov = [0]*len(items)
    tot = [len(tokens)]*len(items)
    for i, txt in enumerate(items):
        toks = set(_tokens(txt))
        cov[i] = len(set(tokens) & toks)
    return cov, tot

def _coverage_or_fuzzy(tokens: List[str], items: List[str]) -> Tuple[List[int], List[int]]:
    # very light fuzzy (edit distance 1) to tolerate minor typos
    def ed1(a: str, b: str) -> bool:
        if abs(len(a)-len(b)) > 1:
            return False
        # simple DP-free check
        mismatches = 0
        ia = ib = 0
        while ia < len(a) and ib < len(b):
            if a[ia] == b[ib]:
                ia += 1; ib += 1
            else:
                mismatches += 1
                if mismatches > 1: return False
                if len(a) == len(b):
                    ia += 1; ib += 1
                elif len(a) > len(b):
                    ia += 1
                else:
                    ib += 1
        mismatches += (len(a)-ia) + (len(b)-ib)
        return mismatches <= 1

    cov = [0]*len(items)
    tot = [len(tokens)]*len(items)
    for i, txt in enumerate(items):
        toks = set(_tokens(txt))
        score = 0
        for t in tokens:
            if t in toks or any(ed1(t, w) for w in toks):
                score += 1
        cov[i] = score
    return cov, tot

def _normalize_cov(cov: List[int], tot: List[int]) -> List[float]:
    out: List[float] = []
    for c, t in zip(cov, tot):
        if t <= 0:
            out.append(0.0)
        else:
            out.append(max(0.0, min(1.0, c / t)))
    return out

# -----------------------------
# Public entry point
# -----------------------------
def hybrid_question_search(
    qdf: pd.DataFrame,
    query: str,
    *,
    top_k: int = 120,
    min_score: float = 0.40,
) -> pd.DataFrame:
    """
    Hybrid search over the question metadata (qdf).
    Returns DataFrame[code, text, display, score].
    - Lexical cascade ensures multi-keyword queries yield results.
    - Semantic (if available) surfaces related questions beyond exact words.
    - Dedupe by code; strict threshold (> min_score).
    """
    if qdf is None or qdf.empty or not query or not str(query).strip():
        return qdf.head(0)

    # Columns must exist
    for col in ("code", "text", "display"):
        if col not in qdf.columns:
            raise ValueError(f"qdf missing required column: {col}")

    # Prepare core arrays
    codes: List[str]   = qdf["code"].astype(str).tolist()
    texts: List[str]   = qdf["text"].astype(str).tolist()
    shows: List[str]   = qdf["display"].astype(str).tolist()

    q_raw = str(query).strip()
    q_norm = _norm(q_raw)

    # ----------------- LEXICAL SCORE -----------------
    qtoks = _tokens(q_norm)
    qtoks = _uniq_preserve(qtoks)

    # Phrase hits (quoted phrases only)
    phrase_marks = _phrase_hits(q_raw, texts)

    # AND coverage
    cov_and, tot_and = _coverage_and(qtoks, texts)
    and_norm = _normalize_cov(cov_and, tot_and)

    # OR coverage (graceful fallback if AND yields zero)
    cov_or, tot_or = _coverage_or(qtoks, texts)
    or_norm = _normalize_cov(cov_or, tot_or)

    # If AND totally empty and OR too sparse, allow fuzzy OR
    if max(and_norm or [0]) == 0 and max(or_norm or [0]) <= 0.34:
        cov_fz, tot_fz = _coverage_or_fuzzy(qtoks, texts)
        fz_norm = _normalize_cov(cov_fz, tot_fz)
    else:
        fz_norm = [0.0]*len(texts)

    # Coverage gate: require at least ceil(60% of tokens) for OR/fuzzy paths if AND fails
    need = max(1, math.ceil(0.60 * max(1, len(qtoks))))
    or_gate = [c >= need for c in cov_or]
    fz_gate = [c >= need for c in cov_fz] if 'cov_fz' in locals() else [False]*len(texts)

    # Lexical score: max of (phrase boost, AND, gated OR, gated fuzzy)
    lex_scores: List[float] = []
    for i in range(len(texts)):
        base = and_norm[i]
        if or_gate[i]:
            base = max(base, or_norm[i] * 0.9)
        if fz_gate[i]:
            base = max(base, fz_norm[i] * 0.8)
        if phrase_marks[i]:
            base = max(base, 1.0)  # strong signal
        lex_scores.append(max(0.0, min(1.0, base)))

    # If semantic is unavailable, lightly expand query with synonyms to avoid literal-only bias
    if not _ST_OK and qtoks:
        expanded = _expand_with_synonyms(qtoks)
        if len(expanded) > len(qtoks):
            cov_syn, tot_syn = _coverage_or(expanded, texts)
            syn_norm = _normalize_cov(cov_syn, tot_syn)
            lex_scores = [max(a, b * 0.85) for a, b in zip(lex_scores, syn_norm)]

    # ----------------- SEMANTIC SCORE (optional) -----------------
    if _ST_OK:
        mat = _get_semantic_matrix(texts)
        if mat is not None:
            try:
                global _ST_MODEL
                qvec = _ST_MODEL.encode([q_raw], normalize_embeddings=True)[0]
                sim = _cosine_sim(qvec, mat)  # [-1,1] but normalized vectors → [-1,1], clamp to [0,1]
                sim = (sim + 1.0) / 2.0
                sem_scores = sim.tolist()
            except Exception:
                sem_scores = [0.0]*len(texts)
        else:
            sem_scores = [0.0]*len(texts)
    else:
        sem_scores = [0.0]*len(texts)

    # ----------------- BLEND & RANK -----------------
    # Blend prioritizes meaning while keeping lexical precision
    blended = [min(1.0, max(0.0, 0.60 * s + 0.40 * l)) for s, l in zip(sem_scores, lex_scores)]

    # Build frame
    df = pd.DataFrame({
        "code": codes,
        "text": texts,
        "display": shows,
        "score": blended,
    })

    # Dedupe by code (keep highest score)
    df = df.sort_values("score", ascending=False).drop_duplicates(subset=["code"], keep="first")

    # Strict threshold (> min_score)
    df = df[df["score"] > float(min_score)]

    # Top-K
    if top_k is not None and top_k > 0:
        df = df.head(top_k)

    # Stable order for ties
    df = df.sort_values(["score", "code"], ascending=[False, True]).reset_index(drop=True)

    return df
