# utils/hybrid_search.py
# -------------------------------------------------------------------------
# Hybrid search for survey questions (LOCAL, API-free, stricter)
# - Semantic: sentence-transformer embeddings (if available)
# - Lexical: exact code, substring, token coverage, Jaccard, bigram phrase
# - STRICT LEXICAL GATE: require at least one lexical anchor from the query
# - Synonym bridge only when embeddings are unavailable
# - Blended score; fixed threshold (>0.40)
# -------------------------------------------------------------------------

from __future__ import annotations
import os
import re
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Try to load a local sentence-embeddings model; fall back to lexical-only if unavailable
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

# -----------------------------
# Normalization / token helpers
# -----------------------------
_word_re = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "the","and","of","to","in","for","with","on","at","by","from",
    "a","an","is","are","was","were","be","been","being",
    "as","it","its","this","that","these","those","overall"
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", str(s).lower()).strip()

def _tokens(s: str) -> List[str]:
    return _word_re.findall(_norm(s))

def _tokset(s: str) -> set:
    return set(_tokens(s))

# -----------------------------
# Lightweight synonym map (used ONLY if embeddings are not available)
# -----------------------------
_SYNONYMS: Dict[str, List[str]] = {
    "career": ["career", "job", "work", "profession", "employment", "occupational"],
    "advancement": ["advancement", "progression", "promotion", "advance", "advancing", "development", "growth"],
    "progression": ["progression", "advancement", "promotion", "career growth", "development"],
    "promotion": ["promotion", "promotions", "promote", "promoted", "advancement", "progression"],
    "satisfaction": ["satisfaction", "satisfied", "satisfying", "overall satisfaction"],
    "recognition": ["recognition", "appreciation", "acknowledgment", "acknowledgement"],
    "leadership": ["leadership", "leaders", "management", "manager"],
    "inclusion": ["inclusion", "inclusive", "belonging", "equity"],
    "wellbeing": ["wellbeing", "well-being", "well being", "wellness"],
}

def _alts_for(tok: str) -> List[str]:
    t = tok.lower()
    return _SYNONYMS.get(t, [t])

# -----------------------------
# Embedding model & cache
# -----------------------------
_DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("PSES_EMBED_MODEL") or "",
    "paraphrase-multilingual-MiniLM-L12-v2",   # multilingual
    "all-MiniLM-L6-v2",                        # fast English model
]

_MODEL: Optional["SentenceTransformer"] = None
_MODEL_NAME: Optional[str] = None
_INDEX_CACHE: Dict[str, Dict[str, object]] = {}

def _load_model() -> Optional["SentenceTransformer"]:
    global _MODEL, _MODEL_NAME
    if not _ST_AVAILABLE:
        return None
    if _MODEL is not None:
        return _MODEL
    for name in _DEFAULT_MODEL_CANDIDATES:
        if not name:
            continue
        try:
            m = SentenceTransformer(name)
            _MODEL = m
            _MODEL_NAME = name
            return _MODEL
        except Exception:
            continue
    try:
        m = SentenceTransformer("all-MiniLM-L6-v2")
        _MODEL = m
        _MODEL_NAME = "all-MiniLM-L6-v2"
        return _MODEL
    except Exception:
        _MODEL = None
        _MODEL_NAME = None
        return None

def _hash_qdf(qdf: pd.DataFrame) -> str:
    if qdf is None or qdf.empty:
        return "empty"
    parts = (qdf["code"].astype(str) + "||" + qdf["text"].astype(str)).tolist()
    return hashlib.sha256(("\n".join(parts)).encode("utf-8")).hexdigest()

def _build_index(qdf: pd.DataFrame) -> Dict[str, object]:
    model = _load_model()
    codes = qdf["code"].astype(str).tolist()
    texts = qdf["text"].astype(str).tolist()
    displays = qdf["display"].astype(str).tolist()
    norms = [f"{_norm(c)} — {_norm(t)}" for c, t in zip(codes, texts)]
    emb_strings = [f"{c} — {t}" for c, t in zip(codes, texts)]

    vecs = None
    if model is not None:
        try:
            arr = np.array(model.encode(emb_strings, convert_to_numpy=True, show_progress_bar=False))
            norms_arr = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            vecs = arr / norms_arr
        except Exception:
            vecs = None

    return {
        "codes": codes,
        "texts": texts,
        "displays": displays,
        "norms": norms,
        "vecs": vecs,                 # np.ndarray or None
        "model_name": _MODEL_NAME,
    }

def _get_index(qdf: pd.DataFrame) -> Dict[str, object]:
    key = _hash_qdf(qdf)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    idx = _build_index(qdf)
    _INDEX_CACHE[key] = idx
    return idx

# -----------------------------
# Lexical features + strict anchor
# -----------------------------
def _lexical_features(q_norm: str, q_tokens_nostop: List[str], bigrams: List[str], item_norm: str, *, use_synonyms: bool) -> Dict[str, float]:
    """
    Compute:
      - substr: whole-query substring present (0/1)
      - coverage: fraction of NON-STOPWORD tokens present (prefix tolerant; synonyms only if use_synonyms=True)
      - jaccard: symmetric token overlap (using all tokens, not expanded)
      - bigram: any adjacent token phrase present (0/1)
      - anchor: strict boolean gate → require substr OR coverage>0 OR bigram
    """
    substr = 1.0 if q_norm and (q_norm in item_norm) else 0.0
    item_words = _tokset(item_norm)

    # coverage over non-stopword tokens
    matched = 0
    for tok in q_tokens_nostop:
        alts = _alts_for(tok) if use_synonyms else [tok]
        hit = False
        for a in alts:
            if (a in item_norm) or any(w.startswith(a) or a.startswith(w) for w in item_words):
                hit = True
                break
        if hit:
            matched += 1
    coverage = matched / max(len(q_tokens_nostop), 1)

    # jaccard (unexpanded, all tokens)
    qset_all = set(q_tokens_nostop)  # already non-stop
    inter = len(qset_all & item_words)
    union = len(qset_all | item_words) or 1
    jaccard = inter / union

    bigram_hit = 1.0 if any(bg in item_norm for bg in bigrams) else 0.0

    anchor = (substr >= 1.0) or (coverage > 0.0) or (bigram_hit > 0.0)
    return {"substr": substr, "coverage": coverage, "jaccard": jaccard, "bigram": bigram_hit, "anchor": float(anchor)}

# -----------------------------
# Main search (blended with strict gating)
# -----------------------------
def hybrid_question_search(
    qdf: pd.DataFrame,
    query: str,
    top_k: int = 120,
    min_score: float = 0.40
) -> pd.DataFrame:
    """
    Hybrid search over the question metadata (qdf).
    Returns DataFrame[score, hit_type, code, text, display].
    Fully LOCAL. No OpenAI or external APIs are called.
    """
    if qdf is None or qdf.empty or not query or not str(query).strip():
        return qdf.head(0)

    q_raw = str(query).strip()
    q_norm = _norm(q_raw)
    # non-stopword tokens for stricter gating/coverage
    q_tokens_nostop = [t for t in _tokens(q_raw) if t not in STOPWORDS]
    bigrams = [" ".join(q_tokens_nostop[i:i+2]) for i in range(len(q_tokens_nostop) - 1)] if len(q_tokens_nostop) >= 2 else []

    # Build/retrieve index & embeddings
    idx = _get_index(qdf)
    codes: List[str] = idx["codes"]      # type: ignore
    texts: List[str] = idx["texts"]      # type: ignore
    displays: List[str] = idx["displays"]# type: ignore
    norms: List[str] = idx["norms"]      # type: ignore
    vecs = idx["vecs"]                   # type: ignore
    has_semantic = isinstance(vecs, np.ndarray)

    # Query embedding if model present
    qvec = None
    if has_semantic:
        try:
            model = _load_model()
            qarr = np.array(model.encode([q_raw], convert_to_numpy=True, show_progress_bar=False))  # type: ignore
            qvec = qarr[0]
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        except Exception:
            qvec = None
            has_semantic = False

    rows: List[Tuple[float, str, str, str, str]] = []
    use_synonyms = not has_semantic  # synonym expansion only without embeddings

    for code, text, disp, item_norm, in_vec in zip(codes, texts, displays, norms, (vecs if has_semantic else [None]*len(codes))):
        exact_code = 1.0 if q_norm.upper() == code.upper() else 0.0

        # Lexical features & STRICT ANCHOR
        lf = _lexical_features(q_norm, q_tokens_nostop, bigrams, item_norm, use_synonyms=use_synonyms)
        if lf["anchor"] < 1.0:
            # Reject purely semantic matches with no lexical anchor from the query
            continue

        # Semantic cosine sim (0..1)
        sim = 0.0
        if has_semantic and qvec is not None and in_vec is not None:
            sim = float(np.dot(in_vec, qvec))

        # Blend (semantic dominates once anchor is satisfied)
        if has_semantic:
            score = (
                0.75 * sim +
                0.12 * lf["substr"] +
                0.08 * lf["coverage"] +
                0.03 * lf["bigram"] +
                0.02 * lf["jaccard"]
            )
            if exact_code >= 1.0:
                score += 0.30
        else:
            score = (
                0.25 * lf["substr"] +
                0.50 * lf["coverage"] +
                0.15 * lf["bigram"] +
                0.10 * lf["jaccard"]
            )
            if exact_code >= 1.0:
                score += 0.30

        if score <= min_score:
            continue

        if exact_code >= 1.0:
            hit_type = "exact"
        elif has_semantic and sim >= 0.60:
            hit_type = "semantic"
        elif lf["substr"] >= 1.0:
            hit_type = "substring"
        elif lf["bigram"] > 0.0:
            hit_type = "phrase"
        elif lf["coverage"] > 0.0:
            hit_type = "token"
        else:
            hit_type = "other"

        rows.append((float(score), hit_type, str(code), str(text), str(disp)))

    if not rows:
        return qdf.head(0)

    out = pd.DataFrame(rows, columns=["score", "hit_type", "code", "text", "display"])
    out = out.sort_values(["score", "code"], ascending=[False, True], kind="mergesort").head(top_k)
    return out.reset_index(drop=True)

# --- Status for Diagnostics ---
def get_embedding_status() -> dict:
    status = {
        "sentence_transformers_installed": bool(_ST_AVAILABLE),
        "model_loaded": False,
        "model_name": _MODEL_NAME,
        "catalogues_indexed": len(_INDEX_CACHE),
    }
    if _ST_AVAILABLE:
        try:
            m = _load_model()
            status["model_loaded"] = bool(m)
        except Exception:
            pass
    return status
