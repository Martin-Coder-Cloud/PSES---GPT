# utils/hybrid_search.py
# -------------------------------------------------------------------------
# Hybrid search for survey questions (LOCAL, API-free)
# - Semantic: sentence-transformer embeddings (if available)
# - Lexical: exact code, substring, token coverage, Jaccard, bigram phrase
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

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", str(s).lower()).strip()

def _tokens(s: str) -> List[str]:
    return _word_re.findall(_norm(s))

def _tokset(s: str) -> set:
    return set(_tokens(s))

# -----------------------------
# Embedding model & cache
# -----------------------------
# Pick a small, reliable model; allow override via env var PSES_EMBED_MODEL
_DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("PSES_EMBED_MODEL") or "",  # explicit override first
    "paraphrase-multilingual-MiniLM-L12-v2",   # multilingual (good for EN/FR)
    "all-MiniLM-L6-v2",                        # fast, popular English model
]

_MODEL: Optional["SentenceTransformer"] = None
_INDEX_CACHE: Dict[str, Dict[str, object]] = {}  # keyed by hash of (code|text)

def _load_model() -> Optional["SentenceTransformer"]:
    """Load a sentence-transformer model once (if available)."""
    global _MODEL
    if not _ST_AVAILABLE:
        return None
    if _MODEL is not None:
        return _MODEL
    for name in _DEFAULT_MODEL_CANDIDATES:
        if not name:
            continue
        try:
            _MODEL = SentenceTransformer(name)
            return _MODEL
        except Exception:
            continue
    # final attempt with a very small model name (if candidates fail)
    try:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL
    except Exception:
        return None

def _hash_qdf(qdf: pd.DataFrame) -> str:
    """Stable hash for the current code/text catalogue to reuse cached index."""
    if qdf is None or qdf.empty:
        return "empty"
    parts = (qdf["code"].astype(str) + "||" + qdf["text"].astype(str)).tolist()
    h = hashlib.sha256(("\n".join(parts)).encode("utf-8")).hexdigest()
    return h

def _build_index(qdf: pd.DataFrame) -> Dict[str, object]:
    """
    Build (or rebuild) the in-memory index for this qdf.
    Returns dict with: codes, texts, displays, norms, vecs (L2-normalized), model_name
    """
    model = _load_model()
    codes = qdf["code"].astype(str).tolist()
    texts = qdf["text"].astype(str).tolist()
    displays = qdf["display"].astype(str).tolist()
    norms = [f"{_norm(c)} — {_norm(t)}" for c, t in zip(codes, texts)]
    emb_strings = [f"{c} — {t}" for c, t in zip(codes, texts)]

    vecs = None
    model_name = None
    if model is not None:
        try:
            arr = np.array(model.encode(emb_strings, convert_to_numpy=True, show_progress_bar=False))
            # L2 normalize for cosine via dot product
            norms_arr = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            vecs = arr / norms_arr
            model_name = getattr(model, "model_name_or_path", "sentence-transformers")
        except Exception:
            vecs = None
            model_name = None

    return {
        "codes": codes,
        "texts": texts,
        "displays": displays,
        "norms": norms,   # normalized text for lexical features
        "vecs": vecs,     # np.ndarray [N, D] (normalized), or None
        "model_name": model_name,
    }

def _get_index(qdf: pd.DataFrame) -> Dict[str, object]:
    """Fetch cached index for this qdf (by hash), building if needed."""
    key = _hash_qdf(qdf)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    idx = _build_index(qdf)
    _INDEX_CACHE[key] = idx
    return idx

# -----------------------------
# Lexical scoring (local)
# -----------------------------
def _lexical_features(q_norm: str, q_tokens: List[str], bigrams: List[str], item_norm: str) -> Dict[str, float]:
    """
    Compute lexical features in [0,1]:
      - in_code/in_text are inferred via item_norm containing code/text; we rely on item_norm already being "code — text"
      - token coverage: fraction of query tokens present (substring or prefix tolerant)
      - jaccard: symmetric token overlap
      - bigram_hit: any adjacent token phrase present
    """
    # item_norm is "<code_n> — <text_n>"
    in_item = 1.0 if q_norm and (q_norm in item_norm) else 0.0

    # token coverage (prefix tolerant)
    item_words = _tokset(item_norm)
    matched = 0
    for tok in q_tokens:
        if (tok in item_norm) or any(w.startswith(tok) or tok.startswith(w) for w in item_words):
            matched += 1
    coverage = matched / max(len(q_tokens), 1)

    # jaccard
    qset = set(q_tokens)
    inter = len(qset & item_words)
    union = len(qset | item_words) or 1
    jaccard = inter / union

    # bigram phrase
    bigram_hit = 1.0 if any(bg in item_norm for bg in bigrams) else 0.0

    return {
        "substr": in_item,
        "coverage": coverage,
        "jaccard": jaccard,
        "bigram": bigram_hit,
    }

# -----------------------------
# Main search
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
    Note: This is fully LOCAL. No OpenAI or external APIs are called.
    """
    if qdf is None or qdf.empty or not query or not str(query).strip():
        return qdf.head(0)

    q_raw = str(query).strip()
    q_norm = _norm(q_raw)
    q_tokens = _tokens(q_raw)
    bigrams = [" ".join(q_tokens[i:i+2]) for i in range(len(q_tokens) - 1)] if len(q_tokens) >= 2 else []

    # Build/retrieve index & (maybe) embeddings
    idx = _get_index(qdf)
    codes: List[str] = idx["codes"]  # type: ignore
    texts: List[str] = idx["texts"]  # type: ignore
    displays: List[str] = idx["displays"]  # type: ignore
    norms: List[str] = idx["norms"]  # type: ignore
    vecs = idx["vecs"]               # type: ignore
    has_semantic = isinstance(vecs, np.ndarray)

    # Precompute query embedding if available
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
    for code, text, disp, item_norm, in_vec in zip(codes, texts, displays, norms, (vecs if has_semantic else [None]*len(codes))):
        # Exact code match signal (strong)
        exact_code = 1.0 if q_norm.upper() == code.upper() else 0.0

        # Lexical features (0..1)
        lf = _lexical_features(q_norm, q_tokens, bigrams, item_norm)

        # Semantic sim (0..1)
        sim = 0.0
        if has_semantic and qvec is not None and in_vec is not None:
            sim = float(np.dot(in_vec, qvec))  # cosine because both L2-normalized

        # Blend (weights tuned for small catalogues)
        # Keep scale ~[0,1.4] with an exact-code boost.
        score = (
            0.60 * sim +
            0.20 * lf["substr"] +
            0.10 * lf["coverage"] +
            0.05 * lf["jaccard"] +
            0.05 * lf["bigram"]
        )
        if exact_code >= 1.0:
            score += 0.40  # push exact code to the top

        if score <= min_score:
            continue

        # Hit type for UI/debug
        if exact_code >= 1.0:
            hit_type = "exact"
        elif lf["substr"] >= 1.0:
            hit_type = "substring"
        elif sim >= 0.60:
            hit_type = "semantic"
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
# --- at the bottom of utils/hybrid_search.py ---

def get_embedding_status() -> dict:
    """
    Return a lightweight status dict about local embeddings usage.
    Keys:
      - sentence_transformers_installed: bool
      - model_loaded: bool
      - model_name: str|None
      - catalogues_indexed: int   (# of qdf catalogues cached in memory)
    """
    status = {
        "sentence_transformers_installed": bool(_ST_AVAILABLE),
        "model_loaded": False,
        "model_name": None,
        "catalogues_indexed": len(_INDEX_CACHE),
    }
    if _ST_AVAILABLE:
        try:
            m = _load_model()
            status["model_loaded"] = bool(m)
            status["model_name"] = getattr(m, "model_name_or_path", None) if m else None
        except Exception:
            pass
    return status
