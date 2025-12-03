# hybrid_search.py
# -------------------------------------------------------------------------
# Lexical-first questionnaire search with AI Semantic Search (OpenAI).
#
# - Input:  qdf with columns ['code', 'text', 'display'] from Survey Questions.xlsx
# - Output: DataFrame with columns:
#       code, text, display, score, origin
#   where origin is 'lex' for lexical matches and 'sem' for AI semantic matches.
#
# Semantic engine:
#   - Uses OpenAI embeddings (e.g. text-embedding-3-small).
#   - Embeds the full catalogue of question texts once per process and caches.
#   - Embeds each user query once and computes cosine similarity.
#
# Diagnostics:
#   - get_embedding_status()      -> semantic engine status
#   - get_last_search_metrics()   -> last search metrics
# -------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Set
import hashlib
import os
import re
import time

import pandas as pd

# -------------------------------------------------------------------------
# Semantic engine flags / globals (OpenAI backend)
# -------------------------------------------------------------------------

_ST_OK: bool = False          # True if NumPy is available
_ST_CLIENT = None             # OpenAI client (lazy-init)
_ST_NAME = os.environ.get(
    "PSES_EMBED_MODEL",       # optional override for this app
    os.environ.get(           # optional override for menu1
        "MENU1_EMBED_MODEL",
        "text-embedding-3-small",  # default OpenAI embedding model
    ),
)
_ST_LAST_ERROR: Optional[str] = None

try:
    import numpy as np  # type: ignore
    _ST_OK = True
except Exception:
    np = None  # type: ignore
    _ST_OK = False

# -------------------------------------------------------------------------
# Normalization / tokenization helpers
# -------------------------------------------------------------------------

_word_re = re.compile(r"[a-z0-9']+")

_stop = {
    "the", "and", "of", "to", "in", "for", "with", "by", "on", "at", "from",
    "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "as", "that", "this", "these", "those", "or", "but", "if",
    "i", "me", "my", "we", "our", "you", "your", "they", "them", "their",
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _tokens(text: str) -> List[str]:
    text = _norm(text)
    return _word_re.findall(text)


def _stems(tokens: List[str]) -> List[str]:
    """
    Very lightweight stemmer:
    - drops plural 's', 'es'
    - handles 'ies' -> 'y'
    - skips stopwords
    """
    stems: List[str] = []
    for t in tokens:
        if t in _stop:
            continue
        if t.endswith("ies") and len(t) > 4:
            stems.append(t[:-3] + "y")
        elif t.endswith("es") and len(t) > 3:
            stems.append(t[:-2])
        elif t.endswith("s") and len(t) > 3:
            stems.append(t[:-1])
        else:
            stems.append(t)
    return stems


def _char4(token: str) -> List[str]:
    token = "#" + token + "#"
    grams: List[str] = []
    for i in range(len(token) - 3):
        grams.append(token[i : i + 4])
    return grams


# -------------------------------------------------------------------------
# Question code hints (e.g., "Q16", "question 27b")
# -------------------------------------------------------------------------

_CODE_HINT_RE = re.compile(r"(?i)(?:q|question)\s*([0-9]{1,3})([a-z]?)")


def _parse_qcode_hint(text: str) -> Tuple[Optional[int], str]:
    m = _CODE_HINT_RE.search(text or "")
    if not m:
        return None, ""
    num, suffix = m.groups()
    try:
        n = int(num)
    except Exception:
        return None, ""
    return n, (suffix or "")


def _extract_code_hints(raw_query: str) -> List[Tuple[int, str]]:
    hints: List[Tuple[int, str]] = []
    for num, suf in _CODE_HINT_RE.findall(raw_query or ""):
        try:
            hints.append((int(num), _norm(suf)))
        except Exception:
            pass

    collapsed = _norm((raw_query or "")).replace(" ", "")
    m = re.match(r"(?i)^(?:q|question)([0-9]{1,3})([a-z]?)$", collapsed)
    if m:
        num, suf = m.groups()
        try:
            hints.append((int(num), _norm(suf)))
        except Exception:
            pass

    return hints


# -------------------------------------------------------------------------
# 4-gram / IDF-style weighting for lexical similarity
# -------------------------------------------------------------------------

def _qgrams(text: str) -> Set[str]:
    grams: Set[str] = set()
    toks = _stems(_tokens(text))
    for t in toks:
        grams.update(_char4(t))
    return grams


_EMBED_CACHE: Dict[str, "np.ndarray"] = {}      # key -> embedding matrix
_TXT_CACHE: Dict[str, List[str]] = {}           # key -> list[texts]
_LAST_SEARCH_METRICS: Dict[str, object] = {}    # for diagnostics

_GRAM_DF: Dict[str, int] = {}
_GRAM_INFORMATIVE: Set[str] = set()
_GRAM_READY: bool = False


def _build_gram_df(texts: List[str]) -> None:
    """
    Build a simple DF-like document frequency for character 4-grams and
    identify "informative" grams (not too rare or too common).
    """
    global _GRAM_DF, _GRAM_INFORMATIVE, _GRAM_READY
    if _GRAM_READY:
        return

    df: Dict[str, int] = {}
    for txt in texts:
        grams = set()
        toks = _stems(_tokens(txt))
        for t in toks:
            grams.update(_char4(t))
        for g in grams:
            df[g] = df.get(g, 0) + 1

    _GRAM_DF = df
    total = len(texts) or 1
    informative: Set[str] = set()
    for g, c in df.items():
        freq = c / total
        # Keep grams that are neither too rare nor too common
        if 0.01 <= freq <= 0.7:
            informative.add(g)

    _GRAM_INFORMATIVE = informative
    _GRAM_READY = True


def _jaccard_informative_grams(qgrams: Set[str], tgrams: Set[str]) -> float:
    if not _GRAM_READY:
        return 0.0
    iq = qgrams & _GRAM_INFORMATIVE
    it = tgrams & _GRAM_INFORMATIVE
    if not iq and not it:
        return 0.0
    inter = len(iq & it)
    union = len(iq | it)
    return inter / union if union else 0.0


# -------------------------------------------------------------------------
# Embedding helpers (OpenAI backend)
# -------------------------------------------------------------------------

def _index_key(texts: List[str]) -> str:
    """
    Stable hash key for a specific ordered list of texts.
    Used to cache embeddings for the questionnaire.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update((_norm(t) + "\n").encode("utf-8"))
    return h.hexdigest()


def _get_openai_client():
    """
    Lazily initialize and return an OpenAI client for embeddings.
    Returns None if OPENAI_API_KEY is missing or client init fails.
    """
    global _ST_CLIENT, _ST_LAST_ERROR

    if not _ST_OK:
        _ST_LAST_ERROR = "NumPy not available for embeddings."
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        _ST_LAST_ERROR = "OPENAI_API_KEY missing for AI Semantic Search."
        return None

    if _ST_CLIENT is None:
        try:
            from openai import OpenAI  # type: ignore
            _ST_CLIENT = OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover - environment-specific
            _ST_LAST_ERROR = f"OpenAI client init failed: {type(exc).__name__}"
            _ST_CLIENT = None
            return None

    return _ST_CLIENT


def _get_semantic_matrix(texts: List[str]) -> Optional["np.ndarray"]:
    """
    Build (or reuse) a normalized embedding matrix for the given texts
    using OpenAI embeddings. Results are cached by a stable hash key so
    subsequent calls are cheap.

    Returns:
        np.ndarray of shape (N, D) with L2-normalized rows, or None on error.
    """
    if not _ST_OK:
        return None

    client = _get_openai_client()
    if client is None:
        return None

    key = _index_key(texts)

    # Reuse cached embeddings if the catalogue hasn't changed
    if key in _EMBED_CACHE and _TXT_CACHE.get(key) == texts:
        return _EMBED_CACHE[key]

    try:
        rsp = client.embeddings.create(model=_ST_NAME, input=texts)
        vecs = [item.embedding for item in rsp.data]
        mat = np.asarray(vecs, dtype="float32")

        # L2 normalize rows so cosine similarity is just a dot product
        norms = (mat ** 2).sum(axis=1, keepdims=True) ** 0.5
        norms[norms == 0.0] = 1.0
        mat = mat / norms

    except Exception as exc:  # pragma: no cover - network / API dependent
        global _ST_LAST_ERROR
        _ST_LAST_ERROR = f"Embedding build error: {type(exc).__name__}"
        return None

    _EMBED_CACHE[key] = mat
    _TXT_CACHE[key] = texts
    return mat


def _cosine_sim(vecA: "np.ndarray", matB: "np.ndarray") -> "np.ndarray":
    """
    Cosine similarity for normalized vectors: dot product of query with matrix.
    """
    return matB @ vecA


# -------------------------------------------------------------------------
# Public entry point: hybrid_question_search
# -------------------------------------------------------------------------

def hybrid_question_search(
    qdf: pd.DataFrame,
    raw_query: str,
    *,
    max_rows: int = 100,
    min_score: float = 0.40,
) -> pd.DataFrame:
    """
    Hybrid (lexical + AI Semantic Search) questionnaire search.

    Parameters
    ----------
    qdf : DataFrame
        Must contain columns: 'code', 'text', 'display'
        (coming from Survey Questions.xlsx metadata).
    raw_query : str
        User-entered search query (keywords, codes, etc.).
    max_rows : int
        Maximum number of rows to return.
    min_score : float
        Minimum lexical score threshold to count as a lexical hit.

    Returns
    -------
    DataFrame with columns:
        code, text, display, score, origin
    where:
        origin = 'lex' for lexical hits
        origin = 'sem' for AI semantic hits (OpenAI embeddings)
    """
    global _LAST_SEARCH_METRICS

    t0 = time.time()
    raw_query = raw_query or ""
    q = _norm(raw_query)

    if not q.strip():
        # Empty query -> empty result, but keep columns for safety
        return qdf.head(0).assign(score=[], origin=[])

    # Prepare base catalogue from metadata
    base = qdf.copy()
    base["__display__"] = base.get("display", base.get("text", ""))
    base["__text__"] = base.get("text", base.get("display", ""))

    # Build gram DF once for lexical similarity
    _build_gram_df(base["__text__"].tolist())

    # Query lexical signals
    q_tokens = _tokens(q)
    q_stems = _stems(q_tokens)
    q_grams = _qgrams(q)

    codes = base["code"].astype(str).fillna("").tolist()
    texts = base["__text__"].astype(str).fillna("").tolist()
    displays = base["__display__"].astype(str).fillna("").tolist()

    lex_scores: List[float] = []
    has_lex: List[bool] = []

    # Code hints (e.g., "Q16", "question 32b")
    code_hints = _extract_code_hints(raw_query)
    hint_nums = {n for (n, _) in code_hints}

    # Lexical scoring loop
    for code, text, disp in zip(codes, texts, displays):
        score = 0.0

        # Direct question code hint (e.g., user refers to "Q16")
        try:
            m = re.match(r"(?i)q?([0-9]{1,3})([a-z]?)", code or "")
            if m:
                num = int(m.group(1))
                suf = _norm(m.group(2))
                if num in hint_nums:
                    score = max(score, 0.95)
        except Exception:
            pass

        # Token overlap (Jaccard on stems)
        toks_t = _tokens(text + " " + disp)
        stems_t = _stems(toks_t)
        if q_stems and stems_t:
            inter = len(set(q_stems) & set(stems_t))
            union = len(set(q_stems) | set(stems_t))
            if union:
                score = max(score, inter / union)

        # 4-gram Jaccard (informative grams only)
        grams_t = _qgrams(text + " " + disp)
        gram_score = _jaccard_informative_grams(q_grams, grams_t)
        score = max(score, gram_score)

        lex_scores.append(score)
        has_lex.append(score >= min_score)

    df_all = pd.DataFrame(
        {
            "code": codes,
            "text": texts,
            "display": displays,
            "lex_score": lex_scores,
            "has_lex": has_lex,
        }
    )

    # Lexical rows
    df_lex = df_all[df_all["has_lex"]].copy()
    df_lex["score"] = df_lex["lex_score"]
    df_lex["origin"] = "lex"

    # Semantic side (AI Semantic Search) for NON-lex items
    N = len(df_all)
    df_nonlex = df_all[~df_all["has_lex"]].reset_index(drop=True)

    if _ST_OK and not df_nonlex.empty:
        try:
            # Build / reuse embedding matrix for the full catalogue
            mat = _get_semantic_matrix(texts)
            client = _get_openai_client()

            if mat is not None and client is not None:
                # Embed the raw query once (OpenAI)
                rsp = client.embeddings.create(model=_ST_NAME, input=[raw_query])
                qvec = np.asarray(rsp.data[0].embedding, dtype="float32")
                norm = float((qvec ** 2).sum() ** 0.5)
                if norm == 0.0:
                    norm = 1.0
                qvec = qvec / norm

                sim = _cosine_sim(qvec, mat)        # [-1, 1]
                sem_all = ((sim + 1.0) / 2.0).tolist()  # [0, 1]
            else:
                sem_all = [0.0] * N

        except Exception as exc:  # defensive: never crash semantic
            global _ST_LAST_ERROR
            _ST_LAST_ERROR = f"Query embedding error: {type(exc).__name__}"
            sem_all = [0.0] * N
    else:
        sem_all = [0.0] * N

    SEM_FLOOR = 0.43  # semantic similarity threshold

    sem_rows: List[Dict[str, object]] = []
    for i in range(N):
        if has_lex[i]:
            continue  # semantic only used for non-lex rows
        s = sem_all[i]
        if s >= SEM_FLOOR:
            sem_rows.append(
                {
                    "code": codes[i],
                    "text": texts[i],
                    "display": displays[i],
                    "score": s,
                    "origin": "sem",  # AI Semantic Search
                }
            )

    df_sem = pd.DataFrame(sem_rows)

    # Combine, sort, cap rows
    out = pd.concat(
        [
            df_lex[["code", "text", "display", "score", "origin"]],
            df_sem[["code", "text", "display", "score", "origin"]],
        ],
        ignore_index=True,
    )
    out = out.sort_values(["score", "code"], ascending=[False, True])
    out = out.head(max_rows).reset_index(drop=True)

    t1 = time.time()
    _LAST_SEARCH_METRICS = {
        "query": raw_query,
        "rows_lex": int(df_lex.shape[0]),
        "rows_sem": int(df_sem.shape[0]),
        "t_total_ms": int((t1 - t0) * 1000),
        "semantic_enabled": bool(_ST_OK),
        "semantic_model": _ST_NAME,
    }

    return out


# -------------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------------

def get_embedding_status() -> Dict[str, object]:
    """
    Return a snapshot of the AI Semantic Search engine status.
    """
    status: Dict[str, object] = {
        "semantic_library_installed": bool(_ST_OK),
        "sentence_transformers_version": None,   # no longer used (OpenAI backend)
        "torch_version": None,
        "device": "n/a",                         # purely API-based now
        "model_name": _ST_NAME,
        "model_loaded": bool(_ST_CLIENT is not None),
        "embedding_index_ready": bool(_EMBED_CACHE),
        "catalogues_indexed": len(_EMBED_CACHE) or 0,
    }
    if _ST_LAST_ERROR:
        status["last_error"] = _ST_LAST_ERROR
    return status


def get_last_search_metrics() -> Dict[str, object]:
    """
    Return metrics of the last hybrid_question_search() call.
    """
    return dict(_LAST_SEARCH_METRICS)
