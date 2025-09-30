# utils/hybrid_search.py
# -------------------------------------------------------------------------
# Lexical-first search with semantic backfill (term-agnostic)
# Policy:
#   1) Return ALL lexical hits (any stem OR 4-gram overlap) with a score floor
#      so they pass the global threshold (> min_score).
#   2) ONLY if lexical hits < 5, add ALL semantic hits meeting a high cosine
#      threshold (and > min_score). Dedupe by code. Rank by score desc.
# Notes:
#   - Generic morphology: lightweight stemmer + char 4-grams (no hardcoded word lists)
#   - Optional semantic: sentence-transformers if available; otherwise lexical-only
#   - Operators: +include / -exclude (substring on normalized display+text)
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Optional
import hashlib
import os
import re
import pandas as pd

# Optional semantic support (graceful degradation)
_ST_OK: bool = False
_ST_MODEL = None
_ST_NAME = os.environ.get("MENU1_EMBED_MODEL", "all-MiniLM-L6-v2")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    _ST_OK = True
except Exception:
    _ST_OK = False

# -----------------------------
# Normalization / tokenization
# -----------------------------
_word_re = re.compile(r"[a-z0-9']+")
_stop = {
    "the","and","of","to","in","for","with","on","at","by","from",
    "a","an","is","are","was","were","be","been","being","or","as",
    "it","that","this","these","those","i","you","we","they","he","she"
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _tokens(s: str) -> List[str]:
    return [t for t in _word_re.findall(_norm(s)) if t and t not in _stop]

def _uniq(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# -----------------------------
# Lightweight stemming + 4-grams
# -----------------------------
def _stem(tok: str) -> str:
    # simple suffix stripper, term-agnostic
    for suf in ("ments","ment","ings","ing","ities","ity","ions","ion",
                "ness","ships","ship","ably","able","ally","al","ed","es","s","y"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[: -len(suf)]
    return tok

def _stems(tokens: List[str]) -> List[str]:
    return [_stem(t) for t in tokens]

def _char4(tok: str) -> List[str]:
    t = tok
    return [t[i:i+4] for i in range(len(t)-3)] if len(t) >= 4 else [t]

# -----------------------------
# +include / -exclude parsing
# -----------------------------
def _parse_req_exc(raw_q: str) -> Tuple[str, List[str], List[str]]:
    parts = raw_q.split()
    inc, exc, kept = [], [], []
    for p in parts:
        if p.startswith("+") and len(p) > 1: inc.append(_norm(p[1:]))
        elif p.startswith("-") and len(p) > 1: exc.append(_norm(p[1:]))
        else: kept.append(p)
    return " ".join(kept).strip(), inc, exc

# -----------------------------
# Semantic embedding cache
# -----------------------------
_EMBED_CACHE = {}
_TXT_CACHE = {}

def _index_key(texts: List[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update((_norm(t)+"\n").encode("utf-8"))
    return h.hexdigest()

def _get_semantic_matrix(texts: List[str]) -> Optional["np.ndarray"]:
    if not _ST_OK: return None
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

def _cosine_sim(vecA, matB):  # both normalized
    return (matB @ vecA)

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
    Returns DataFrame[code, text, display, score] per the policy above.
    """
    if qdf is None or qdf.empty or not query or not str(query).strip():
        return qdf.head(0)

    for col in ("code","text","display"):
        if col not in qdf.columns:
            raise ValueError(f"qdf missing required column: {col}")

    codes  = qdf["code"].astype(str).tolist()
    texts  = qdf["text"].astype(str).tolist()
    shows  = qdf["display"].astype(str).tolist()
    haystacks = [_norm(f"{d} {t}") for d, t in zip(shows, texts)]

    # Parse operators first
    q_raw = str(query).strip()
    q_clean, includes, excludes = _parse_req_exc(q_raw)

    # Query tokens, stems, and char-4 grams
    qtoks  = _uniq(_tokens(q_clean))
    qstems = _stems(qtoks)
    qgrams = set(g for t in qstems for g in _char4(t))

    N = len(texts)
    lex_scores = [0.0] * N
    has_lex    = [False] * N

    # --- Lexical evidence (any stem overlap OR any 4-gram overlap) ---
    # Coverage on stems (matched stems / query stems), with a floor if any evidence exists.
    qstem_set = set(qstems)
    for i, txt in enumerate(texts):
        toks   = _stems(_tokens(txt))
        stem_o = len(qstem_set & set(toks))
        cov    = (stem_o / max(1, len(qstems))) if qstems else 0.0

        # char-4 gram overlap adds robust morphological/phrase linkage
        grams_i = set(g for t in toks for g in _char4(t))
        gram_hit = bool(qgrams & grams_i)

        if stem_o > 0 or gram_hit:
            # "good lexical match" without exception → ensure passes global min later
            cov = max(cov, 0.50)  # lexical floor so > 0.40 overall
            has_lex[i] = True

        lex_scores[i] = min(1.0, max(0.0, cov))

    # --- Apply include/exclude (generic substrings on normalized display+text) ---
    def _contains_any(hay: str, needles: List[str]) -> bool:
        return any(n and n in hay for n in needles)

    for i, hay in enumerate(haystacks):
        if excludes and _contains_any(hay, excludes):
            lex_scores[i] = 0.0; has_lex[i] = False
        if includes and not all(inc in hay for inc in includes):
            lex_scores[i] = 0.0; has_lex[i] = False

    # Build lexical-only frame first
    df_lex = pd.DataFrame({
        "code": codes, "text": texts, "display": shows, "score": lex_scores, "has_lex": has_lex
    })

    # Keep lexical hits that cleanly pass > min_score
    df_lex_hits = df_lex[(df_lex["has_lex"]) & (df_lex["score"] > float(min_score))] \
        .sort_values(["score","code"], ascending=[False, True]) \
        .drop_duplicates("code", keep="first")

    # If we already have 5 or more lexical hits, return them (respect top_k)
    if len(df_lex_hits) >= 5:
        out = df_lex_hits[["code","text","display","score"]]
        if top_k and top_k > 0:
            out = out.head(top_k)
        return out.reset_index(drop=True)

    # --- Semantic backfill ONLY for items with no lexical evidence ---
    if _ST_OK:
        try:
            mat = _get_semantic_matrix(texts)
            if mat is not None:
                global _ST_MODEL
                qvec = _ST_MODEL.encode([q_raw], normalize_embeddings=True)[0]
                sim  = _cosine_sim(qvec, mat)            # [-1,1]
                sem  = ((sim + 1.0) / 2.0).tolist()       # [0,1]
            else:
                sem = [0.0]*N
        except Exception:
            sem = [0.0]*N
    else:
        sem = [0.0]*N

    # High semantic threshold for eligibility (normalized cosine)
    SEM_FLOOR = 0.78  # strict: includes only closely related items

    # Only consider items with NO lexical evidence
    backfill_rows = []
    for i in range(N):
        if has_lex[i]:
            continue
        s = sem[i]
        if s >= SEM_FLOOR:
            # light shaping so strong semantics dominate; still bounded [0,1]
            s_shaped = min(1.0, max(0.0, s * s))  # compress mid, keep high strong
            backfill_rows.append((codes[i], texts[i], shows[i], s_shaped))

    df_sem = pd.DataFrame(backfill_rows, columns=["code","text","display","score"]) if backfill_rows else \
             pd.DataFrame(columns=["code","text","display","score"])

    # Combine lexical + semantic; dedupe by code; enforce > min_score; rank
    out = pd.concat([
        df_lex_hits[["code","text","display","score"]],
        df_sem
    ], ignore_index=True)

    if out.empty:
        return out

    out = out.sort_values("score", ascending=False).drop_duplicates("code", keep="first")
    out = out[out["score"] > float(min_score)]
    if top_k and top_k > 0:
        out = out.head(top_k)
    out = out.sort_values(["score","code"], ascending=[False, True]).reset_index(drop=True)
    return out
