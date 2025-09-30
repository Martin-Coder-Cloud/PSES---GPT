# utils/hybrid_search.py
# -------------------------------------------------------------------------
# Hybrid search for survey questions (LOCAL, API-free by default)
# - Lexical: stemming + char n-grams + IDF-weighted coverage cascade
# - Semantic: sentence-transformer embeddings if available
# - Adaptive blending: lexical vs semantic weights depend on evidence strength
# - Anchor cap: blocks pure semantic drift without any lexical/n-gram overlap
# - Supports +include/-exclude operators
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Optional
import hashlib
import math
import os
import re
import pandas as pd

# Optional semantic support
_ST_OK = False
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
_stop = {"the","and","of","to","in","for","with","on","at","by","from",
         "a","an","is","are","was","were","be","been","being","or","as",
         "it","that","this","these","those","i","you","we","they","he","she"}

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
# Light stemming + n-grams
# -----------------------------
def _stem(tok: str) -> str:
    for suf in ("ments","ment","ings","ing","ities","ity","ions","ion","ness",
                "ships","ship","ed","es","s","y"):
        if tok.endswith(suf) and len(tok) > len(suf)+2:
            return tok[: -len(suf)]
    return tok

def _stems(tokens: List[str]) -> List[str]:
    return [_stem(t) for t in tokens]

def _char_ngrams(tok: str, n: int = 4) -> List[str]:
    return [tok[i:i+n] for i in range(len(tok)-n+1)] if len(tok) >= n else [tok]

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
# IDF weighting
# -----------------------------
_IDF_CACHE: Optional[dict] = None

def _idf_build(texts: List[str]) -> dict:
    import math
    N = len(texts)
    df = {}
    for txt in texts:
        for t in set(_stems(_tokens(txt))):
            df[t] = df.get(t,0)+1
    return {t: math.log((N+1)/(c+1))+1 for t,c in df.items()}

def _idf_score(tokens: List[str]) -> float:
    if not tokens: return 0.0
    vals = [_IDF_CACHE.get(t,1.0) for t in _stems(tokens)]
    return sum(vals)/len(vals)

# -----------------------------
# Semantic embedding cache
# -----------------------------
_EMBED_CACHE = {}
_TXT_CACHE = {}

def _index_key(texts: List[str]) -> str:
    h = hashlib.md5()
    for t in texts: h.update((_norm(t)+"\n").encode("utf-8"))
    return h.hexdigest()

def _get_semantic_matrix(texts: List[str]) -> Optional["np.ndarray"]:
    if not _ST_OK: return None
    global _ST_MODEL
    if _ST_MODEL is None:
        try: _ST_MODEL = SentenceTransformer(_ST_NAME)
        except Exception: return None
    key = _index_key(texts)
    if key in _EMBED_CACHE and _TXT_CACHE.get(key)==texts:
        return _EMBED_CACHE[key]
    try: mat = _ST_MODEL.encode(texts, normalize_embeddings=True)
    except Exception: return None
    _EMBED_CACHE[key]=mat; _TXT_CACHE[key]=texts
    return mat

def _cosine_sim(vecA, matB): return (matB @ vecA)

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

    if qdf is None or qdf.empty or not query.strip():
        return qdf.head(0)
    for c in ("code","text","display"):
        if c not in qdf.columns: raise ValueError(f"qdf missing {c}")

    codes, texts, shows = qdf["code"].astype(str).tolist(), qdf["text"].astype(str).tolist(), qdf["display"].astype(str).tolist()

    # Build IDF cache if needed
    global _IDF_CACHE
    if _IDF_CACHE is None: _IDF_CACHE = _idf_build(texts)

    # Parse operators
    q_raw = str(query).strip()
    q_clean, includes, excludes = _parse_req_exc(q_raw)

    qtoks = _uniq(_tokens(q_clean))
    qstems = _stems(qtoks)
    qgrams = set(g for t in qstems for g in _char_ngrams(t,4))

    # Lexical coverage: simple token overlap on stems
    lex_scores = []
    for txt in texts:
        toks = _stems(_tokens(txt))
        overlap = len(set(qstems)&set(toks))
        cov = overlap/max(1,len(qstems))
        # boost if any char-ngrams overlap
        grams = set(g for t in toks for g in _char_ngrams(t,4))
        if qgrams & grams: cov = max(cov,0.5)
        lex_scores.append(cov)

    # Semantic score
    if _ST_OK:
        mat = _get_semantic_matrix(texts)
        if mat is not None:
            try:
                global _ST_MODEL
                qvec = _ST_MODEL.encode([q_raw], normalize_embeddings=True)[0]
                sim = _cosine_sim(qvec, mat)
                sem_scores = ((sim+1)/2).tolist()
            except Exception:
                sem_scores = [0.0]*len(texts)
        else: sem_scores=[0.0]*len(texts)
    else:
        sem_scores=[0.0]*len(texts)

    # Adaptive blending
    spec = _idf_score(qtoks)
    blended=[]
    for l,s in zip(lex_scores,sem_scores):
        if l>=0.5: wL,wS=0.6,0.4
        elif l>0: wL,wS=0.5,0.5
        else: wL,wS=0.4,0.6
        sc=wL*l+wS*s
        # Anchor cap: if no lexical AND no ngram overlap, cap
        if l==0 and sc>0.45: sc=0.45
        blended.append(min(1.0,sc))

    # Apply +include/-exclude
    norm_texts=[_norm(f"{d} {t}") for d,t in zip(shows,texts)]
    def contains(hay,needles): return any(n in hay for n in needles)
    for i,hay in enumerate(norm_texts):
        if excludes and contains(hay,excludes): blended[i]=0.0
        if includes and not all(inc in hay for inc in includes): blended[i]=0.0

    df=pd.DataFrame({"code":codes,"text":texts,"display":shows,"score":blended})
    df=df.sort_values("score",ascending=False).drop_duplicates("code",keep="first")
    df=df[df["score"]>min_score]
    if top_k: df=df.head(top_k)
    return df.reset_index(drop=True)
