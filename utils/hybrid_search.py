# utils/hybrid_search.py
# -------------------------------------------------------------------------
# Lightweight hybrid search for survey questions
# - Exact code match
# - Substring match in code/text
# - Token-overlap (Jaccard) as a proxy for semantic similarity
# No external dependencies beyond pandas/re
# -------------------------------------------------------------------------

import re
import pandas as pd

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", str(s).lower()).strip()

def _tokset(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t}

def hybrid_question_search(
    qdf: pd.DataFrame,
    query: str,
    top_k: int = 120,
    min_score: float = 0.40
) -> pd.DataFrame:
    """
    Search the question metadata (qdf) by keyword or code.
    Returns a DataFrame of matches with columns: score, hit_type, code, text, display.
    Expects qdf to have columns: code, text, display.

    - top_k: return at most this many rows (default 120)
    - min_score: filter out weak hits below this score (default 0.40)
    """
    if not query or not query.strip():
        return qdf.head(0)

    q = query.strip()
    q_norm = _norm(q)
    q_tok = _tokset(q)

    rows = []
    for _, r in qdf.iterrows():
        code = str(r["code"])
        text = str(r["text"])
        disp = str(r["display"])

        # Exact/substring signals
        exact_code = (q.upper() == code.upper())
        in_code = (q_norm in _norm(code))
        in_text = (q_norm in _norm(text))

        # Token overlap (Jaccard)
        jt = 0.0
        if q_tok:
            tset = _tokset(text + " " + code)
            if tset:
                inter = len(q_tok & tset)
                union = len(q_tok | tset)
                jt = inter / union if union else 0.0

        # Scoring heuristic (exact >> substring >> token overlap)
        score = (3.0 if exact_code else 0.0) \
              + (1.0 if in_code else 0.0) \
              + (1.5 if in_text else 0.0) \
              + jt
        if score < min_score:
            continue

        hit_type = "exact" if exact_code else "substring" if (in_code or in_text) else "semantic"
        rows.append((score, hit_type, code, text, disp))

    if not rows:
        return qdf.head(0)

    out = pd.DataFrame(rows, columns=["score", "hit_type", "code", "text", "display"])
    out = out.sort_values(["score", "code"], ascending=[False, True], kind="mergesort").head(top_k)
    return out.reset_index(drop=True)
