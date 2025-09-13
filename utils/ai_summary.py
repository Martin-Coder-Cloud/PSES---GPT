# utils/ai_summary.py
# Turns your compact display table (df_disp) into a JSON payload for the model,
# and asks for BOTH a narrative and a small "story table" via Structured Outputs.

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI


def _col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Find the first present column among candidates (case-sensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    # Be a bit more tolerant: try case-insensitive match
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _years_sorted(df: pd.DataFrame, year_col: str) -> List[int]:
    ys = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(ys)


def _compact_payload(
    df_disp: pd.DataFrame,
    question_code: str,
    question_text: str,
) -> Dict[str, Any]:
    year_col = _col(df_disp, "Year") or "Year"
    demo_col = _col(df_disp, "Demographic") or "Demographic"
    pos_col  = _col(df_disp, "POSITIVE", "Positive") or "POSITIVE"
    n_col    = _col(df_disp, "ANSCOUNT", "AnsCount", "N")  # optional

    years = _years_sorted(df_disp, year_col)
    latest = years[-1] if years else None
    baseline = years[0] if len(years) >= 2 else (years[-1] if years else None)

    groups: List[Dict[str, Any]] = []
    if demo_col in df_disp.columns:
        for gname, gdf in df_disp.groupby(demo_col, dropna=False):
            series = []
            for _, r in gdf.sort_values(year_col).iterrows():
                yr = int(pd.to_numeric(r[year_col], errors="coerce"))
                if pd.isna(yr):
                    continue
                pos = float(pd.to_numeric(r.get(pos_col, None), errors="coerce"))
                n = None
                if n_col in gdf.columns:
                    n_val = pd.to_numeric(r.get(n_col, None), errors="coerce")
                    n = int(n_val) if pd.notna(n_val) else None
                series.append({"year": yr, "positive": pos, "n": n})
            groups.append({"name": str(gname) if pd.notna(gname) else "", "series": series})
    else:
        # overall only
        series = []
        for _, r in df_disp.sort_values(year_col).iterrows():
            yr = int(pd.to_numeric(r[year_col], errors="coerce"))
            pos = float(pd.to_numeric(r.get(pos_col, None), errors="coerce"))
            n = None
            if n_col and n_col in df_disp.columns:
                n_val = pd.to_numeric(r.get(n_col, None), errors="coerce")
                n = int(n_val) if pd.notna(n_val) else None
            series.append({"year": yr, "positive": pos, "n": n})
        groups = [{"name": "All respondents", "series": series}]

    payload = {
        "question_code": str(question_code),
        "question_text": str(question_text),
        "years": years,
        "latest_year": latest,
        "baseline_year": baseline,
        "groups": groups,
        "metric": "POSITIVE"
    }
    return payload


def ai_narrative_and_storytable(
    df_disp: pd.DataFrame,
    question_code: str,
    question_text: str,
    *,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        "narrative": str,
        "table": [ { "segment": str, "positive_2024": number|null, "delta_vs_baseline_pts": number|null, "note": str|null }, ... ]
      }
    """
    client = OpenAI()  # uses OPENAI_API_KEY

    data = _compact_payload(df_disp, question_code, question_text)

    # JSON schema for Structured Outputs (forces valid, parseable JSON)
    schema = {
        "name": "SurveySummary",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "narrative": { "type": "string" },
                "table": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "segment": { "type": "string" },
                            "positive_2024": { "type": ["number", "null"] },
                            "delta_vs_baseline_pts": { "type": ["number", "null"] },
                            "note": { "type": ["string", "null"] }
                        },
                        "required": ["segment"]
                    }
                }
            },
            "required": ["narrative", "table"]
        },
        "strict": True
    }

    # Instructions tuned to your brief
    system = (
        "You are a survey insights writer. Produce an executive-ready summary for senior management.\n"
        "Use the POSITIVE metric only. Rules:\n"
        "• Start with the latest year (prefer 2024) overall point, if available.\n"
        "• Call out top and bottom subgroup in the latest year, the gap between them, and how that gap changed vs the baseline year (earliest selected).\n"
        "• Summarize trend concisely (no long lists): typical change range in points, plus biggest increase/decrease; name the subgroup(s).\n"
        "• Treat |Δ| ≥ 5 pts as notable and ≥ 3 pts as worth mentioning if relevant.\n"
        "• Use whole percents and 'pts' for deltas. Keep to ~4–6 sentences, plain language.\n"
        "Also produce a compact 'story table' that mirrors the narrative (overall, top/bottom, gap, standout movers)."
    )

    user = (
        "Here is the results table as JSON. Only use the numbers provided.\n"
        "Return JSON that matches the schema exactly.\n\n"
        f"{json.dumps(data, ensure_ascii=False)}"
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",  # adjust if you prefer another model
        temperature=temperature,
        input=[{"role": "system", "content": system},
               {"role": "user", "content": user}],
        response_format={ "type": "json_schema", "json_schema": schema }
    )

    # The model must return valid JSON per schema; parse it:
    text = resp.output_text
    try:
        out = json.loads(text)
    except Exception:
        # Fallback: return a minimal object to avoid breaking the UI
        out = {"narrative": text, "table": []}
    return out
