# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# PS-wide only. Big-file single-pass loader. All data read as TEXT. 999/9999 suppressed.
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_results2024_filtered,
    get_results2024_schema,
    get_results2024_schema_inferred,
)

# Ensure OpenAI key is available from Streamlit secrets (no hardcoding)
os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# ── Debug/diagnostic visibility toggle ─────────────────────────────────────────
SHOW_DEBUG = False  # <- set to True to show parameters preview + diagnostics

# Stable alias to avoid any accidental local-shadowing of `pd` in functions
PD = pd


# ─────────────────────────────
# Cached metadata
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_demographics_metadata() -> pd.DataFrame:
    df = pd.read_excel("metadata/Demographics.xlsx")
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_questions_metadata() -> pd.DataFrame:
    qdf = pd.read_excel("metadata/Survey Questions.xlsx")
    qdf.columns = [c.strip().lower() for c in qdf.columns]
    if "question" in qdf.columns and "english" in qdf.columns:
        qdf = qdf.rename(columns={"question": "code", "english": "text"})
    qdf["code"] = qdf["code"].astype(str).str.strip()
    qdf["qnum"] = qdf["code"].str.extract(r"Q?(\d+)", expand=False)
    with pd.option_context("mode.chained_assignment", None):
        qdf["qnum"] = pd.to_numeric(qdf["qnum"], errors="coerce")
    qdf = qdf.sort_values(["qnum", "code"], na_position="last")
    qdf["display"] = qdf["code"] + " – " + qdf["text"].astype(str)
    return qdf[["code", "text", "display"]]


def _normalize_qcode(s: str) -> str:
    """Uppercase and strip all non-alphanumeric (so Q19a, q19A, Q19-a, Q19_a -> Q19A; Q51_1, Q51-1 -> Q511)."""
    s = "" if s is None else str(s)
    s = s.upper()
    return "".join(ch for ch in s if ch.isalnum())


def _norm_q(x: str) -> str:
    """
    Normalize a question code:
      - uppercase
      - remove spaces, underscores, dashes, and periods
      - apply known aliases (e.g., D57_1 -> Q57_1)
    """
    if x is None:
        return ""
    s = str(x).upper().strip()
    s = s.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
    aliases = {"D571": "Q571", "D572": "Q572"}
    return aliases.get(s, s)


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
# Helpers
# ─────────────────────────────
def _find_demcode_col(demo_df: pd.DataFrame) -> str | None:
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            return c
    return None


def _four_digit(s: str) -> str:
    s = "".join(ch for ch in str(s) if ch.isdigit())
    return s.zfill(4) if s else ""


def resolve_demographic_codes_from_metadata(
    demo_df: pd.DataFrame,
    category_label: str | None,
    subgroup_label: str | None,
):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    code_col = _find_demcode_col(demo_df)

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    df_cat = (
        demo_df[demo_df[DEMO_CAT_COL] == category_label]
        if DEMO_CAT_COL in demo_df.columns
        else demo_df.copy()
    )
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
            raw_code = str(r[code_col])
            label = str(r[LABEL_COL])
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
    """
    STRICT scale matching with normalization.
    Returns list of tuples: [(answer1, label1), (answer2, label2), ...] for non-empty labels in metadata.
    """
    if scales_df is None or scales_df.empty:
        return None

    qnorm = _normalize_qcode(question_code)
    if "__code_norm__" not in scales_df.columns:
        return None

    match = scales_df[scales_df["__code_norm__"] == qnorm]
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
    cols = [f"answer{i}" for i in range(1, 7 + 1)] + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE", "YES"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    out = df.copy()
    keep = PD.Series(True, index=out.index)
    for c in present:
        s = out[c].astype(str).str.strip()
        keep &= (s != "999") & (s != "9999")
    return out.loc[keep].copy()


def format_display_table_raw(
    df: pd.DataFrame, category_in_play: bool, dem_disp_map: dict, scale_pairs
) -> pd.DataFrame:
    """
    Build the display table using ONLY the scale columns provided by scale_pairs (strict),
    plus commonly used summary columns if present.
    Human labels from metadata are applied to the answer columns.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["Year"] = out["SURVEYR"].astype(str)

    if category_in_play:
        def to_label(code):
            key = "" if code is None else str(code).strip()
            if key == "":
                return "All respondents"
            return dem_disp_map.get(key, str(code))
        out["Demographic"] = out["DEMCODE"].apply(to_label)

    # STRICT: only the answer columns specified by metadata scale
    dist_cols_raw = []
    rename_map = {}
    if scale_pairs:
        for k, v in scale_pairs:
            ku = k.upper()
            if ku in out.columns:
                dist_cols_raw.append(ku)
                rename_map[ku] = v
            elif k in out.columns:
                dist_cols_raw.append(k)
                rename_map[k] = v

    keep_cols = (
        ["Year"]
        + (["Demographic"] if category_in_play else [])
        + dist_cols_raw
        + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE"]
    )
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    sort_cols = ["Year"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

    return out


# ─────────────────────────────
# Metric decision (FINAL rule)
# ─────────────────────────────
def detect_metric_mode(df_disp: pd.DataFrame, scale_pairs) -> dict:
    """
    FINAL rule:
      1) POSITIVE if usable
      2) else AGREE if usable
      3) else Answer1 (human label from scale metadata)
    Returns:
      {
        "mode": "positive" | "agree" | "answer1",
        "metric_col": str,         # column present in df_disp
        "ui_label": str,           # string to display under title and summary table
        "metric_label": str,       # human phrasing to pass to AI (“% positive” | “% agree” | “% <Answer1 label>”)
        "answer1_label": str | None
      }
    """
    cols_l = {c.lower(): c for c in df_disp.columns}

    # 1) POSITIVE
    if "positive" in cols_l:
        col = cols_l["positive"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {
                "mode": "positive",
                "metric_col": col,
                "ui_label": "(% positive answers)",
                "metric_label": "% positive",
                "answer1_label": None,
            }

    # 2) AGREE
    if "agree" in cols_l:
        col = cols_l["agree"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {
                "mode": "agree",
                "metric_col": col,
                "ui_label": "(% agree)",
                "metric_label": "% agree",
                "answer1_label": None,
            }

    # 3) Answer1 (fallback)
    answer1_label = None
    if scale_pairs:
        for k, v in scale_pairs:
            if k.lower() == "answer1":
                answer1_label = v
                break
    # Use the human-labeled column in df_disp that matches answer1_label
    if answer1_label and answer1_label in df_disp.columns:
        if PD.to_numeric(df_disp[answer1_label], errors="coerce").notna().any():
            return {
                "mode": "answer1",
                "metric_col": answer1_label,  # human-labeled column
                "ui_label": f"(% {answer1_label})",
                "metric_label": f"% {answer1_label}",
                "answer1_label": answer1_label,
            }

    # Fallback (UI can still render gracefully)
    return {
        "mode": "positive",
        "metric_col": cols_l.get("positive", "POSITIVE"),
        "ui_label": "(% positive answers)",
        "metric_label": "% positive",
        "answer1_label": None,
    }


# ─────────────────────────────
# AI helpers (auto-run)
# ─────────────────────────────
def _effective_model_name() -> str:
    # use OPENAI_MODEL from secrets if set; default to gpt-4o-mini
    return (st.secrets.get("OPENAI_MODEL", "") or "gpt-4o-mini").strip()


def _ai_build_payload_single_metric(
    df_disp: pd.DataFrame,
    question_code: str,
    question_text: str,
    category_in_play: bool,
    metric_col: str,
) -> dict:
    def col(df, *cands):
        for c in cands:
            if c in df.columns:
                return c
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None

    year_col = col(df_disp, "Year") or "Year"
    demo_col = col(df_disp, "Demographic") or "Demographic"
    n_col    = col(df_disp, "ANSCOUNT", "AnsCount", "N")

    ys = PD.to_numeric(df_disp[year_col], errors="coerce").dropna().astype(int).unique().tolist()
    ys = sorted(ys)
    latest = ys[-1] if ys else None
    baseline = ys[0] if ys else None

    overall_label = "All respondents"

    # Overall series (All respondents)
    if category_in_play and demo_col in df_disp.columns:
        base = df_disp[df_disp[demo_col] == overall_label].copy()
    else:
        base = df_disp.copy()

    overall_series = []
    for _, r in base.sort_values(year_col).iterrows():
        yr = PD.to_numeric(r[year_col], errors="coerce")
        if PD.isna(yr):
            continue
        val = PD.to_numeric(r.get(metric_col, None), errors="coerce")
        n = PD.to_numeric(r.get(n_col, None), errors="coerce") if n_col in base.columns else None
        overall_series.append({
            "year": int(yr),
            "value": (float(val) if PD.notna(val) else None),
            "n": (int(n) if PD.notna(n) else None) if n is not None else None
        })

    # Group series (exclude overall)
    groups = []
    if category_in_play and demo_col in df_disp.columns:
        for gname, gdf in df_disp.groupby(demo_col, dropna=False):
            if str(gname) == overall_label:
                continue
            series = []
            for _, r in gdf.sort_values(year_col).iterrows():
                yr = PD.to_numeric(r[year_col], errors="coerce")
                if PD.isna(yr):
                    continue
                val = PD.to_numeric(r.get(metric_col, None), errors="coerce")
                n = PD.to_numeric(r.get(n_col, None), errors="coerce") if n_col in gdf.columns else None
                series.append({
                    "year": int(yr),
                    "value": (float(val) if PD.notna(val) else None),
                    "n": (int(n) if PD.notna(n) else None) if n is not None else None
                })
            groups.append({"name": (str(gname) if PD.notna(gname) else ""), "series": series})

    return {
        "question_code": str(question_code),
        "question_text": str(question_text),
        "years": ys,
        "latest_year": latest,
        "baseline_year": baseline,
        "overall_label": overall_label,
        "overall_series": overall_series,
        "groups": groups,
        "has_groups": bool(groups),
    }


def _class_name(e: Exception) -> str:
    return type(e).__name__


def _call_openai_with_retry(client, **kwargs) -> tuple[str, str]:
    """
    Calls OpenAI with a 60s timeout and retries once on failure.
    Returns (content, error_hint) where:
      - content: the returned message content (or "")
      - error_hint: short human-friendly hint if it failed
    """
    # First attempt (JSON mode)
    try:
        comp = client.chat.completions.create(timeout=60.0, **kwargs)
        content = comp.choices[0].message.content if comp.choices else ""
        if content:
            return content, ""
        return "", "empty response"
    except Exception:
        # If the SDK/env doesn't like response_format, retry without it
        try:
            kwargs2 = {k: v for k, v in kwargs.items() if k != "response_format"}
            comp = client.chat.completions.create(timeout=60.0, **kwargs2)
            content = comp.choices[0].message.content if comp.choices else ""
            if content:
                return content, ""
            return "", "empty response"
        except Exception as e2:
            name2 = _class_name(e2).lower()
            if "authentication" in name2 or "auth" in name2:
                return "", "invalid_api_key"
            if "timeout" in name2 or "timedout" in name2:
                return "", "timeout"
            if "rate" in name2 and "limit" in name2:
                return "", "rate_limit"
            if "connection" in name2 or "network" in name2:
                return "", "network_error"
            if "badrequest" in name2 or "invalidrequest" in name2:
                return "", "invalid_request"
            if "typeerror" in name2:
                return "", "type_error"
            return "", name2 or "unknown_error"


def _ai_narrative_and_storytable(
    df_disp: pd.DataFrame,
    question_code: str,
    question_text: str,
    category_in_play: bool,
    metric_col: str,
    metric_label: str,
    temperature: float = 0.2,
) -> dict:
    """
    Generates the AI narrative with your approved context and rules.
    - Uses ONLY the values present in df_disp (the displayed tabulation).
    - Always starts with 2024 All respondents (if present) as context.
    - Trend classification thresholds: stable ≤1, slight >1–2, notable >2.
    - Demographic comparisons are pairwise (group-to-group) only.
    - Gap qualification: ≤2 normal; >2–5 notable; >5 important.
    - Skip missing/N/A. No word limit.
    - Returns JSON with only 'narrative'.
    - 60s timeout + one retry; safe parse; friendly hint on failure.
    """
    try:
        from openai import OpenAI
    except Exception:
        st.error(
            "AI summary requires the OpenAI SDK. Add `openai>=1.40.0` to requirements.txt and set `OPENAI_API_KEY` in Streamlit secrets."
        )
        return {"narrative": "", "hint": "missing_sdk"}

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        st.info("AI disabled: missing OpenAI API key in Streamlit secrets.")
        return {"narrative": "", "hint": "missing_api_key"}

    client = OpenAI()
    model_name = _effective_model_name()

    data = _ai_build_payload_single_metric(df_disp, question_code, question_text, category_in_play, metric_col)

    # ----- System prompt with fixed project context + refined rules (no hyperlink) -----
    system = (
        "You are preparing insights for the Government of Canada’s Public Service Employee Survey (PSES).\n"
        "Scope: Public Service–wide results only (no departmental results).\n"
        "The survey measures federal public servants’ opinions on engagement, leadership, workforce, workplace, "
        "workplace well-being, and compensation.\n"
        "In 2024, 186,635 employees across 93 departments and agencies responded (50.5% response rate).\n"
        "\n"
        "DATA PROVENANCE (STRICT):\n"
        "All statistics, trends, and comparisons must be derived strictly from the values in the provided JSON payload "
        "(which mirrors the displayed tabulation). Do not guess, infer, average, impute, or otherwise estimate any number. "
        "If a required value is not present, omit that statement as out of scope. Do not mention missing or 'n/a'.\n"
        "\n"
        "NARRATIVE RULES (REFINED):\n"
        "• Start with the 2024 All respondents value for the chosen metric (context only).\n"
        "• Trend over time (if ≥2 years): compare latest to baseline. Classify as Stable (|Δ| ≤ 1 pt), "
        "Slight increase/decrease (>1–2 pts), or Notable increase/decrease (>2 pts). Use whole percents and say 'points'. "
        "You may use natural synonyms (e.g., 'remained steady', 'saw a slight rise').\n"
        "• Demographics (if present): focus on the latest year. Compute pairwise gaps between groups with data. "
        "Highlight the largest pairwise gap (name both groups, state values, and the gap in points). "
        "Qualify gaps: ≤2 pts normal; >2–5 pts notable; >5 pts important. "
        "If baseline exists for both groups in that pair, indicate whether the gap widened or narrowed since baseline and by how many points. "
        "Optionally mention one or two other notable/important pairwise gaps. Do NOT compare any group to 'All respondents'.\n"
        "• Metric phrasing is exact: use the provided metric_label (e.g., '% positive', '% agree', '% Always'). "
        "If metric_label is '% agree', it reflects agreement with the literal wording (AGREE ≠ POSITIVE for negatively-worded items).\n"
        "• Executive and fact-based tone. No speculation about causes. No word limit.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object with ONLY one key: 'narrative'.\n"
    )

    user = json.dumps(
        {
            "metric_label": metric_label,
            "payload": data,
            "notes": {
                "trend_thresholds": {"stable": 1.0, "slight": (1.0, 2.0), "notable": 2.0},
                "gap_qualification": {"normal": 2.0, "notable": (2.0, 5.0), "important": 5.0},
            },
        },
        ensure_ascii=False,
    )

    # Save the exact request for Diagnostics → AI prompt visibility
    st.session_state["ai_request_debug"] = {
        "model": model_name,
        "system": system,
        "user": user,
    }

    # Perform request with a 60s timeout and one retry; safe JSON parse; return hint on failure
    kwargs = dict(
        model=model_name,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content, hint = _call_openai_with_retry(client, **kwargs)
    if not content:
        return {"narrative": "", "hint": hint or "no_content"}

    try:
        out = json.loads(content)
    except Exception:
        return {"narrative": "", "hint": "json_decode_error"}

    if not isinstance(out, dict):
        return {"narrative": "", "hint": "non_dict_json"}

    return {"narrative": out.get("narrative", "").strip(), "hint": ""}


# ─────────────────────────────
# 🔎 AI Health Check (now used inside Diagnostics → AI prompt visibility)
# ─────────────────────────────
def run_ai_health_check():
    """Small, non-data test call to verify key + connectivity + model reachability."""
    try:
        from openai import OpenAI
    except Exception:
        st.error("Health check requires the OpenAI SDK. Add `openai>=1.40.0` to requirements.txt.")
        return

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        st.warning("OPENAI_API_KEY not set in Streamlit secrets.")
        return

    client = OpenAI()
    model_name = _effective_model_name()
    system = "You are a minimal health check. Reply with exactly: OK."
    user = "Say OK."

    t0 = time.perf_counter()
    try:
        comp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            timeout=10.0,
        )
        content = comp.choices[0].message.content.strip() if comp.choices else ""
        dt = (time.perf_counter() - t0) * 1000
        if content.upper().startswith("OK"):
            st.success(f"AI health check passed (model={model_name}, ~{dt:.0f} ms).")
        else:
            st.info(f"AI reachable but unexpected reply (~{dt:.0f} ms): {content[:60]}")
    except Exception as e:
        name = type(e).__name__.lower()
        if "authentication" in name or "auth" in name:
            st.error("AI health check failed: invalid/missing API key.")
        elif "timeout" in name or "timedout" in name:
            st.error("AI health check timed out (10s).")
        elif "rate" in name and "limit" in name:
            st.error("AI health check hit rate limit.")
        elif "connection" in name or "network" in name:
            st.error("AI health check failed: network/connectivity issue.")
        else:
            st.error(f"AI health check error: {type(e).__name__}")


# ─────────────────────────────
# Summary trend table builder (Years as columns)
# ─────────────────────────────
def build_trend_summary_table(
    df_disp: pd.DataFrame,
    category_in_play: bool,
    metric_col: str,
    selected_years: list[str] | None = None,
) -> pd.DataFrame:
    """
    Always build a single-metric table with Demographics as rows and Years as columns.
    Values are whole % strings ('n/a' for missing).
    """
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

    # Column order: years ascending
    if selected_years:
        years = sorted([str(y) for y in selected_years], key=lambda x: int(x))
    else:
        years = sorted(df["Year"].astype(str).unique().tolist(), key=lambda x: int(x))

    # Pivot to Segment x Year for the single metric
    pivot = df.pivot_table(index="Demographic", columns="Year", values=metric_col, aggfunc="first").copy()
    pivot.index.name = "Segment"

    # Ensure all years exist as columns
    for y in years:
        if y not in pivot.columns:
            pivot[y] = PD.NA

    # Format as whole % or 'n/a'
    for c in pivot.columns:
        vals = PD.to_numeric(pivot[c], errors="coerce").round(0)
        out = PD.Series("n/a", index=pivot.index, dtype="object")
        mask = vals.notna()
        out.loc[mask] = vals.loc[mask].astype(int).astype(str) + "%"
        pivot[c] = out

    pivot = pivot.reset_index()
    pivot = pivot[["Segment"] + years]
    return pivot


# ─────────────────────────────
# Report (PDF) builder
# ─────────────────────────────
def build_pdf_report(
    question_code: str,
    question_text: str,
    selected_years: list[str],
    dem_display: list[str],
    narrative: str,
    df_summary: pd.DataFrame,
    ui_label: str,   # reflects chosen metric
) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        st.error("PDF export requires `reportlab`. Please add `reportlab==3.6.13` to requirements.txt.")
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=36, bottomMargin=36, leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet()
    title = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, textColor="#555555")

    flow = []
    flow.append(Paragraph("PSES Analysis Report", title))
    flow.append(Paragraph(f"{question_code} — {question_text}", body))
    flow.append(Paragraph(ui_label, small))
    flow.append(Spacer(1, 10))

    ctx = f"""
    <b>Years:</b> {', '.join(selected_years)}<br/>
    <b>Demographic selection:</b> {', '.join(dem_display)}
    """
    flow.append(Paragraph(ctx, body))
    flow.append(Spacer(1, 10))

    flow.append(Paragraph("Analysis Summary", h2))
    if (narrative or "").strip():
        for chunk in narrative.split("\n"):
            if chunk.strip():
                flow.append(Paragraph(chunk.strip(), body))
                flow.append(Spacer(1, 4))
    else:
        flow.append(Paragraph("AI analysis was unavailable for this run.", small))

    if df_summary is not None and not df_summary.empty:
        flow.append(Spacer(1, 10))
        flow.append(Paragraph("Summary Table", h2))
        data = [df_summary.columns.tolist()] + df_summary.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                ]
            )
        )
        flow.append(tbl)

    doc.build(flow)
    return buf.getvalue()


# ─────────────────────────────
# UI
# ─────────────────────────────
def run_menu1():
    st.markdown(
        """
    <style>
      .custom-header{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }
      .custom-instruction{ font size: 15px; line-height: 1.4; margin-bottom: 8px; color: #333; }
      .field-label{ font-size: 16px; font-weight: 600; margin: 10px 0 2px; color: #222; }
      .big-button button{ font-size: 16px; padding: 0.6em 1.6em; margin-top: 16px; }
      .tiny-note{ font-size: 12px; color: #666; margin-top: -4px; margin-bottom: 10px; }
      .q-sub{ font-size: 14px; color: #333; margin-top: -4px; margin-bottom: 2px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            "<img style='width:75%;max-width:740px;height:auto;display:block;margin:0 auto 16px;' "
            "src='https://raw.githubusercontent.com/Martin-Coder-Cloud/PSES---GPT/main/PSES%20Banner%20New.png'>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="custom-header">🔍 Search by Survey Question</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="custom-instruction">Please select a question you are interested in, the survey year and, optionally, a demographic breakdown.<br>'
            'This application provides only Public Service-wide results. The output is a result table, a short analysis and a summary table.</div>',
            unsafe_allow_html=True,
        )

        # 🔧 Toggle for tech parameters & diagnostics (persist in session)
        show_debug = st.toggle(
            "🔧 Show technical parameters & diagnostics",
            value=st.session_state.get("show_debug", SHOW_DEBUG),
        )
        st.session_state["show_debug"] = show_debug

        # 🧠 Toggle for AI (persist in session)
        ai_enabled = st.toggle(
            "🧠 Enable AI analysis (OpenAI)",
            value=st.session_state.get("ai_enabled", False),
            help="Turn OFF while testing to avoid API usage.",
        )
        st.session_state["ai_enabled"] = ai_enabled

        # Question
        st.markdown('<div class="field-label">Select a survey question:</div>', unsafe_allow_html=True)
        question_options = qdf["display"].tolist()
        selected_label = st.selectbox("Question", question_options, key="question_dropdown", label_visibility="collapsed")
        question_code = qdf.loc[qdf["display"] == selected_label, "code"].values[0]
        question_text = qdf.loc[qdf["display"] == selected_label, "text"].values[0]

        # Years (strings)
        st.markdown('<div class="field-label">Select survey year(s):</div>', unsafe_allow_html=True)
        all_years = ["2024", "2022", "2020", "2019"]
        select_all = st.checkbox("All years", value=True, key="select_all_years")
        selected_years: list[str] = []
        year_cols = st.columns(len(all_years))
        for idx, yr in enumerate(all_years):
            with year_cols[idx]:
                checked = True if select_all else False
                if st.checkbox(yr, value=checked, key=f"year_{yr}"):
                    selected_years.append(yr)
        selected_years = sorted(selected_years)
        if not selected_years:
            st.warning("⚠️ Please select at least one year.")
            return

        # Demographic category/subgroup
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
        st.markdown('<div class="field-label">Select a demographic category (or All respondents):</div>', unsafe_allow_html=True)
        demo_categories = ["All respondents"] + sorted(
            demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist()
        )
        demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

        sub_selection = None
        if demo_selection != "All respondents":
            st.markdown(
                f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>',
                unsafe_allow_html=True,
            )
            sub_items = (
                demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            sub_items = sorted(sub_items)
            sub_selection = st.selectbox(
                "(leave blank to include all subgroups in this category)",
                [""] + sub_items,
                key=f"sub_{demo_selection.replace(' ', '_')}",
                label_visibility="collapsed",
            )
            if sub_selection == "":
                sub_selection = None

        # Resolve DEMCODEs once (as text)
        demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(
            demo_df, demo_selection, sub_selection
        )
        # ALWAYS include overall row when a category is in play
        if category_in_play and (None not in demcodes):
            demcodes = [None] + demcodes
        dem_display = ["All respondents" if c is None else str(c).strip() for c in demcodes]

        # ──────────────────────────────────────────────────────────────────────
        # Diagnostics panel (exactly three sections)
        # ──────────────────────────────────────────────────────────────────────
        if show_debug:
            with st.expander("🛠 Diagnostics", expanded=False):
                # 1) Parameters sent to the database
                with st.expander("1) Parameters sent to the database", expanded=True):
                    params_df = PD.DataFrame(
                        {
                            "Parameter": ["QUESTION (from metadata)", "SURVEYR (years)", "DEMCODE(s) (from metadata)"],
                            "Value": [question_code, ", ".join(selected_years), ", ".join(dem_display)],
                        }
                    )
                    st.dataframe(params_df, use_container_width=True, hide_index=True)

                # 2) Environment diagnostics (automatic)
                with st.expander("2) Environment diagnostics", expanded=False):
                    try:
                        info = {}
                        info["OPENAI_API_KEY_present"] = bool(os.environ.get("OPENAI_API_KEY", "").strip())
                        info["OPENAI_MODEL_effective"] = _effective_model_name()

                        try:
                            import openai  # type: ignore
                            info["openai_version"] = getattr(openai, "__version__", "unknown")
                        except Exception:
                            info["openai_version"] = "not installed"

                        info["pandas_version"] = pd.__version__
                        info["streamlit_version"] = st.__version__

                        files = {
                            "metadata/Survey Scales.xlsx": os.path.exists("metadata/Survey Scales.xlsx"),
                            "metadata/Survey Questions.xlsx": os.path.exists("metadata/Survey Questions.xlsx"),
                            "metadata/Demographics.xlsx": os.path.exists("metadata/Demographics.xlsx"),
                        }
                        info["metadata_files_exist"] = files

                        st.write(info)
                    except Exception as e:
                        st.error(f"Environment diagnostics failed: {e}")

                # 3) AI prompt visibility (includes Health Check first)
                with st.expander("3) AI prompt visibility", expanded=False):
                    # Health check button at top
                    if st.button("Run AI health check", key="btn_ai_healthcheck"):
                        run_ai_health_check()

                    # Show last request (if any)
                    dbg = st.session_state.get("ai_request_debug") or {}
                    model_used = dbg.get("model", "(no request this session)")
                    system_prompt = dbg.get("system", "")
                    user_json = dbg.get("user", "")

                    st.markdown(f"**Model used (last request):** `{model_used}`")

                    st.markdown("**System prompt (last request):**")
                    if system_prompt:
                        st.code(system_prompt)
                    else:
                        st.caption("No AI request has been made yet in this session.")

                    st.markdown("**User payload JSON (last request):**")
                    if user_json:
                        # Truncate preview with a toggle
                        MAX_PREVIEW = 12000
                        if len(user_json) > MAX_PREVIEW:
                            st.code(user_json[:MAX_PREVIEW] + "\n… (truncated)")
                            if st.toggle("Show full payload JSON"):
                                st.code(user_json)
                        else:
                            st.code(user_json)
                    else:
                        st.caption("No AI request has been made yet in this session.")

        # Run query (single pass, cached big file)
        if st.button("🔎 Run query"):
            with st.spinner("Processing data..."):
                # STRICT scale matching — if we cannot find an exact normalized match, stop.
                scale_pairs = get_scale_labels(load_scales_metadata(), question_code)
                if scale_pairs is None or len(scale_pairs) == 0:
                    qnorm = _normalize_qcode(question_code)
                    st.error(
                        f"Metadata scale not found for question '{question_code}' (normalized '{qnorm}'). "
                        "No results are displayed to avoid mislabeling. Please verify 'Survey Scales.xlsx'."
                    )
                    return

                # Preferred fast path (new loader signature)
                try:
                    df_raw = load_results2024_filtered(
                        question_code=question_code,
                        years=selected_years,
                        group_values=demcodes,  # includes None for overall when category_in_play
                    )
                except TypeError:
                    parts = []
                    for gv in demcodes:
                        try:
                            parts.append(
                                load_results2024_filtered(
                                    question_code=question_code,
                                    years=selected_years,
                                    group_value=(None if gv is None else str(gv).strip()),
                                )
                            )
                        except TypeError:
                            continue
                    df_raw = (
                        PD.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
                        if parts
                        else PD.DataFrame()
                    )

                if df_raw is None or df_raw.empty:
                    st.info("No data found for this selection.")
                    return

                df_raw = exclude_999_raw(df_raw)
                if df_raw.empty:
                    st.info("Data exists, but all rows are not applicable (999/9999).")
                    return

                if "SURVEYR" in df_raw.columns:
                    df_raw = df_raw.sort_values(by="SURVEYR", ascending=False)

                # 🔧 show raw results before formatting when toggle is ON
                if show_debug:
                    st.markdown("#### Raw results (debug — pre-formatting)")
                    st.caption(f"Rows: {len(df_raw):,} | Columns: {len(df_raw.columns)}")
                    st.dataframe(df_raw, use_container_width=True)

                st.subheader(f"{question_code} — {question_text}")
                dem_map_clean = {None: "All respondents"}
                try:
                    for k, v in (disp_map or {}).items():
                        dem_map_clean[(None if k is None else str(k).strip())] = v
                except Exception:
                    pass

                df_disp = format_display_table_raw(
                    df=df_raw,
                    category_in_play=category_in_play,
                    dem_disp_map=dem_map_clean,
                    scale_pairs=scale_pairs,
                )

            # Results table
            st.dataframe(df_disp, use_container_width=True)

            # Decide metric per FINAL rule
            decision = detect_metric_mode(df_disp, scale_pairs)
            mode = decision["mode"]
            metric_col = decision["metric_col"]
            ui_label = decision["ui_label"]
            metric_label = decision["metric_label"]
            answer1_label = decision["answer1_label"]

            # === Analysis Summary (AI) ===
            st.markdown("### Analysis Summary")
            st.markdown(
                f"<div class='q-sub'>{question_code} — {question_text}</div>"
                f"<div class='tiny-note'>{ui_label}</div>",
                unsafe_allow_html=True,
            )

            narrative = ""
            ai_cols_used_note = ""
            if ai_enabled:
                # Show which model we're using right on the spinner
                model_name = _effective_model_name()
                with st.spinner(f"Contacting AI (model: {model_name})…"):
                    ai_out = _ai_narrative_and_storytable(
                        df_disp=df_disp,
                        question_code=question_code,
                        question_text=question_text,
                        category_in_play=category_in_play,
                        metric_col=metric_col,
                        metric_label=metric_label,
                        temperature=0.2,
                    )
                narrative = (ai_out.get("narrative") or "").strip()
                hint = (ai_out.get("hint") or "").strip()
                if narrative:
                    st.write(narrative)
                else:
                    # Friendlier hints for common cases
                    hint_map = {
                        "invalid_api_key": "Invalid or missing API key.",
                        "timeout": "AI request timed out (took longer than 60s).",
                        "rate_limit": "Rate limit reached. Try again shortly.",
                        "network_error": "Network error contacting the AI endpoint.",
                        "invalid_request": "Invalid request payload.",
                        "json_decode_error": "AI returned a malformed response.",
                        "type_error": "Client configuration issue (TypeError).",
                    }
                    msg = hint_map.get(hint, "AI unavailable right now.")
                    st.info(f"{msg} Tables remain available.")

                # Explicitly state which metric/column was passed to AI
                if mode in ("positive", "agree"):
                    ai_cols_used_note = f"AI was passed: {metric_label} (column: {metric_col})."
                else:
                    ai_cols_used_note = f"AI was passed: % {answer1_label} (column: {metric_col})."
                st.caption(ai_cols_used_note)
            else:
                st.caption("AI analysis is disabled (toggle above).")

            # === Summary Table ===
            st.markdown("### Summary Table")
            st.markdown(
                f"<div class='q-sub'>{question_code} — {question_text}</div>"
                f"<div class='tiny-note'>{ui_label}</div>",
                unsafe_allow_html=True,
            )
            trend_df = build_trend_summary_table(
                df_disp=df_disp,
                category_in_play=category_in_play,
                metric_col=metric_col,
                selected_years=selected_years,
            )
            if trend_df is not None and not trend_df.empty:
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
            else:
                st.info("No summary table could be generated for the current selection.")

            # Downloads
            with io.BytesIO() as xbuf:
                with PD.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
                    df_disp.to_excel(writer, sheet_name="Results", index=False)
                    if trend_df is not None and not trend_df.empty:
                        trend_df.to_excel(writer, sheet_name="Summary Table", index=False)
                    ctx = {
                        "QUESTION": question_code,
                        "SURVEYR (years)": ", ".join(selected_years),
                        "DEMCODE(s)": ", ".join(dem_display),
                        "Metric used": metric_label,
                        "AI enabled": "Yes" if ai_enabled else "No",
                        "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    PD.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(
                        writer, sheet_name="Context", index=False
                    )
                xdata = xbuf.getvalue()

            st.download_button(
                label="⬇️ Download data (Excel)",
                data=xdata,
                file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            pdf_bytes = build_pdf_report(
                question_code=question_code,
                question_text=question_text,
                selected_years=selected_years,
                dem_display=dem_display,
                narrative=narrative,  # empty when AI disabled or failed
                df_summary=trend_df if trend_df is not None else PD.DataFrame(),
                ui_label=ui_label,
            )
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download summary report (PDF)",
                    data=pdf_bytes,
                    file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("PDF export unavailable (install `reportlab` in requirements to enable).")


if __name__ == "__main__":
    run_menu1()
