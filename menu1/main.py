# menu1/main.py — PSES AI Explorer (Menu 1: Search by Question)
# PS-wide only. Big-file single-pass loader. All data read as TEXT. 999/9999 suppressed.
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import streamlit as st

# Import loader module and functions (diagnostics are optional)
import utils.data_loader as _dl

# Safe imports: these may or may not exist depending on your loader version
try:
    from utils.data_loader import get_last_query_diag  # diagnostics (optional)
except Exception:
    def get_last_query_diag():
        return {}

try:
    from utils.data_loader import (
        load_results2024_filtered,
        get_results2024_schema,
        get_results2024_schema_inferred,
        _resolve_results_path,  # diagnostics only
    )
except Exception:
    # At minimum we need load_results2024_filtered. If not present, raise.
    from utils.data_loader import load_results2024_filtered  # type: ignore
    def get_results2024_schema(): return {}
    def get_results2024_schema_inferred(): return {}
    def _resolve_results_path(): return None

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
    """
    cols_l = {c.lower(): c for c in df_disp.columns}

    # 1) POSITIVE
    if "positive" in cols_l:
        col = cols_l["positive"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {"mode": "positive", "metric_col": col, "ui_label": "(% positive answers)", "metric_label": "% positive", "answer1_label": None}

    # 2) AGREE
    if "agree" in cols_l:
        col = cols_l["agree"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {"mode": "agree", "metric_col": col, "ui_label": "(% agree)", "metric_label": "% agree", "answer1_label": None}

    # 3) Answer1 (fallback)
    answer1_label = None
    if scale_pairs:
        for k, v in scale_pairs:
            if k.lower() == "answer1":
                answer1_label = v
                break
    if answer1_label and answer1_label in df_disp.columns:
        if PD.to_numeric(df_disp[answer1_label], errors="coerce").notna().any():
            return {"mode": "answer1", "metric_col": answer1_label, "ui_label": f"(% {answer1_label})", "metric_label": f"% {answer1_label}", "answer1_label": answer1_label}

    # Fallback
    return {"mode": "positive", "metric_col": cols_l.get("positive", "POSITIVE"), "ui_label": "(% positive answers)", "metric_label": "% positive", "answer1_label": None}


# ─────────────────────────────
# AI helpers (auto-run)
# ─────────────────────────────
def _class_name(e: Exception) -> str:
    return type(e).__name__


def _call_openai_with_retry(client, **kwargs) -> tuple[str, str]:
    """60s timeout + one retry; returns (content, error_hint)."""
    try:
        comp = client.chat.completions.create(timeout=60.0, **kwargs)
        content = comp.choices[0].message.content if comp.choices else ""
        if content:
            return content, ""
        return "", "empty response"
    except Exception:
        try:
            kwargs2 = {k: v for k, v in kwargs.items() if k != "response_format"}
            comp = client.chat.completions.create(timeout=60.0, **kwargs2)
            content = comp.choices[0].message.content if comp.choices else ""
            if content:
                return content, ""
            return "", "empty response"
        except Exception as e2:
            name2 = _class_name(e2).lower()
            if "authentication" in name2 or "auth" in name2: return "", "invalid_api_key"
            if "timeout" in name2 or "timedout" in name2:   return "", "timeout"
            if "rate" in name2 and "limit" in name2:        return "", "rate_limit"
            if "connection" in name2 or "network" in name2: return "", "network_error"
            if "badrequest" in name2 or "invalidrequest" in name2: return "", "invalid_request"
            if "typeerror" in name2: return "", "type_error"
            return "", name2 or "unknown_error"


def _ai_build_payload_single_metric(df_disp: pd.DataFrame, question_code: str, question_text: str, category_in_play: bool, metric_col: str) -> dict:
    def col(df, *cands):
        for c in cands:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None
    year_col = col(df_disp, "Year") or "Year"
    demo_col = col(df_disp, "Demographic") or "Demographic"
    n_col    = col(df_disp, "ANSCOUNT", "AnsCount", "N")

    ys = PD.to_numeric(df_disp[year_col], errors="coerce").dropna().astype(int).unique().tolist()
    ys = sorted(ys); latest = ys[-1] if ys else None; baseline = ys[0] if ys else None

    overall_label = "All respondents"
    base = df_disp[df_disp[demo_col] == overall_label].copy() if (category_in_play and demo_col in df_disp.columns) else df_disp.copy()

    overall_series = []
    for _, r in base.sort_values(year_col).iterrows():
        yr = PD.to_numeric(r[year_col], errors="coerce")
        if PD.isna(yr): continue
        val = PD.to_numeric(r.get(metric_col, None), errors="coerce")
        n = PD.to_numeric(r.get(n_col, None), errors="coerce") if n_col in base.columns else None
        overall_series.append({"year": int(yr), "value": (float(val) if PD.notna(val) else None), "n": (int(n) if PD.notna(n) else None) if n is not None else None})

    groups = []
    if category_in_play and demo_col in df_disp.columns:
        for gname, gdf in df_disp.groupby(demo_col, dropna=False):
            if str(gname) == overall_label: continue
            series = []
            for _, r in gdf.sort_values(year_col).iterrows():
                yr = PD.to_numeric(r[year_col], errors="coerce")
                if PD.isna(yr): continue
                val = PD.to_numeric(r.get(metric_col, None), errors="coerce")
                n = PD.to_numeric(r.get(n_col, None), errors="coerce") if n_col in gdf.columns else None
                series.append({"year": int(yr), "value": (float(val) if PD.notna(val) else None), "n": (int(n) if PD.notna(n) else None) if n is not None else None})
            groups.append({"name": (str(gname) if PD.notna(gname) else ""), "series": series})

    return {"question_code": str(question_code), "question_text": str(question_text), "years": ys, "latest_year": latest, "baseline_year": baseline, "overall_label": "All respondents", "overall_series": overall_series, "groups": groups, "has_groups": bool(groups)}


def _ai_narrative_and_storytable(df_disp: pd.DataFrame, question_code: str, question_text: str, category_in_play: bool, metric_col: str, metric_label: str, temperature: float = 0.2) -> dict:
    try:
        from openai import OpenAI
    except Exception:
        st.error("AI summary requires the OpenAI SDK. Add `openai>=1.40.0` to requirements.txt and set `OPENAI_API_KEY` in Streamlit secrets.")
        return {"narrative": "", "hint": "missing_sdk"}

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        st.info("AI disabled: missing OpenAI API key in Streamlit secrets.")
        return {"narrative": "", "hint": "missing_api_key"}

    client = OpenAI()
    data = _ai_build_payload_single_metric(df_disp, question_code, question_text, category_in_play, metric_col)
    model_name = (st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

    system = (
        "You are preparing insights for the Government of Canada’s Public Service Employee Survey (PSES).\n"
        "Scope: Public Service–wide results only (no departmental results).\n"
        "The survey measures federal public servants’ opinions on engagement, leadership, workforce, workplace, "
        "workplace well-being, and compensation.\n"
        "In 2024, 186,635 employees across 93 departments and agencies responded (50.5% response rate).\n"
        "\n"
        "DATA PROVENANCE (STRICT):\n"
        "All statistics, trends, and comparisons must be derived strictly from the values in the provided JSON payload "
        "(which mirrors the displayed tabulation). Do not guess or impute. If a required value is not present, omit it.\n"
        "\n"
        "NARRATIVE RULES (REFINED):\n"
        "• Start with the 2024 All respondents value for the chosen metric.\n"
        "• Trend over time (if ≥2 years): compare latest to baseline. Stable (≤1 pt), Slight (>1–2 pts), Notable (>2 pts).\n"
        "• Demographics (if present): focus on latest year, largest pairwise gap, classify gaps (≤2 normal, >2–5 notable, >5 important).\n"
        "• Use exact metric_label wording; executive, factual tone; no speculation.\n"
        "OUTPUT: JSON object with only 'narrative'.\n"
    )
    user_payload = {"metric_label": metric_label, "payload": data, "notes": {"trend_thresholds": {"stable": 1.0, "slight": (1.0, 2.0), "notable": 2.0}, "gap_qualification": {"normal": 2.0, "notable": (2.0, 5.0), "important": 5.0}}}
    user = json.dumps(user_payload, ensure_ascii=False)

    st.session_state["last_ai_model"] = model_name
    st.session_state["last_ai_system"] = system
    st.session_state["last_ai_user"] = user

    kwargs = dict(model=model_name, temperature=temperature, response_format={"type": "json_object"}, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
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
# 🔎 AI Health Check (auto, no button)
# ─────────────────────────────
def run_ai_health_check():
    try:
        from openai import OpenAI
    except Exception:
        return {"status": "error", "detail": "missing_sdk"}
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return {"status": "warn", "detail": "missing_api_key"}
    model_name = (st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI()
    system = "You are a minimal health check. Reply with exactly: OK."
    user = "Say OK."
    t0 = time.perf_counter()
    try:
        comp = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0, timeout=10.0)
        content = comp.choices[0].message.content.strip() if comp.choices else ""
        dt_ms = int((time.perf_counter() - t0) * 1000)
        if content.upper().startswith("OK"):
            return {"status": "ok", "detail": f"model={model_name}, ~{dt_ms} ms"}
        return {"status": "info", "detail": f"unexpected reply (~{dt_ms} ms): {content[:60]}"}
    except Exception as e:
        name = type(e).__name__.lower()
        if "authentication" in name or "auth" in name: return {"status": "error", "detail": "invalid_api_key"}
        if "timeout" in name or "timedout" in name:   return {"status": "error", "detail": "timeout"}
        if "rate" in name and "limit" in name:        return {"status": "error", "detail": "rate_limit"}
        if "connection" in name or "network" in name: return {"status": "error", "detail": "network_error"}
        return {"status": "error", "detail": name or "unknown_error"}


# ─────────────────────────────
# Summary trend table builder (Years as columns)
# ─────────────────────────────
def build_trend_summary_table(df_disp: pd.DataFrame, category_in_play: bool, metric_col: str, selected_years: list[str] | None = None) -> pd.DataFrame:
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
    years = sorted([str(y) for y in (selected_years or df["Year"].astype(str).unique().tolist())], key=lambda x: int(x))
    pivot = df.pivot_table(index="Demographic", columns="Year", values=metric_col, aggfunc="first").copy()
    pivot.index.name = "Segment"
    for y in years:
        if y not in pivot.columns:
            pivot[y] = PD.NA
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
def build_pdf_report(question_code: str, question_text: str, selected_years: list[str], dem_display: list[str], narrative: str, df_summary: pd.DataFrame, ui_label: str) -> bytes | None:
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
    title = styles["Heading1"]; h2 = styles["Heading2"]; body = styles["BodyText"]; small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, textColor="#555555")

    flow = []
    flow.append(Paragraph("PSES Analysis Report", title))
    flow.append(Paragraph(f"{question_code} — {question_text}", body))
    flow.append(Paragraph(ui_label, small))
    flow.append(Spacer(1, 10))

    ctx = f"<b>Years:</b> {', '.join(selected_years)}<br/><b>Demographic selection:</b> {', '.join(dem_display)}"
    flow.append(Paragraph(ctx, body)); flow.append(Spacer(1, 10))

    flow.append(Paragraph("Analysis Summary", h2))
    if (narrative or "").strip():
        for chunk in narrative.split("\n"):
            if chunk.strip():
                flow.append(Paragraph(chunk.strip(), body)); flow.append(Spacer(1, 4))
    else:
        flow.append(Paragraph("AI analysis was unavailable for this run.", small))

    if df_summary is not None and not df_summary.empty:
        flow.append(Spacer(1, 10)); flow.append(Paragraph("Summary Table", h2))
        data = [df_summary.columns.tolist()] + df_summary.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F0F0")), ("TEXTCOLOR",(0,0),(-1,0),colors.black), ("ALIGN",(0,0),(-1,-1),"LEFT"), ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("BOTTOMPADDING",(0,0),(-1,0),6), ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#CCCCCC"))]))
        flow.append(tbl)

    doc.build(flow)
    return buf.getvalue()


# ─────────────────────────────
# Small helper to detect backend (non-fatal if unavailable)
# ─────────────────────────────
def _detect_backend():
    try:
        if hasattr(_dl, "LAST_BACKEND"):
            return getattr(_dl, "LAST_BACKEND")
        if hasattr(_dl, "get_last_backend") and callable(_dl.get_last_backend):
            return _dl.get_last_backend()
        if hasattr(_dl, "BACKEND_IN_USE"):
            return getattr(_dl, "BACKEND_IN_USE")
    except Exception:
        pass
    return "DuckDB" if os.environ.get("USE_DUCKDB", "1") == "1" else "pandas"


# ─────────────────────────────
# Lightweight profiler for step timing
# ─────────────────────────────
class Profiler:
    def __init__(self):
        self.steps: list[tuple[str, float]] = []

    @contextmanager
    def step(self, name: str, live=None, engine: str = "", t0_global: float | None = None):
        t0 = time.perf_counter()
        if live is not None and t0_global is not None:
            live.caption(f"Processing… {name} • engine: {engine} • {time.perf_counter() - t0_global:.1f}s")
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.steps.append((name, dt))


# ─────────────────────────────
# UI
# ─────────────────────────────
def run_menu1():
    st.markdown(
        """
    <style>
      .custom-header{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }
      .custom-instruction{ font-size: 15px; line-height: 1.4; margin-bottom: 8px; color: #333; }
      .field-label{ font-size: 16px; font-weight: 600; margin: 10px 0 2px; color: #222; }
      .big-button button{ font-size: 16px; padding: 0.6em 1.6em; margin-top: 16px; }
      .tiny-note{ font-size: 12px; color: #666; margin-top: -4px; margin-bottom: 10px; }
      .q-sub{ font-size: 14px; color: #333; margin-top: -4px; margin-bottom: 2px; }
      .codebox { font-family: monospace; white-space: pre-wrap; border: 1px solid #ddd; padding: 8px; border-radius: 6px; background: #fafafa; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    # Auto-run AI health check once per session (no button)
    if st.session_state.get("ai_health_checked") is None:
        st.session_state["ai_health_result"] = run_ai_health_check()
        st.session_state["ai_health_checked"] = True

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

        # Toggles
        show_debug = st.toggle("🔧 Show technical parameters & diagnostics", value=st.session_state.get("show_debug", SHOW_DEBUG))
        st.session_state["show_debug"] = show_debug
        ai_enabled = st.toggle("🧠 Enable AI analysis (OpenAI)", value=st.session_state.get("ai_enabled", False), help="Turn OFF while testing to avoid API usage.")
        st.session_state["ai_enabled"] = ai_enabled

        # Question selector
        st.markdown('<div class="field-label">Select a survey question:</div>', unsafe_allow_html=True)
        question_options = qdf["display"].tolist()
        selected_label = st.selectbox("Question", question_options, key="question_dropdown", label_visibility="collapsed")
        question_code = qdf.loc[qdf["display"] == selected_label, "code"].values[0]
        question_text = qdf.loc[qdf["display"] == selected_label, "text"].values[0]

        # Years
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

        # Demographic selection
        DEMO_CAT_COL = "DEMCODE Category"
        LABEL_COL = "DESCRIP_E"
        st.markdown('<div class="field-label">Select a demographic category (or All respondents):</div>', unsafe_allow_html=True)
        demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
        demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

        sub_selection = None
        if demo_selection != "All respondents":
            st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
            sub_items = (
                demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL]
                .dropna().astype(str).unique().tolist()
            )
            sub_items = sorted(sub_items)
            sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=f"sub_{demo_selection.replace(' ', '_')}", label_visibility="collapsed")
            if sub_selection == "":
                sub_selection = None

        # Resolve DEMCODEs once (as text)
        demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(demo_df, demo_selection, sub_selection)
        if category_in_play and (None not in demcodes):
            demcodes = [None] + demcodes
        dem_display = ["All respondents" if c is None else str(c).strip() for c in demcodes]

        # Diagnostics: parameter/environment/AI prompt (unchanged)
        if show_debug:
            with st.expander("🛠 Diagnostics", expanded=False):
                tabs = st.tabs(["1) Parameters sent to the database", "2) Environment diagnostics", "3) AI prompt visibility"])
                with tabs[0]:
                    params_df = PD.DataFrame({"Parameter": ["QUESTION (from metadata)", "SURVEYR (years)", "DEMCODE(s) (from metadata)"], "Value": [question_code, ", ".join(selected_years), ", ".join(dem_display)]})
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                with tabs[1]:
                    info = {}
                    info["OPENAI_API_KEY_present"] = bool(os.environ.get("OPENAI_API_KEY", "").strip())
                    info["OPENAI_MODEL_secret_present"] = bool(st.secrets.get("OPENAI_MODEL", ""))
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
                with tabs[2]:
                    hc = st.session_state.get("ai_health_result") or {}
                    status = hc.get("status", "info"); detail = hc.get("detail", "")
                    if status == "ok": st.success(f"AI health check passed ({detail}).")
                    elif status == "warn": st.warning(f"AI health check: {detail}.")
                    elif status == "error": st.error(f"AI health check error: {detail}.")
                    else: st.info(f"AI health check: {detail or 'no details'}")
                    model_used = (st.session_state.get("last_ai_model") or st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini")
                    st.caption(f"Last AI model used: {model_used}")
                    sys_txt = st.session_state.get("last_ai_system", ""); usr_txt = st.session_state.get("last_ai_user", "")
                    if not sys_txt and not usr_txt:
                        st.info("No AI prompt available yet. Run a query with AI enabled to populate this view.")
                    else:
                        st.markdown("**System prompt (exact):**"); st.code(sys_txt, language="markdown")
                        st.markdown("**User payload (JSON sent to the model):**")
                        kb = len(usr_txt.encode("utf-8")) / 1024.0; st.caption(f"Approx size: ~{kb:.1f} KB")
                        st.code(usr_txt, language="json")

        # Run query (single pass, cached big file)
        if st.button("🔎 Run query"):
            engine_used = _detect_backend()
            status_line = st.empty()  # live line for engine + elapsed seconds

            prof = Profiler()
            t0_global = time.perf_counter()

            with st.spinner("Processing data…"):
                status_line.caption(f"Processing… engine: {engine_used} • 0.0s")

                # 1) Match scales
                with prof.step("Match scales", live=status_line, engine=engine_used, t0_global=t0_global):
                    scale_pairs = get_scale_labels(load_scales_metadata(), question_code)
                    if scale_pairs is None or len(scale_pairs) == 0:
                        qnorm = _normalize_qcode(question_code)
                        st.error(f"Metadata scale not found for question '{question_code}' (normalized '{qnorm}'). Please verify 'Survey Scales.xlsx'.")
                        return

                # 2) Load data via loader
                with prof.step("Load data", live=status_line, engine=engine_used, t0_global=t0_global):
                    try:
                        df_raw = load_results2024_filtered(question_code=question_code, years=selected_years, group_values=demcodes)
                    except TypeError:
                        parts = []
                        for gv in demcodes:
                            try:
                                parts.append(load_results2024_filtered(question_code=question_code, years=selected_years, group_value=(None if gv is None else str(gv).strip())))
                            except TypeError:
                                continue
                        df_raw = (PD.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else PD.DataFrame())

                if df_raw is None or df_raw.empty:
                    status_line.caption(f"Processing complete • engine: {engine_used} • {time.perf_counter()-t0_global:.1f}s")
                    st.info("No data found for this selection.")
                    return

                # 3) Suppression
                with prof.step("Suppress 999/9999", live=status_line, engine=engine_used, t0_global=t0_global):
                    df_raw = exclude_999_raw(df_raw)
                    if df_raw.empty:
                        status_line.caption(f"Processing complete • engine: {engine_used} • {time.perf_counter()-t0_global:.1f}s")
                        st.info("Data exists, but all rows are not applicable (999/9999).")
                        return

                # 4) Sort & format
                with prof.step("Sort & format table", live=status_line, engine=engine_used, t0_global=t0_global):
                    if "SURVEYR" in df_raw.columns:
                        df_raw = df_raw.sort_values(by="SURVEYR", ascending=False)

                    dem_map_clean = {None: "All respondents"}
                    try:
                        for k, v in (disp_map or {}).items():
                            dem_map_clean[(None if k is None else str(k).strip())] = v
                    except Exception:
                        pass

                    df_disp = format_display_table_raw(df=df_raw, category_in_play=category_in_play, dem_disp_map=dem_map_clean, scale_pairs=scale_pairs)

            total_s = time.perf_counter() - t0_global
            status_line.caption(f"Processing complete • engine: {engine_used} • {total_s:.1f}s")

            # ===== Tabs: Results | Diagnostics =====
            results_tab, diag_tab = st.tabs(["Results", "Diagnostics"])

            # ---------------- Results tab ----------------
            with results_tab:
                # Results table
                st.subheader(f"{question_code} — {question_text}")
                st.dataframe(df_disp, use_container_width=True)

                # Data source + engine
                st.caption("Data source: https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f")
                st.caption(f"Backend engine: {engine_used}")

                # Decide metric
                decision = detect_metric_mode(df_disp, scale_pairs)
                metric_col = decision["metric_col"]; ui_label = decision["ui_label"]; metric_label = decision["metric_label"]

                # Analysis Summary (AI)
                st.markdown("### Analysis Summary")
                st.markdown(f"<div class='q-sub'>{question_code} — {question_text}</div><div class='tiny-note'>{ui_label}</div>", unsafe_allow_html=True)

                narrative = ""
                if st.session_state.get("ai_enabled", False):
                    ai_t0 = time.perf_counter()
                    ai_out = _ai_narrative_and_storytable(df_disp=df_disp, question_code=question_code, question_text=question_text, category_in_play=category_in_play, metric_col=metric_col, metric_label=metric_label, temperature=0.2)
                    narrative = (ai_out.get("narrative") or "").strip()
                    hint = (ai_out.get("hint") or "").strip()
                    if narrative:
                        model_used = (st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
                        narrative += f"\n\n_Powered by OpenAI model {model_used}_"
                        st.write(narrative)
                    else:
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
                    prof.steps.append(("AI summary", time.perf_counter() - ai_t0))
                else:
                    st.caption("AI analysis is disabled (toggle above).")

                # Summary Table
                st.markdown("### Summary Table")
                st.markdown(f"<div class='q-sub'>{question_code} — {question_text}</div><div class='tiny-note'>{ui_label}</div>", unsafe_allow_html=True)
                trend_t0 = time.perf_counter()
                trend_df = build_trend_summary_table(df_disp=df_disp, category_in_play=category_in_play, metric_col=metric_col, selected_years=selected_years)
                prof.steps.append(("Build summary table", time.perf_counter() - trend_t0))

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
                        ctx = {"QUESTION": question_code, "SURVEYR (years)": ", ".join(selected_years), "DEMCODE(s)": ", ".join(dem_display), "Metric used": metric_label, "AI enabled": "Yes" if st.session_state.get("ai_enabled", False) else "No", "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        PD.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Context", index=False)
                    xdata = xbuf.getvalue()
                st.download_button(label="⬇️ Download data (Excel)", data=xdata, file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                pdf_bytes = build_pdf_report(question_code=question_code, question_text=question_text, selected_years=selected_years, dem_display=dem_display, narrative=narrative, df_summary=trend_df if trend_df is not None else PD.DataFrame(), ui_label=ui_label)
                if pdf_bytes:
                    st.download_button(label="⬇️ Download summary report (PDF)", data=pdf_bytes, file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.pdf", mime="application/pdf")
                else:
                    st.caption("PDF export unavailable (install `reportlab` in requirements to enable).")

            # ---------------- Diagnostics tab ----------------
            with diag_tab:
                st.subheader("Loading & Backend Details")

                # 1) Engine / paths / elapsed / rows (if loader exposes diagnostics)
                diag = {}
                try:
                    diag = get_last_query_diag() or {}
                except Exception:
                    diag = {}

                if diag:
                    st.markdown(
                        f"**Engine**: `{diag.get('engine','?')}` &nbsp;&nbsp; "
                        f"**Elapsed**: `{diag.get('elapsed_ms','?')} ms` &nbsp;&nbsp; "
                        f"**Rows**: `{diag.get('rows','?')}`"
                    )
                    cols = ["engine","elapsed_ms","rows","question_code","years","group_value","parquet_dir","csv_path","parquet_error"]
                    diag_rows = [{"Field": k, "Value": diag.get(k)} for k in cols if k in diag]
                    if diag_rows:
                        st.table(pd.DataFrame(diag_rows))
                else:
                    # Fall back to static backend info if diagnostics are not implemented in the loader
                    try:
                        info = _dl.get_backend_info() if hasattr(_dl, "get_backend_info") else {}
                    except Exception:
                        info = {}
                    if info:
                        st.write(info)
                    else:
                        st.info("No loader diagnostics available for this session.")

                # 2) Step timings (moved here from Results page)
                if prof.steps:
                    st.markdown("**Step timings (ms)**")
                    prof_df = PD.DataFrame(
                        [{"Step": name, "Duration (ms)": int(dt * 1000)} for name, dt in prof.steps]
                    ).sort_values("Duration (ms)", ascending=False, ignore_index=True)
                    total_ms = int(sum(dt for _, dt in prof.steps) * 1000)
                    st.caption(f"Total (profiled): ~{total_ms} ms")
                    st.table(prof_df)

                # 3) Optional: show resolved raw path if available
                try:
                    path_hint = _resolve_results_path()
                    if path_hint:
                        st.caption(f"Resolved results path: `{path_hint}`")
                except Exception:
                    pass


if __name__ == "__main__":
    run_menu1()
