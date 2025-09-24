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

# Safe imports from loader: use only helpers that exist in your loader file
# (get_backend_info + prewarm_fastpath are present; get_last_query_diag is NOT.)
try:
    from utils.data_loader import (
        load_results2024_filtered,
        get_results2024_schema,              # may or may not exist; we don't use it here
        get_results2024_schema_inferred,     # may or may not exist; we don't use it here
    )
except Exception:
    from utils.data_loader import load_results2024_filtered  # type: ignore
def get_results2024_schema(): return {}
def get_results2024_schema_inferred(): return {}

# ✅ Available in your loader (used for diagnostics + warmup)
try:
    from utils.data_loader import get_backend_info, prewarm_fastpath  # <-- exists in your loader
except Exception:
    def get_backend_info(): return {}
    def prewarm_fastpath(): return "csv"

# Ensure OpenAI key is available from Streamlit secrets (no hardcoding)
os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# ── Debug/diagnostic visibility toggle ─────────────────────────────────────────
SHOW_DEBUG = False  # <- set to True to show parameters preview + diagnostics

# Stable alias
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
    s = "" if s is None else str(s)
    s = s.upper()
    return "".join(ch for ch in s if ch.isalnum())

def _norm_q(x: str) -> str:
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
    s = "".join(ch for ch in str(s) if s is not None and ch.isdigit())
    return s.zfill(4) if s else ""

def resolve_demographic_codes_from_metadata(demo_df, category_label, subgroup_label):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"
    code_col = _find_demcode_col(demo_df)

    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
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
            raw_code = str(r[code_col]); label = str(r[LABEL_COL])
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
    if scales_df is None or scales_df.empty:
        return None
    qnorm = _normalize_qcode(question_code)
    if "__code_norm__" not in scales_df.columns:
        return None
    match = scales_df[sdf := (scales_df["__code_norm__"] == qnorm)]
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
    """
    Replace 999/9999 with NA (do NOT drop rows), so partially valid rows remain.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    candidates = [f"answer{i}" for i in range(1, 7 + 1)] + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE", "YES",
                                                             "positive_pct", "neutral_pct", "negative_pct", "n"]
    present = [c for c in candidates if c in out.columns]
    for c in present:
        s = out[c].astype(str).str.strip()
        mask = (s == "999") | (s == "9999")
        # also catch numeric 999/9999
        mask |= pd.to_numeric(out[c], errors="coerce").isin([999, 9999])
        out.loc[mask, c] = pd.NA
    return out

def format_display_table_raw(df, category_in_play, dem_disp_map, scale_pairs) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["SURVEYR"] = pd.to_numeric(out.get("SURVEYR", out.get("year")), errors="coerce").astype("Int64")
    out["Year"] = out["SURVEYR"].astype(str)

    if category_in_play:
        def to_label(code):
            key = "" if code is None else str(code).strip()
            if key == "":
                return "All respondents"
            return dem_disp_map.get(key, str(code))
        # DEMCODE in CSV fallback, group_value in Parquet
        dem_src = "DEMCODE" if "DEMCODE" in out.columns else "group_value"
        out["Demographic"] = out[dem_src].apply(to_label)

    dist_cols_raw, rename_map = [], {}
    if scale_pairs:
        for k, v in scale_pairs:
            for kcand in (k.upper(), k):
                if kcand in out.columns:
                    dist_cols_raw.append(kcand); rename_map[kcand] = v; break

    keep_cols = (["Year"] + (["Demographic"] if category_in_play else []) + dist_cols_raw +
                 ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE",
                  "positive_pct","neutral_pct","negative_pct"])
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map).copy()

    # Drop answer columns that are entirely NA
    answer_label_cols = [v for v in rename_map.values() if v in out.columns]
    drop_all_na = [c for c in answer_label_cols if PD.to_numeric(out[c], errors="coerce").isna().all()]
    if drop_all_na:
        out = out.drop(columns=drop_all_na)

    # NEW: filter out rows where ALL core metrics are NA (after 9999→NA)
    core_candidates = []
    core_candidates += ["POSITIVE", "AGREE"]
    core_candidates += [c for c in answer_label_cols if c in out.columns]
    core_candidates = [c for c in core_candidates if c in out.columns]
    if core_candidates:
        mask_any = PD.Series(False, index=out.index)
        for c in core_candidates:
            mask_any = mask_any | PD.to_numeric(out[c], errors="coerce").notna()
        out = out.loc[mask_any].copy()

    sort_cols = ["Year"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    return out


# ─────────────────────────────
# Metric decision (matches requested rule)
# ─────────────────────────────
def detect_metric_mode(df_disp: pd.DataFrame, scale_pairs) -> dict:
    """
    Choose metric for both Summary (if applicable) and AI analysis using the rule:
      1) POSITIVE if it has data
      2) else AGREE if it has data
      3) else first Answer (Answer1..7 label) with data
    Also returns a flag 'summary_allowed' which is True only if mode is POSITIVE or AGREE.
    """
    cols_l = {c.lower(): c for c in df_disp.columns}

    # POSITIVE
    if "positive" in cols_l:
        col = cols_l["positive"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {"mode":"positive","metric_col":col,"ui_label":"(% positive answers)","metric_label":"% positive","summary_allowed":True}

    # AGREE
    if "agree" in cols_l:
        col = cols_l["agree"]
        if PD.to_numeric(df_disp[col], errors="coerce").notna().any():
            return {"mode":"agree","metric_col":col,"ui_label":"(% agree)","metric_label":"% agree","summary_allowed":True}

    # First answer label with data
    if scale_pairs:
        for k, v in scale_pairs:
            label = v  # renamed label appearing in df_disp
            if label in df_disp.columns and PD.to_numeric(df_disp[label], errors="coerce").notna().any():
                return {"mode":k.lower(),"metric_col":label,"ui_label":f"(% {label})","metric_label":f"% {label}","summary_allowed":False}

    # Nothing found — default to POSITIVE but summary not allowed
    return {"mode":"none","metric_col":cols_l.get("positive","POSITIVE"),"ui_label":"(% positive answers)","metric_label":"% positive","summary_allowed":False}


# ─────────────────────────────
# Arrow-safe display helper
# ─────────────────────────────
def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object/mixed columns to string for clean Arrow serialization."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out


# ─────────────────────────────
# AI helpers (compact)
# ─────────────────────────────
def _class_name(e: Exception) -> str: return type(e).__name__

def _call_openai_with_retry(client, **kwargs) -> tuple[str, str]:
    try:
        comp = client.chat.completions.create(timeout=60.0, **kwargs)
        content = comp.choices[0].message.content if comp.choices else ""
        return (content or "", "" if content else "empty response")
    except Exception:
        try:
            kwargs2 = {k: v for k, v in kwargs.items() if k != "response_format"}
            comp = client.chat.completions.create(timeout=60.0, **kwargs2)
            content = comp.choices[0].message.content if comp.choices else ""
            return (content or "", "" if content else "empty response")
        except Exception as e2:
            name2 = _class_name(e2).lower()
            if "authentication" in name2 or "auth" in name2: return "", "invalid_api_key"
            if "timeout" in name2 or "timedout" in name2:   return "", "timeout"
            if "rate" in name2 and "limit" in name2:        return "", "rate_limit"
            if "connection" in name2 or "network" in name2: return "", "network_error"
            if "badrequest" in name2 or "invalidrequest" in name2: return "", "invalid_request"
            if "typeerror" in name2: return "", "type_error"
            return "", name2 or "unknown_error"

def _ai_build_payload_single_metric(df_disp, question_code, question_text, category_in_play, metric_col):
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

def _ai_narrative_and_storytable(df_disp, question_code, question_text, category_in_play, metric_col, metric_label, temperature: float = 0.2) -> dict:
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

    # Prompt change: "minimal ≤2" (was "normal ≤2")
    system = (
        "You are preparing insights for the Government of Canada’s Public Service Employee Survey (PSES).\n\n"
        "Context\n"
        "- The PSES provides information to improve people management practices in the federal public service.\n"
        "- Results help departments and agencies identify strengths and concerns in areas such as employee engagement, anti-racism, equity and inclusion, and workplace well-being.\n"
        "- The survey tracks progress over time to refine action plans. Employees’ voices guide improvements to workplace quality, which leads to better results for the public service and Canadians.\n"
        "- Each cycle includes recurring questions (for tracking trends) and new/modified questions reflecting evolving priorities (e.g., updated Employment Equity questions and streamlined hybrid-work items in 2024).\n"
        "- Statistics Canada administers the survey with the Treasury Board of Canada Secretariat. Confidentiality is guaranteed under the Statistics Act (grouped reporting; results for groups <10 are suppressed).\n\n"
        "Data-use rules (hard constraints)\n"
        "- Use ONLY the provided JSON payload/table. DO NOT invent, assume, extrapolate, infer, or generalize beyond the numbers present. No speculation or hypotheses.\n"
        "- Public Service–wide scope ONLY; do not reference specific departments unless present in the payload.\n"
        "- Express percentages as whole numbers (e.g., “75%”). Use “points” for differences/changes.\n\n"
        "Analysis rules\n"
        "- Begin with the 2024 result for the selected question (metric_label).\n"
        "- Describe trend over time: compare 2024 with the earliest year available, using thresholds:\n"
        "  • stable ≤1 point\n"
        "  • slight >1–2 points\n"
        "  • notable >2 points\n"
        "- Compare demographic groups in 2024:\n"
        "  • Focus on the most relevant comparisons (largest gap(s), or those crossing thresholds).\n"
        "  • Report gaps in points and classify them: minimal ≤2, notable >2–5, important >5.\n"
        "- If multiple groups are present, highlight only the most meaningful contrasts instead of exhaustively listing all.\n"
        "- Mention whether gaps observed in 2024 have widened, narrowed, or remained stable compared with earlier years.\n"
        "- Conclude with a concise overall statement (e.g., “Overall, results have remained steady and demographic gaps are unchanged”).\n\n"
        "Style & output\n"
        "- Professional, concise, neutral. Narrative style (1–3 short paragraphs, no lists).\n"
        "- Output VALID JSON with exactly one key: \"narrative\".\n"
    )

    user_payload = {"metric_label": metric_label, "payload": data}
    user = json.dumps(user_payload, ensure_ascii=False)

    st.session_state["last_ai_model"] = model_name
    st.session_state["last_ai_system"] = system
    st.session_state["last_ai_user"] = user

    kwargs = dict(model=model_name, temperature=temperature, response_format={"type": "json_object"}, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    content, hint = _call_openai_with_retry(client, **kwargs)
    if not content: return {"narrative": "", "hint": hint or "no_content"}
    try:
        out = json.loads(content)
    except Exception:
        return {"narrative": "", "hint": "json_decode_error"}
    if not isinstance(out, dict):
        return {"narrative": "", "hint": "non_dict_json"}

    n = out.get("narrative", "")
    if isinstance(n, (dict, list)):
        n = (n.get("text", "") if isinstance(n, dict) and "text" in n
             else json.dumps(n, ensure_ascii=False))
    else:
        n = str(n)

    return {"narrative": n.strip(), "hint": ""}


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
# Summary trend table builder
# ─────────────────────────────
def build_trend_summary_table(df_disp, category_in_play, metric_col, selected_years: list[str] | None = None) -> pd.DataFrame:
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
    return pivot[["Segment"] + years]


# ─────────────────────────────
# Report (PDF) builder
# ─────────────────────────────
def build_pdf_report(question_code, question_text, selected_years, dem_display, narrative, df_summary, ui_label) -> bytes | None:
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
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F0F0F0")),("TEXTCOLOR",(0,0),(-1,0),colors.black),("ALIGN",(0,0),(-1,-1),"LEFT"),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("BOTTOMPADDING",(0,0),(-1,0),6),("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#CCCCCC"))]))
        flow.append(tbl)

    doc.build(flow)
    return buf.getvalue()


# ─────────────────────────────
# Backend detect (soft)
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
    # soft guess
    try:
        import pyarrow  # noqa
        if get_backend_info().get("parquet_ready"):
            return "parquet"
    except Exception:
        pass
    return "csv"


# ─────────────────────────────
# Lightweight profiler
# ─────────────────────────────
class Profiler:
    def __init__(self):
        self.steps: list[tuple[str, float]] = []
    from contextlib import contextmanager
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

    # 🔸 Optional one-time prewarm on first open of Menu 1
    if not st.session_state.get("prewarmed_once"):
        try:
            with st.spinner("⚡ Preparing fast path (one-time)…"):
                prewarm_fastpath()  # builds Parquet if possible, else ensures CSV
            st.session_state["prewarmed_once"] = True
        except Exception:
            # Non-fatal: we still run on CSV fallback
            st.session_state["prewarmed_once"] = True

    demo_df = load_demographics_metadata()
    qdf = load_questions_metadata()
    sdf = load_scales_metadata()

    # Auto-run AI health check once per session
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
        # Description (updated previously)
        st.markdown(
            '<div class="custom-instruction">Please select a question you are interested in, the survey year and, optionally, a demographic breakdown.<br>'
            'This application provides only Public Service-wide results. The output is a summary and a detailed result table with a short analysis.</div>',
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
        DEMO_CAT_COL = "DEMCODE Category"; LABEL_COL = "DESCRIP_E"
        st.markdown('<div class="field-label">Select a demographic category (or All respondents):</div>', unsafe_allow_html=True)
        demo_categories = ["All respondents"] + sorted(demo_df[DEMO_CAT_COL].dropna().astype(str).unique().tolist())
        demo_selection = st.selectbox("Demographic category", demo_categories, key="demo_main", label_visibility="collapsed")

        sub_selection = None
        if demo_selection != "All respondents":
            st.markdown(f'<div class="field-label">Subgroup ({demo_selection}) (optional):</div>', unsafe_allow_html=True)
            sub_items = sorted(
                demo_df.loc[demo_df[DEMO_CAT_COL] == demo_selection, LABEL_COL]
                .dropna().astype(str).unique().tolist()
            )
            sub_selection = st.selectbox("(leave blank to include all subgroups in this category)", [""] + sub_items, key=f"sub_{demo_selection.replace(' ', '_')}", label_visibility="collapsed")
            if sub_selection == "":
                sub_selection = None

        # Resolve DEMCODEs once (as text)
        demcodes, disp_map, category_in_play = resolve_demographic_codes_from_metadata(demo_df, demo_selection, sub_selection)
        if category_in_play and (None not in demcodes):
            demcodes = [None] + demcodes
        dem_display = ["All respondents" if c is None else str(c).strip() for c in demcodes]

        # ──────────────────────────────────────────────────────────────────
        # Run query (single pass, cached big file)
        # ──────────────────────────────────────────────────────────────────
        if st.button("🔎 Run query"):
            engine_guess = _detect_backend()
            status_line = st.empty()
            prof = Profiler()
            t0_global = time.perf_counter()

            with st.spinner("Processing data…"):
                status_line.caption(f"Processing… engine: {engine_guess} • 0.0s")

                # 1) Match scales
                with prof.step("Match scales", live=status_line, engine=engine_guess, t0_global=t0_global):
                    scale_pairs = get_scale_labels(load_scales_metadata(), question_code)
                    if not scale_pairs:
                        qnorm = _normalize_qcode(question_code)
                        st.error(f"Metadata scale not found for question '{question_code}' (normalized '{qnorm}'). Please verify 'Survey Scales.xlsx'.")
                        return

                # 2) Load data
                with prof.step("Load data", live=status_line, engine=engine_guess, t0_global=t0_global):
                    try:
                        df_raw = load_results2024_filtered(question_code=question_code, years=selected_years, group_values=demcodes)  # type: ignore[arg-type]
                    except TypeError:
                        parts = []
                        for gv in demcodes:
                            try:
                                parts.append(load_results2024_filtered(question_code=question_code, years=selected_years, group_value=(None if gv is None else str(gv).strip())))
                            except TypeError:
                                continue
                        df_raw = (PD.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else PD.DataFrame())

                if df_raw is None or df_raw.empty:
                    status_line.caption(f"Processing complete • engine: {engine_guess} • {time.perf_counter()-t0_global:.1f}s")
                    st.info("No data found for this selection.")
                    try:
                        backend = get_backend_info() or {}
                    except Exception:
                        backend = {}
                    st.session_state["last_loader_diag"] = {
                        "engine": engine_guess,
                        "elapsed_ms": int((time.perf_counter() - t0_global) * 1000),
                        "rows": 0,
                        "question_code": question_code,
                        "years": ",".join(selected_years),
                        "group_value": ("multiple" if len(demcodes) > 1 else str(demcodes[0])),
                        "parquet_dir": backend.get("parquet_dir"),
                        "csv_path": backend.get("csv_path"),
                        "parquet_error": None,
                    }
                    st.session_state["last_prof_steps"] = list(prof.steps)
                    return

                # 3) Replace 999/9999 with NA (keep rows)
                with prof.step("999/9999 → NA", live=status_line, engine=engine_guess, t0_global=t0_global):
                    df_raw = exclude_999_raw(df_raw)

                # 4) Sort, format & filter-out NA rows
                with prof.step("Sort & format table", live=status_line, engine=engine_guess, t0_global=t0_global):
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
            status_line.caption(f"Processing complete • engine: {engine_guess} • {total_s:.1f}s")

            # ---------------- Results ----------------
            st.subheader(f"{question_code} — {question_text}")

            # Decide metric (POSITIVE → AGREE → first Answer)
            decision = detect_metric_mode(df_disp, scale_pairs)
            metric_col = decision["metric_col"]; ui_label = decision["ui_label"]; metric_label = decision["metric_label"]
            summary_allowed = bool(decision.get("summary_allowed", False))

            # Build trend summary only if allowed (POSITIVE/AGREE)
            trend_t0 = time.perf_counter()
            trend_df = PD.DataFrame()
            if summary_allowed:
                trend_df = build_trend_summary_table(df_disp=df_disp, category_in_play=category_in_play, metric_col=metric_col, selected_years=selected_years)
            prof.steps.append(("Build summary table", time.perf_counter() - trend_t0))

            # Two tabs with Summary first, Detailed second
            tab_summary, tab_detail = st.tabs(["Summary results", "Detailed results"])

            with tab_summary:
                st.markdown(f"<div class='q-sub'>{question_code} — {question_text}</div><div class='tiny-note'>{ui_label}</div>", unsafe_allow_html=True)
                if summary_allowed and trend_df is not None and not trend_df.empty:
                    st.dataframe(make_arrow_safe(trend_df), use_container_width=True, hide_index=True)
                else:
                    st.info("Summary table is unavailable for this selection.")
                # Source link
                st.markdown(
                    "Source: [2024 Public Service Employee Survey Results - Open Government Portal]"
                    "(https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f)"
                )

            with tab_detail:
                df_disp_display = make_arrow_safe(df_disp)
                st.dataframe(df_disp_display, use_container_width=True)
                st.caption(f"Backend engine: {engine_guess}")

            # Narrative AFTER tabs with requested title — uses the same metric rule
            st.markdown("### Analysis Summary")
            st.markdown(f"<div class='q-sub'>{question_code} — {question_text}</div><div class='tiny-note'>{ui_label}</div>", unsafe_allow_html=True)

            narrative = ""
            if st.session_state.get("ai_enabled", False):
                ai_t0 = time.perf_counter()
                ai_status = st.empty()
                ai_status.info("Preparing AI summary…")
                with st.spinner("🤖 Contacting OpenAI… generating analysis"):
                    ai_out = _ai_narrative_and_storytable(
                        df_disp=df_disp,
                        question_code=question_code,
                        question_text=question_text,
                        category_in_play=category_in_play,
                        metric_col=metric_col,      # POSITIVE→AGREE→first Answer with data
                        metric_label=metric_label,
                        temperature=0.2
                    )
                ai_status.empty()
                narrative = str(ai_out.get("narrative") or "").strip()
                hint = str(ai_out.get("hint") or "").strip()
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

            # Downloads (Summary sheet only when summary_allowed)
            with io.BytesIO() as xbuf:
                with PD.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
                    df_disp.to_excel(writer, sheet_name="Results", index=False)
                    if summary_allowed and trend_df is not None and not trend_df.empty:
                        trend_df.to_excel(writer, sheet_name="Summary Table", index=False)
                    ctx = {
                        "QUESTION": question_code,
                        "SURVEYR (years)": ", ".join(selected_years),
                        "DEMCODE(s)": ", ".join(dem_display),
                        "Metric used": metric_label,
                        "AI enabled": "Yes" if st.session_state.get("ai_enabled", False) else "No",
                        "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    PD.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Context", index=False)
                xdata = xbuf.getvalue()
            st.download_button(
                label="⬇️ Download data (Excel)",
                data=xdata,
                file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            pdf_bytes = build_pdf_report(
                question_code=question_code,
                question_text=question_text,
                selected_years=selected_years,
                dem_display=dem_display,
                narrative=narrative,
                df_summary=(trend_df if (summary_allowed and trend_df is not None) else PD.DataFrame()),
                ui_label=ui_label
            )
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download summary report (PDF)",
                    data=pdf_bytes,
                    file_name=f"PSES_{question_code}_{'-'.join(selected_years)}.pdf",
                    mime="application/pdf"
                )
            else:
                st.caption("PDF export unavailable (install `reportlab` in requirements to enable).")

            # ── Persist diagnostics for Diagnostics → Loading details tab
            try:
                backend = get_backend_info() or {}
            except Exception:
                backend = {}
            rows_count = 0
            if isinstance(df_raw, pd.DataFrame):
                rows_count = int(df_raw.shape[0])
            st.session_state["last_loader_diag"] = {
                "engine": engine_guess,
                "elapsed_ms": int(total_s * 1000),
                "rows": rows_count,
                "question_code": question_code,
                "years": ",".join(selected_years),
                "group_value": ("multiple" if len(demcodes) > 1 else str(demcodes[0])),
                "parquet_dir": backend.get("parquet_dir"),
                "csv_path": backend.get("csv_path"),
                "parquet_error": None,
            }
            st.session_state["last_prof_steps"] = list(prof.steps)


if __name__ == "__main__":
    run_menu1()
