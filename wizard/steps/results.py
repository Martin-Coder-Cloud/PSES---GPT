# wizard/results.py — run the query and render results (trimmed Menu 1 logic)
from typing import Dict, Any, List
import json, time, os
import pandas as pd
import streamlit as st

# ——— data loader (your existing module) ———
try:
    from utils.data_loader import load_results2024_filtered
except Exception:
    load_results2024_filtered = None  # type: ignore

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
    def _normalize_qcode(s: str) -> str:
        s = "" if s is None else str(s)
        s = s.upper()
        return "".join(ch for ch in s if ch.isalnum())
    sdf["__code_norm__"] = sdf[code_col].astype(str).map(_normalize_qcode)
    return sdf

def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    if scales_df is None or scales_df.empty:
        return None
    def _normalize_qcode(s: str) -> str:
        s = "" if s is None else str(s)
        s = s.upper()
        return "".join(ch for ch in s if ch.isalnum())
    qnorm = _normalize_qcode(question_code)
    if "__code_norm__" not in scales_df.columns:
        return None
    match = scales_df[sf := (scales_df["__code_norm__"] == qnorm)]
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
    if df is None or df.empty:
        return df
    out = df.copy()
    candidates = [f"answer{i}" for i in range(1, 7 + 1)] + ["POSITIVE", "NEUTRAL", "NEGATIVE", "ANSCOUNT", "AGREE",
                                                             "positive_pct", "neutral_pct", "negative_pct", "n"]
    present = [c for c in candidates if c in out.columns]
    for c in present:
        s = out[c].astype(str).str.strip()
        mask = (s == "999") | (s == "9999")
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
    drop_all_na = [c for c in answer_label_cols if pd.to_numeric(out[c], errors="coerce").isna().all()]
    if drop_all_na:
        out = out.drop(columns=drop_all_na)

    # Filter out rows where ALL core metrics are NA (after 9999→NA)
    core_candidates = []
    core_candidates += ["POSITIVE", "AGREE"]
    core_candidates += [c for c in answer_label_cols if c in out.columns]
    core_candidates = [c for c in core_candidates if c in out.columns]
    if core_candidates:
        mask_any = pd.Series(False, index=out.index)
        for c in core_candidates:
            mask_any = mask_any | pd.to_numeric(out[c], errors="coerce").notna()
        out = out.loc[mask_any].copy()

    sort_cols = ["Year"] + (["Demographic"] if category_in_play else [])
    sort_asc = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    return out

def detect_metric_mode(df_disp: pd.DataFrame, scale_pairs) -> dict:
    cols_l = {c.lower(): c for c in df_disp.columns}
    if "positive" in cols_l and pd.to_numeric(df_disp[cols_l["positive"]], errors="coerce").notna().any():
        return {"mode":"positive","metric_col":cols_l["positive"],"ui_label":"(% positive answers)","metric_label":"% positive","summary_allowed":True}
    if "agree" in cols_l and pd.to_numeric(df_disp[cols_l["agree"]], errors="coerce").notna().any():
        return {"mode":"agree","metric_col":cols_l["agree"],"ui_label":"(% agree)","metric_label":"% agree","summary_allowed":True}
    if scale_pairs:
        for k, v in scale_pairs:
            label = v
            if label in df_disp.columns and pd.to_numeric(df_disp[label], errors="coerce").notna().any():
                return {"mode":k.lower(),"metric_col":label,"ui_label":f"(% {label})","metric_label":f"% {label}","summary_allowed":False}
    return {"mode":"none","metric_col":cols_l.get("positive","POSITIVE"),"ui_label":"(% positive answers)","metric_label":"% positive","summary_allowed":False}

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out

def build_trend_summary_table(df_disp, category_in_play, metric_col, selected_years: List[str] | None=None) -> pd.DataFrame:
    if df_disp is None or df_disp.empty or "Year" not in df_disp.columns:
        return pd.DataFrame()
    if metric_col not in df_disp.columns:
        low = {c.lower(): c for c in df_disp.columns}
        if metric_col.lower() in low:
            metric_col = low[metric_col.lower()]
        else:
            return pd.DataFrame()
    df = df_disp.copy()
    if "Demographic" not in df.columns:
        df["Demographic"] = "All respondents"
    years = sorted([str(y) for y in (selected_years or df["Year"].astype(str).unique().tolist())], key=lambda x: int(x))
    pivot = df.pivot_table(index="Demographic", columns="Year", values=metric_col, aggfunc="first").copy()
    pivot.index.name = "Segment"
    for y in years:
        if y not in pivot.columns:
            pivot[y] = pd.NA
    for c in pivot.columns:
        vals = pd.to_numeric(pivot[c], errors="coerce").round(0)
        out = pd.Series("n/a", index=pivot.index, dtype="object")
        mask = vals.notna()
        out.loc[mask] = vals.loc[mask].astype(int).astype(str) + "%"
        pivot[c] = out
    pivot = pivot.reset_index()
    return pivot[["Segment"] + years]

def render(wizard_state: Dict[str, Any]):
    q = wizard_state.get("question") or {}
    years: List[str] = wizard_state.get("years") or []
    demcodes = wizard_state.get("demcodes") or [None]
    category_in_play = bool(wizard_state.get("category_in_play"))
    disp_map = wizard_state.get("dem_disp_map") or {None: "All respondents"}

    if not q or not years:
        st.info("Please complete the steps above, then run the query.")
        return

    question_code = q.get("code")
    question_text = q.get("label") or ""

    # 1) Scales
    scale_pairs = get_scale_labels(load_scales_metadata(), question_code)
    if not scale_pairs:
        st.error(f"Metadata scale not found for question '{question_code}'. Verify 'Survey Scales.xlsx'.")
        return

    # 2) Data
    if load_results2024_filtered is None:
        st.error("Data loader is unavailable (utils.data_loader.load_results2024_filtered).")
        return

    try:
        df_raw = load_results2024_filtered(question_code=question_code, years=years, group_values=demcodes)  # type: ignore[arg-type]
    except TypeError:
        # older signature: group_value (single); join parts
        parts = []
        for gv in demcodes:
            try:
                parts.append(load_results2024_filtered(question_code=question_code, years=years, group_value=(None if gv is None else str(gv).strip())))
            except TypeError:
                continue
        df_raw = (pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame())

    if df_raw is None or df_raw.empty:
        st.info("No data found for this selection.")
        return

    # 3) Clean & display
    df_raw = exclude_999_raw(df_raw)
    df_disp = format_display_table_raw(df=df_raw, category_in_play=category_in_play, dem_disp_map=disp_map, scale_pairs=scale_pairs)

    st.subheader(f"{question_code} — {question_text}")

    decision = detect_metric_mode(df_disp, scale_pairs)
    metric_col = decision["metric_col"]; ui_label = decision["ui_label"]; metric_label = decision["metric_label"]
    summary_allowed = bool(decision.get("summary_allowed", False))

    tab_summary, tab_detail = st.tabs(["Summary results", "Detailed results"])

    with tab_summary:
        st.markdown(f"<div class='tiny-note'>{ui_label}</div>", unsafe_allow_html=True)
        if summary_allowed:
            trend_df = build_trend_summary_table(df_disp=df_disp, category_in_play=category_in_play, metric_col=metric_col, selected_years=years)
            if trend_df is not None and not trend_df.empty:
                st.dataframe(make_arrow_safe(trend_df), use_container_width=True, hide_index=True)
            else:
                st.info("Summary table is unavailable for this selection, please see the detailed results.")
        else:
            st.info("Summary table is unavailable for this selection, please see the detailed results.")

        st.markdown(
            "Source: [2024 Public Service Employee Survey Results - Open Government Portal]"
            "(https://open.canada.ca/data/en/dataset/7f625e97-9d02-4c12-a756-1ddebb50e69f)"
        )

    with tab_detail:
        st.dataframe(make_arrow_safe(df_disp), use_container_width=True)
