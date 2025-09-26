# utils/menu1_helpers.py
# -------------------------------------------------------------------------
# Extracted, no-logic-change helpers for Menu 1 to keep menu1/main.py short.
# -------------------------------------------------------------------------

from __future__ import annotations
import pandas as pd

# ─────────────────────────────
# Helpers (identical behavior)
# ─────────────────────────────
def resolve_demographic_codes(demo_df: pd.DataFrame, category_label: str | None, subgroup_label: str | None):
    DEMO_CAT_COL = "DEMCODE Category"
    LABEL_COL = "DESCRIP_E"

    code_col = None
    for c in ["DEMCODE", "DemCode", "CODE", "Code", "CODE_E", "Demographic code"]:
        if c in demo_df.columns:
            code_col = c
            break

    # Overall
    if not category_label or category_label == "All respondents":
        return [None], {None: "All respondents"}, False

    # Category present
    df_cat = demo_df[demo_df[DEMO_CAT_COL] == category_label] if DEMO_CAT_COL in demo_df.columns else demo_df.copy()
    if df_cat.empty:
        return [None], {None: "All respondents"}, False

    # Subgroup selected
    if subgroup_label:
        if code_col and LABEL_COL in df_cat.columns:
            row = df_cat[df_cat[LABEL_COL] == subgroup_label]
            if not row.empty:
                code = str(row.iloc[0][code_col])
                return [code], {code: subgroup_label}, True
        return [subgroup_label], {subgroup_label: subgroup_label}, True

    # No subgroup -> all codes in category
    if code_col and LABEL_COL in df_cat.columns:
        codes = df_cat[code_col].astype(str).tolist()
        labels = df_cat[LABEL_COL].astype(str).tolist()
        keep = [(c, l) for c, l in zip(codes, labels) if str(c).strip() != ""]
        codes = [c for c, _ in keep]
        disp_map = {c: l for c, l in keep}
        return codes, disp_map, True

    # Fallback: use labels as identifiers
    if LABEL_COL in df_cat.columns:
        labels = df_cat[LABEL_COL].astype(str).tolist()
        disp_map = {l: l for l in labels}
        return labels, disp_map, True

    return [None], {None: "All respondents"}, False


def get_scale_labels(scales_df: pd.DataFrame, question_code: str):
    sdf = scales_df.copy()
    candidates = pd.DataFrame()
    for key in ["code", "question"]:
        if key in sdf.columns:
            candidates = sdf[sdf[key].astype(str).str.upper() == str(question_code).upper()]
            if not candidates.empty:
                break
    labels = []
    for i in range(1, 7 + 1):
        col = f"answer{i}"
        lbl = None
        if not candidates.empty and col in candidates.columns:
            vals = candidates[col].dropna().astype(str)
            if not vals.empty:
                lbl = vals.iloc[0].strip()
        if not lbl:
            lbl = f"Answer {i}"
        labels.append((col, lbl))
    return labels


def drop_na_999(df: pd.DataFrame) -> pd.DataFrame:
    # Support both raw and normalized names
    pos_col = "positive_pct" if "positive_pct" in df.columns else "POSITIVE" if "POSITIVE" in df.columns else None
    neu_col = "neutral_pct" if "neutral_pct" in df.columns else "NEUTRAL" if "NEUTRAL" in df.columns else None
    neg_col = "negative_pct" if "negative_pct" in df.columns else "NEGATIVE" if "NEGATIVE" in df.columns else None
    n_col   = "n" if "n" in df.columns else "ANSCOUNT" if "ANSCOUNT" in df.columns else None

    cols_check = [c for c in [f"answer{i}" for i in range(1, 8)] if c in df.columns]
    cols_check += [c for c in [pos_col, neu_col, neg_col, n_col] if c]

    if not cols_check:
        return df

    mask_keep = pd.Series(True, index=df.index)
    for c in cols_check:
        vals = pd.to_numeric(df[c], errors="coerce")
        mask_keep &= (vals != 999)
    return df.loc[mask_keep].copy()


def normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have these canonical columns (creating/renaming as needed):
      question_code, year, group_value, positive_pct, neutral_pct, negative_pct, n
    """
    out = df.copy()

    # QUESTION -> question_code
    if "question_code" not in out.columns:
        if "QUESTION" in out.columns:
            out = out.rename(columns={"QUESTION": "question_code"})
        else:
            for c in out.columns:
                if c.strip().lower() == "question":
                    out = out.rename(columns={c: "question_code"})
                    break

    # SURVEYR -> year
    if "year" not in out.columns:
        if "SURVEYR" in out.columns:
            out = out.rename(columns={"SURVEYR": "year"})
        else:
            for c in out.columns:
                if c.strip().lower() in ("surveyr", "year"):
                    out = out.rename(columns={c: "year"})
                    break

    # DEMCODE -> group_value
    if "group_value" not in out.columns:
        if "DEMCODE" in out.columns:
            out = out.rename(columns={"DEMCODE": "group_value"})
        else:
            for c in out.columns:
                if c.strip().lower() == "demcode":
                    out = out.rename(columns={c: "group_value"})
                    break

    # POSITIVE/NEUTRAL/NEGATIVE -> *_pct
    if "positive_pct" not in out.columns and "POSITIVE" in out.columns:
        out = out.rename(columns={"POSITIVE": "positive_pct"})
    if "neutral_pct" not in out.columns and "NEUTRAL" in out.columns:
        out = out.rename(columns={"NEUTRAL": "neutral_pct"})
    if "negative_pct" not in out.columns and "NEGATIVE" in out.columns:
        out = out.rename(columns={"NEGATIVE": "negative_pct"})

    # ANSCOUNT -> n
    if "n" not in out.columns and "ANSCOUNT" in out.columns:
        out = out.rename(columns={"ANSCOUNT": "n"})

    return out


def format_table_for_display(
    df_slice: pd.DataFrame,
    dem_disp_map: dict,
    category_in_play: bool,
    scale_pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    if df_slice.empty:
        return df_slice
    out = df_slice.copy()

    # Numeric year for sorting, string year for display
    out["YearNum"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["Year"] = out["YearNum"].astype(str)

    if category_in_play:
        def lbl(code):
            if code is None or (isinstance(code, float) and pd.isna(code)) or str(code).strip() == "":
                return "All respondents"
            return dem_disp_map.get(code, dem_disp_map.get(str(code), str(code)))
        out["Demographic"] = out["group_value"].apply(lbl)

    dist_cols = [k for k, _ in scale_pairs if k in out.columns]
    rename_map = {k: v for k, v in scale_pairs if k in out.columns}
    keep_cols = ["YearNum", "Year"] + (["Demographic"] if category_in_play else []) \
                + dist_cols + ["positive_pct", "neutral_pct", "negative_pct", "n"]
    out = out[keep_cols].copy()
    out = out.rename(columns=rename_map)
    out = out.rename(columns={"positive_pct": "Positive", "neutral_pct": "Neutral", "negative_pct": "Negative", "n": "n"})

    # Sort by Year desc, then Demographic asc
    sort_cols = ["YearNum"] + (["Demographic"] if category_in_play else [])
    sort_asc  = [False] + ([True] if category_in_play else [])
    out = out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

    # Drop helper
    out = out.drop(columns=["YearNum"])

    # Numeric formatting
    for c in out.columns:
        if c not in ("Year", "Demographic"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    pct_like = [c for c in out.columns if c not in ("Year", "Demographic", "n")]
    out[pct_like] = out[pct_like].round(1)
    if "n" in out.columns:
        out["n"] = out["n"].astype("Int64")

    return out


def build_positive_only_narrative(df_disp: pd.DataFrame, category_in_play: bool) -> str:
    if df_disp.empty or "Positive" not in df_disp.columns:
        return "No results available to summarize."

    t = df_disp.copy()
    t["_Y"] = pd.to_numeric(t["Year"], errors="coerce")
    latest_year = int(t["_Y"].max())
    df_latest = t[t["_Y"] == latest_year]

    lines = []

    # Across demographics
    if category_in_play and "Demographic" in t.columns:
        groups = df_latest.dropna(subset=["Positive"]).sort_values("Positive", ascending=False)
        if len(groups) >= 2:
            top = groups.iloc[0]; bot = groups.iloc[-1]
            lines.append(
                f"In {latest_year}, {top['Demographic']} is highest on Positive ({top['Positive']:.1f}%), "
                f"while {bot['Demographic']} is lowest ({bot['Positive']:.1f}%)."
            )
        elif len(groups) == 1:
            g = groups.iloc[0]
            lines.append(f"In {latest_year}, {g['Demographic']} has Positive at {g['Positive']:.1f}%.")

    # Over time (latest vs previous)
    def prev_year(subdf: pd.DataFrame):
        ys = sorted(pd.to_numeric(subdf["Year"], errors="coerce").dropna().unique().tolist())
        return int(ys[-2]) if len(ys) >= 2 else None

    if category_in_play and "Demographic" in t.columns:
        top3 = df_latest.sort_values("Positive", ascending=False).head(3)["Demographic"].tolist()
        for g in top3:
            s = t[t["Demographic"] == g]
            prev = prev_year(s)
            if prev is not None:
                latest_pos = s[s["Year"] == str(latest_year)]["Positive"].dropna()
                prev_pos   = s[s["Year"] == str(prev)]["Positive"].dropna()
                if not latest_pos.empty and not prev_pos.empty:
                    delta = latest_pos.iloc[0] - prev_pos.iloc[0]
                    lines.append(f"{g}: {latest_year} {latest_pos.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")
    else:
        prev = prev_year(t)
        if prev is not None:
            latest_pos = df_latest["Positive"].dropna()
            prev_pos   = t[t["Year"] == str(prev)]["Positive"].dropna()
            if not latest_pos.empty and not prev_pos.empty:
                delta = latest_pos.iloc[0] - prev_pos.iloc[0]
                lines.append(f"Overall: {latest_year} {latest_pos.iloc[0]:.1f}% ({delta:+.1f} pts vs {prev}).")

    if not lines:
        return "No notable changes to report based on Positive."
    return " ".join(lines)
