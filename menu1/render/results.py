# menu1/results.py  — Phase 1 standardized headings & indentation
from __future__ import annotations
from typing import Dict, Callable, Any, Tuple, List, Set, Optional
import io, json, hashlib, re
import pandas as pd, streamlit as st
from ..ai import AI_SYSTEM_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
# helpers (unchanged from previous working build)
# ─────────────────────────────────────────────────────────────────────────────
def _hash_key(obj: Any) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            payload = obj.to_csv(index=True, na_rep="")
        else:
            payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _ai_cache_get(k): return st.session_state.get("menu1_ai_cache", {}).get(k)
def _ai_cache_put(k,v):
    c=st.session_state.get("menu1_ai_cache",{}); c[k]=v; st.session_state["menu1_ai_cache"]=c
def _source_link_line(title,url):
    st.markdown(
        f"<div style='margin-top:6px;font-size:0.9rem;'>Source: "
        f"<a href='{url}' target='_blank'>{title}</a></div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
#  AI-fact-check computation (identical logic)
# ─────────────────────────────────────────────────────────────────────────────
_INT_RE=re.compile(r"-?\d+")
def _is_year_like(n:int)->bool: return 1900<=n<=2100
def _safe_int(x): 
    try:
        if x in (None,"","na","n/a","none","nan","null",9999): return None
        v=pd.to_numeric(x,errors="coerce")
        return None if pd.isna(v) else int(round(float(v)))
    except: return None
def _allowed_numbers_from_disp(df,metric):
    if df.empty:return set(),set()
    if "Year" not in df.columns or metric not in df.columns:
        ycols=[c for c in df.columns if _is_year_like(int(str(c)[:4]))]
        if not ycols:return set(),set()
        idc=[c for c in df.columns if c not in ycols]
        df=df.melt(id_vars=idc,value_vars=ycols,var_name="Year",value_name="val")
        metric="val"
    df["val"]=df[metric].apply(_safe_int)
    yrs={_safe_int(y) for y in pd.to_numeric(df["Year"],errors="coerce") if _safe_int(y)}
    vals={v for v in df["val"] if v is not None}
    return vals,yrs
def _extract_ints(text):
    sents=re.split(r'(?<=[.!?])\s+',text.strip())
    out=[]
    for s in sents:
        for m in _INT_RE.findall(s):
            try:
                n=int(m)
                if not _is_year_like(n): out.append((n,s))
            except:...
    return out
def _validate(text,allowed):
    if not text:return True,[]
    bad=[]
    for n,s in _extract_ints(text):
        if n not in allowed: bad.append(f"{n} — {s}")
    return len(bad)==0,bad[:5]
def _compute_factcheck_results(tab_labels,per_q_disp,per_q_metric_col,per_q_narratives):
    any_issue=False;details=[]
    for q in tab_labels:
        df=per_q_disp.get(q)
        if df is None or df.empty:
            details.append(("caption",f"{q}: fact check skipped (no table)."));continue
        metric=per_q_metric_col.get(q) or "AGREE"
        allowed,_=_allowed_numbers_from_disp(df.copy(),metric)
        ok,bad=_validate(per_q_narratives.get(q,""),allowed)
        if not ok:
            any_issue=True
            nums=", ".join(re.findall(r"\d+", " ".join(bad))) or ""
            details.append(("warning",f"{q}: potential mismatches detected ({nums})."))
        else:
            details.append(("caption",f"{q}: no numeric inconsistencies detected."))
    return any_issue,details

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def tabs_summary_and_per_q(*,payload,ai_on,build_overall_prompt,build_per_q_prompt,
                           call_openai_json,source_url,source_title):

    per_q_disp=payload["per_q_disp"]; per_q_metric_col=payload["per_q_metric_col"]
    per_q_metric_label=payload["per_q_metric_label"]; pivot=payload["pivot"]
    tab_labels=payload["tab_labels"]; years=payload["years"]
    demo_sel=payload["demo_selection"]; code_to_text=payload["code_to_text"]

    sig={"tab_labels":tab_labels,"years":years,"demo_selection":demo_sel,
         "pivot_sig":_hash_key(pivot)}
    key="menu1_ai_"+_hash_key(sig)

    # ─────────────── Results  (Title 2 / H2 no indent) ───────────────
    st.markdown("## Results")
    tab_titles=["Summary table"]+tab_labels+["Technical notes"]
    tabs=st.tabs(tab_titles)

    # Summary table
    with tabs[0]:
        st.markdown("<div style='margin-left:8%;'>",unsafe_allow_html=True)
        st.markdown("### Summary table")
        st.dataframe(pivot.reset_index(),use_container_width=True)
        _source_link_line(source_title,source_url)
        st.markdown("</div>",unsafe_allow_html=True)

    # Per-question tabs
    for i,q in enumerate(tab_labels,1):
        with tabs[i]:
            st.markdown("<div style='margin-left:8%;'>",unsafe_allow_html=True)
            st.markdown(f"### {q} — {code_to_text.get(q,'')}")
            st.dataframe(per_q_disp[q],use_container_width=True)
            _source_link_line(source_title,source_url)
            st.markdown("</div>",unsafe_allow_html=True)

    # Technical notes
    with tabs[-1]:
        st.markdown("<div style='margin-left:8%;'>",unsafe_allow_html=True)
        st.markdown("### Technical notes")
        st.markdown(
            """
1. **Summary results** reflect positive answers.  
2. **Weights/adjustment:** Results are weighted for non-response.  
3. **Rounding:** Percentages may not sum to 100.  
4. **Suppression:** Small cells (<10) suppressed.
            """
        )
        st.markdown("</div>",unsafe_allow_html=True)

    # ─────────────── AI Summary  (Title 2 / H2 no indent) ───────────────
    if not ai_on: return
    st.markdown("## AI Summary")

    # Load or compute narratives (logic unchanged)
    per_q_narr=st.session_state.get("menu1_ai_narr_per_q",{}) or {}
    overall=st.session_state.get("menu1_ai_narr_overall")
    if not per_q_narr:
        cached=_ai_cache_get(key)
        if cached:
            per_q_narr=cached.get("per_q",{}); overall=cached.get("overall")

    if per_q_narr:
        for q,txt in per_q_narr.items():
            if txt:
                st.markdown(f"**{q} — {code_to_text.get(q,'')}**")
                st.write(txt)
        if overall and len(tab_labels)>1:
            st.markdown("**Overall**"); st.write(overall)

    # ─────────────── AI Fact Check (Title 3 / H3 indented) ───────────────
    any_issue,details=_compute_factcheck_results(
        tab_labels,per_q_disp,per_q_metric_col,per_q_narr)

    st.markdown("<div style='margin-left:8%;'>",unsafe_allow_html=True)
    st.markdown("### AI Fact Check")
    if not any_issue:
        st.markdown("✅ **The data points in the summaries were validated and correspond to the data provided.**")
    else:
        st.markdown("❌ **Issues detected in one or more statements. Review details below.**")
    with st.expander("Details",expanded=False):
        for lvl,msg in details:
            (st.warning if lvl=="warning" else st.caption)(msg)
    st.markdown("</div>",unsafe_allow_html=True)
