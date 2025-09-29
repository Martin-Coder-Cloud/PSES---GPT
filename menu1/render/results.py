# menu1/render/results.py
"""
Results rendering for Menu 1:
- Summary table tab (rows: question code only, or code×demographic if applicable)
- One tab per question with the detailed distribution table
- Source caption appears directly under each table (before AI narratives)
- Selected questions & metrics legend (code → text · metric) under the Summary table (small font)
- AI narratives:
    • On Summary tab:
        - Per-question "Summary Analysis" for every question shown
        - If >1 questions selected → add an "Overall Summary Analysis"
    • On each per-question tab:
        - Per-question "Summary Analysis" (reuses the Summary tab narrative if available)
- Excel export (Summary + each question + Context)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import io
from datetime import datetime

import pandas as pd
import streamlit as st

from ..ai import AI_SYSTEM_PROMPT, extract_narrative
from ..constants import DEFAULT_OPENAI_MODEL


# -----------------------------------------------------------------------------
# Source caption
# -----------------------------------------------------------------------------
def source_caption(*, source_url: str, source_title: str) -> None:
    """Show only a clickable title (no raw URL), placed directly under a table."""
    st.caption(f"Source: [{source_title}]({source_url})")


# -----------------------------------------------------------------------------
# Main tabs renderer
# -----------------------------------------------------------------------------
def tabs_summary_and_per_q(
    *,
    payload: Dict[str, Any],
    ai_on: bool,
    build_overall_prompt,
    build_per_q_prompt,
    call_openai_json,
    source_url: str,
    source_title: str,
) -> None:
    """
    Render Summary + per-question tabs from stashed payload.
    Required payload keys:
      - tab_labels: List[str]               # question codes in order
      - pivot: pd.DataFrame                 # summary table (index=question code or question×demo; columns=years)
      - per_q_disp: Dict[str, pd.DataFrame] # display tables per question
      - per_q_metric_col: Dict[str, str]
      - per_q_metric_label: Dict[str, str]
      - code_to_text: Dict[str, str]
      - years: List[int]
      - demo_selection: str
      - sub_selection: Optional[str]
    """
    tab_labels: List[str] = payload.get("tab_labels", [])
    pivot: pd.DataFrame = payload.get("pivot")
    per_q_disp: Dict[str, pd.DataFrame] = payload.get("per_q_disp", {})
    per_q_metric_col: Dict[str, str] = payload.get("per_q_metric_col", {})
    per_q_metric_label: Dict[str, str] = payload.get("per_q_metric_label", {})
    code_to_text: Dict[str, str] = payload.get("code_to_text", {})
    years: List[int] = payload.get("years", [])
    demo_selection: Optional[str] = payload.get("demo_selection")
    sub_selection: Optional[str] = payload.get("sub_selection")

    if pivot is None or not isinstance(pivot, pd.DataFrame):
        st.info("No data found for your selection.")
        return

    # Build tabs (Summary first, then each question code)
    tabs = st.tabs(["Summary table"] + tab_labels)

    # Cache to avoid duplicate LLM calls across tabs
    ai_summaries: Dict[str, str] = {}

    # -----------------------
    # Summary tab
    # -----------------------
    with tabs[0]:
        st.markdown("### Summary table")
        st.dataframe(pivot.round(1).reset_index(), use_container_width=True)
        # Source directly under the table (before AI)
        source_caption(source_url=source_url, source_title=source_title)

        # Legend: code → text · metric (SMALL FONT, compact)
        if tab_labels:
            items_html = "".join(
                f"<li><strong>{q}</strong> — {code_to_text.get(q, '')} "
                f"<span style='opacity:0.8'>· {per_q_metric_label.get(q, '% positive')}</span></li>"
                for q in tab_labels
            )
            st.markdown(
                f"""
                <div style="font-size:13px;color:#444;margin-top:6px;margin-bottom:8px;">
                  <div style="font-weight:600;margin-bottom:4px;">Selected questions &amp; metrics</div>
                  <ul style="margin:0 0 0 18px;padding:0;list-style:disc;">
                    {items_html}
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- AI Summary section (per question, then optional overall) ---
        if ai_on and tab_labels:
            st.markdown("### AI Summary")

            # Per-question narratives (for each question shown in the Summary table)
            category_in_play = (demo_selection is not None) and (demo_selection != "All respondents")
            for q in tab_labels:
                df_disp = per_q_disp.get(q)
                if df_disp is None or df_disp.empty:
                    continue

                metric_col = per_q_metric_col.get(q, "Positive")
                metric_label = per_q_metric_label.get(q, "% positive")
                q_text = code_to_text.get(q, "")

                with st.spinner(f"Generating Summary Analysis for {q}…"):
                    user_payload = build_per_q_prompt(
                        question_code=q,
                        question_text=q_text,
                        df_disp=df_disp,
                        metric_col=metric_col,
                        metric_label=metric_label,
                        category_in_play=category_in_play,
                    )
                    content, hint = call_openai_json(
                        system=AI_SYSTEM_PROMPT,
                        user=user_payload,
                        model=DEFAULT_OPENAI_MODEL,
                        temperature=0.2,
                        max_retries=2,
                    )
                narrative = extract_narrative(content)
                if narrative:
                    ai_summaries[q] = narrative
                    st.markdown(f"**{q} — {q_text}**")
                    st.write(narrative)
                    st.caption(f"Generated by OpenAI • model: {DEFAULT_OPENAI_MODEL}")
                else:
                    st.caption(f"{q}: AI unavailable or returned no narrative ({hint or 'no content'}).")

            # Overall narrative only when multiple questions are selected
            if len(tab_labels) > 1:
                with st.spinner("Generating Overall Summary Analysis…"):
                    q_to_metric = {q: per_q_metric_label.get(q, "% positive") for q in tab_labels}
                    user_payload = build_overall_prompt(
                        tab_labels=tab_labels,
                        pivot_df=pivot,
                        q_to_metric=q_to_metric,
                    )
                    content, hint = call_openai_json(
                        system=AI_SYSTEM_PROMPT,
                        user=user_payload,
                        model=DEFAULT_OPENAI_MODEL,
                        temperature=0.2,
                        max_retries=2,
                    )
                narrative = extract_narrative(content)
                if narrative:
                    st.markdown("**Overall Summary Analysis**")
                    st.write(narrative)
                    st.caption(f"Generated by OpenAI • model: {DEFAULT_OPENAI_MODEL}")
                else:
                    st.caption(f"Overall summary: AI unavailable or returned no narrative ({hint or 'no content'}).")

    # -----------------------
    # Per-question tabs
    # -----------------------
    category_in_play = (demo_selection is not None) and (demo_selection != "All respondents")

    for idx, qcode in enumerate(tab_labels, start=1):
        with tabs[idx]:
            qtext = code_to_text.get(qcode, "")
            st.subheader(f"{qcode} — {qtext}")

            df_disp = per_q_disp.get(qcode)
            if df_disp is None or df_disp.empty:
                st.info("No data for this question.")
                continue

            st.dataframe(df_disp, use_container_width=True)
            # Source directly under the table (before AI)
            source_caption(source_url=source_url, source_title=source_title)

            if ai_on:
                # Reuse narrative computed on the Summary tab if available
                if qcode in ai_summaries:
                    st.markdown("### Summary Analysis")
                    st.write(ai_summaries[qcode])
                    st.caption(f"Generated by OpenAI • model: {DEFAULT_OPENAI_MODEL}")
                else:
                    with st.spinner("Generating Summary Analysis…"):
                        metric_col = per_q_metric_col.get(qcode, "Positive")
                        metric_label = per_q_metric_label.get(qcode, "% positive")
                        user_payload = build_per_q_prompt(
                            question_code=qcode,
                            question_text=qtext,
                            df_disp=df_disp,
                            metric_col=metric_col,
                            metric_label=metric_label,
                            category_in_play=category_in_play
                        )
                        content, hint = call_openai_json(
                            system=AI_SYSTEM_PROMPT,
                            user=user_payload,
                            model=DEFAULT_OPENAI_MODEL,
                            temperature=0.2,
                            max_retries=2,
                        )
                    narrative = extract_narrative(content)
                    if narrative:
                        st.markdown("### Summary Analysis")
                        st.write(narrative)
                        st.caption(f"Generated by OpenAI • model: {DEFAULT_OPENAI_MODEL}")
                    else:
                        st.caption(f"AI unavailable or returned no narrative ({hint or 'no content'}).")
            else:
                st.info("No AI summary generated.")

    # -----------------------
    # Excel export
    # -----------------------
    _download_excel_section(
        per_q_disp=per_q_disp,
        pivot=pivot,
        tab_labels=tab_labels,
        years=years,
        demo_selection=demo_selection,
        sub_selection=sub_selection,
    )


# -----------------------------------------------------------------------------
# Excel download (Summary + each Q + Context)
# -----------------------------------------------------------------------------
def _download_excel_section(
    *,
    per_q_disp: Dict[str, pd.DataFrame],
    pivot: pd.DataFrame,
    tab_labels: List[str],
    years: List[int],
    demo_selection: Optional[str],
    sub_selection: Optional[str],
) -> None:
    """Create and expose an Excel file containing the summary and each question tab."""
    if (not tab_labels) or (pivot is None):
        return

    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            # Summary sheet
            pivot.round(1).reset_index().to_excel(writer, sheet_name="Summary_Table", index=False)
            # One sheet per question (use a short, safe name)
            for q, df_disp in per_q_disp.items():
                safe = q[:28] or "Q"
                # avoid boolean evaluation of DataFrame
                df_to_write = df_disp if isinstance(df_disp, pd.DataFrame) else pd.DataFrame()
                df_to_write.to_excel(writer, sheet_name=f"{safe}", index=False)
            # Context sheet
            ctx = {
                "Questions": ", ".join(tab_labels),
                "Years": ", ".join(map(str, years or [])),
                "Category": demo_selection or "All respondents",
                "Subgroup": (
                    sub_selection or "(all in category)"
                    if (demo_selection and demo_selection != "All respondents")
                    else "All respondents"
                ),
                "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            pd.DataFrame(list(ctx.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Context", index=False)
        data = buf.getvalue()

    st.download_button(
        label="Download Excel (Summary + all tabs)",
        data=data,
        file_name=f"PSES_multiQ_{'-'.join(map(str, years or []))}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
