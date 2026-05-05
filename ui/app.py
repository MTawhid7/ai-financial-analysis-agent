"""Streamlit UI — real-time Thought/Action/Observation stream + dry-run replay.

Run with:
    streamlit run ui/app.py
    streamlit run ui/app.py -- --dry-run path/to/run_trace.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on the path when run directly.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="AI Financial Analyst Agent",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Financial Analyst Agent")
    st.caption("ReAct + Multi-Agent | LangGraph | Gemini Free Tier")
    st.divider()

    dry_run_enabled = st.checkbox("Dry-run mode (replay trace)", value=False)
    trace_file = None
    if dry_run_enabled:
        uploaded = st.file_uploader("Upload run_trace.json", type="json")
        if uploaded:
            trace_file = json.load(uploaded)

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- LangGraph pipeline\n"
        "- Gemini 2.0 Flash (primary)\n"
        "- Gemini 2.0 Flash-Lite (sub-tasks)\n"
        "- yfinance + DuckDuckGo (free tier)\n"
        "- LangSmith tracing\n"
    )

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------

st.title("AI Financial Analyst Agent")
st.caption(
    "Enter one or more stock tickers to receive an autonomous, multi-page "
    "research report. All data is fetched from free-tier APIs."
)

col_input, col_info = st.columns([3, 1])

with col_input:
    ticker_input = st.text_input(
        "Tickers (comma-separated)",
        placeholder="AAPL, NVDA",
        help="Enter up to 3 tickers for best performance on the free tier.",
    )

with col_info:
    st.metric("Free tier budget", "1,500 req/day")

run_button = st.button("Run Analysis", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Transparency panel (shown during and after runs)
# ------------------------------------------------------------------

transparency_container = st.container()

with transparency_container:
    budget_col, cache_col, error_col, status_col = st.columns(4)
    budget_placeholder = budget_col.empty()
    cache_placeholder = cache_col.empty()
    error_placeholder = error_col.empty()
    status_placeholder = status_col.empty()


def _render_transparency(budget_stats: dict, errors: list, status: str) -> None:
    budget_placeholder.metric(
        "API Calls",
        f"{budget_stats.get('total_calls', 0)} / {budget_stats.get('daily_budget', 1500)}",
        delta=f"{budget_stats.get('budget_used_pct', 0):.1f}%",
    )
    cache_placeholder.metric("Cache Hits", budget_stats.get("cache_hits", 0))
    error_placeholder.metric("Errors", len(errors), delta_color="inverse")
    status_placeholder.metric("Status", status)


# ------------------------------------------------------------------
# TAO stream area
# ------------------------------------------------------------------

tao_container = st.expander("Agent Thought / Action / Observation Stream", expanded=True)

# ------------------------------------------------------------------
# Report area
# ------------------------------------------------------------------

report_placeholder = st.empty()

# ------------------------------------------------------------------
# Dry-run replay
# ------------------------------------------------------------------


def _replay_trace(trace: dict) -> None:
    """Replay a run_trace.json without making any API calls."""
    st.info("Replaying from saved trace — no API calls made.")
    tool_calls = trace.get("tool_calls", [])
    errors = trace.get("errors", [])
    budget_stats = trace.get("budget_stats", {})
    status = trace.get("status", "COMPLETE")

    with tao_container:
        for entry in tool_calls:
            step = entry.get("step", "?")
            agent = entry.get("agent", "?")
            tool = entry.get("tool", "?")
            cache = " (cached)" if entry.get("cache_hit") else ""
            st.markdown(f"**[STEP {step}]** `{agent}` → `{tool}`{cache}")
            with st.expander(f"Input / Output — step {step}", expanded=False):
                st.json({"input": entry.get("input"), "output": entry.get("output")})

    _render_transparency(budget_stats, errors, status)
    st.success("Dry-run replay complete.")


# ------------------------------------------------------------------
# Live run
# ------------------------------------------------------------------


def _run_analysis(tickers: list[str]) -> None:
    from ai_financial_analyst.agents.orchestrator import run_pipeline

    tao_lines: list[str] = []

    with tao_container:
        tao_output = st.empty()

    def _append_tao(line: str) -> None:
        tao_lines.append(line)
        with tao_container:
            tao_output.markdown("\n\n".join(tao_lines[-30:]))

    _append_tao(f"**Starting analysis for: {', '.join(tickers)}**")
    _render_transparency({}, [], "RUNNING")

    try:
        final_state, trace_path = asyncio.run(
            run_pipeline(
                query=f"Analyse {', '.join(tickers)}",
                tickers=tickers,
                dry_run=False,
                trace_output_dir=".",
            )
        )
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        return

    # Render TAO log
    for entry in final_state.get("iteration_log", []):
        step = entry.get("step", "?")
        agent = entry.get("agent", "?")
        tool = entry.get("tool", "?")
        cache = " (cached)" if entry.get("cache_hit") else ""
        _append_tao(f"**[STEP {step}]** `{agent}` → `{tool}`{cache}")

    # Transparency panel
    errors = final_state.get("errors", [])
    status = final_state.get("status", "COMPLETE")
    # Budget stats available from trace file
    try:
        trace_data = json.loads(Path(trace_path).read_text())
        budget_stats = trace_data.get("budget_stats", {})
    except Exception:
        budget_stats = {}

    _render_transparency(budget_stats, errors, status)

    # Render report
    report = final_state.get("report_markdown", "")
    if report:
        report_placeholder.markdown(report)
        st.download_button(
            "Download Report (Markdown)",
            data=report,
            file_name=f"report_{'_'.join(tickers)}.md",
            mime="text/markdown",
        )
    else:
        st.warning("No report generated — check the error log above.")

    # Download trace
    if trace_path and Path(trace_path).exists():
        st.download_button(
            "Download run_trace.json",
            data=Path(trace_path).read_text(),
            file_name=Path(trace_path).name,
            mime="application/json",
        )

    if errors:
        with st.expander("Error Details", expanded=False):
            for err in errors:
                st.json(err)


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

if dry_run_enabled and trace_file:
    _replay_trace(trace_file)
elif run_button:
    if not ticker_input.strip():
        st.warning("Please enter at least one ticker symbol.")
    else:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if len(tickers) > 3:
            st.warning("For free-tier rate limits, we recommend ≤ 3 tickers. Proceeding anyway.")
        _run_analysis(tickers)
