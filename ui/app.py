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

from ai_financial_analyst.core.log_capture import capture_logs

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

    debug_mode = st.checkbox("Debug mode", value=False,
                              help="Show intermediate agent outputs and pipeline logs after each run.")

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- LangGraph pipeline\n"
        "- Gemini 3 Flash (primary)\n"
        "- Gemini 3.1 Flash-Lite (sub-tasks)\n"
        "- yfinance + Tavily (free tier)\n"
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

    # Live step callback — called by each agent immediately after each tool call.
    def _on_step(event: dict) -> None:
        step = event.get("step", "?")
        agent = event.get("agent", "?")
        tool = event.get("tool", "?")
        cache = " *(cached)*" if event.get("cache_hit") else ""
        status_icon = "✓" if event.get("ok", True) else "✗"
        _append_tao(f"{status_icon} **[STEP {step}]** `{agent}` → `{tool}`{cache}")

    _append_tao(f"**Starting analysis for: {', '.join(tickers)}**")
    _render_transparency({}, [], "RUNNING")

    try:
        with capture_logs() as log_handler:
            final_state, trace_path, artifacts_path = asyncio.run(
                run_pipeline(
                    query=f"Analyse {', '.join(tickers)}",
                    tickers=tickers,
                    dry_run=False,
                    step_callback=_on_step,
                )
            )
        captured_logs = log_handler.records
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        import traceback
        st.code(traceback.format_exc(), language="text")
        return

    _append_tao("**Pipeline complete.**")

    # Transparency panel
    errors = final_state.get("errors", [])
    status = final_state.get("status", "COMPLETE")
    trace_data: dict = {}
    artifacts_data: dict = {}
    try:
        trace_data = json.loads(Path(trace_path).read_text())
        budget_stats = trace_data.get("budget_stats", {})
    except Exception:
        budget_stats = {}
    try:
        artifacts_data = json.loads(Path(artifacts_path).read_text())
    except Exception:
        pass

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

    # Download trace + artifacts
    dl_col1, dl_col2 = st.columns(2)
    if trace_path and Path(trace_path).exists():
        dl_col1.download_button(
            "Download run_trace.json",
            data=Path(trace_path).read_text(),
            file_name=Path(trace_path).name,
            mime="application/json",
        )
    if artifacts_path and Path(artifacts_path).exists():
        dl_col2.download_button(
            "Download run_artifacts.json (Full API & LLM Responses)",
            data=Path(artifacts_path).read_text(),
            file_name=Path(artifacts_path).name,
            mime="application/json",
        )

    if errors:
        with st.expander(f"⚠️ Errors ({len(errors)})", expanded=True):
            for err in errors:
                st.json(err)

    # Debug mode: intermediate state inspection
    if debug_mode:
        st.divider()
        st.subheader("Debug: Pipeline Inspection")

        agent_events = trace_data.get("agent_events", [])
        if agent_events:
            with st.expander("Agent Events Timeline", expanded=True):
                for evt in agent_events:
                    icon = "▶" if evt["event"] == "agent_start" else "✓"
                    st.markdown(
                        f"`{evt['timestamp']}` **{icon} {evt['agent']}** "
                        f"— {evt['event'].replace('_', ' ')}"
                    )
                    summary = evt.get("input_summary") or evt.get("output_summary", {})
                    if summary:
                        st.json(summary)

        raw_data = final_state.get("raw_data", {})
        if raw_data:
            with st.expander("Researcher Output — raw_data keys per ticker", expanded=False):
                for ticker, data in raw_data.items():
                    st.markdown(f"**{ticker}**: {list(data.keys())}")
                    for dtype, dval in data.items():
                        st.markdown(f"*{dtype}*")
                        if isinstance(dval, dict):
                            st.json(dval)
                        else:
                            st.text(str(dval)[:500])

        analysis = final_state.get("analysis", {})
        if analysis:
            with st.expander("Quant Analyst Output — analysis per ticker", expanded=False):
                st.json(analysis)

        if captured_logs:
            with st.expander(f"Pipeline Logs ({len(captured_logs)} entries)", expanded=False):
                level_colors = {
                    "DEBUG": "gray", "INFO": "green",
                    "WARNING": "orange", "ERROR": "red", "CRITICAL": "red"
                }
                for rec in captured_logs:
                    color = level_colors.get(rec["level"], "gray")
                    st.markdown(
                        f"<span style='color:{color};font-size:0.85em'>"
                        f"[{rec['time']}] **{rec['level']}** {rec['message']}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

        # ------------------------------------------------------------------
        # Full API & LLM response inspection (from run_artifacts.json)
        # ------------------------------------------------------------------

        tool_responses = artifacts_data.get("tool_responses", [])
        if tool_responses:
            with st.expander(
                f"Full API Responses ({len(tool_responses)} tool calls)", expanded=False
            ):
                for resp in tool_responses:
                    label = (
                        f"Step {resp.get('step')} — "
                        f"`{resp.get('agent')}` → `{resp.get('tool')}`"
                        + (" *(cached)*" if resp.get("cache_hit") else "")
                    )
                    with st.expander(label, expanded=False):
                        st.markdown("**Input**")
                        st.json(resp.get("input", {}))
                        st.markdown("**Full Output**")
                        parsed = resp.get("output_parsed")
                        if parsed is not None:
                            st.json(parsed)
                        else:
                            st.text(resp.get("full_output_str", "")[:3000])

        llm_exchanges = artifacts_data.get("llm_exchanges", [])
        if llm_exchanges:
            with st.expander(
                f"Full LLM Exchanges ({len(llm_exchanges)} calls)", expanded=False
            ):
                for i, ex in enumerate(llm_exchanges):
                    ticker_label = f" [{ex.get('ticker')}]" if ex.get("ticker") else ""
                    label = f"{ex.get('agent')} — {ex.get('purpose')}{ticker_label}"
                    with st.expander(label, expanded=False):
                        for msg in ex.get("prompt_messages", []):
                            role = msg.get("role", "?")
                            st.markdown(f"**Prompt ({role})**")
                            st.text(str(msg.get("content", ""))[:4000])
                        st.markdown("**Raw LLM Response**")
                        st.text(ex.get("raw_response", "")[:6000])


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
