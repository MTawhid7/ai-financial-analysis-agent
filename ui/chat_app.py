"""Conversational chat UI — Phase 1 + Phase 2 (memory system).

Run with:
    streamlit run ui/chat_app.py

Keeps the original ui/app.py intact for dry-run replay and classic mode.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when run directly.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Page config — must be first Streamlit call
# ------------------------------------------------------------------

st.set_page_config(
    page_title="AI Financial Analyst — Chat",
    page_icon="💬",
    layout="wide",
)

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------


def _init_session() -> None:
    """Initialise all session-state keys exactly once per browser session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conv_state" not in st.session_state:
        from ai_financial_analyst.core.conversation_state import new_session
        st.session_state.conv_state = new_session()

    if "agent" not in st.session_state:
        from ai_financial_analyst.agents.conversational_agent import ConversationalAgent
        st.session_state.agent = ConversationalAgent()

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    if "last_trace_path" not in st.session_state:
        st.session_state.last_trace_path = None

    if "last_artifacts_path" not in st.session_state:
        st.session_state.last_artifacts_path = None


_init_session()

# ------------------------------------------------------------------
# Helper: run async from the synchronous Streamlit context
# ------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from the synchronous Streamlit script context."""
    return asyncio.run(coro)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("💬 AI Financial Analyst")
    st.caption("Conversational · Multi-Agent · Gemini Free Tier")
    st.divider()

    # --- Budget metrics ---
    budget_stats = st.session_state.agent.budget.get_stats()
    st.metric(
        "API Calls (session)",
        f"{budget_stats.get('total_calls', 0)} / {budget_stats.get('daily_budget', 1500)}",
    )
    st.metric("Cache Hits", budget_stats.get("cache_hits", 0))

    if budget_stats.get("model_degraded"):
        st.warning(
            "⚠️ **Flash rate-limited** — switched to Flash-Lite. "
            "Response quality may be reduced. Wait ~1 min for recovery.",
            icon="⚠️",
        )

    st.divider()

    # --- Memory panel ---
    st.subheader("🧠 Memory")

    try:
        prefs: dict = _run_async(st.session_state.agent._memory.get_preferences())
        analysis_count: int = _run_async(st.session_state.agent._memory.count_analyses())
    except Exception:
        prefs = {}
        analysis_count = 0

    st.caption(f"Past analyses stored: **{analysis_count}**")

    if prefs:
        st.caption("Known preferences:")
        for key, value in prefs.items():
            st.markdown(f"- **{key}**: {value}")
    else:
        st.caption("No preferences stored yet. Try saying *\"I prefer conservative analysis\"*.")

    if st.button("🗑️ Clear memory", use_container_width=True):
        try:
            _run_async(st.session_state.agent._memory.clear_all())
            st.success("Memory cleared.")
        except Exception as exc:
            st.error(f"Could not clear memory: {exc}")
        st.rerun()

    st.divider()

    # --- Settings ---
    debug_mode = st.checkbox(
        "Debug mode",
        value=st.session_state.debug_mode,
        help="Show trace/artifacts download links after pipeline runs.",
    )
    st.session_state.debug_mode = debug_mode

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        from ai_financial_analyst.core.conversation_state import new_session
        st.session_state.messages = []
        st.session_state.conv_state = new_session()
        st.session_state.last_trace_path = None
        st.session_state.last_artifacts_path = None
        st.rerun()

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- LangGraph pipeline\n"
        "- Gemini 3 Flash (primary)\n"
        "- Gemini 3.1 Flash-Lite (sub-tasks)\n"
        "- yfinance + Tavily (free tier)\n"
        "- LangSmith tracing\n"
    )
    st.divider()
    st.caption(
        "**Capabilities**\n\n"
        "- Stock analysis: *Analyse AAPL*\n"
        "- General questions: *What is CAGR?*\n"
        "- Calculations: *Calculate 15^3*\n\n"
        "Off-topic requests will be politely declined."
    )

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

st.title("AI Financial Analyst")
st.caption(
    "Chat naturally. Ask for stock analysis, financial concepts, or calculations. "
    "The agent uses memory to personalise responses across sessions."
)

# Render existing conversation history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Debug: trace/artifacts downloads (shown after a pipeline run)
# ------------------------------------------------------------------

if st.session_state.debug_mode:
    trace_path = st.session_state.get("last_trace_path")
    artifacts_path = st.session_state.get("last_artifacts_path")
    if trace_path or artifacts_path:
        st.divider()
        st.subheader("Debug: Last Pipeline Run")
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
                "Download run_artifacts.json",
                data=Path(artifacts_path).read_text(),
                file_name=Path(artifacts_path).name,
                mime="application/json",
            )

# ------------------------------------------------------------------
# Chat input and response handling
# ------------------------------------------------------------------

if prompt := st.chat_input("Ask about stocks, financial concepts, or request analysis…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()
        response_placeholder = st.empty()

        tao_lines: list[str] = []

        def _on_step(event: dict) -> None:
            step = event.get("step", "?")
            agent_name = event.get("agent", "?")
            tool = event.get("tool", "?")
            cache = " *(cached)*" if event.get("cache_hit") else ""
            icon = "✓" if event.get("ok", True) else "✗"
            tao_lines.append(f"{icon} **[Step {step}]** `{agent_name}` → `{tool}`{cache}")
            progress_placeholder.markdown(
                "**Running analysis…**\n\n" + "\n\n".join(tao_lines[-20:])
            )

        try:
            response, new_state = asyncio.run(
                st.session_state.agent.process_message(
                    prompt,
                    st.session_state.conv_state,
                    step_callback=_on_step,
                )
            )
        except Exception as exc:
            import traceback
            progress_placeholder.empty()
            error_msg = f"An unexpected error occurred: `{exc}`"
            response_placeholder.error(error_msg)
            response = error_msg
            new_state = st.session_state.conv_state
            if st.session_state.debug_mode:
                st.code(traceback.format_exc(), language="text")
        else:
            progress_placeholder.empty()
            response_placeholder.markdown(response)

            _debug_dir = Path("debug_artifacts")
            if _debug_dir.exists():
                trace_files = sorted(
                    _debug_dir.glob("run_trace_*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                artifacts_files = sorted(
                    _debug_dir.glob("run_artifacts_*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                if trace_files:
                    st.session_state.last_trace_path = str(trace_files[0])
                if artifacts_files:
                    st.session_state.last_artifacts_path = str(artifacts_files[0])

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.conv_state = new_state
    st.rerun()
