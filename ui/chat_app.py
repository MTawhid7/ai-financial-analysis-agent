"""Conversational chat UI — Phase 1 + Phase 2 + Phase 2.5.

Phase 2.5 additions:
  - Conversation persistence: every turn is saved to SQLite
  - Conversation sidebar: resume past chats, start new ones, delete old ones
  - Memory-query intent: retrieves stored analyses instead of re-running the pipeline

Run with:
    streamlit run ui/chat_app.py
"""

from __future__ import annotations

import asyncio
import time
import uuid
import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Financial Analyst — Chat",
    page_icon="💬",
    layout="wide",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from the synchronous Streamlit script context."""
    return asyncio.run(coro)


def _format_timestamp(ts: float) -> str:
    """Return a human-readable date label for a Unix timestamp."""
    if not ts:
        return ""
    now = time.time()
    delta = now - ts
    if delta < 86400:
        return "Today"
    if delta < 172800:
        return "Yesterday"
    return time.strftime("%b %d", time.localtime(ts))


def _conversation_title(message: str) -> str:
    """Generate a concise conversation title from the first user message."""
    text = message.strip().replace("\n", " ")
    return text[:55] + "…" if len(text) > 55 else text


# ------------------------------------------------------------------
# Conversation load / create helpers (called before sidebar renders)
# ------------------------------------------------------------------


def _load_conversation(conversation_id: str) -> None:
    """Load a past conversation from DB into session state."""
    try:
        messages_from_db = _run_async(
            st.session_state.agent._memory.load_conversation(conversation_id)
        )
    except Exception:
        messages_from_db = []

    from ai_financial_analyst.core.conversation_state import ChatMessage, new_session

    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages_from_db
    ]

    conv_messages = [
        ChatMessage(
            role=m["role"],
            content=m["content"],
            metadata={"intent": m.get("intent", ""), "tickers": m.get("tickers", "")},
            timestamp=m.get("created_at", 0.0),
        )
        for m in messages_from_db
    ]
    new_state = new_session()
    new_state["session_id"] = conversation_id
    new_state["messages"] = conv_messages
    st.session_state.conv_state = new_state
    st.session_state.conversation_id = conversation_id


def _start_new_conversation() -> None:
    """Create a fresh in-memory conversation (DB row created on first message)."""
    from ai_financial_analyst.core.conversation_state import new_session
    conv_state = new_session()
    st.session_state.conversation_id = None  # created in DB on first message
    st.session_state.messages = []
    st.session_state.conv_state = conv_state
    st.session_state.last_trace_path = None
    st.session_state.last_artifacts_path = None


# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------


def _init_session() -> None:
    """Initialise all session-state keys exactly once per browser session."""
    if "agent" not in st.session_state:
        from ai_financial_analyst.agents.conversational_agent import ConversationalAgent
        st.session_state.agent = ConversationalAgent()

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    if "last_trace_path" not in st.session_state:
        st.session_state.last_trace_path = None

    if "last_artifacts_path" not in st.session_state:
        st.session_state.last_artifacts_path = None

    # Conversation state — initialised after agent so we can query the DB.
    if "conversation_id" not in st.session_state:
        try:
            convs = _run_async(st.session_state.agent._memory.list_conversations(limit=1))
            if convs:
                _load_conversation(convs[0]["id"])
            else:
                _start_new_conversation()
        except Exception:
            _start_new_conversation()


_init_session()

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
        st.caption("No preferences yet. Try: *\"I prefer conservative analysis\"*")

    if st.button("🗑️ Clear memory", use_container_width=True, key="clear_memory"):
        try:
            _run_async(st.session_state.agent._memory.clear_all())
            st.success("Memory cleared.")
        except Exception as exc:
            st.error(f"Could not clear memory: {exc}")
        st.rerun()

    st.divider()

    # --- Conversation history ---
    st.subheader("💬 Conversations")

    if st.button("➕ New conversation", use_container_width=True, key="new_conv"):
        _start_new_conversation()
        st.rerun()

    try:
        conversations = _run_async(
            st.session_state.agent._memory.list_conversations(limit=15)
        )
    except Exception:
        conversations = []

    current_conv_id = st.session_state.get("conversation_id")
    for conv in conversations:
        is_active = conv["id"] == current_conv_id
        label = f"{'▶ ' if is_active else ''}{conv['title']}"
        date_label = _format_timestamp(conv.get("updated_at", 0))

        col_btn, col_del = st.columns([5, 1])
        with col_btn:
            if st.button(
                label,
                key=f"conv_{conv['id']}",
                use_container_width=True,
                help=date_label,
            ):
                if not is_active:
                    _load_conversation(conv["id"])
                    st.rerun()
        with col_del:
            if st.button("✕", key=f"del_{conv['id']}", help="Delete this conversation"):
                try:
                    _run_async(
                        st.session_state.agent._memory.delete_conversation(conv["id"])
                    )
                    if is_active:
                        _start_new_conversation()
                except Exception:
                    pass
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
        "- Recall past analyses: *What did we find about AAPL?*\n"
        "- General questions: *What is CAGR?*\n"
        "- Calculations: *Calculate 15^3*\n\n"
        "Off-topic requests will be politely declined."
    )

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

st.title("AI Financial Analyst")
st.caption(
    "Chat naturally. Analyses and preferences are remembered across sessions. "
    "Ask *\"What did we find about AAPL earlier?\"* to recall past results."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Debug: trace/artifacts downloads
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

if prompt := st.chat_input("Ask about stocks, concepts, or recall a past analysis…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Ensure a conversation exists in the DB ---
    if not st.session_state.conversation_id:
        conv_id = st.session_state.conv_state.get("session_id") or str(uuid.uuid4())
        title = _conversation_title(prompt)
        try:
            _run_async(st.session_state.agent._memory.create_conversation(conv_id, title))
        except Exception as exc:
            st.toast(f"Could not save conversation: {exc}", icon="⚠️")
        st.session_state.conversation_id = conv_id
        # Sync the session_id in conv_state so analysis summaries link to this conversation.
        st.session_state.conv_state["session_id"] = conv_id

    conv_id = st.session_state.conversation_id

    # --- Save user message ---
    try:
        _run_async(st.session_state.agent._memory.save_message(conv_id, "user", prompt))
    except Exception:
        pass

    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Generate response ---
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

    # --- Save assistant response ---
    intent = new_state.get("current_intent", "")
    tickers_str = ", ".join(new_state.get("pending_tickers", []))
    try:
        _run_async(st.session_state.agent._memory.save_message(
            conv_id, "assistant", response, intent, tickers_str
        ))
        _run_async(st.session_state.agent._memory.update_conversation_updated_at(conv_id))
    except Exception:
        pass

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.conv_state = new_state
    st.rerun()
