"""Conversation-level state for the multi-turn chat agent.

Kept entirely separate from AgentState (the inner pipeline's state).
ConversationState is owned by the ConversationalAgent; AgentState is owned
by the Researcher → Quant → Editor pipeline.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict


class ChatMessage(TypedDict):
    role: str       # "user" | "assistant" | "system"
    content: str
    metadata: dict[str, Any]
    timestamp: float


class ConversationState(TypedDict, total=False):
    session_id: str
    messages: list[ChatMessage]
    current_intent: str | None   # last classified intent
    pending_tickers: list[str]   # tickers extracted from last user turn


def new_session() -> ConversationState:
    """Create a fresh ConversationState for a new chat session."""
    return ConversationState(
        session_id=str(uuid.uuid4()),
        messages=[],
        current_intent=None,
        pending_tickers=[],
    )


def append_messages(
    state: ConversationState,
    user_content: str,
    assistant_content: str,
    intent: str | None = None,
    tickers: list[str] | None = None,
) -> ConversationState:
    """Return a new ConversationState with both turns appended."""
    now = time.time()
    new_messages = list(state.get("messages", [])) + [
        ChatMessage(
            role="user",
            content=user_content,
            metadata={"intent": intent} if intent else {},
            timestamp=now,
        ),
        ChatMessage(
            role="assistant",
            content=assistant_content,
            metadata={},
            timestamp=now,
        ),
    ]
    return ConversationState(
        **{
            **state,
            "messages": new_messages,
            "current_intent": intent,
            "pending_tickers": tickers or [],
        }
    )


def get_recent_context(state: ConversationState, max_messages: int = 6) -> list[ChatMessage]:
    """Return the last `max_messages` messages for LLM context injection.

    Drops oldest messages first (never drops system messages if any exist).
    """
    messages = state.get("messages", [])
    return messages[-max_messages:] if len(messages) > max_messages else messages
