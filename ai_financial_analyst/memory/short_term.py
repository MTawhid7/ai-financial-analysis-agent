"""Short-term in-session memory utilities.

Purely stateless (static methods). Manages the context window for the LLM
by selecting the most recent messages that fit within a token budget.
No I/O — all state lives in ConversationState.messages.
"""

from __future__ import annotations

from ..core.conversation_state import ChatMessage

# Consistent with the existing heuristic used throughout the codebase.
_CHARS_PER_TOKEN = 4


class ShortTermMemory:
    """Context-window manager for in-session conversation history."""

    DEFAULT_MAX_TOKENS = 3000

    @staticmethod
    def get_windowed_messages(
        messages: list[ChatMessage],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[ChatMessage]:
        """Return the most recent messages that fit within the token budget.

        Walks backwards from the most recent message, accumulating cost until
        the budget is exhausted. Earlier messages are dropped first.
        """
        budget = max_tokens
        result: list[ChatMessage] = []
        for msg in reversed(messages):
            cost = len(msg.get("content", "")) // _CHARS_PER_TOKEN
            if budget - cost < 0:
                break
            result.insert(0, msg)
            budget -= cost
        return result
