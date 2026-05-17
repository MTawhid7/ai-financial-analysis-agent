"""Short-term in-session memory utilities.

Purely stateless (static methods). Manages the context window for the LLM
by selecting the most recent messages that fit within a token budget.
No I/O — all state lives in ConversationState.messages.
"""

from __future__ import annotations

from ..core.conversation_state import ChatMessage
from ..core.utils import estimate_tokens as _estimate_tokens  # re-exported for callers


class ShortTermMemory:
    """Context-window manager for in-session conversation history."""

    DEFAULT_MAX_TOKENS = 3000

    @staticmethod
    def get_windowed_messages(
        messages: list[ChatMessage],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[ChatMessage]:
        """Return the most recent messages that fit within the token budget.

        Walks backwards from the most recent message in turn-pairs
        (user + assistant). A pair is never split — both are included or
        neither is. Standalone user messages (no preceding assistant) are
        included individually. JSON-heavy content is estimated at 2 chars/token
        instead of 4 to avoid over-counting tool responses.
        """
        budget = max_tokens
        result: list[ChatMessage] = []
        i = len(messages) - 1

        while i >= 0:
            msg = messages[i]
            role = msg.get("role", "")

            if role == "assistant" and i > 0 and messages[i - 1].get("role") == "user":
                # Pair: user (i-1) + assistant (i)
                user_msg = messages[i - 1]
                cost = (
                    _estimate_tokens(user_msg.get("content", ""))
                    + _estimate_tokens(msg.get("content", ""))
                )
                if budget - cost < 0:
                    break
                result.insert(0, msg)
                result.insert(0, user_msg)
                budget -= cost
                i -= 2
            else:
                # Standalone message (first user turn, system, or orphaned assistant)
                cost = _estimate_tokens(msg.get("content", ""))
                if budget - cost < 0:
                    break
                result.insert(0, msg)
                budget -= cost
                i -= 1

        return result
