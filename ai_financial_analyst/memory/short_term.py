"""Short-term in-session memory utilities.

Purely stateless (static methods). Manages the context window for the LLM
by selecting the most recent messages that fit within a token budget.
No I/O — all state lives in ConversationState.messages.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from ..core.conversation_state import ChatMessage
from ..core.utils import estimate_tokens as _estimate_tokens  # re-exported for callers

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Context-window manager for in-session conversation history."""

    DEFAULT_MAX_TOKENS = 3000

    @staticmethod
    def get_windowed_messages(
        messages: list[ChatMessage],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[ChatMessage]:
        """Return the most recent messages that fit within the token budget.

        System-role messages are always included first and their cost is
        deducted before windowing begins — they are never dropped under
        budget pressure. The remainder of the budget is then spent on the
        most recent user/assistant turn-pairs (newest first). A pair is
        never split — both are included or neither is.
        """
        # Pre-pass: pin system messages (always include, deduct from budget).
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        system_cost = sum(_estimate_tokens(m.get("content", "")) for m in system_msgs)
        budget = max(0, max_tokens - system_cost)

        result: list[ChatMessage] = []
        i = len(non_system) - 1

        while i >= 0:
            msg = non_system[i]
            role = msg.get("role", "")

            if role == "assistant" and i > 0 and non_system[i - 1].get("role") == "user":
                # Pair: user (i-1) + assistant (i)
                user_msg = non_system[i - 1]
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
                # Standalone message (first user turn or orphaned assistant)
                cost = _estimate_tokens(msg.get("content", ""))
                if budget - cost < 0:
                    break
                result.insert(0, msg)
                budget -= cost
                i -= 1

        # System messages always appear before conversation history.
        return system_msgs + result


async def get_windowed_with_summary(
    messages: list[ChatMessage],
    max_tokens: int = ShortTermMemory.DEFAULT_MAX_TOKENS,
    summarizer: Callable[[list[ChatMessage]], Awaitable[str]] | None = None,
) -> list[ChatMessage]:
    """Return windowed messages, optionally summarising what was dropped.

    When *summarizer* is provided and messages are dropped due to the token
    budget, the dropped turn-pairs are passed to the summarizer.  The summary
    is prepended as a synthetic ``system`` message so the LLM retains the gist
    of earlier context without paying the full token cost.

    When *summarizer* is None the behaviour is identical to the synchronous
    ``ShortTermMemory.get_windowed_messages()``.
    """
    windowed = ShortTermMemory.get_windowed_messages(messages, max_tokens)

    if summarizer is None:
        return windowed

    # Identify dropped non-system messages (get_windowed_messages always keeps
    # the most-recent pairs, so anything not in windowed was dropped from the start).
    non_system_all      = [m for m in messages if m.get("role") != "system"]
    non_system_windowed = [m for m in windowed  if m.get("role") != "system"]
    n_kept   = len(non_system_windowed)
    dropped  = non_system_all[: len(non_system_all) - n_kept]

    if not dropped:
        return windowed

    try:
        summary_text = await summarizer(dropped)
        if summary_text:
            summary_msg: ChatMessage = {
                "role":    "system",
                "content": f"[Earlier conversation summary] {summary_text}",
            }
            # Ordering: original system messages → summary → recent conversation
            system_msgs = [m for m in windowed if m.get("role") == "system"]
            conv_msgs   = [m for m in windowed if m.get("role") != "system"]
            return system_msgs + [summary_msg] + conv_msgs
    except Exception as exc:
        logger.debug("Conversation summarisation skipped: %s", exc)

    return windowed
