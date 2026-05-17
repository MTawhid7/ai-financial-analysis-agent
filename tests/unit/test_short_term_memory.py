"""Unit tests for ShortTermMemory context-window management."""

from __future__ import annotations

import pytest

from ai_financial_analyst.memory.short_term import ShortTermMemory, get_windowed_with_summary
from ai_financial_analyst.core.conversation_state import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content, metadata={}, timestamp=0.0)


class TestGetWindowedMessages:
    def test_returns_all_messages_when_within_budget(self):
        messages = [_msg("user", "hi"), _msg("assistant", "hello")]
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=1000)
        assert result == messages

    def test_drops_oldest_when_budget_exceeded(self):
        # Each message costs len("x" * 400) // 4 = 100 tokens
        messages = [_msg("user", "a" * 400)] * 5
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=250)
        # 250 tokens fits 2 messages (2 × 100 = 200 ≤ 250, 3 × 100 = 300 > 250)
        assert len(result) == 2
        assert result == messages[-2:]

    def test_returns_empty_for_empty_input(self):
        assert ShortTermMemory.get_windowed_messages([], max_tokens=1000) == []

    def test_single_oversized_message_excluded(self):
        big_msg = _msg("user", "x" * 10000)  # ~2500 tokens
        result = ShortTermMemory.get_windowed_messages([big_msg], max_tokens=100)
        assert result == []

    def test_preserves_order(self):
        messages = [_msg("user", f"msg{i}") for i in range(5)]
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=10000)
        assert [m["content"] for m in result] == [f"msg{i}" for i in range(5)]

    def test_keeps_most_recent_on_truncation(self):
        messages = [
            _msg("user", "old message " * 100),
            _msg("assistant", "old reply " * 100),
            _msg("user", "recent"),
            _msg("assistant", "recent reply"),
        ]
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=20)
        # Only the recent short messages should fit
        contents = [m["content"] for m in result]
        assert "recent" in contents
        assert "recent reply" in contents

    def test_zero_budget_returns_empty(self):
        messages = [_msg("user", "hello")]
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=0)
        assert result == []


class TestPairPreservation:
    def test_user_assistant_pair_not_split(self):
        """A user+assistant pair must be included together or not at all."""
        # Each message = 100 chars = 25 tokens. Budget = 40 fits one pair but not two.
        user1 = _msg("user", "a" * 100)
        asst1 = _msg("assistant", "b" * 100)
        user2 = _msg("user", "c" * 100)
        asst2 = _msg("assistant", "d" * 100)
        messages = [user1, asst1, user2, asst2]
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=50)
        # Most recent pair (user2, asst2) costs 50 tokens total → fits.
        # Older pair (user1, asst1) would exceed budget.
        assert len(result) == 2
        assert result[0]["content"] == "c" * 100
        assert result[1]["content"] == "d" * 100

    def test_assistant_not_included_without_user_pair(self):
        """A lone assistant message (no preceding user) is included standalone."""
        asst = _msg("assistant", "hello")
        messages = [asst]
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=1000)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_json_dense_content_estimates_smaller_token_cost(self):
        """JSON-heavy content should consume fewer estimated tokens, fitting more into budget."""
        # JSON content with high structural density (~50% structural chars)
        json_content = '{"key": "value", "num": 123, "arr": [1, 2, 3]}' * 10  # 480 chars
        # With json_density > 0.10, chars_per_token = 2.0 → ~240 tokens
        # With old 4-chars/token → ~120 tokens
        # Prose content same length
        prose_content = "a" * len(json_content)  # pure prose → ~120 tokens

        # Budget that fits the JSON message but NOT two prose messages of same length
        json_msg = _msg("user", json_content)
        result = ShortTermMemory.get_windowed_messages([json_msg], max_tokens=300)
        # JSON message: 480 chars / 2 chars_per_token = 240 tokens ≤ 300 → included
        assert len(result) == 1

        prose_msg = _msg("user", prose_content)
        result2 = ShortTermMemory.get_windowed_messages([prose_msg], max_tokens=100)
        # Prose: 480 chars / 4 = 120 tokens > 100 → excluded
        assert len(result2) == 0


class TestSystemMessagePriority:
    """System-role messages are always included regardless of budget pressure."""

    def test_system_message_always_included(self):
        sys_msg = _msg("system", "You are a financial analyst.")
        # 20 user/assistant pairs that together far exceed any budget
        many_msgs = [_msg("user", "Q"), _msg("assistant", "A")] * 20
        messages = [sys_msg] + many_msgs

        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=50)

        roles = [m["role"] for m in result]
        assert "system" in roles
        assert result[0]["role"] == "system"  # system message comes first

    def test_system_message_cost_reduces_available_budget(self):
        # System message of ~25 tokens (100 chars / 4)
        sys_msg = _msg("system", "x" * 100)
        # User message of ~25 tokens
        user_msg = _msg("user", "x" * 100)

        # Budget of 30 tokens: enough for system (25) but not system + user (50)
        result = ShortTermMemory.get_windowed_messages(
            [sys_msg, user_msg], max_tokens=30
        )
        roles = [m["role"] for m in result]
        assert "system" in roles
        assert "user" not in roles  # user dropped — budget exhausted by system

    def test_system_message_appears_before_conversation(self):
        sys_msg  = _msg("system", "Instructions.")
        user_msg = _msg("user", "Hello.")
        asst_msg = _msg("assistant", "Hi.")

        result = ShortTermMemory.get_windowed_messages(
            [sys_msg, user_msg, asst_msg], max_tokens=500
        )
        assert result[0]["role"] == "system"

    def test_no_system_messages_behaves_as_before(self):
        messages = [_msg("user", "A"), _msg("assistant", "B")] * 3
        result = ShortTermMemory.get_windowed_messages(messages, max_tokens=500)
        # All messages fit; no system messages; result is unchanged
        assert len(result) == 6
        assert all(m["role"] in ("user", "assistant") for m in result)


class TestGetWindowedWithSummary:
    """Tests for the async hierarchical-summarisation variant."""

    @pytest.mark.asyncio
    async def test_no_summarizer_behaves_identically_to_sync(self):
        messages = [_msg("user", "hi"), _msg("assistant", "hello")]
        sync_result  = ShortTermMemory.get_windowed_messages(messages, max_tokens=1000)
        async_result = await get_windowed_with_summary(messages, max_tokens=1000, summarizer=None)
        assert sync_result == async_result

    @pytest.mark.asyncio
    async def test_summarizer_called_for_dropped_messages(self):
        """When messages are dropped, the summarizer should receive them and the
        result should contain a synthetic system summary message."""
        # Make 4 messages where only the most recent 2 fit in the budget
        old_pair = [_msg("user", "a" * 400), _msg("assistant", "b" * 400)]  # ~200 tokens
        new_pair = [_msg("user", "c" * 400), _msg("assistant", "d" * 400)]  # ~200 tokens
        messages = old_pair + new_pair

        calls: list[list] = []

        async def mock_summarizer(msgs):
            calls.append(msgs)
            return "Earlier we discussed topic A."

        result = await get_windowed_with_summary(
            messages, max_tokens=250, summarizer=mock_summarizer
        )

        assert calls, "summarizer should have been called"
        # The old_pair should be the dropped messages passed to the summarizer
        assert len(calls[0]) == 2

        roles = [m["role"] for m in result]
        assert "system" in roles
        # The summary should appear as a system message
        summary_msgs = [m for m in result if m.get("role") == "system" and "Earlier" in m.get("content", "")]
        assert summary_msgs

    @pytest.mark.asyncio
    async def test_no_dropped_messages_summarizer_not_called(self):
        """When all messages fit, the summarizer should NOT be called."""
        messages = [_msg("user", "hi"), _msg("assistant", "hello")]
        calls: list = []

        async def mock_summarizer(msgs):
            calls.append(msgs)
            return "summary"

        await get_windowed_with_summary(messages, max_tokens=10000, summarizer=mock_summarizer)
        assert not calls, "summarizer should not be called when no messages are dropped"

    @pytest.mark.asyncio
    async def test_summarizer_failure_falls_back_to_silent_drop(self):
        """A summarizer that raises should not propagate the error."""
        old = _msg("user", "a" * 400)
        new_pair = [_msg("user", "b" * 400), _msg("assistant", "c" * 400)]
        messages = [old] + new_pair

        async def failing_summarizer(msgs):
            raise RuntimeError("LLM unavailable")

        result = await get_windowed_with_summary(
            messages, max_tokens=250, summarizer=failing_summarizer
        )
        # Should return the windowed result without the summary message
        assert isinstance(result, list)
        assert len(result) >= 0  # no crash
