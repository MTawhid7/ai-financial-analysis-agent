"""Unit tests for ShortTermMemory context-window management."""

from __future__ import annotations

import pytest

from ai_financial_analyst.memory.short_term import ShortTermMemory
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
