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
