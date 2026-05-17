"""Unit tests for MemoryBackend Protocol and InMemoryBackend implementation.

Verifies that InMemoryBackend:
  - Satisfies the MemoryBackend Protocol (runtime_checkable check)
  - Implements all required methods correctly
  - Is fully independent of file I/O (no aiosqlite, no tmp_path)
"""

from __future__ import annotations

import pytest

from ai_financial_analyst.memory.protocol import MemoryBackend
from ai_financial_analyst.memory.in_memory import InMemoryBackend


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend(user_id="test-user")


class TestProtocolCompliance:
    def test_in_memory_backend_satisfies_protocol(self, backend):
        """isinstance check uses @runtime_checkable on the Protocol."""
        assert isinstance(backend, MemoryBackend)

    def test_protocol_methods_all_present(self, backend):
        required = [
            "save_preference", "get_all_preferences",
            "save_analysis_summary", "search_summaries",
            "get_recent_summaries", "count_summaries",
            "save_conversation", "update_conversation_title",
            "list_conversations", "delete_conversation",
            "save_message", "get_conversation_messages",
            "delete_all",
        ]
        for method in required:
            assert hasattr(backend, method), f"Missing method: {method}"


class TestInMemoryPreferences:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, backend):
        await backend.save_preference("style", "conservative")
        prefs = await backend.get_all_preferences()
        assert prefs["style"] == "conservative"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, backend):
        await backend.save_preference("style", "aggressive")
        await backend.save_preference("style", "conservative")
        prefs = await backend.get_all_preferences()
        assert prefs["style"] == "conservative"
        assert len(prefs) == 1

    @pytest.mark.asyncio
    async def test_empty_returns_empty_dict(self, backend):
        assert await backend.get_all_preferences() == {}


class TestInMemorySummaries:
    @pytest.mark.asyncio
    async def test_save_and_count(self, backend):
        await backend.save_analysis_summary("s1", ["AAPL"], "Apple reported strong results.", "r1")
        assert await backend.count_summaries() == 1

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, backend):
        await backend.save_analysis_summary("s1", ["AAPL"], "Apple iPhone revenue grew.", "r1")
        await backend.save_analysis_summary("s2", ["MSFT"], "Microsoft Azure dominated.", "r2")
        results = await backend.search_summaries("Apple")
        assert len(results) == 1
        assert "Apple" in results[0]["summary"]

    @pytest.mark.asyncio
    async def test_search_no_match_returns_empty(self, backend):
        await backend.save_analysis_summary("s1", ["AAPL"], "Apple results.", "r1")
        results = await backend.search_summaries("Microsoft")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_recent_summaries(self, backend):
        for i in range(3):
            await backend.save_analysis_summary(f"s{i}", ["T{i}"], f"Summary {i}", f"r{i}")
        recent = await backend.get_recent_summaries(limit=2)
        assert len(recent) == 2


class TestInMemoryConversations:
    @pytest.mark.asyncio
    async def test_save_and_list(self, backend):
        await backend.save_conversation("c1", "AAPL Analysis")
        convos = await backend.list_conversations()
        assert len(convos) == 1
        assert convos[0]["title"] == "AAPL Analysis"

    @pytest.mark.asyncio
    async def test_update_title(self, backend):
        await backend.save_conversation("c1", "Old Title")
        await backend.update_conversation_title("c1", "New Title")
        convos = await backend.list_conversations()
        assert convos[0]["title"] == "New Title"

    @pytest.mark.asyncio
    async def test_delete_removes_conversation_and_messages(self, backend):
        await backend.save_conversation("c1", "Test")
        await backend.save_message("c1", "user", "Hello")
        await backend.delete_conversation("c1")
        convos = await backend.list_conversations()
        msgs   = await backend.get_conversation_messages("c1")
        assert convos == []
        assert msgs   == []


class TestInMemoryMessages:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_ordered(self, backend):
        await backend.save_conversation("c1", "Test")
        await backend.save_message("c1", "user",      "Hello")
        await backend.save_message("c1", "assistant", "Hi there!")
        msgs = await backend.get_conversation_messages("c1")
        assert len(msgs) == 2
        assert msgs[0]["role"]    == "user"
        assert msgs[1]["role"]    == "assistant"
        assert msgs[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_empty_conversation(self, backend):
        msgs = await backend.get_conversation_messages("nonexistent")
        assert msgs == []


class TestDeleteAll:
    @pytest.mark.asyncio
    async def test_clears_everything(self, backend):
        await backend.save_preference("k", "v")
        await backend.save_analysis_summary("s1", ["A"], "text", "r1")
        await backend.save_conversation("c1", "Title")
        await backend.save_message("c1", "user", "Hello")
        await backend.delete_all()
        assert await backend.get_all_preferences() == {}
        assert await backend.count_summaries()      == 0
        assert await backend.list_conversations()   == []
