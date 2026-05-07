"""Unit tests for MemoryManager.

LongTermMemory uses a real temporary SQLite database.
LLM calls are mocked — no API quota consumed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_financial_analyst.memory.long_term import LongTermMemory
from ai_financial_analyst.memory.memory_manager import MemoryManager
from ai_financial_analyst.core.conversation_state import new_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "manager_test.db")


@pytest.fixture
def lt(db_path) -> LongTermMemory:
    return LongTermMemory(db_path=db_path)


def _mock_subllm(text: str) -> MagicMock:
    response = MagicMock()
    response.content = text
    subllm = MagicMock()
    subllm.ainvoke = AsyncMock(return_value=response)
    return subllm


# ---------------------------------------------------------------------------
# build_memory_context
# ---------------------------------------------------------------------------


class TestBuildMemoryContext:
    @pytest.mark.asyncio
    async def test_empty_when_nothing_stored(self, lt):
        mgr = MemoryManager(lt)
        ctx = await mgr.build_memory_context([], "AAPL")
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_includes_preferences(self, lt):
        await lt.save_preference("investment_style", "conservative")
        mgr = MemoryManager(lt)
        ctx = await mgr.build_memory_context([], "anything")
        assert "conservative" in ctx
        assert "investment_style" in ctx

    @pytest.mark.asyncio
    async def test_includes_relevant_past_summary(self, lt):
        await lt.save_analysis_summary("sess1", ["AAPL"], "Apple had 15% CAGR.", "r1")
        mgr = MemoryManager(lt)
        ctx = await mgr.build_memory_context([], "AAPL")
        assert "AAPL" in ctx
        assert "CAGR" in ctx

    @pytest.mark.asyncio
    async def test_excludes_irrelevant_summary(self, lt):
        await lt.save_analysis_summary("sess1", ["MSFT"], "Microsoft revenue grew.", "r1")
        mgr = MemoryManager(lt)
        # Searching for AAPL should not surface an MSFT summary
        ctx = await mgr.build_memory_context([], "AAPL")
        assert "MSFT" not in ctx

    @pytest.mark.asyncio
    async def test_context_truncated_to_max_chars(self, lt):
        # Flood the DB with long summaries
        for i in range(10):
            await lt.save_analysis_summary("s", ["AAPL"], "x" * 500, f"r{i}")
        mgr = MemoryManager(lt)
        ctx = await mgr.build_memory_context([], "AAPL")
        # _MAX_CONTEXT_CHARS = 2000 + possible "…"
        assert len(ctx) <= 2002

    @pytest.mark.asyncio
    async def test_empty_query_skips_summary_search(self, lt):
        await lt.save_analysis_summary("s", ["AAPL"], "Apple summary.", "r1")
        mgr = MemoryManager(lt)
        ctx = await mgr.build_memory_context([], "")
        # Preferences section won't exist (none stored); summary search skipped
        assert ctx == ""


# ---------------------------------------------------------------------------
# maybe_extract_preferences
# ---------------------------------------------------------------------------


class TestMaybeExtractPreferences:
    @pytest.mark.asyncio
    async def test_saves_extracted_preference(self, lt):
        subllm = _mock_subllm('{"investment_style": "conservative"}')
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("I prefer conservative analysis")
        prefs = await lt.get_all_preferences()
        assert prefs.get("investment_style") == "conservative"

    @pytest.mark.asyncio
    async def test_skips_llm_when_no_signal(self, lt):
        subllm = _mock_subllm('{"k": "v"}')
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("What is a P/E ratio?")
        subllm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_subllm_skips_silently(self, lt):
        mgr = MemoryManager(lt, subllm=None)
        await mgr.maybe_extract_preferences("I prefer conservative analysis")
        # Should not raise; just skips
        assert await lt.get_all_preferences() == {}

    @pytest.mark.asyncio
    async def test_malformed_llm_response_does_not_crash(self, lt):
        subllm = _mock_subllm("not valid json at all")
        mgr = MemoryManager(lt, subllm=subllm)
        # Should not raise
        await mgr.maybe_extract_preferences("I prefer conservative analysis")

    @pytest.mark.asyncio
    async def test_strips_markdown_fences_before_parsing(self, lt):
        subllm = _mock_subllm('```json\n{"investor_type": "long-term"}\n```')
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("I'm a long-term investor")
        prefs = await lt.get_all_preferences()
        assert prefs.get("investor_type") == "long-term"

    @pytest.mark.asyncio
    async def test_empty_dict_response_stores_nothing(self, lt):
        subllm = _mock_subllm("{}")
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("I prefer careful analysis")
        assert await lt.get_all_preferences() == {}


# ---------------------------------------------------------------------------
# maybe_save_analysis_summary
# ---------------------------------------------------------------------------


class TestMaybeSaveAnalysisSummary:
    @pytest.mark.asyncio
    async def test_saves_summary(self, lt):
        subllm = _mock_subllm("Apple showed 15% CAGR and traded at a P/E of 28x.")
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_save_analysis_summary("sess1", ["AAPL"], "# Full report...", "run1")
        assert await lt.count_summaries() == 1

    @pytest.mark.asyncio
    async def test_skips_when_no_report(self, lt):
        subllm = _mock_subllm("Summary.")
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_save_analysis_summary("sess1", ["AAPL"], "", "run1")
        subllm.ainvoke.assert_not_called()
        assert await lt.count_summaries() == 0

    @pytest.mark.asyncio
    async def test_skips_when_no_subllm(self, lt):
        mgr = MemoryManager(lt, subllm=None)
        await mgr.maybe_save_analysis_summary("sess1", ["AAPL"], "# Report", "run1")
        assert await lt.count_summaries() == 0

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_crash(self, lt):
        subllm = MagicMock()
        subllm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mgr = MemoryManager(lt, subllm=subllm)
        # Should not raise
        await mgr.maybe_save_analysis_summary("sess1", ["AAPL"], "# Report", "run1")


# ---------------------------------------------------------------------------
# UI accessors
# ---------------------------------------------------------------------------


class TestUIAccessors:
    @pytest.mark.asyncio
    async def test_get_preferences_delegates(self, lt):
        await lt.save_preference("k", "v")
        mgr = MemoryManager(lt)
        assert await mgr.get_preferences() == {"k": "v"}

    @pytest.mark.asyncio
    async def test_count_analyses_delegates(self, lt):
        await lt.save_analysis_summary("s", ["AAPL"], "Summary.", "r1")
        mgr = MemoryManager(lt)
        assert await mgr.count_analyses() == 1

    @pytest.mark.asyncio
    async def test_clear_all_delegates(self, lt):
        await lt.save_preference("k", "v")
        await lt.save_analysis_summary("s", ["AAPL"], "Summary.", "r1")
        mgr = MemoryManager(lt)
        await mgr.clear_all()
        assert await mgr.get_preferences() == {}
        assert await mgr.count_analyses() == 0
