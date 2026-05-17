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
    """Mock subllm whose direct ainvoke returns text (used by maybe_save_analysis_summary)."""
    response = MagicMock()
    response.content = text
    subllm = MagicMock()
    subllm.ainvoke = AsyncMock(return_value=response)
    return subllm


def _mock_subllm_structured(preferences: dict) -> MagicMock:
    """Mock subllm that supports with_structured_output (used by maybe_extract_preferences)."""
    from ai_financial_analyst.memory.memory_manager import _PreferenceOutput

    structured_result = _PreferenceOutput(preferences=preferences)
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=structured_result)

    subllm = MagicMock()
    subllm.with_structured_output = MagicMock(return_value=structured_llm)
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
        subllm = _mock_subllm_structured({"investment_style": "conservative"})
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("I prefer conservative analysis")
        prefs = await lt.get_all_preferences()
        assert prefs.get("investment_style") == "conservative"

    @pytest.mark.asyncio
    async def test_uses_structured_output_schema(self, lt):
        from ai_financial_analyst.memory.memory_manager import _PreferenceOutput
        subllm = _mock_subllm_structured({"investor_type": "long-term"})
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("I'm a long-term investor")
        subllm.with_structured_output.assert_called_once_with(_PreferenceOutput)

    @pytest.mark.asyncio
    async def test_skips_llm_when_no_signal(self, lt):
        subllm = _mock_subllm_structured({"k": "v"})
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("What is a P/E ratio?")
        subllm.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_subllm_skips_silently(self, lt):
        mgr = MemoryManager(lt, subllm=None)
        await mgr.maybe_extract_preferences("I prefer conservative analysis")
        # Should not raise; just skips
        assert await lt.get_all_preferences() == {}

    @pytest.mark.asyncio
    async def test_llm_error_does_not_crash(self, lt):
        from ai_financial_analyst.memory.memory_manager import _PreferenceOutput
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        subllm = MagicMock()
        subllm.with_structured_output = MagicMock(return_value=structured_llm)
        mgr = MemoryManager(lt, subllm=subllm)
        # Should not raise
        await mgr.maybe_extract_preferences("I prefer conservative analysis")

    @pytest.mark.asyncio
    async def test_empty_preferences_stores_nothing(self, lt):
        subllm = _mock_subllm_structured({})
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_extract_preferences("I prefer careful analysis")
        assert await lt.get_all_preferences() == {}


# ---------------------------------------------------------------------------
# maybe_save_analysis_summary
# ---------------------------------------------------------------------------


class TestBuildMemoryContextConfig:
    @pytest.mark.asyncio
    async def test_limit_uses_settings_value(self, lt):
        """build_memory_context passes settings.memory_context_summaries_limit, not hardcoded 2."""
        from ai_financial_analyst.config import settings

        mock_search = AsyncMock(return_value=[])
        lt.search_summaries = mock_search

        mgr = MemoryManager(lt)
        await mgr.build_memory_context([], "AAPL")

        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        assert kwargs.get("limit") == settings.memory_context_summaries_limit

    @pytest.mark.asyncio
    async def test_save_summary_no_truncation(self, lt):
        """maybe_save_analysis_summary passes the full report, not [:3000]."""
        long_report = "x" * 10_000
        subllm = _mock_subllm("One-paragraph summary.")
        mgr = MemoryManager(lt, subllm=subllm)
        await mgr.maybe_save_analysis_summary("sess1", ["AAPL"], long_report, "run1")

        # The ainvoke call should have received the full 10 000-char report in its prompt
        call_args = subllm.ainvoke.call_args
        prompt_text = call_args[0][0][0].content  # HumanMessage.content
        assert "x" * 9_000 in prompt_text  # far beyond old 3000-char limit


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


# ---------------------------------------------------------------------------
# Preference signal detection — expanded regex
# ---------------------------------------------------------------------------


class TestPreferenceSignals:
    """_PREFERENCE_SIGNALS regex covers novel phrasings beyond the original set."""

    def _matches(self, message: str) -> bool:
        from ai_financial_analyst.memory.memory_manager import _PREFERENCE_SIGNALS
        return bool(_PREFERENCE_SIGNALS.search(message))

    # Original signals — must still work
    def test_original_i_prefer_detected(self):
        assert self._matches("I prefer conservative analysis")

    def test_original_focus_on_detected(self):
        assert self._matches("Focus on dividend stocks")

    # New signals
    def test_would_prefer_detected(self):
        assert self._matches("I would prefer detailed reports")

    def test_id_prefer_detected(self):
        assert self._matches("I'd prefer a brief summary")

    def test_as_a_detected(self):
        assert self._matches("As a long-term investor I need macro data")

    def test_give_me_detected(self):
        assert self._matches("Give me brief summaries")

    def test_my_preference_detected(self):
        assert self._matches("My preference is detailed analysis")

    def test_i_tend_to_detected(self):
        assert self._matches("I tend to focus on value stocks")

    def test_make_sure_detected(self):
        assert self._matches("Make sure to include risk metrics")

    def test_always_give_detected(self):
        assert self._matches("Always give me a brief executive summary")

    # Non-preference messages — must NOT be detected
    def test_plain_question_not_detected(self):
        assert not self._matches("What is the P/E ratio for AAPL?")

    def test_analysis_request_not_detected(self):
        assert not self._matches("Analyse Apple stock for me")

    def test_generic_greeting_not_detected(self):
        assert not self._matches("Hello, can you help me?")
