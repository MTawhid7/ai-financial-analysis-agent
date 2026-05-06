"""Unit tests for the intent classifier.

All LLM calls are mocked — no API quota consumed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_financial_analyst.agents.intent_classifier import (
    IntentType,
    _VALID_INTENTS,
    _extract_tickers_regex,
    classify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subllm(intent: str, tickers: list[str]) -> AsyncMock:
    """Return a mock subllm whose ainvoke returns a classification JSON."""
    payload = json.dumps({"intent": intent, "tickers": tickers})
    response = MagicMock()
    response.content = payload
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=response)
    return mock_llm


# ---------------------------------------------------------------------------
# _extract_tickers_regex
# ---------------------------------------------------------------------------


class TestExtractTickersRegex:
    def test_single_ticker(self):
        assert _extract_tickers_regex("Analyse AAPL") == ["AAPL"]

    def test_multiple_tickers(self):
        result = _extract_tickers_regex("Compare AAPL and MSFT")
        assert "AAPL" in result
        assert "MSFT" in result

    def test_excludes_common_words(self):
        result = _extract_tickers_regex("IS THE AND OR BUT")
        assert result == []

    def test_excludes_single_letters(self):
        result = _extract_tickers_regex("I think A is good")
        assert "I" not in result
        assert "A" not in result

    def test_empty_message(self):
        assert _extract_tickers_regex("") == []

    def test_no_caps(self):
        assert _extract_tickers_regex("what is a p/e ratio?") == []


# ---------------------------------------------------------------------------
# classify() — mocked LLM
# ---------------------------------------------------------------------------


class TestClassify:
    @pytest.mark.asyncio
    async def test_financial_analysis_with_ticker(self):
        llm = _make_subllm("financial_analysis", ["AAPL"])
        intent, tickers = await classify("Analyse AAPL", llm)
        assert intent == "financial_analysis"
        assert tickers == ["AAPL"]

    @pytest.mark.asyncio
    async def test_financial_analysis_multi_ticker(self):
        llm = _make_subllm("financial_analysis", ["AAPL", "MSFT"])
        intent, tickers = await classify("Compare AAPL and MSFT", llm)
        assert intent == "financial_analysis"
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    @pytest.mark.asyncio
    async def test_financial_question(self):
        llm = _make_subllm("financial_question", [])
        intent, tickers = await classify("What is a P/E ratio?", llm)
        assert intent == "financial_question"
        assert tickers == []

    @pytest.mark.asyncio
    async def test_off_topic(self):
        llm = _make_subllm("off_topic", [])
        intent, tickers = await classify("What is the weather today?", llm)
        assert intent == "off_topic"
        assert tickers == []

    @pytest.mark.asyncio
    async def test_clarification_needed(self):
        llm = _make_subllm("clarification_needed", [])
        intent, tickers = await classify("Tell me about that thing", llm)
        assert intent == "clarification_needed"

    @pytest.mark.asyncio
    async def test_invalid_intent_falls_back_to_financial_question(self):
        """An unexpected intent string from the LLM should fall back gracefully."""
        response = MagicMock()
        response.content = json.dumps({"intent": "unknown_intent_xyz", "tickers": []})
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=response)

        intent, tickers = await classify("some message", llm)
        assert intent == "financial_question"
        assert tickers == []

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_gracefully(self):
        """If the LLM raises an exception, classify() falls back without crashing."""
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("Network error"))

        intent, tickers = await classify("Analyse TSLA", llm)
        assert intent == "financial_question"  # safe fallback
        assert tickers == []

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back(self):
        """If the LLM returns non-JSON, classify() falls back gracefully."""
        response = MagicMock()
        response.content = "I cannot classify this message."
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=response)

        intent, tickers = await classify("something", llm)
        assert intent == "financial_question"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        """LLM wrapping JSON in ```json fences should still be parsed."""
        payload = '```json\n{"intent": "financial_analysis", "tickers": ["NVDA"]}\n```'
        response = MagicMock()
        response.content = payload
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=response)

        intent, tickers = await classify("Analyse NVDA", llm)
        assert intent == "financial_analysis"
        assert tickers == ["NVDA"]

    @pytest.mark.asyncio
    async def test_regex_fallback_when_tickers_missing(self):
        """If LLM returns financial_analysis but empty tickers, regex should fill in."""
        llm = _make_subllm("financial_analysis", [])
        intent, tickers = await classify("Tell me about TSLA today", llm)
        assert intent == "financial_analysis"
        assert "TSLA" in tickers
