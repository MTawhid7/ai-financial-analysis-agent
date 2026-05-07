"""Integration tests for ConversationalAgent.

The inner pipeline (run_pipeline) and all LLM calls are mocked.
Tests verify routing, state management, and error-handling behaviour.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_financial_analyst.core.conversation_state import new_session
from ai_financial_analyst.agents.conversational_agent import (
    ConversationalAgent,
    _REJECTION_RESPONSE,
    _CLARIFICATION_RESPONSE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    """Return a ConversationalAgent with mocked LLM instances."""
    with (
        patch("ai_financial_analyst.agents.conversational_agent.get_primary_llm_with_fallback") as mock_primary,
        patch("ai_financial_analyst.agents.conversational_agent.get_subllm") as mock_sub,
    ):
        mock_primary.return_value = MagicMock()
        mock_sub.return_value = MagicMock()
        a = ConversationalAgent()
    return a


def _mock_classify(intent: str, tickers: list[str]):
    """Return a coroutine that resolves to (intent, tickers)."""
    async def _fn(*args, **kwargs):
        return intent, tickers
    return _fn


def _mock_pipeline(report: str = "# Report\n\nSample report.") -> AsyncMock:
    """Return a mock run_pipeline that returns a state with report_markdown."""
    final_state = {"report_markdown": report, "errors": [], "status": "COMPLETE"}
    mock = AsyncMock(return_value=(final_state, "/tmp/trace.json", "/tmp/artifacts.json"))
    return mock


def _mock_llm_response(text: str) -> MagicMock:
    """Return a mock LLM whose ainvoke returns a response with the given text."""
    response = MagicMock()
    response.content = text
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


# ---------------------------------------------------------------------------
# Financial analysis routing
# ---------------------------------------------------------------------------


class TestFinancialAnalysisRouting:
    @pytest.mark.asyncio
    async def test_calls_pipeline_on_financial_analysis_intent(self, agent):
        state = new_session()
        mock_pipeline = _mock_pipeline()

        with (
            patch(
                "ai_financial_analyst.agents.conversational_agent.classify",
                side_effect=_mock_classify("financial_analysis", ["AAPL"]),
            ),
            patch(
                "ai_financial_analyst.agents.conversational_agent.run_pipeline",
                mock_pipeline,
            ),
        ):
            response, new_state = await agent.process_message("Analyse AAPL", state)

        assert new_state["current_intent"] == "financial_analysis"
        assert new_state["pending_tickers"] == ["AAPL"]

    @pytest.mark.asyncio
    async def test_pipeline_report_included_in_response(self, agent):
        state = new_session()
        report_text = "# AAPL Analysis\n\nMarket cap: $3T"
        mock_pipeline = _mock_pipeline(report=report_text)

        with patch(
            "ai_financial_analyst.agents.conversational_agent.run_pipeline",
            mock_pipeline,
        ):
            response = await agent._handle_financial_analysis(
                "Analyse AAPL", ["AAPL"], state, None
            )
            assert report_text in response

    @pytest.mark.asyncio
    async def test_no_tickers_prompts_for_clarification(self, agent):
        state = new_session()
        pipeline_called = []

        async def _never_called(*a, **kw):
            pipeline_called.append(True)
            return ({}, "", "")

        with patch(
            "ai_financial_analyst.agents.conversational_agent.run_pipeline",
            _never_called,
        ):
            response = await agent._handle_financial_analysis(
                "Tell me about stuff", [], state, None
            )

        assert not pipeline_called, "Pipeline should not be called with no tickers"
        assert "specify" in response.lower() or "which stock" in response.lower()

    @pytest.mark.asyncio
    async def test_pipeline_exception_returns_error_message(self, agent):
        state = new_session()

        async def _failing_pipeline(*a, **kw):
            raise RuntimeError("Simulated API failure")

        with patch(
            "ai_financial_analyst.agents.conversational_agent.run_pipeline",
            _failing_pipeline,
        ):
            response = await agent._handle_financial_analysis(
                "Analyse TSLA", ["TSLA"], state, None
            )
        assert "error" in response.lower()
        assert "TSLA" in response


# ---------------------------------------------------------------------------
# Off-topic and clarification routing
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.asyncio
    async def test_off_topic_returns_rejection(self, agent):
        state = new_session()
        with patch(
            "ai_financial_analyst.agents.conversational_agent.classify",
            side_effect=_mock_classify("off_topic", []),
        ):
            response, new_state = await agent.process_message("What's the weather?", state)

        assert response == _REJECTION_RESPONSE
        assert new_state["current_intent"] == "off_topic"

    @pytest.mark.asyncio
    async def test_clarification_needed_returns_clarification(self, agent):
        state = new_session()
        with patch(
            "ai_financial_analyst.agents.conversational_agent.classify",
            side_effect=_mock_classify("clarification_needed", []),
        ):
            response, new_state = await agent.process_message("Tell me about that", state)

        assert response == _CLARIFICATION_RESPONSE
        assert new_state["current_intent"] == "clarification_needed"


# ---------------------------------------------------------------------------
# Memory query routing
# ---------------------------------------------------------------------------


class TestMemoryQueryRouting:
    @pytest.mark.asyncio
    async def test_memory_query_does_not_call_pipeline(self, agent):
        """memory_query should return stored summaries, never invoke run_pipeline."""
        state = new_session()
        pipeline_called = []

        async def _pipeline_spy(*a, **kw):
            pipeline_called.append(True)
            return ({}, "", "")

        with (
            patch(
                "ai_financial_analyst.agents.conversational_agent.classify",
                side_effect=_mock_classify("memory_query", ["AAPL"]),
            ),
            patch(
                "ai_financial_analyst.agents.conversational_agent.run_pipeline",
                _pipeline_spy,
            ),
        ):
            response, new_state = await agent.process_message(
                "What did we find about AAPL earlier?", state
            )

        assert not pipeline_called, "Pipeline must NOT be called for memory_query"
        assert new_state["current_intent"] == "memory_query"

    @pytest.mark.asyncio
    async def test_memory_query_no_stored_summaries_returns_helpful_message(self, agent):
        state = new_session()
        # Mock _lt.search_summaries to return nothing
        agent._memory._lt.search_summaries = AsyncMock(return_value=[])

        with patch(
            "ai_financial_analyst.agents.conversational_agent.classify",
            side_effect=_mock_classify("memory_query", ["AAPL"]),
        ):
            response, _ = await agent.process_message(
                "What did we find about AAPL earlier?", state
            )

        assert "don't have" in response.lower() or "stored" in response.lower()

    @pytest.mark.asyncio
    async def test_memory_query_with_stored_summary_returns_it(self, agent):
        state = new_session()
        agent._memory._lt.search_summaries = AsyncMock(return_value=[
            {"tickers": "AAPL", "summary": "Apple had 17% CAGR, P/E 34x.", "created_at": 0}
        ])

        with patch(
            "ai_financial_analyst.agents.conversational_agent.classify",
            side_effect=_mock_classify("memory_query", ["AAPL"]),
        ):
            response, _ = await agent.process_message(
                "What did we find about AAPL earlier?", state
            )

        assert "17% CAGR" in response or "AAPL" in response


# ---------------------------------------------------------------------------
# Financial question answering
# ---------------------------------------------------------------------------


class TestFinancialQuestion:
    @pytest.mark.asyncio
    async def test_financial_question_calls_primary_llm(self, agent):
        state = new_session()
        llm_answer = "A P/E ratio is the price-to-earnings ratio."
        agent._primary_llm = _mock_llm_response(llm_answer)

        response = await agent._handle_financial_question("What is a P/E ratio?", state)
        assert llm_answer in response

    @pytest.mark.asyncio
    async def test_llm_failure_returns_error_message(self, agent):
        state = new_session()
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        agent._primary_llm = llm

        response = await agent._handle_financial_question("What is ROE?", state)
        assert "error" in response.lower()

    @pytest.mark.asyncio
    async def test_conversation_history_passed_to_llm(self, agent):
        """Verify that recent conversation history is included in the LLM call."""
        from ai_financial_analyst.core.conversation_state import append_messages

        state = new_session()
        state = append_messages(state, "Previous user turn", "Previous assistant turn")

        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            response = MagicMock()
            response.content = "Answer"
            return response

        agent._primary_llm = MagicMock()
        agent._primary_llm.ainvoke = _capture_invoke

        await agent._handle_financial_question("Follow-up question", state)

        # Should include system prompt + 2 history messages + current question
        assert len(captured_messages) >= 3
        contents = [m.content for m in captured_messages]
        assert any("Previous user turn" in c for c in contents)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    @pytest.mark.asyncio
    async def test_messages_appended_after_turn(self, agent):
        state = new_session()
        with patch(
            "ai_financial_analyst.agents.conversational_agent.classify",
            side_effect=_mock_classify("off_topic", []),
        ):
            _, new_state = await agent.process_message("Random message", state)

        messages = new_state.get("messages", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[0]["content"] == "Random message"

    @pytest.mark.asyncio
    async def test_session_id_preserved_across_turns(self, agent):
        state = new_session()
        original_id = state["session_id"]

        with patch(
            "ai_financial_analyst.agents.conversational_agent.classify",
            side_effect=_mock_classify("off_topic", []),
        ):
            _, new_state = await agent.process_message("Hello", state)

        assert new_state["session_id"] == original_id
