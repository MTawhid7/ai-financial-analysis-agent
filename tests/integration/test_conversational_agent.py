"""Integration tests for ConversationalAgent.

The ConversationalAgent now delegates all routing to the Manager LLM
(which uses tool-use / function-calling).  These tests verify:
- process_message correctly delegates to Manager.run()
- State (messages, session_id) is updated correctly after each turn
- Preference extraction is attempted on each message
- last_analysis_state is exposed after a financial analysis
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_financial_analyst.core.conversation_state import new_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    """Return a ConversationalAgent with mocked LLMs and Manager."""
    with (
        patch("ai_financial_analyst.agents.conversational_agent.get_primary_llm_with_fallback") as mock_primary,
        patch("ai_financial_analyst.agents.conversational_agent.get_subllm") as mock_sub,
        patch("ai_financial_analyst.agents.conversational_agent.ManagerAgent") as mock_manager_cls,
    ):
        mock_primary.return_value = MagicMock()
        mock_sub.return_value = MagicMock()
        mock_manager_cls.return_value = MagicMock()
        from ai_financial_analyst.agents.conversational_agent import ConversationalAgent
        a = ConversationalAgent()
    return a


def _setup_manager(agent, response: str) -> None:
    """Configure the agent's Manager mock to return a given response string."""
    agent._manager.run = AsyncMock(return_value=response)


# ---------------------------------------------------------------------------
# Core delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    @pytest.mark.asyncio
    async def test_process_message_delegates_to_manager(self, agent):
        _setup_manager(agent, "The analysis for AAPL shows strong growth.")
        state = new_session()

        response, _ = await agent.process_message("Analyse AAPL", state)

        agent._manager.run.assert_awaited_once()
        assert "AAPL" in response or "growth" in response.lower()

    @pytest.mark.asyncio
    async def test_manager_receives_message_and_state(self, agent):
        _setup_manager(agent, "Here is the analysis.")
        state = new_session()

        await agent.process_message("Compare AAPL vs MSFT", state)

        call_kwargs = agent._manager.run.call_args
        assert call_kwargs is not None
        # First positional arg is the message
        args, kwargs = call_kwargs
        message_arg = args[0] if args else kwargs.get("message", "")
        assert "AAPL" in message_arg or "MSFT" in message_arg

    @pytest.mark.asyncio
    async def test_manager_error_propagates(self, agent):
        agent._manager.run = AsyncMock(side_effect=RuntimeError("LLM failed"))
        state = new_session()

        with pytest.raises(RuntimeError, match="LLM failed"):
            await agent.process_message("Analyse AAPL", state)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    @pytest.mark.asyncio
    async def test_messages_appended_after_turn(self, agent):
        _setup_manager(agent, "Here is your answer.")
        state = new_session()

        _, new_state = await agent.process_message("Hello", state)

        messages = new_state.get("messages", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Here is your answer."

    @pytest.mark.asyncio
    async def test_session_id_preserved_across_turns(self, agent):
        _setup_manager(agent, "Answer.")
        state = new_session()
        original_id = state["session_id"]

        _, new_state = await agent.process_message("Hello", state)

        assert new_state["session_id"] == original_id

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate_messages(self, agent):
        _setup_manager(agent, "First answer.")
        state = new_session()

        _, state = await agent.process_message("Turn 1", state)
        _setup_manager(agent, "Second answer.")
        _, state = await agent.process_message("Turn 2", state)

        assert len(state["messages"]) == 4  # 2 turns × (user + assistant)


# ---------------------------------------------------------------------------
# Preference extraction
# ---------------------------------------------------------------------------


class TestPreferenceExtraction:
    @pytest.mark.asyncio
    async def test_preference_extraction_called_on_each_message(self, agent):
        _setup_manager(agent, "OK.")
        agent._memory.maybe_extract_preferences = AsyncMock()
        state = new_session()

        await agent.process_message("I prefer conservative analysis", state)

        agent._memory.maybe_extract_preferences.assert_awaited_once_with(
            "I prefer conservative analysis"
        )

    @pytest.mark.asyncio
    async def test_preference_extraction_failure_does_not_crash(self, agent):
        _setup_manager(agent, "OK.")
        agent._memory.maybe_extract_preferences = AsyncMock(side_effect=RuntimeError("Network error"))
        state = new_session()

        # Should not raise — preference extraction is best-effort
        response, _ = await agent.process_message("I prefer conservative analysis", state)
        assert response == "OK."


# ---------------------------------------------------------------------------
# Analysis state exposure
# ---------------------------------------------------------------------------


class TestAnalysisState:
    @pytest.mark.asyncio
    async def test_last_analysis_state_is_none_by_default(self, agent):
        assert agent.last_analysis_state is None

    @pytest.mark.asyncio
    async def test_last_analysis_state_set_after_analysis(self, agent):
        fake_state = {"report_markdown": "# Report", "tickers": ["AAPL"], "errors": [], "run_id": "test"}
        _setup_manager(agent, "Analysis complete.")
        agent.last_analysis_state = fake_state  # Manager tool would set this
        state = new_session()

        # With last_analysis_state pre-set, process_message should attempt summary save
        agent._memory.maybe_save_analysis_summary = AsyncMock()
        await agent.process_message("Follow-up question", state)

        # Summary save is attempted when last_analysis_state is set
        agent._memory.maybe_save_analysis_summary.assert_awaited()
