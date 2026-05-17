"""Integration tests for ConversationalAgent.

Tests use the DI constructor (injecting mock LLMRegistry + mock MemoryManager)
instead of patching module-level imports. This is the canonical test pattern
for the refactored ConversationalAgent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_financial_analyst.agents.conversational_agent import ConversationalAgent
from ai_financial_analyst.core.conversation_state import new_session


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """ConversationalAgent with fully injected mocks — no real LLMs or DB."""
    # Mock LLMRegistry so no Gemini API calls are made
    mock_registry = MagicMock()
    mock_primary  = MagicMock()
    mock_subllm   = MagicMock()
    mock_registry.get_primary_with_fallback.return_value = mock_primary
    mock_registry.get_subllm.return_value               = mock_subllm
    mock_registry._budget                               = MagicMock()

    # Mock MemoryManager so no SQLite is touched
    mock_memory = MagicMock()
    mock_memory.maybe_extract_preferences   = AsyncMock()
    mock_memory.maybe_save_analysis_summary = AsyncMock()
    mock_memory.build_memory_context        = AsyncMock(return_value="")

    # Inject via DI constructor — no module patching needed
    with patch("ai_financial_analyst.agents.conversational_agent.ManagerAgent") as mock_mgr_cls:
        mock_manager_instance = MagicMock()
        mock_manager_instance.run = AsyncMock(return_value="default response")
        mock_mgr_cls.return_value = mock_manager_instance

        a = ConversationalAgent(
            user_id      = "test-user",
            llm_registry = mock_registry,
            memory       = mock_memory,
        )
    return a


def _setup_manager(agent, response: str) -> None:
    """Configure the agent's Manager mock to return a given response string."""
    agent._manager.run = AsyncMock(return_value=response)


# ── Core delegation ───────────────────────────────────────────────────────────

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
        _setup_manager(agent, "Response.")
        state = new_session()

        await agent.process_message("Hello", state)

        call_kwargs = agent._manager.run.call_args.kwargs
        assert call_kwargs["message"] == "Hello"
        assert "session_id" in call_kwargs["state"]

    @pytest.mark.asyncio
    async def test_manager_error_propagates(self, agent):
        agent._manager.run = AsyncMock(side_effect=RuntimeError("LLM error"))
        state = new_session()

        with pytest.raises(RuntimeError, match="LLM error"):
            await agent.process_message("Analyse AAPL", state)


# ── State management ──────────────────────────────────────────────────────────

class TestStateManagement:
    @pytest.mark.asyncio
    async def test_messages_appended_after_turn(self, agent):
        _setup_manager(agent, "Here is the analysis.")
        state = new_session()

        _, new_state = await agent.process_message("Analyse AAPL", state)

        messages = new_state.get("messages", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_session_id_preserved_across_turns(self, agent):
        _setup_manager(agent, "Response.")
        state    = new_session()
        orig_sid = state["session_id"]

        _, state = await agent.process_message("Message 1", state)
        _, state = await agent.process_message("Message 2", state)

        assert state["session_id"] == orig_sid

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate_messages(self, agent):
        _setup_manager(agent, "OK.")
        state = new_session()

        for _ in range(3):
            _, state = await agent.process_message("Question", state)

        assert len(state.get("messages", [])) == 6  # 3 user + 3 assistant


# ── Preference extraction ─────────────────────────────────────────────────────

class TestPreferenceExtraction:
    @pytest.mark.asyncio
    async def test_preference_extraction_called_on_each_message(self, agent):
        _setup_manager(agent, "Response.")
        state = new_session()

        await agent.process_message("I prefer conservative analysis", state)

        agent._memory.maybe_extract_preferences.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_preference_extraction_failure_does_not_crash(self, agent):
        _setup_manager(agent, "Response.")
        agent._memory.maybe_extract_preferences = AsyncMock(side_effect=Exception("DB error"))
        state = new_session()

        response, _ = await agent.process_message("Hello", state)
        # Should still return a response despite extraction failure
        assert response == "Response."


# ── Analysis state ────────────────────────────────────────────────────────────

class TestAnalysisState:
    @pytest.mark.asyncio
    async def test_last_analysis_state_is_none_by_default(self, agent):
        assert agent.last_analysis_state is None

    @pytest.mark.asyncio
    async def test_last_analysis_state_set_after_analysis(self, agent):
        fake_state = {
            "tickers": ["AAPL"],
            "report_markdown": "# AAPL Report\nThis is not financial advice.",
            "run_id": "test-run",
        }
        _setup_manager(agent, "Here is the report.")
        agent.last_analysis_state = fake_state
        state = new_session()

        await agent.process_message("Analyse AAPL", state)

        agent._memory.maybe_save_analysis_summary.assert_awaited_once()


# ── Factory method ────────────────────────────────────────────────────────────

class TestCreateFactory:
    def test_create_returns_conversational_agent(self):
        """ConversationalAgent.create() should return a ConversationalAgent instance."""
        with (
            patch("ai_financial_analyst.agents.conversational_agent.LLMRegistry") as mock_reg_cls,
            patch("ai_financial_analyst.agents.conversational_agent.LongTermMemory"),
            patch("ai_financial_analyst.agents.conversational_agent.MemoryManager"),
            patch("ai_financial_analyst.agents.conversational_agent.ManagerAgent"),
        ):
            mock_reg = MagicMock()
            mock_reg._budget = MagicMock()
            mock_reg.get_primary_with_fallback.return_value = MagicMock()
            mock_reg.get_subllm.return_value                = MagicMock()
            mock_reg_cls.return_value = mock_reg

            agent = ConversationalAgent.create(user_id="test-user")

        assert isinstance(agent, ConversationalAgent)
        assert agent.user_id == "test-user"

    def test_di_constructor_accepts_injected_deps(self):
        """Injected dependencies should be stored without creating defaults."""
        mock_registry = MagicMock()
        mock_primary  = MagicMock()
        mock_subllm   = MagicMock()
        mock_registry.get_primary_with_fallback.return_value = mock_primary
        mock_registry.get_subllm.return_value               = mock_subllm
        mock_registry._budget                               = MagicMock()

        mock_memory = MagicMock()

        with patch("ai_financial_analyst.agents.conversational_agent.ManagerAgent"):
            agent = ConversationalAgent(
                user_id      = "user-42",
                llm_registry = mock_registry,
                memory       = mock_memory,
            )

        assert agent.user_id   == "user-42"
        assert agent._memory   is mock_memory
        assert agent._registry is mock_registry
