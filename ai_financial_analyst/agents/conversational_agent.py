"""ConversationalAgent — session-scoped wrapper around the Manager LLM orchestrator.

The Manager (agents/manager.py) uses tool-use / function-calling to autonomously
decide which capability to invoke for any user message.  This class initialises
the session resources (LLMs, memory, budget tracking) and exposes a single
`process_message()` entry point used by the FastAPI chat endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from dotenv import load_dotenv

from ..core.budget_tracker import RequestBudgetTracker
from ..core.conversation_state import (
    ConversationState,
    append_messages,
    new_session,
)
from ..core.llm import get_primary_llm_with_fallback, get_subllm
from ..memory.long_term import LongTermMemory
from ..memory.memory_manager import MemoryManager
from .manager import ManagerAgent

load_dotenv()

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Session-scoped agent — one instance per user, cached by session_manager.

    Holds LLMs, memory, budget tracker, and the Manager orchestrator.
    The Manager replaces the old hardcoded intent classifier; it uses
    LangChain tool-use (function calling) to route requests dynamically.
    """

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        self.budget = RequestBudgetTracker()
        self._primary_llm = get_primary_llm_with_fallback(budget_tracker=self.budget)
        self._subllm = get_subllm(budget_tracker=self.budget)
        self._memory = MemoryManager(LongTermMemory(user_id=user_id), subllm=self._subllm)
        self._manager = ManagerAgent(primary_llm=self._primary_llm, agent=self)

        # Set by Manager tools to expose pipeline state for chart generation / export
        self.last_analysis_state: dict | None = None
        # Set by Manager.run() before each turn so edit_report_section can find the right DB row
        self._current_session_id: str = ""
        self._current_conversation_id: str | None = None

    async def process_message(
        self,
        message: str,
        state: ConversationState,
        step_callback: Callable[[dict], None] | None = None,
        conversation_id: str | None = None,
    ) -> tuple[str, ConversationState]:
        """Process one user turn via the Manager LLM and return (response, updated_state).

        Args:
            message: Raw user message text.
            state: Current conversation state (history, session_id, etc.)
            step_callback: Called by pipeline tools after each tool invocation.
            conversation_id: DB conversation ID (for refinement to find stored report).

        Returns:
            (response_text, new_conversation_state)
        """
        # Best-effort preference extraction (non-blocking)
        try:
            await self._memory.maybe_extract_preferences(message)
        except Exception as exc:
            logger.debug("Preference extraction skipped: %s", exc)

        # Delegate to Manager LLM orchestrator
        response = await self._manager.run(
            message=message,
            state=state,
            step_callback=step_callback,
            conversation_id=conversation_id,
        )

        # Persist a summary if the Manager ran a financial analysis
        if self.last_analysis_state:
            final_state = self.last_analysis_state
            tickers = final_state.get("tickers", [])
            report = final_state.get("report_markdown", "")
            if tickers and report:
                try:
                    await self._memory.maybe_save_analysis_summary(
                        session_id=state.get("session_id", ""),
                        tickers=tickers,
                        report_markdown=report,
                        run_id=final_state.get("run_id", ""),
                    )
                except Exception as exc:
                    logger.debug("Analysis summary save skipped: %s", exc)

        new_state = append_messages(
            state,
            user_content=message,
            assistant_content=response,
            intent=None,    # Manager doesn't expose a single intent string
            tickers=None,
        )
        return response, new_state
