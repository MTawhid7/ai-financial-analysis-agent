"""ConversationalAgent — session-scoped wrapper around the Manager LLM orchestrator.

Design (Phase 4 DI refactor):
- All dependencies are injected via constructor or the `create()` factory.
- No direct calls to module-level LLM factories inside __init__.
- `create(user_id, cfg)` is the canonical entry point for production use.
- `ConversationalAgent(user_id, llm_registry, memory)` is the injectable form for tests.
- Backward compat: `ConversationalAgent(user_id)` still works (creates defaults).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..config import settings as _default_settings
from ..config.settings import Settings
from ..core.budget_tracker import RequestBudgetTracker
from ..core.conversation_state import (
    ConversationState,
    append_messages,
    new_session,
)
from ..core.llm import LLMRegistry, RateLimitFallbackLLM, get_subllm
from ..memory.long_term import LongTermMemory
from ..memory.memory_manager import MemoryManager
from ..memory.protocol import MemoryBackend
from .manager import ManagerAgent

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Session-scoped agent — one instance per user, cached by session_manager.

    Dependencies (all injected):
        llm_registry  — provides primary + fallback LLM pair with circuit breaker
        memory        — MemoryManager wrapping any MemoryBackend implementation

    Use `ConversationalAgent.create(user_id)` in production.
    Use `ConversationalAgent(user_id, llm_registry=mock, memory=mock)` in tests.
    """

    def __init__(
        self,
        user_id:      str = "default",
        *,
        llm_registry: LLMRegistry | None = None,
        memory:       MemoryManager | None = None,
    ) -> None:
        self.user_id = user_id

        # Resolve dependencies — inject if provided, otherwise create defaults.
        # Tests inject mocks; production uses the `create()` factory.
        if llm_registry is not None:
            self._registry = llm_registry
        else:
            budget        = RequestBudgetTracker()
            self._registry = LLMRegistry(budget_tracker=budget)

        self.budget       = self._registry._budget
        self._primary_llm = self._registry.get_primary_with_fallback()
        self._subllm      = self._registry.get_subllm()

        if memory is not None:
            self._memory = memory
        else:
            self._memory = MemoryManager(
                LongTermMemory(user_id=user_id),
                subllm=self._subllm,
            )

        self._manager = ManagerAgent(primary_llm=self._primary_llm, agent=self)

        # Shared mutable state set by Manager tools — exposed as public attributes
        # so tools can read/write without breaking the interface.
        self.last_analysis_state: dict | None = None
        self._current_session_id:       str = ""
        self._current_conversation_id:  str | None = None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, user_id: str, cfg: Settings | None = None) -> "ConversationalAgent":
        """Production factory: creates a fully wired agent from settings.

        Args:
            user_id: The authenticated user's ID (scopes all memory queries).
            cfg:     Settings instance (defaults to the module-level singleton).

        This is the only place that instantiates concrete dependencies from
        configuration — business-logic code never calls os.environ directly.
        """
        cfg    = cfg or _default_settings
        budget = RequestBudgetTracker(daily_budget=cfg.llm_daily_budget)
        reg    = LLMRegistry(budget_tracker=budget)
        mem    = MemoryManager(
            LongTermMemory(user_id=user_id, db_path=cfg.memory_db_path),
            subllm=reg.get_subllm(),
        )
        return cls(user_id=user_id, llm_registry=reg, memory=mem)

    # ── Main entry point ──────────────────────────────────────────────────────

    async def process_message(
        self,
        message:         str,
        state:           ConversationState,
        step_callback:   Callable[[dict], None] | None = None,
        conversation_id: str | None = None,
    ) -> tuple[str, ConversationState]:
        """Process one user turn via the Manager LLM and return (response, updated_state)."""
        try:
            await self._memory.maybe_extract_preferences(message)
        except Exception as exc:
            logger.debug("Preference extraction skipped: %s", exc)

        response = await self._manager.run(
            message         = message,
            state           = state,
            step_callback   = step_callback,
            conversation_id = conversation_id,
        )

        # Persist a summary if the Manager ran a financial analysis
        if self.last_analysis_state:
            final_state = self.last_analysis_state
            tickers     = final_state.get("tickers", [])
            report      = final_state.get("report_markdown", "")
            if tickers and report:
                try:
                    await self._memory.maybe_save_analysis_summary(
                        session_id    = state.get("session_id", ""),
                        tickers       = tickers,
                        report_markdown = report,
                        run_id        = final_state.get("run_id", ""),
                    )
                except Exception as exc:
                    logger.debug("Analysis summary save skipped: %s", exc)

        new_state = append_messages(
            state,
            user_content      = message,
            assistant_content = response,
            intent            = None,
            tickers           = None,
        )
        return response, new_state
