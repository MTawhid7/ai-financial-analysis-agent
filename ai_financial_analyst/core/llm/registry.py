"""LLMRegistry — factory that creates configured LLM instances with injected deps.

Usage:
    registry = LLMRegistry(budget_tracker=budget)
    primary  = registry.get_primary_with_fallback()   # RateLimitFallbackLLM
    subllm   = registry.get_subllm()                  # plain Gemini Flash-Lite

Each registry owns its own circuit breaker — fully isolated between sessions and tests.
"""

from __future__ import annotations

import logging
from typing import Any

from .circuit_breaker import CircuitBreaker
from .gemini import RateLimitFallbackLLM, _make_primary_llm, _make_subllm

logger = logging.getLogger(__name__)


class LLMRegistry:
    """Creates and caches configured LLM instances for a single session.

    Pass a `budget_tracker` to track API call counts; omit for testing.
    Each registry gets its own circuit breaker — call counts and trip states
    are never shared between registries.
    """

    def __init__(self, budget_tracker: Any = None) -> None:
        self._budget = budget_tracker
        self._primary_cb  = CircuitBreaker.from_settings()
        self._fallback_cb = CircuitBreaker.from_settings()
        self._primary_with_fallback: RateLimitFallbackLLM | None = None
        self._subllm: Any | None = None

    def get_primary_with_fallback(self) -> RateLimitFallbackLLM:
        """Return a Flash LLM that automatically falls back to Flash-Lite on rate limits.

        Lazily creates the LLM on first call; subsequent calls return the cached instance.
        """
        if self._primary_with_fallback is None:
            primary  = _make_primary_llm(
                budget_tracker   = self._budget,
                circuit_breaker  = self._primary_cb,
            )
            fallback = _make_subllm(
                budget_tracker   = self._budget,
                circuit_breaker  = self._fallback_cb,
            )
            self._primary_with_fallback = RateLimitFallbackLLM(
                primary         = primary,
                fallback        = fallback,
                budget_tracker  = self._budget,
                circuit_breaker = self._primary_cb,
            )
            logger.debug("LLMRegistry: created primary-with-fallback LLM")
        return self._primary_with_fallback

    def get_subllm(self) -> Any:
        """Return Gemini Flash-Lite LLM for summarisation and sub-task calls.

        Lazily created and cached per registry.
        """
        if self._subllm is None:
            self._subllm = _make_subllm(
                budget_tracker  = self._budget,
                circuit_breaker = self._fallback_cb,
            )
            logger.debug("LLMRegistry: created sub-LLM")
        return self._subllm

    def reset_circuit_breakers(self) -> None:
        """Reset both circuit breakers to CLOSED state (useful in tests)."""
        self._primary_cb.reset()
        self._fallback_cb.reset()
        # Invalidate cached instances so they're recreated with fresh breakers
        self._primary_with_fallback = None
        self._subllm                = None
