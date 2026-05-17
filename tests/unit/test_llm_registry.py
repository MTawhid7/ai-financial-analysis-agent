"""Unit tests for the LLM layer refactor: LLMRegistry, CircuitBreaker, per-instance isolation."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_financial_analyst.core.llm import (
    CircuitBreaker,
    CircuitBreakerError,
    LLMRegistry,
    RateLimitFallbackLLM,
)


# ── CircuitBreaker (per-instance) ─────────────────────────────────────────────

class TestCircuitBreakerPerInstance:
    def test_two_instances_are_independent(self):
        """Tripping one instance does not affect another."""
        cb1 = CircuitBreaker(max_failures=2, window_s=60)
        cb2 = CircuitBreaker(max_failures=2, window_s=60)

        for _ in range(2):
            try:
                cb1.record_failure()
            except CircuitBreakerError:
                pass

        assert cb1.is_open() is True
        assert cb2.is_open() is False  # independent

    def test_from_settings_returns_new_instance(self):
        cb1 = CircuitBreaker.from_settings()
        cb2 = CircuitBreaker.from_settings()
        assert cb1 is not cb2   # not the same object

    def test_reset_clears_state(self):
        cb = CircuitBreaker(max_failures=2, window_s=60)
        for _ in range(2):
            try:
                cb.record_failure()
            except CircuitBreakerError:
                pass
        assert cb.is_open() is True
        cb.reset()
        assert cb.is_open() is False


# ── LLMRegistry ───────────────────────────────────────────────────────────────

class TestLLMRegistry:
    def test_registry_creates_primary_with_fallback(self):
        registry = LLMRegistry()
        # Should return a RateLimitFallbackLLM (with no budget tracker, no real API call)
        llm = registry.get_primary_with_fallback()
        assert isinstance(llm, RateLimitFallbackLLM)

    def test_registry_caches_primary_instance(self):
        registry = LLMRegistry()
        llm1 = registry.get_primary_with_fallback()
        llm2 = registry.get_primary_with_fallback()
        assert llm1 is llm2  # same object on second call

    def test_registry_caches_subllm_instance(self):
        registry = LLMRegistry()
        sub1 = registry.get_subllm()
        sub2 = registry.get_subllm()
        assert sub1 is sub2

    def test_two_registries_have_independent_circuit_breakers(self):
        r1 = LLMRegistry()
        r2 = LLMRegistry()
        assert r1._primary_cb is not r2._primary_cb

    def test_reset_circuit_breakers_invalidates_cache(self):
        registry = LLMRegistry()
        llm1 = registry.get_primary_with_fallback()
        registry.reset_circuit_breakers()
        llm2 = registry.get_primary_with_fallback()
        assert llm1 is not llm2   # new instance after reset

    def test_reset_restores_closed_state(self):
        registry = LLMRegistry()
        cb = registry._primary_cb
        for _ in range(cb.max_failures):
            try:
                cb.record_failure()
            except CircuitBreakerError:
                pass
        assert cb.is_open() is True
        registry.reset_circuit_breakers()
        assert registry._primary_cb.is_open() is False


# ── RateLimitFallbackLLM circuit breaker injection ────────────────────────────

class TestRateLimitFallbackLLMWithInjectedBreaker:
    @pytest.mark.asyncio
    async def test_open_circuit_breaker_routes_to_fallback(self):
        primary  = MagicMock()
        fallback = MagicMock()
        fallback.ainvoke = AsyncMock(return_value="fallback_result")

        cb = CircuitBreaker(max_failures=1, window_s=60)
        # Trip the breaker
        try:
            cb.record_failure()
        except CircuitBreakerError:
            pass
        assert cb.is_open()

        llm    = RateLimitFallbackLLM(primary, fallback, circuit_breaker=cb)
        result = await llm.ainvoke("input")

        assert result == "fallback_result"
        primary.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_closed_breaker_uses_primary(self):
        primary  = MagicMock()
        primary.ainvoke = AsyncMock(return_value="primary_result")
        fallback = MagicMock()

        cb  = CircuitBreaker(max_failures=5, window_s=60)   # not tripped
        llm = RateLimitFallbackLLM(primary, fallback, circuit_breaker=cb)

        result = await llm.ainvoke("input")

        assert result == "primary_result"
        fallback.ainvoke.assert_not_called()

    def test_bind_tools_preserves_circuit_breaker(self):
        primary  = MagicMock()
        fallback = MagicMock()
        primary.bind_tools.return_value  = MagicMock()
        fallback.bind_tools.return_value = MagicMock()

        cb  = CircuitBreaker(max_failures=3, window_s=30)
        llm = RateLimitFallbackLLM(primary, fallback, circuit_breaker=cb)

        bound = llm.bind_tools([])
        assert isinstance(bound, RateLimitFallbackLLM)
        assert bound._cb is cb   # same circuit breaker propagated
