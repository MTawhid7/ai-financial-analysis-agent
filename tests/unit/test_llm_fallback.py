"""Unit tests for RateLimitFallbackLLM and _CircuitBreaker.

Verifies that the fallback wrapper:
  - Delegates to the primary on success
  - Falls back to Flash-Lite on CircuitBreakerError
  - Falls back to Flash-Lite on tenacity.RetryError
  - Does NOT catch non-rate-limit exceptions
  - Records model degradation in the budget tracker
  - Works correctly in LangChain chain composition (prompt | llm)
  - Recovers via half-open probe after the breaker trips
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tenacity

from ai_financial_analyst.core.budget_tracker import RequestBudgetTracker
from ai_financial_analyst.core.llm import CircuitBreakerError, RateLimitFallbackLLM, _CircuitBreaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sync_llm(return_value=None, side_effect=None) -> MagicMock:
    mock = MagicMock()
    if side_effect is not None:
        mock.invoke.side_effect = side_effect
    else:
        mock.invoke.return_value = return_value
    return mock


def _make_async_llm(return_value=None, side_effect=None) -> MagicMock:
    mock = MagicMock()
    if side_effect is not None:
        mock.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        mock.ainvoke = AsyncMock(return_value=return_value)
    return mock


def _retry_error() -> tenacity.RetryError:
    """Construct a RetryError as tenacity would raise it."""
    @tenacity.retry(stop=tenacity.stop_after_attempt(1))
    def _fail():
        raise RuntimeError("429 resource exhausted")
    try:
        _fail()
    except tenacity.RetryError as e:
        return e
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# invoke — synchronous path
# ---------------------------------------------------------------------------


class TestInvoke:
    def test_delegates_to_primary_on_success(self):
        primary = _make_sync_llm(return_value="primary result")
        fallback = _make_sync_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = llm.invoke("input")

        assert result == "primary result"
        primary.invoke.assert_called_once_with("input", config=None)
        fallback.invoke.assert_not_called()

    def test_falls_back_on_circuit_breaker_error(self):
        primary = _make_sync_llm(side_effect=CircuitBreakerError("tripped"))
        fallback = _make_sync_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = llm.invoke("input")

        assert result == "fallback result"
        fallback.invoke.assert_called_once_with("input", config=None)

    def test_falls_back_on_retry_error(self):
        primary = _make_sync_llm(side_effect=_retry_error())
        fallback = _make_sync_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = llm.invoke("input")

        assert result == "fallback result"

    def test_does_not_catch_non_rate_limit_exceptions(self):
        primary = _make_sync_llm(side_effect=ValueError("malformed input"))
        fallback = _make_sync_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        with pytest.raises(ValueError, match="malformed input"):
            llm.invoke("input")

        fallback.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# ainvoke — async path
# ---------------------------------------------------------------------------


class TestAinvoke:
    @pytest.mark.asyncio
    async def test_delegates_to_primary_on_success(self):
        primary = _make_async_llm(return_value="primary result")
        fallback = _make_async_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = await llm.ainvoke("input")

        assert result == "primary result"
        primary.ainvoke.assert_awaited_once_with("input", config=None)
        fallback.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_circuit_breaker_error(self):
        primary = _make_async_llm(side_effect=CircuitBreakerError("tripped"))
        fallback = _make_async_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = await llm.ainvoke("input")

        assert result == "fallback result"
        fallback.ainvoke.assert_awaited_once_with("input", config=None)

    @pytest.mark.asyncio
    async def test_falls_back_on_retry_error(self):
        primary = _make_async_llm(side_effect=_retry_error())
        fallback = _make_async_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = await llm.ainvoke("input")

        assert result == "fallback result"

    @pytest.mark.asyncio
    async def test_does_not_catch_non_rate_limit_exceptions(self):
        primary = _make_async_llm(side_effect=RuntimeError("unexpected crash"))
        fallback = _make_async_llm(return_value="fallback result")
        llm = RateLimitFallbackLLM(primary, fallback)

        with pytest.raises(RuntimeError, match="unexpected crash"):
            await llm.ainvoke("input")

        fallback.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Budget tracker integration
# ---------------------------------------------------------------------------


class TestBudgetTrackerIntegration:
    def test_records_degradation_on_fallback(self):
        budget = RequestBudgetTracker()
        primary = _make_sync_llm(side_effect=CircuitBreakerError("tripped"))
        fallback = _make_sync_llm(return_value="ok")
        llm = RateLimitFallbackLLM(primary, fallback, budget_tracker=budget)

        llm.invoke("input")

        assert budget.model_degraded is True
        assert budget.get_stats()["model_degraded"] is True

    def test_no_degradation_when_primary_succeeds(self):
        budget = RequestBudgetTracker()
        primary = _make_sync_llm(return_value="ok")
        fallback = _make_sync_llm(return_value="fallback")
        llm = RateLimitFallbackLLM(primary, fallback, budget_tracker=budget)

        llm.invoke("input")

        assert budget.model_degraded is False

    def test_degradation_recorded_only_once(self):
        budget = RequestBudgetTracker()
        primary = _make_sync_llm(side_effect=CircuitBreakerError("tripped"))
        fallback = _make_sync_llm(return_value="ok")
        llm = RateLimitFallbackLLM(primary, fallback, budget_tracker=budget)

        llm.invoke("a")
        llm.invoke("b")

        # model_degraded is a one-way flag; get_stats still reports True
        assert budget.model_degraded is True

    def test_works_without_budget_tracker(self):
        """No budget_tracker should not cause any error."""
        primary = _make_sync_llm(side_effect=CircuitBreakerError("tripped"))
        fallback = _make_sync_llm(return_value="ok")
        llm = RateLimitFallbackLLM(primary, fallback, budget_tracker=None)

        result = llm.invoke("input")
        assert result == "ok"


# ---------------------------------------------------------------------------
# LangChain chain composition
# ---------------------------------------------------------------------------


class TestChainComposition:
    def test_is_runnable(self):
        from langchain_core.runnables.base import Runnable
        primary = _make_sync_llm(return_value="ok")
        fallback = _make_sync_llm(return_value="ok")
        llm = RateLimitFallbackLLM(primary, fallback)
        assert isinstance(llm, Runnable)

    def test_pipe_operator_creates_sequence(self):
        """prompt | llm should produce a RunnableSequence."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableSequence

        prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
        primary = _make_sync_llm(return_value="ok")
        fallback = _make_sync_llm(return_value="ok")
        llm = RateLimitFallbackLLM(primary, fallback)

        chain = prompt | llm
        assert isinstance(chain, RunnableSequence)

    def test_with_structured_output_delegates_to_both_models(self):
        """with_structured_output() must wrap both primary and fallback."""
        from pydantic import BaseModel

        class _Schema(BaseModel):
            value: str

        # Mocks that record whether with_structured_output was called
        primary_structured  = _make_sync_llm(return_value=_Schema(value="primary"))
        fallback_structured = _make_sync_llm(return_value=_Schema(value="fallback"))

        primary  = MagicMock()
        fallback = MagicMock()
        primary.with_structured_output.return_value  = primary_structured
        fallback.with_structured_output.return_value = fallback_structured

        llm = RateLimitFallbackLLM(primary, fallback)
        structured_llm = llm.with_structured_output(_Schema)

        # Both underlying models must have been called with the schema
        primary.with_structured_output.assert_called_once_with(_Schema)
        fallback.with_structured_output.assert_called_once_with(_Schema)

        # Returned object is still a RateLimitFallbackLLM (preserves fallback logic)
        assert isinstance(structured_llm, RateLimitFallbackLLM)

    def test_with_structured_output_fallback_path(self):
        """CircuitBreakerError on structured primary still falls back to structured fallback."""
        from pydantic import BaseModel

        class _Schema(BaseModel):
            value: str

        primary_structured  = _make_sync_llm(
            side_effect=CircuitBreakerError("tripped")
        )
        fallback_structured = _make_sync_llm(
            return_value=_Schema(value="from_fallback")
        )

        primary  = MagicMock()
        fallback = MagicMock()
        primary.with_structured_output.return_value  = primary_structured
        fallback.with_structured_output.return_value = fallback_structured

        llm = RateLimitFallbackLLM(primary, fallback)
        structured_llm = llm.with_structured_output(_Schema)

        result = structured_llm.invoke("input")
        assert result.value == "from_fallback"


# ---------------------------------------------------------------------------
# _CircuitBreaker half-open recovery
# ---------------------------------------------------------------------------


class TestCircuitBreakerHalfOpen:
    """Verify the half-open probe mechanism on _CircuitBreaker directly."""

    def _tripped_breaker(self) -> _CircuitBreaker:
        """Return a breaker that has just tripped (max failures reached)."""
        cb = _CircuitBreaker(max_failures=2, window_s=60)
        # Force two failures to trip it
        for _ in range(2):
            try:
                cb.record_failure()
            except CircuitBreakerError:
                pass
        return cb

    def test_is_open_when_tripped(self):
        cb = self._tripped_breaker()
        assert cb.is_open() is True

    def test_probe_allowed_after_delay(self):
        cb = self._tripped_breaker()
        # Simulate time passing: set _half_open_at to the past
        cb._half_open_at = time.monotonic() - 1.0
        # First call to is_open should allow the probe (return False)
        assert cb.is_open() is False
        assert cb._probe_in_flight is True

    def test_only_one_probe_in_flight(self):
        cb = self._tripped_breaker()
        cb._half_open_at = time.monotonic() - 1.0
        # First probe allowed
        assert cb.is_open() is False  # probe in flight
        # Second concurrent check is blocked
        assert cb.is_open() is True

    def test_probe_success_resets_breaker(self):
        cb = self._tripped_breaker()
        cb._half_open_at = time.monotonic() - 1.0
        cb.is_open()  # enter half-open
        cb.probe_succeeded()
        assert cb.is_open() is False
        assert cb._failures == []
        assert cb._half_open_at is None
        assert cb._probe_in_flight is False

    def test_probe_failure_extends_timer(self):
        cb = self._tripped_breaker()
        cb._half_open_at = time.monotonic() - 1.0
        cb.is_open()  # enter half-open
        before = time.monotonic()
        cb.probe_failed()
        # Timer should be extended into the future
        assert cb._half_open_at is not None
        assert cb._half_open_at > before
        assert cb._probe_in_flight is False
        # Breaker remains open
        assert cb.is_open() is True
