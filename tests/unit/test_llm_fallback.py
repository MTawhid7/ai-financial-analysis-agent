"""Unit tests for RateLimitFallbackLLM.

Verifies that the fallback wrapper:
  - Delegates to the primary on success
  - Falls back to Flash-Lite on CircuitBreakerError
  - Falls back to Flash-Lite on tenacity.RetryError
  - Does NOT catch non-rate-limit exceptions
  - Records model degradation in the budget tracker
  - Works correctly in LangChain chain composition (prompt | llm)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest
import tenacity

from ai_financial_analyst.core.budget_tracker import RequestBudgetTracker
from ai_financial_analyst.core.llm import CircuitBreakerError, RateLimitFallbackLLM


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
