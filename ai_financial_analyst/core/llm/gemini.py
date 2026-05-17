"""Gemini LLM client with tenacity retry, jitter, and rate-limit fallback.

Key design decisions:
- RateLimitFallbackLLM owns its circuit breaker (injected at construction).
  No module-level singletons — each LLMRegistry gets isolated state.
- _PrimaryLLM and _SubLLM use per-module circuit breakers for their retry
  decorators (an implementation detail of the retry mechanism; not exposed).
- content_to_str normalises Gemini's typed-block response format to plain str.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import tenacity
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.base import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from ...config import settings
from .circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class RateLimitError(RuntimeError):
    """HTTP 429 received from the Gemini API."""


# ── Retry helpers ─────────────────────────────────────────────────────────────

def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        term in msg
        for term in ("429", "resource exhausted", "quota", "503", "unavailable", "500")
    )


def _make_retry_decorator(circuit_breaker: CircuitBreaker) -> Callable:
    """Return a tenacity retry decorator bound to the given circuit breaker instance."""

    def before_retry(retry_state: tenacity.RetryCallState) -> None:
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if _is_rate_limit(exc):
                circuit_breaker.record_failure()

    def after_success(_retry_state: tenacity.RetryCallState) -> None:
        circuit_breaker.reset()

    return tenacity.retry(
        retry=tenacity.retry_if_exception(_is_rate_limit),
        wait=tenacity.wait_exponential_jitter(initial=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        before=before_retry,
        after=after_success,
        reraise=False,
    )


# ── Budget callback ───────────────────────────────────────────────────────────

class _BudgetCallbackHandler(BaseCallbackHandler):
    """Records each LLM invocation to the budget tracker via LangChain callbacks."""

    def __init__(self, record_fn: Callable[[], None]) -> None:
        super().__init__()
        self._record_fn = record_fn

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self._record_fn()


# ── Low-level Gemini wrappers ─────────────────────────────────────────────────
# These classes exist only to apply the tenacity retry decorator at the
# ChatGoogleGenerativeAI layer. They are not exported from the package;
# consumers use LLMRegistry.

def _make_primary_llm(budget_tracker=None, circuit_breaker: CircuitBreaker | None = None):
    """Create a retrying Gemini Flash LLM."""
    cb = circuit_breaker or CircuitBreaker.from_settings()
    retry = _make_retry_decorator(cb)

    class _PrimaryLLM(ChatGoogleGenerativeAI):
        @retry  # type: ignore[arg-type]
        async def ainvoke(self, *args, **kwargs):
            return await super().ainvoke(*args, **kwargs)

        @retry  # type: ignore[arg-type]
        def invoke(self, *args, **kwargs):
            return super().invoke(*args, **kwargs)

    llm = _PrimaryLLM(
        model=settings.llm_primary_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
        streaming=True,
        # max_retries=1: disable SDK-layer retry so only our tenacity layer retries.
        max_retries=1,
    )
    if budget_tracker:
        llm = llm.with_config(
            {"callbacks": [_BudgetCallbackHandler(budget_tracker.record_primary_call)]}
        )
    return llm


def _make_subllm(budget_tracker=None, circuit_breaker: CircuitBreaker | None = None):
    """Create a retrying Gemini Flash-Lite LLM for sub-tasks."""
    cb = circuit_breaker or CircuitBreaker.from_settings()
    retry = _make_retry_decorator(cb)

    class _SubLLM(ChatGoogleGenerativeAI):
        @retry  # type: ignore[arg-type]
        async def ainvoke(self, *args, **kwargs):
            return await super().ainvoke(*args, **kwargs)

        @retry  # type: ignore[arg-type]
        def invoke(self, *args, **kwargs):
            return super().invoke(*args, **kwargs)

    llm = _SubLLM(
        model=settings.llm_fallback_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
        streaming=False,
        max_retries=1,
    )
    if budget_tracker:
        llm = llm.with_config(
            {"callbacks": [_BudgetCallbackHandler(budget_tracker.record_sub_call)]}
        )
    return llm


# ── RateLimitFallbackLLM ──────────────────────────────────────────────────────

class RateLimitFallbackLLM(Runnable):
    """LangChain Runnable that falls back to Flash-Lite when Flash is rate-limited.

    Owns its circuit breaker instance — no module-level singleton.
    Each LLMRegistry creates one RateLimitFallbackLLM with its own breaker,
    so test runs are fully isolated without import tricks.

    Compatible with LangChain chain composition: `prompt | llm` works because
    this class extends Runnable and inherits __or__ / __ror__.
    """

    def __init__(
        self,
        primary:         Any,
        fallback:        Any,
        budget_tracker:  Any = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._primary        = primary
        self._fallback       = fallback
        self._budget_tracker = budget_tracker
        self._cb             = circuit_breaker or CircuitBreaker.from_settings()

    # ── LangChain Runnable interface ──────────────────────────────────────────

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if self._cb.is_open():
            logger.debug("Circuit breaker OPEN — routing to fallback (sync)")
            return self._fallback.invoke(input, config=config, **kwargs)
        is_probe = self._cb._probe_in_flight
        try:
            result = self._primary.invoke(input, config=config, **kwargs)
            if is_probe:
                self._cb.probe_succeeded()
            return result
        except (CircuitBreakerError, tenacity.RetryError) as exc:
            if is_probe:
                self._cb.probe_failed()
            self._on_rate_limit_fallback(exc)
            return self._fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if self._cb.is_open():
            logger.debug("Circuit breaker OPEN — routing to fallback (async)")
            return await self._fallback.ainvoke(input, config=config, **kwargs)
        is_probe = self._cb._probe_in_flight
        try:
            coro   = self._primary.ainvoke(input, config=config, **kwargs)
            result = await asyncio.wait_for(coro, timeout=settings.llm_call_timeout_s)
            if is_probe:
                self._cb.probe_succeeded()
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "Primary model call timed out after %.0fs — falling back to Flash-Lite",
                settings.llm_call_timeout_s,
            )
            if is_probe:
                self._cb.probe_failed()
            return await self._fallback.ainvoke(input, config=config, **kwargs)
        except (CircuitBreakerError, tenacity.RetryError) as exc:
            if is_probe:
                self._cb.probe_failed()
            self._on_rate_limit_fallback(exc)
            return await self._fallback.ainvoke(input, config=config, **kwargs)

    # ── LangChain composition helpers ─────────────────────────────────────────

    def bind_tools(self, tools: list, **kwargs: Any) -> "RateLimitFallbackLLM":
        """Delegate bind_tools to both underlying models; preserve circuit breaker."""
        new_primary  = self._primary.bind_tools(tools, **kwargs)
        new_fallback = self._fallback.bind_tools(tools, **kwargs)
        return RateLimitFallbackLLM(new_primary, new_fallback, self._budget_tracker, self._cb)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "RateLimitFallbackLLM":
        """Delegate with_structured_output to both models; preserve circuit breaker."""
        new_primary  = self._primary.with_structured_output(schema, **kwargs)
        new_fallback = self._fallback.with_structured_output(schema, **kwargs)
        return RateLimitFallbackLLM(new_primary, new_fallback, self._budget_tracker, self._cb)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_rate_limit_fallback(self, exc: BaseException) -> None:
        logger.warning(
            "Primary model rate-limited — falling back to Flash-Lite. Cause: %s",
            str(exc)[:120],
        )
        if self._budget_tracker is not None:
            self._budget_tracker.record_model_degradation()


# ── Utilities ─────────────────────────────────────────────────────────────────

def content_to_str(content: Any) -> str:
    """Normalise an LLM response content value to a plain string.

    In langchain-google-genai 4.x the google-genai SDK returns content as a
    list of typed blocks: [{'type': 'text', 'text': '...'}].  Plain strings
    are passed through unchanged.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def with_retry(fn: Callable, circuit_breaker: CircuitBreaker | None = None) -> Callable:
    """Wrap an async callable with retry + circuit breaker logic."""
    cb = circuit_breaker or CircuitBreaker.from_settings()
    return _make_retry_decorator(cb)(fn)
