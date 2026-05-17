"""Gemini LLM client with tenacity retry, jitter, circuit breaker, and rate-limit fallback."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import tenacity
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.base import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings

logger = logging.getLogger(__name__)

# Model identifiers — sourced from settings (env-configurable, not hardcoded).
_PRIMARY_MODEL = settings.llm_primary_model
_SUB_MODEL     = settings.llm_fallback_model

# Circuit breaker settings — all from settings.
_CB_MAX_CONSECUTIVE_429 = settings.llm_cb_max_failures
_CB_WINDOW_SECONDS      = settings.llm_cb_window_s
_CB_HALF_OPEN_DELAY     = settings.llm_cb_half_open_delay_s
_PRIMARY_CALL_TIMEOUT   = settings.llm_call_timeout_s


class RateLimitError(RuntimeError):
    """HTTP 429 received from Gemini API."""


class CircuitBreakerError(RuntimeError):
    """Circuit breaker tripped: too many consecutive 429s in the window."""


@dataclass
class _CircuitBreaker:
    """Tracks 429s within a rolling time window with half-open recovery.

    States:
    - CLOSED  : normal operation (_half_open_at is None, failures < max)
    - OPEN    : blocking calls (_half_open_at is set, probe not yet due)
    - HALF-OPEN: one probe in flight (_probe_in_flight=True)
    """

    max_failures: int = _CB_MAX_CONSECUTIVE_429
    window_seconds: float = _CB_WINDOW_SECONDS
    _failures: list[float] = field(default_factory=list)
    _half_open_at: float | None = field(default=None)
    _probe_in_flight: bool = field(default=False)

    def _active_failures(self) -> list[float]:
        now = time.monotonic()
        return [t for t in self._failures if now - t < self.window_seconds]

    def is_open(self) -> bool:
        """Return True if the breaker should block the call.

        Returns False (allow) in two cases:
        1. Normal closed state — failures below threshold.
        2. Half-open probe window — allows exactly one probe through.
        """
        active = self._active_failures()
        self._failures = active

        if len(active) < self.max_failures:
            # Closed — healthy
            return False

        # Breaker tripped — check if probe is due
        now = time.monotonic()
        if self._half_open_at is not None and now >= self._half_open_at and not self._probe_in_flight:
            # Enter half-open: let exactly one probe through
            self._probe_in_flight = True
            logger.info("Circuit breaker half-open: sending probe to primary model")
            return False

        return True  # Still open

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failures = [t for t in self._failures if now - t < self.window_seconds]
        self._failures.append(now)
        if len(self._failures) >= self.max_failures:
            if self._half_open_at is None:
                self._half_open_at = time.monotonic() + _CB_HALF_OPEN_DELAY
            raise CircuitBreakerError(
                f"Circuit breaker open: {self.max_failures} rate-limit errors "
                f"within {self.window_seconds}s. Probe allowed in {_CB_HALF_OPEN_DELAY:.0f}s."
            )

    def probe_succeeded(self) -> None:
        """Call when the half-open probe request succeeded — close the breaker."""
        self.reset()
        logger.info("Circuit breaker closed: primary model probe succeeded")

    def probe_failed(self) -> None:
        """Call when the half-open probe request failed — extend the open period."""
        self._probe_in_flight = False
        self._half_open_at = time.monotonic() + _CB_HALF_OPEN_DELAY
        logger.warning(
            "Circuit breaker probe failed — staying open for another %.0fs",
            _CB_HALF_OPEN_DELAY,
        )

    def reset(self) -> None:
        self._failures.clear()
        self._half_open_at = None
        self._probe_in_flight = False


_primary_cb = _CircuitBreaker()
_sub_cb = _CircuitBreaker()


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        term in msg for term in ("429", "resource exhausted", "quota", "503", "unavailable", "500")
    )


def _retry_on_rate_limit(circuit_breaker: _CircuitBreaker) -> Callable:
    """Return a tenacity retry decorator bound to the given circuit breaker."""

    def before_retry(retry_state: tenacity.RetryCallState) -> None:
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if _is_rate_limit(exc):
                circuit_breaker.record_failure()

    def after_success(retry_state: tenacity.RetryCallState) -> None:
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


class _BudgetCallbackHandler(BaseCallbackHandler):
    """Records each LLM call to the budget tracker via LangChain's callback system.

    Using callbacks instead of monkey-patching avoids Pydantic v2's strict
    __setattr__ validation on ChatGoogleGenerativeAI.
    """

    def __init__(self, record_fn: Callable[[], None]) -> None:
        super().__init__()
        self._record_fn = record_fn

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self._record_fn()


class _PrimaryLLM(ChatGoogleGenerativeAI):
    @_retry_on_rate_limit(_primary_cb)
    async def ainvoke(self, *args, **kwargs):
        return await super().ainvoke(*args, **kwargs)

    @_retry_on_rate_limit(_primary_cb)
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs)


class _SubLLM(ChatGoogleGenerativeAI):
    @_retry_on_rate_limit(_sub_cb)
    async def ainvoke(self, *args, **kwargs):
        return await super().ainvoke(*args, **kwargs)

    @_retry_on_rate_limit(_sub_cb)
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs)


def get_primary_llm(budget_tracker=None):
    """Return Gemini Flash LLM for the core ReAct reasoning loop."""
    llm = _PrimaryLLM(
        model=_PRIMARY_MODEL,
        google_api_key=settings.google_api_key,
        temperature=0.1,
        streaming=True,
        # max_retries=1 disables the SDK's own retry so only our tenacity layer retries.
        # In langchain-google-genai 4.x, max_retries=0 means "use SDK default (5 retries)",
        # not zero — setting 1 means exactly one attempt with no SDK-level retry.
        max_retries=1,
    )
    if budget_tracker:
        llm = llm.with_config(
            {"callbacks": [_BudgetCallbackHandler(budget_tracker.record_primary_call)]}
        )
    return llm


def get_subllm(budget_tracker=None):
    """Return Gemini Flash-Lite LLM for summarisation and sanitisation sub-tasks."""
    llm = _SubLLM(
        model=_SUB_MODEL,
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


class RateLimitFallbackLLM(Runnable):
    """LangChain Runnable that transparently falls back to Flash-Lite when the
    primary model (Flash) is rate-limited.

    Catches CircuitBreakerError (circuit tripped after 3× 429 in 30s) and
    tenacity.RetryError (5 retries exhausted) from the primary, then delegates
    the same call to the fallback model. Non-rate-limit exceptions propagate
    normally.

    Compatible with LangChain chain composition:  prompt | llm  works because
    this class extends Runnable and inherits __or__ / __ror__.
    """

    def __init__(self, primary: Any, fallback: Any, budget_tracker: Any = None) -> None:
        self._primary = primary
        self._fallback = fallback
        self._budget_tracker = budget_tracker

    # ------------------------------------------------------------------
    # LangChain Runnable interface
    # ------------------------------------------------------------------

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if _primary_cb.is_open():
            logger.debug("Circuit breaker open — routing directly to fallback (sync)")
            return self._fallback.invoke(input, config=config, **kwargs)
        is_probe = _primary_cb._probe_in_flight
        try:
            result = self._primary.invoke(input, config=config, **kwargs)
            if is_probe:
                _primary_cb.probe_succeeded()
            return result
        except (CircuitBreakerError, tenacity.RetryError) as exc:
            if is_probe:
                _primary_cb.probe_failed()
            self._on_rate_limit_fallback(exc)
            return self._fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if _primary_cb.is_open():
            logger.debug("Circuit breaker open — routing directly to fallback (async)")
            return await self._fallback.ainvoke(input, config=config, **kwargs)
        is_probe = _primary_cb._probe_in_flight
        try:
            coro = self._primary.ainvoke(input, config=config, **kwargs)
            result = await asyncio.wait_for(coro, timeout=_PRIMARY_CALL_TIMEOUT)
            if is_probe:
                _primary_cb.probe_succeeded()
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "Primary model call timed out after %.0fs — falling back to Flash-Lite",
                _PRIMARY_CALL_TIMEOUT,
            )
            if is_probe:
                _primary_cb.probe_failed()
            return await self._fallback.ainvoke(input, config=config, **kwargs)
        except (CircuitBreakerError, tenacity.RetryError) as exc:
            if is_probe:
                _primary_cb.probe_failed()
            self._on_rate_limit_fallback(exc)
            return await self._fallback.ainvoke(input, config=config, **kwargs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def bind_tools(self, tools, **kwargs):
        """Return a new RateLimitFallbackLLM with both models bound to the same tools.

        Required so that manager.py can call self._primary_llm.bind_tools(tools)
        without hitting AttributeError — RateLimitFallbackLLM wraps two
        ChatGoogleGenerativeAI instances, both of which support bind_tools.
        """
        new_primary = self._primary.bind_tools(tools, **kwargs)
        new_fallback = self._fallback.bind_tools(tools, **kwargs)
        return RateLimitFallbackLLM(new_primary, new_fallback, self._budget_tracker)

    def with_structured_output(self, schema, **kwargs):
        """Return a new RateLimitFallbackLLM where both models enforce a Pydantic schema.

        Each model is wrapped via ChatGoogleGenerativeAI.with_structured_output(),
        which uses Gemini's native JSON mode to guarantee schema conformance.
        The returned wrapper's ainvoke() yields a Pydantic model instance directly
        (not an AIMessage), preserving the same rate-limit fallback semantics.
        """
        new_primary = self._primary.with_structured_output(schema, **kwargs)
        new_fallback = self._fallback.with_structured_output(schema, **kwargs)
        return RateLimitFallbackLLM(new_primary, new_fallback, self._budget_tracker)

    def _on_rate_limit_fallback(self, exc: BaseException) -> None:
        logger.warning(
            "Primary model (Flash) rate-limited — falling back to Flash-Lite. Cause: %s",
            str(exc)[:120],
        )
        if self._budget_tracker is not None:
            self._budget_tracker.record_model_degradation()


def get_primary_llm_with_fallback(budget_tracker=None) -> RateLimitFallbackLLM:
    """Return a Flash LLM that automatically falls back to Flash-Lite on rate limits.

    Use this in place of get_primary_llm() for all pipeline and agent entry points.
    The returned object is a LangChain Runnable and works in prompt | llm chains.
    """
    primary = get_primary_llm(budget_tracker)
    fallback = get_subllm(budget_tracker)
    return RateLimitFallbackLLM(primary, fallback, budget_tracker)


def content_to_str(content: Any) -> str:
    """Normalize an LLM response content value to a plain string.

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


def with_retry(fn: Callable, is_primary: bool = True) -> Callable:
    """Wrap an async callable with retry + circuit breaker logic."""
    cb = _primary_cb if is_primary else _sub_cb
    decorator = _retry_on_rate_limit(cb)
    return decorator(fn)
