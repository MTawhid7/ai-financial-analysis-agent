"""Gemini LLM client with tenacity retry, jitter, and circuit breaker."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import tenacity
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Model identifiers (gemini-2.0-* deprecated, shuts down June 1 2026)
_PRIMARY_MODEL = "gemini-3-flash-preview"
_SUB_MODEL = "gemini-3.1-flash-lite-preview"

# Circuit breaker settings
_CB_MAX_CONSECUTIVE_429 = 3
_CB_WINDOW_SECONDS = 30


class RateLimitError(RuntimeError):
    """HTTP 429 received from Gemini API."""


class CircuitBreakerError(RuntimeError):
    """Circuit breaker tripped: too many consecutive 429s in the window."""


@dataclass
class _CircuitBreaker:
    """Tracks consecutive 429s within a rolling time window."""

    max_failures: int = _CB_MAX_CONSECUTIVE_429
    window_seconds: float = _CB_WINDOW_SECONDS
    _failures: list[float] = field(default_factory=list)

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failures = [t for t in self._failures if now - t < self.window_seconds]
        self._failures.append(now)
        if len(self._failures) >= self.max_failures:
            raise CircuitBreakerError(
                f"Circuit breaker open: {self.max_failures} consecutive 429 errors "
                f"within {self.window_seconds}s. Halting to preserve API quota."
            )

    def reset(self) -> None:
        self._failures.clear()


_primary_cb = _CircuitBreaker()
_sub_cb = _CircuitBreaker()


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource exhausted" in msg or "quota" in msg


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


def get_primary_llm(budget_tracker=None):
    """Return Gemini Flash LLM for the core ReAct reasoning loop."""
    llm = ChatGoogleGenerativeAI(
        model=_PRIMARY_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1,
        streaming=True,
        # max_retries=1 disables the SDK's own retry so only our tenacity layer retries.
        # In langchain-google-genai 4.x, max_retries=0 means "use SDK default (5 retries)",
        # not zero — setting 1 means exactly one attempt with no SDK-level retry.
        max_retries=1,
    )
    if budget_tracker:
        return llm.with_config(
            {"callbacks": [_BudgetCallbackHandler(budget_tracker.record_primary_call)]}
        )
    return llm


def get_subllm(budget_tracker=None):
    """Return Gemini Flash-Lite LLM for summarisation and sanitisation sub-tasks."""
    llm = ChatGoogleGenerativeAI(
        model=_SUB_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.0,
        streaming=False,
        max_retries=1,
    )
    if budget_tracker:
        return llm.with_config(
            {"callbacks": [_BudgetCallbackHandler(budget_tracker.record_sub_call)]}
        )
    return llm


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
