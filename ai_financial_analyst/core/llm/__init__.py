"""LLM abstraction layer for the AI Financial Analyst Agent.

Public API — import from here, not from sub-modules:
    from ai_financial_analyst.core.llm import (
        LLMClient,
        RateLimitFallbackLLM,
        CircuitBreakerError,
        RateLimitError,
        LLMRegistry,
        content_to_str,
        with_retry,
    )

Backward-compatibility shims keep existing import sites working without change:
    from ai_financial_analyst.core.llm import get_primary_llm_with_fallback, get_subllm, ...
"""

from .protocols import LLMClient
from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from .gemini import (
    RateLimitError,
    RateLimitFallbackLLM,
    content_to_str,
    with_retry,
)
from .registry import LLMRegistry

# Backward-compat alias: old name was _CircuitBreaker (private), new is CircuitBreaker.
_CircuitBreaker = CircuitBreaker

# ── Backward-compatibility shims ─────────────────────────────────────────────
# Code that imports from core.llm directly (orchestrator, conversational_agent, etc.)
# continues to work without modification.  These shims will be removed when
# those callers are migrated to use LLMRegistry (Phase 4).

from ..budget_tracker import RequestBudgetTracker as _BudgetTracker


def get_primary_llm(budget_tracker=None):
    """Return primary Gemini LLM (Flash). Deprecated: use LLMRegistry instead."""
    from .gemini import _make_primary_llm
    return _make_primary_llm(budget_tracker=budget_tracker)


def get_subllm(budget_tracker=None):
    """Return fallback Gemini LLM (Flash-Lite). Deprecated: use LLMRegistry instead."""
    from .gemini import _make_subllm
    return _make_subllm(budget_tracker=budget_tracker)


def get_primary_llm_with_fallback(budget_tracker=None) -> RateLimitFallbackLLM:
    """Return a Flash LLM that falls back to Flash-Lite on rate limits.

    Deprecated: use LLMRegistry.get_primary_with_fallback() instead.
    """
    registry = LLMRegistry(budget_tracker=budget_tracker)
    return registry.get_primary_with_fallback()


__all__ = [
    "LLMClient",
    "CircuitBreaker",
    "CircuitBreakerError",
    "RateLimitError",
    "RateLimitFallbackLLM",
    "LLMRegistry",
    "content_to_str",
    "with_retry",
    # shims
    "get_primary_llm",
    "get_subllm",
    "get_primary_llm_with_fallback",
]
