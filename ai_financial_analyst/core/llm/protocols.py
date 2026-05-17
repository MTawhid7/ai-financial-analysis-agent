"""LLMClient protocol — the interface contract for all LLM implementations.

Any class implementing this protocol can be used wherever an LLM is expected,
enabling drop-in substitution of Gemini for OpenAI, Anthropic, or a mock.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async-capable LLM interface compatible with LangChain Runnables.

    Implementations:
      - GeminiLLMClient (production)
      - MockLLMClient   (tests)
      - RateLimitFallbackLLM (wraps two LLMClients with circuit-breaker fallback)
    """

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Async invocation — returns an AIMessage or a Pydantic model instance."""
        ...

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Sync invocation — returns an AIMessage or a Pydantic model instance."""
        ...

    def bind_tools(self, tools: list, **kwargs: Any) -> "LLMClient":
        """Return a new LLMClient with the given tools bound for function-calling."""
        ...

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "LLMClient":
        """Return a new LLMClient that enforces the given Pydantic schema via JSON mode."""
        ...
