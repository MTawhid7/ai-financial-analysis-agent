"""Per-session Gemini API call counter with three-tier budget and RPM warnings.

All thresholds and limits are sourced from settings (env-configurable).

Budget thresholds (fraction of daily budget):
  soft_warn (default 60%) — advisory; logged at INFO
  warn      (default 80%) — hard warning; logged at WARNING
  defer     (default 95%) — activate caching-only mode
"""

from __future__ import annotations

import logging
import time

from ..config import settings

logger = logging.getLogger(__name__)


class _RpmBucket:
    """Rolling 60-second window for per-minute rate tracking."""

    def __init__(self, limit: int) -> None:
        self._timestamps: list[float] = []
        self._limit = limit

    def record(self) -> bool:
        """Record a call. Returns True if the current RPM exceeds the limit."""
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < 60.0]
        self._timestamps.append(now)
        return len(self._timestamps) > self._limit

    def current_rpm(self) -> int:
        now = time.monotonic()
        return len([t for t in self._timestamps if now - t < 60.0])


class RequestBudgetTracker:
    """Tracks Gemini API calls for a single pipeline run.

    Counts primary (Flash) and sub-task (Flash-Lite) calls separately.
    Emits warnings at configurable budget thresholds (soft/hard/deferral).
    """

    def __init__(self, daily_budget: int | None = None) -> None:
        self._daily_budget   = daily_budget or settings.llm_daily_budget
        self._primary_calls  = 0
        self._sub_calls      = 0
        self._cache_hits     = 0
        self._cache_credits  = 0   # estimated calls saved by cache hits
        self._soft_warned    = False
        self._warned         = False
        self._model_degraded = False
        self._primary_rpm    = _RpmBucket(settings.llm_primary_rpm_limit)
        self._sub_rpm        = _RpmBucket(settings.llm_fallback_rpm_limit)
        self._per_tool_calls: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record_primary_call(self) -> None:
        self._primary_calls += 1
        self._check_budget()
        if self._primary_rpm.record():
            logger.warning(
                "RPM alert: %d primary calls in last 60s (free-tier limit: %d RPM)",
                self._primary_rpm.current_rpm(),
                settings.llm_primary_rpm_limit,
            )

    def record_sub_call(self) -> None:
        self._sub_calls += 1
        self._check_budget()
        if self._sub_rpm.record():
            logger.warning(
                "RPM alert: %d sub-model calls in last 60s (free-tier limit: %d RPM)",
                self._sub_rpm.current_rpm(),
                settings.llm_fallback_rpm_limit,
            )

    def record_tool_call(self, tool_name: str, is_primary: bool = True) -> None:
        """Record an LLM call attributed to a named tool for per-tool observability."""
        self._per_tool_calls[tool_name] = self._per_tool_calls.get(tool_name, 0) + 1
        if is_primary:
            self.record_primary_call()
        else:
            self.record_sub_call()

    def record_cache_hit(self, tool_name: str = "") -> None:
        """Record a cache hit. Credits back one estimated call to reduce effective usage."""
        self._cache_hits += 1
        self._cache_credits += 1
        if tool_name:
            key = f"cache:{tool_name}"
            self._per_tool_calls[key] = self._per_tool_calls.get(key, 0) + 1

    def record_model_degradation(self) -> None:
        """Record that the primary model was rate-limited and fallback was used."""
        if not self._model_degraded:
            self._model_degraded = True
            logger.warning(
                "Model degradation recorded: primary model rate-limited, "
                "falling back to %s.",
                settings.llm_fallback_model,
            )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def total_calls(self) -> int:
        return self._primary_calls + self._sub_calls

    @property
    def effective_calls(self) -> int:
        """Total calls minus cache credits (estimated actual quota consumed)."""
        return max(0, self.total_calls - self._cache_credits)

    @property
    def model_degraded(self) -> bool:
        """True if the primary model was rate-limited at least once this session."""
        return self._model_degraded

    @property
    def in_deferral_mode(self) -> bool:
        """True if effective budget usage exceeds the deferral threshold (95% by default)."""
        return self.effective_calls / self._daily_budget >= settings.llm_budget_defer_pct

    def get_stats(self) -> dict:
        ratio = self.effective_calls / self._daily_budget
        return {
            "primary_calls":       self._primary_calls,
            "sub_calls":           self._sub_calls,
            "total_calls":         self.total_calls,
            "cache_hits":          self._cache_hits,
            "cache_credits":       self._cache_credits,
            "effective_calls":     self.effective_calls,
            "daily_budget":        self._daily_budget,
            "budget_used_pct":     round(ratio * 100, 1),
            "model_degraded":      self._model_degraded,
            "in_deferral_mode":    self.in_deferral_mode,
            "primary_rpm_current": self._primary_rpm.current_rpm(),
            "sub_rpm_current":     self._sub_rpm.current_rpm(),
            "per_tool_calls":      dict(self._per_tool_calls),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_budget(self) -> None:
        ratio = self.effective_calls / self._daily_budget
        if not self._soft_warned and ratio >= settings.llm_budget_soft_warn_pct:
            logger.info(
                "Budget soft warning: %d/%d effective calls used (%.0f%% of daily limit).",
                self.effective_calls, self._daily_budget, ratio * 100,
            )
            self._soft_warned = True
        if not self._warned and ratio >= settings.llm_budget_warn_pct:
            logger.warning(
                "Budget alert: %d/%d Gemini API calls used (%.0f%% of daily limit).",
                self.effective_calls, self._daily_budget, ratio * 100,
            )
            self._warned = True
