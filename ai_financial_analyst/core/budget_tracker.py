"""Per-session Gemini API call counter with free-tier budget and RPM warnings."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_WARN_THRESHOLD = 0.80  # Warn at 80% of daily budget

# Free-tier RPM limits (requests per minute, 60-second rolling window)
_PRIMARY_RPM_LIMIT = 15
_SUB_RPM_LIMIT = 30


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
    Logs a WARNING when 80% of the estimated daily budget is consumed.
    """

    def __init__(self, daily_budget: int | None = None) -> None:
        self._daily_budget = daily_budget or int(
            os.getenv("GEMINI_DAILY_REQUEST_BUDGET", "1500")
        )
        self._primary_calls = 0
        self._sub_calls = 0
        self._cache_hits = 0
        self._warned = False
        self._model_degraded = False
        self._primary_rpm = _RpmBucket(_PRIMARY_RPM_LIMIT)
        self._sub_rpm = _RpmBucket(_SUB_RPM_LIMIT)

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
                _PRIMARY_RPM_LIMIT,
            )

    def record_sub_call(self) -> None:
        self._sub_calls += 1
        self._check_budget()
        if self._sub_rpm.record():
            logger.warning(
                "RPM alert: %d sub-model calls in last 60s (free-tier limit: %d RPM)",
                self._sub_rpm.current_rpm(),
                _SUB_RPM_LIMIT,
            )

    def record_cache_hit(self) -> None:
        self._cache_hits += 1

    def record_model_degradation(self) -> None:
        """Record that the primary model was rate-limited and Flash-Lite fallback was used."""
        if not self._model_degraded:
            self._model_degraded = True
            logger.warning(
                "Model degradation recorded: Flash rate-limited, falling back to Flash-Lite."
            )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def total_calls(self) -> int:
        return self._primary_calls + self._sub_calls

    @property
    def model_degraded(self) -> bool:
        """True if the primary model was rate-limited at least once this session."""
        return self._model_degraded

    def get_stats(self) -> dict:
        return {
            "primary_calls": self._primary_calls,
            "sub_calls": self._sub_calls,
            "total_calls": self.total_calls,
            "cache_hits": self._cache_hits,
            "daily_budget": self._daily_budget,
            "budget_used_pct": round(
                self.total_calls / self._daily_budget * 100, 1
            ),
            "model_degraded": self._model_degraded,
            "primary_rpm_current": self._primary_rpm.current_rpm(),
            "sub_rpm_current": self._sub_rpm.current_rpm(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_budget(self) -> None:
        if self._warned:
            return
        ratio = self.total_calls / self._daily_budget
        if ratio >= _WARN_THRESHOLD:
            logger.warning(
                "Budget alert: %d/%d Gemini API calls used (%.0f%% of daily limit).",
                self.total_calls,
                self._daily_budget,
                ratio * 100,
            )
            self._warned = True
