"""Per-session Gemini API call counter with free-tier budget warnings."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_WARN_THRESHOLD = 0.80  # Warn at 80% of daily budget


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

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record_primary_call(self) -> None:
        self._primary_calls += 1
        self._check_budget()

    def record_sub_call(self) -> None:
        self._sub_calls += 1
        self._check_budget()

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
