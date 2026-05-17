"""Circuit breaker for LLM rate-limit protection.

Three-state machine:
  CLOSED    — normal operation; requests pass through
  OPEN      — blocking; too many consecutive failures in the window
  HALF-OPEN — one probe request allowed; success → CLOSED, failure → extends OPEN

The circuit breaker is instantiated per LLMRegistry (not a module-level singleton),
so each session/test gets an isolated breaker that can be reset without import tricks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitBreakerError(RuntimeError):
    """Raised when the circuit breaker is open and the call is blocked."""


@dataclass
class CircuitBreaker:
    """Tracks rate-limit errors within a rolling time window.

    All thresholds default to values from settings but can be overridden
    per-instance for testing or fine-grained control.
    """

    max_failures:      int   = 5
    window_s:          float = 60.0
    half_open_delay_s: float = 60.0

    _failures:          list[float] = field(default_factory=list)
    _half_open_at:      float | None = field(default=None)
    _probe_in_flight:   bool          = field(default=False)

    @classmethod
    def from_settings(cls) -> "CircuitBreaker":
        """Create a CircuitBreaker using the global settings defaults."""
        from ...config import settings
        return cls(
            max_failures      = settings.llm_cb_max_failures,
            window_s          = settings.llm_cb_window_s,
            half_open_delay_s = settings.llm_cb_half_open_delay_s,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def is_open(self) -> bool:
        """Return True if the breaker should block the call.

        Returns False (allow) in two cases:
          1. CLOSED state — failure count below threshold.
          2. HALF-OPEN probe window — exactly one probe allowed through.
        """
        active = self._active_failures()
        self._failures = active

        if len(active) < self.max_failures:
            return False  # CLOSED

        now = time.monotonic()
        if (
            self._half_open_at is not None
            and now >= self._half_open_at
            and not self._probe_in_flight
        ):
            self._probe_in_flight = True
            logger.info("Circuit breaker HALF-OPEN: sending probe to primary model")
            return False  # HALF-OPEN probe

        return True  # OPEN

    def record_failure(self) -> None:
        """Record a rate-limit failure. Trips the breaker when threshold is reached."""
        now = time.monotonic()
        self._failures = [t for t in self._failures if now - t < self.window_s]
        self._failures.append(now)
        if len(self._failures) >= self.max_failures:
            if self._half_open_at is None:
                self._half_open_at = time.monotonic() + self.half_open_delay_s
            raise CircuitBreakerError(
                f"Circuit breaker OPEN: {self.max_failures} rate-limit errors "
                f"within {self.window_s:.0f}s. "
                f"Probe allowed in {self.half_open_delay_s:.0f}s."
            )

    def probe_succeeded(self) -> None:
        """Call when a HALF-OPEN probe request succeeded — closes the breaker."""
        self.reset()
        logger.info("Circuit breaker CLOSED: primary model probe succeeded")

    def probe_failed(self) -> None:
        """Call when a HALF-OPEN probe failed — extends the open period."""
        self._probe_in_flight = False
        self._half_open_at = time.monotonic() + self.half_open_delay_s
        logger.warning(
            "Circuit breaker probe FAILED — staying open for another %.0fs",
            self.half_open_delay_s,
        )

    def reset(self) -> None:
        """Reset to CLOSED state — clears all failure history."""
        self._failures.clear()
        self._half_open_at   = None
        self._probe_in_flight = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _active_failures(self) -> list[float]:
        now = time.monotonic()
        return [t for t in self._failures if now - t < self.window_s]
