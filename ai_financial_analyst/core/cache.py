"""diskcache wrapper with configurable per-call TTL.

TTL constants (use these at call sites rather than magic numbers):
  TTL_PRICE            15 min  — real-time-ish prices via fast_info
  TTL_FUNDAMENTALS      6 h   — PE, market cap, margins (refreshed daily by Yahoo)
  TTL_FINANCIALS       24 h   — income statement, balance sheet, cash flow
  TTL_MARKET_BENCHMARK 24 h   — S&P 500 history for beta calculation
  TTL_RISK_FREE         1 h   — risk-free rate (^TNX)
  TTL_DAMODARAN        30 d   — Damodaran sector benchmarks (published annually)
  TTL_WEB_SEARCH        1 h   — financial news (was 4h — too stale for earnings moves)
  TTL_DEFAULT           4 h   — fallback for anything not explicitly categorised
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable

import diskcache

logger = logging.getLogger(__name__)

# ── TTL constants (seconds) ───────────────────────────────────────────────────
TTL_PRICE            = 15 * 60            # 15 minutes
TTL_FUNDAMENTALS     = 6  * 60 * 60       # 6 hours
TTL_FINANCIALS       = 24 * 60 * 60       # 24 hours
TTL_MARKET_BENCHMARK = 24 * 60 * 60       # 24 hours
TTL_RISK_FREE        =  1 * 60 * 60       # 1 hour
TTL_DAMODARAN        = 30 * 24 * 60 * 60  # 30 days
TTL_WEB_SEARCH       =  1 * 60 * 60       # 1 hour (reduced from 4h)
TTL_DEFAULT          =  4 * 60 * 60       # 4 hours (legacy default)

_TTL_SECONDS = TTL_DEFAULT  # kept for any external references


class ResultCache:
    """Persistent on-disk cache keyed by tool name + canonical args hash."""

    def __init__(self, cache_dir: str | None = None) -> None:
        path = cache_dir or os.getenv("CACHE_DIR", ".cache")
        self._cache = diskcache.Cache(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, tool_name: str, args: dict) -> Any | None:
        key = self._make_key(tool_name, args)
        value = self._cache.get(key)
        if value is not None:
            logger.debug("Cache HIT  key=%s", key[:16])
        else:
            logger.debug("Cache MISS key=%s", key[:16])
        return value

    def set(self, tool_name: str, args: dict, value: Any, ttl: int = TTL_DEFAULT) -> None:
        key = self._make_key(tool_name, args)
        self._cache.set(key, value, expire=ttl)

    def get_or_fetch(
        self,
        tool_name: str,
        args: dict,
        fetch_fn: Callable[[], Any],
        budget_tracker=None,
        ttl: int = TTL_DEFAULT,
    ) -> tuple[Any, bool]:
        """Return (result, cache_hit). Calls fetch_fn on miss.

        Pass one of the TTL_* constants as `ttl` to control staleness per data type.
        """
        cached = self.get(tool_name, args)
        if cached is not None:
            if budget_tracker:
                budget_tracker.record_cache_hit()
            return cached, True
        result = fetch_fn()
        self.set(tool_name, args, result, ttl=ttl)
        return result, False

    def close(self) -> None:
        self._cache.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(tool_name: str, args: dict) -> str:
        canonical = json.dumps(args, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()
        return digest
