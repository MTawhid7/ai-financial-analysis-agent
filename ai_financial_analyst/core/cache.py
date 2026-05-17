"""diskcache wrapper with configurable per-call TTL.

TTL values are sourced from settings (env-configurable). Legacy module-level
constants are retained as aliases for backwards compatibility with existing
import sites that do `from .cache import TTL_PRICE`.

All values are now overridable via environment variables — see config/settings.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

import diskcache

from ..config import settings

logger = logging.getLogger(__name__)

# ── TTL aliases (backwards-compatible; sourced from settings) ──────────────────
TTL_PRICE            = settings.ttl_price_s
TTL_FUNDAMENTALS     = settings.ttl_fundamentals_s
TTL_FINANCIALS       = settings.ttl_financials_s
TTL_MARKET_BENCHMARK = settings.ttl_market_benchmark_s
TTL_RISK_FREE        = settings.ttl_risk_free_s
TTL_DAMODARAN        = settings.ttl_damodaran_s
TTL_WEB_SEARCH       = settings.ttl_web_search_s
TTL_DEFAULT          = settings.ttl_default_s

_TTL_SECONDS = TTL_DEFAULT  # kept for any external references


class ResultCache:
    """Persistent on-disk cache keyed by tool name + canonical args hash."""

    def __init__(self, cache_dir: str | None = None) -> None:
        path = cache_dir or settings.cache_dir
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

        Pass one of the TTL_* constants or settings.get_ttl(data_type) as `ttl`.
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
