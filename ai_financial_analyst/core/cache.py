"""diskcache wrapper with 4-hour TTL for yfinance and DuckDuckGo results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable

import diskcache

logger = logging.getLogger(__name__)

_TTL_SECONDS = 4 * 60 * 60  # 4 hours


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

    def set(self, tool_name: str, args: dict, value: Any) -> None:
        key = self._make_key(tool_name, args)
        self._cache.set(key, value, expire=_TTL_SECONDS)

    def get_or_fetch(
        self,
        tool_name: str,
        args: dict,
        fetch_fn: Callable[[], Any],
        budget_tracker=None,
    ) -> tuple[Any, bool]:
        """Return (result, cache_hit). Calls fetch_fn on miss."""
        cached = self.get(tool_name, args)
        if cached is not None:
            if budget_tracker:
                budget_tracker.record_cache_hit()
            return cached, True
        result = fetch_fn()
        self.set(tool_name, args, result)
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
