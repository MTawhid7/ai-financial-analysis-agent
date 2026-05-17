"""Yahoo Finance data access layer — concurrent multi-type fetcher.

Public API:
    fetch_ticker_data(ticker)  → dict[data_type, dict]  (concurrent, adaptive)
    fetch_single(ticker, data_type) → dict              (single data type)

Architecture:
    Phase 1 (parallel): fetch all 3 CORE types with asyncio.gather + semaphore
    Phase 2 (parallel): if any core type succeeded, fetch 4 EXTENDED types
                        → adaptive: skip extended entirely when all core fail

The semaphore limits concurrent yfinance HTTP threads to prevent Yahoo Finance
CDN rate-limiting. yfinance calls consume zero Gemini RPM quota.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config import settings
from ..base import DataResult, null_result
from . import price, fundamentals, balance_sheet, cash_flow, earnings, metrics, trends

logger = logging.getLogger(__name__)

CORE_DATA_TYPES     = ["price_history", "fundamentals", "balance_sheet"]
EXTENDED_DATA_TYPES = ["cash_flow", "earnings", "price_metrics", "financials_trend"]
ALL_DATA_TYPES      = CORE_DATA_TYPES + EXTENDED_DATA_TYPES

_MODULE_MAP = {
    "price_history":    price,
    "fundamentals":     fundamentals,
    "balance_sheet":    balance_sheet,
    "cash_flow":        cash_flow,
    "earnings":         earnings,
    "price_metrics":    metrics,
    "financials_trend": trends,
}

# Semaphore created lazily per event loop to avoid asyncio errors on import
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.yahoo_fetch_concurrency)
    return _semaphore


async def fetch_ticker_data(ticker: str) -> dict[str, dict]:
    """Fetch all data types for ticker concurrently with adaptive early exit.

    Returns a dict mapping data_type → payload dict (same format as before).
    If all 3 core types fail, returns an empty dict (caller detects failure).
    """
    ticker = ticker.upper()
    sem    = _get_semaphore()

    # ── Phase 1: Core types in parallel ─────────────────────────────────────
    core_tasks = [_fetch_one(ticker, dt, sem) for dt in CORE_DATA_TYPES]
    core_results: list[DataResult | Exception] = await asyncio.gather(
        *core_tasks, return_exceptions=True
    )

    core_data: dict[str, dict] = {}
    for dt, result in zip(CORE_DATA_TYPES, core_results):
        if isinstance(result, Exception):
            logger.warning("Core fetch failed for %s/%s: %s", ticker, dt, result)
        elif result.data_quality != "UNAVAILABLE":
            core_data[dt] = result.to_dict()

    # Adaptive: skip extended types if no core data retrieved
    if not core_data:
        logger.warning("All core data types failed for %s — skipping extended types", ticker)
        return {}

    # ── Phase 2: Extended types in parallel ──────────────────────────────────
    ext_tasks = [_fetch_one(ticker, dt, sem) for dt in EXTENDED_DATA_TYPES]
    ext_results: list[DataResult | Exception] = await asyncio.gather(
        *ext_tasks, return_exceptions=True
    )

    for dt, result in zip(EXTENDED_DATA_TYPES, ext_results):
        if isinstance(result, Exception):
            logger.warning("Extended fetch failed for %s/%s: %s", ticker, dt, result)
        elif result.data_quality != "UNAVAILABLE":
            core_data[dt] = result.to_dict()

    return core_data


async def _fetch_one(
    ticker: str,
    data_type: str,
    sem: asyncio.Semaphore,
) -> DataResult:
    """Run a single synchronous yfinance fetch inside an executor thread."""
    module = _MODULE_MAP.get(data_type)
    if module is None:
        return null_result(ticker, data_type, f"Unknown data type: {data_type}")
    async with sem:
        return await asyncio.to_thread(module.fetch, ticker)


def fetch_single(ticker: str, data_type: str) -> dict:
    """Synchronous single-data-type fetch for the tool wrapper layer.

    Returns the same flat dict that the old tool functions produced.
    """
    module = _MODULE_MAP.get(data_type)
    if module is None:
        return null_result(ticker, data_type, f"Unknown data type: {data_type}").to_dict()
    result = module.fetch(ticker.upper())
    return result.to_dict()
