"""S&P 500 index reference data via Yahoo Finance ^GSPC."""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from ...config import settings
from ...core.cache import ResultCache

logger = logging.getLogger(__name__)

_SP500_TICKER = "^GSPC"
_cache = ResultCache()


def get_sp500_data(period: str = "5y") -> dict[str, Any] | None:
    """Return S&P 500 adjusted daily closes and derived metrics for the given period.

    Returns:
        {"prices": [...], "returns": [...], "cagr": 0.123}
        or None on failure.

    Cached per settings.ttl_market_benchmark_s (default 24 hours).
    """
    result, _ = _cache.get_or_fetch(
        "sp500_data", {"period": period}, lambda: _fetch(period),
        ttl=settings.ttl_market_benchmark_s,
    )
    return result  # type: ignore[return-value]


def _fetch(period: str) -> dict | None:
    try:
        hist = yf.Ticker(_SP500_TICKER).history(period=period, interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 30:
            return None
        prices   = hist["Close"].dropna()
        returns  = prices.pct_change().dropna()
        n_years  = max(len(prices) / 252, 0.001)
        cagr     = (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1
        return {
            "prices":  prices.tolist(),
            "returns": returns.tolist(),
            "cagr":    round(float(cagr), 6),
        }
    except Exception as exc:
        logger.warning("S&P 500 fetch failed: %s", exc)
        return None
