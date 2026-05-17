"""US 10-year Treasury yield (risk-free rate) via Yahoo Finance ^TNX."""

from __future__ import annotations

import logging

import yfinance as yf

from ...config import settings
from ...core.cache import ResultCache

logger = logging.getLogger(__name__)

_RISK_FREE_TICKER = "^TNX"
_cache = ResultCache()


def get_risk_free_rate() -> float:
    """Return the current US 10-year Treasury yield as an annualised decimal.

    Example: returns 0.042 when the 10-year yield is 4.20%.
    Cached per settings.ttl_risk_free_s (default 1 hour).
    Returns 0.04 (4%) as a safe fallback on any error.
    """
    result, _ = _cache.get_or_fetch(
        "risk_free_rate", {}, _fetch, ttl=settings.ttl_risk_free_s
    )
    return result if isinstance(result, float) else 0.04


def _fetch() -> float:
    try:
        fi  = yf.Ticker(_RISK_FREE_TICKER).fast_info
        raw = getattr(fi, "last_price", None)
        if raw and isinstance(raw, (int, float)) and 0 < raw < 25:
            return round(raw / 100, 6)
    except Exception:
        pass
    try:
        hist = yf.Ticker(_RISK_FREE_TICKER).history(period="5d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]) / 100, 6)
    except Exception:
        pass
    logger.warning("Risk-free rate fetch failed; using 4.0%% default")
    return 0.04
