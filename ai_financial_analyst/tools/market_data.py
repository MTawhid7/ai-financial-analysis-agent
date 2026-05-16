"""Market benchmark helpers — risk-free rate and S&P 500 reference data.

All data sourced from Yahoo Finance (free, no API key beyond what yfinance provides).
Results are cached with appropriate TTLs so multiple ticker analyses in one session
only pay the network cost once.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from ..core.cache import ResultCache, TTL_RISK_FREE, TTL_MARKET_BENCHMARK

logger = logging.getLogger(__name__)

_cache = ResultCache()

# Tickers used as proxies
_RISK_FREE_TICKER = "^TNX"    # 10-year US Treasury yield (annualised %)
_SP500_TICKER     = "^GSPC"   # S&P 500 index


def get_risk_free_rate() -> float:
    """Return the current US 10-year Treasury yield as an annualised decimal.

    Example: returns 0.042 when the 10-year yield is 4.20%.
    Cached for 1 hour. Returns 0.04 (4%) as a safe default on failure.
    """
    def _fetch() -> float:
        try:
            fi = yf.Ticker(_RISK_FREE_TICKER).fast_info
            raw = getattr(fi, "last_price", None)
            if raw and isinstance(raw, (int, float)) and 0 < raw < 25:
                return round(raw / 100, 6)  # convert % → decimal
        except Exception:
            pass
        # Fallback: try history
        try:
            hist = yf.Ticker(_RISK_FREE_TICKER).history(period="5d")
            if not hist.empty:
                return round(float(hist["Close"].iloc[-1]) / 100, 6)
        except Exception:
            pass
        return 0.04  # safe fallback: 4%

    result, _ = _cache.get_or_fetch(
        "risk_free_rate", {}, _fetch, ttl=TTL_RISK_FREE
    )
    return result if isinstance(result, float) else 0.04


def get_sp500_data(period: str = "5y") -> dict[str, Any] | None:
    """Return S&P 500 adjusted daily closes and derived returns for the given period.

    Returns a dict with keys:
      "prices"   — list of adjusted closing prices (oldest → newest)
      "returns"  — list of daily log-returns (len = len(prices) - 1)
      "cagr"     — annualised total return as a decimal

    Cached for 24 hours. Returns None on failure.
    """
    def _fetch() -> dict | None:
        try:
            hist = yf.Ticker(_SP500_TICKER).history(
                period=period, interval="1d", auto_adjust=True
            )
            if hist.empty or len(hist) < 30:
                return None
            prices = hist["Close"].dropna()
            returns_s = prices.pct_change().dropna()
            n_years = max(len(prices) / 252, 0.001)
            cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1
            return {
                "prices":  prices.tolist(),
                "returns": returns_s.tolist(),
                "cagr":    round(float(cagr), 6),
            }
        except Exception as exc:
            logger.warning("S&P 500 fetch failed: %s", exc)
            return None

    result, _ = _cache.get_or_fetch(
        "sp500_data", {"period": period}, _fetch, ttl=TTL_MARKET_BENCHMARK
    )
    return result  # type: ignore[return-value]
