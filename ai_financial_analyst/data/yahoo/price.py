"""Yahoo Finance — price_history data type.

Fetches adjusted OHLCV with period fallback (5y → 2y → 1y),
true 52-week high/low, freshness warning, data quality grade,
and stock-split corporate events from the last 5 years.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import yfinance as yf

from ..base import DataResult, assess_quality, null_result, safe_float, utc_now

# Expected minimum data points per received period
_MIN_POINTS: dict[str, int] = {"5y": 200, "2y": 80, "1y": 40}


def fetch(ticker: str) -> DataResult:
    """Fetch price history with graceful period fallback and quality grading."""
    stock = yf.Ticker(ticker)
    ts    = utc_now()

    hist, period_received = _fetch_with_fallback(stock)
    if hist is None or hist.empty:
        return null_result(ticker, "price_history", "No price history available")

    closes = hist["Close"].dropna()
    highs  = hist["High"].dropna()
    lows   = hist["Low"].dropna()

    cutoff           = hist.index[-1] - timedelta(days=365)
    last_year_closes = closes.loc[closes.index >= cutoff]
    last_year_highs  = highs.loc[highs.index >= cutoff]
    last_year_lows   = lows.loc[lows.index >= cutoff]

    current_price = safe_float(closes.iloc[-1])
    price_5y_ago  = safe_float(closes.iloc[0])

    # Quality assessment
    if period_received != "5y":
        quality = "PARTIAL"
        degradation = (
            f"Requested 5-year history; only {period_received} available. "
            "CAGR and long-term trend figures will be less reliable."
        )
    elif len(closes) < _MIN_POINTS.get(period_received, 40):
        quality = "PARTIAL"
        degradation = (
            f"Fewer data points than expected ({len(closes)} bars). "
            "May indicate sparse trading or a recently listed stock."
        )
    else:
        quality, degradation = "FULL", None

    # Freshness check
    freshness_warning = None
    try:
        fi_price = getattr(stock.fast_info, "last_price", None)
        if fi_price and current_price and abs(fi_price - current_price) / max(current_price, 1) > 0.05:
            freshness_warning = (
                f"Price may be stale: cached close ${current_price:.2f} vs "
                f"live ~${fi_price:.2f}. Consider re-fetching."
            )
    except Exception:
        pass

    # Corporate events: stock splits (last 5 years)
    corporate_events = _extract_splits(stock, hist)

    return DataResult(
        ticker           = ticker,
        data_type        = "price_history",
        data_quality     = quality,
        data_timestamp   = ts,
        degradation_note = degradation,
        payload          = {
            "period_requested":  "5y",
            "period_received":   period_received,
            "current_price":     current_price,
            "price_5y_ago":      price_5y_ago,
            "52w_high":          safe_float(last_year_highs.max()) if not last_year_highs.empty else None,
            "52w_low":           safe_float(last_year_lows.min())  if not last_year_lows.empty  else None,
            "data_points":       len(closes),
            "price_adjusted":    True,
            "freshness_warning": freshness_warning,
            "corporate_events":  corporate_events,
        },
    )


def _fetch_with_fallback(stock: yf.Ticker) -> tuple[Any, str]:
    """Try 5y → 2y → 1y; return (hist_df, period_received)."""
    for period, interval in [("5y", "1wk"), ("2y", "1wk"), ("1y", "1d")]:
        h = stock.history(period=period, interval=interval, auto_adjust=True)
        if not h.empty:
            return h, period
    return None, "none"


def _extract_splits(stock: yf.Ticker, hist: Any) -> list[dict]:
    """Return stock splits from the last 5 years as structured dicts."""
    events: list[dict] = []
    try:
        splits = stock.splits
        if splits is None or splits.empty:
            return events
        cutoff = hist.index[-1] - timedelta(days=365 * 5)
        recent = splits[splits.index >= cutoff]
        for dt, ratio in recent.items():
            if ratio and float(ratio) != 1.0:
                r = float(ratio)
                desc = f"{r:.0f}-for-1 stock split" if r > 1 else f"1-for-{1/r:.0f} reverse split"
                events.append({"date": str(dt)[:10], "type": "split", "ratio": round(r, 4), "description": desc})
    except Exception:
        pass
    return events
