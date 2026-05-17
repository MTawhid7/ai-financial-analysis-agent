"""Data fetching utilities: ticker aliases, yfinance helpers, earnings annotations."""
from __future__ import annotations

import bisect
import logging
from typing import Any

from ._theme import _AMBER

logger = logging.getLogger(__name__)

# ── Period → interval mapping ─────────────────────────────────────────────────
# Short periods use fine-grained intraday intervals (within yfinance's 60-day limit
# for sub-hourly data).  Longer periods use daily/weekly as before.
_PERIOD_INTERVAL: dict[str, str] = {
    "1d":  "5m",   # ~78 bars at 5-min resolution (was "1h")
    "5d":  "15m",  # ~130 bars at 15-min resolution (was "1h")
    "1mo": "1h",   # ~143 bars hourly (was "1d")
    "3mo": "1d",   "6mo": "1d",
    "1y":  "1d",   "2y": "1wk",  "5y": "1wk",
    "10y": "1mo",  "ytd": "1d",  "max": "1mo",
}


def _interval_for(period: str) -> str:
    return _PERIOD_INTERVAL.get(period, "1d")


def _interval_for_range(start: str | None, end: str | None, period: str) -> str:
    """Pick a sensible yfinance interval for a date range or fall back to period mapping."""
    if not start:
        return _interval_for(period)
    try:
        from datetime import datetime
        s = datetime.strptime(start[:10], "%Y-%m-%d")
        e = datetime.strptime(end[:10], "%Y-%m-%d") if end else datetime.now()
        days = (e - s).days
        if days <= 5:    return "5m"   # intraday: 5-min bars for very short ranges
        if days <= 30:   return "1h"   # hourly for up to 1 month
        if days <= 60:   return "1d"
        if days <= 730:  return "1wk"
        return "1mo"
    except Exception:
        return "1d"


# ── Ticker aliases ─────────────────────────────────────────────────────────────
TICKER_ALIASES: dict[str, str] = {
    # Broad market indices
    "nasdaq": "QQQ",    "nasdaq 100": "QQQ", "nasdaq100": "QQQ", "ndx": "QQQ",
    "s&p 500": "SPY",   "s&p500": "SPY",     "sp500": "SPY",     "s&p": "SPY",  "spx": "SPY",
    "dow": "DIA",        "dow jones": "DIA",  "djia": "DIA",
    "russell": "IWM",    "russell 2000": "IWM", "small cap": "IWM",
    # Sector ETFs
    "tech": "XLK",          "technology": "XLK",
    "semiconductor": "SOXX","semiconductors": "SOXX", "semis": "SOXX",
    "healthcare": "XLV",    "pharma": "XLV",
    "energy": "XLE",        "oil stocks": "XLE",
    "financials": "XLF",    "banks": "XLF",
    "consumer": "XLY",      "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "utilities": "XLU",
    "real estate": "XLRE",  "reits": "XLRE",
    "materials": "XLB",
    "industrials": "XLI",
    "communication": "XLC", "communications": "XLC",
    # Commodities
    "gold": "GLD",   "silver": "SLV",
    "oil": "USO",    "crude oil": "USO",
    # Crypto
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD","eth": "ETH-USD",
    # Fixed income / rates
    "bonds": "TLT",  "treasury": "TLT", "long bonds": "TLT",
    "10 year": "^TNX", "10yr": "^TNX",
    # Volatility
    "vix": "^VIX", "volatility": "^VIX", "fear index": "^VIX",
}


def _resolve_ticker(name: str) -> str:
    """Convert a common name or alias to a canonical ticker symbol."""
    return TICKER_ALIASES.get(name.lower().strip(), name.upper().strip())


# ── yfinance fetch helper ──────────────────────────────────────────────────────

def _fetch_hist(
    ticker: str,
    period: str = "1y",
    interval: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[Any, str]:
    """Fetch OHLCV from yfinance.

    Returns (DataFrame, resolved_ticker). Resolves ticker aliases.
    Uses date range when start is provided, otherwise uses period string.
    """
    import yfinance as yf
    resolved = _resolve_ticker(ticker)
    if interval is None:
        interval = _interval_for_range(start, end, period)
    yf_ticker = yf.Ticker(resolved)
    if start:
        kwargs: dict[str, Any] = {"start": start, "interval": interval}
        if end:
            kwargs["end"] = end
        hist = yf_ticker.history(**kwargs)
    else:
        hist = yf_ticker.history(period=period, interval=interval)
    return hist, resolved


# ── Earnings annotation helper ─────────────────────────────────────────────────

def _earnings_annotations(
    ticker: str,
    dates: list[str],
    y0_paper: float = 0.0,
    y1_paper: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    """Return (shapes, annotations) for earnings dates within the chart's date range.

    Uses bisect to snap each earnings date to the nearest trading day in
    `dates` so category-axis shapes align correctly.
    """
    try:
        import yfinance as yf
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            return [], []
        if "Reported EPS" in ed.columns:
            ed = ed[ed["Reported EPS"].notna()]
        chart_start = dates[0] if dates else ""
        chart_end   = dates[-1] if dates else ""
        shapes: list[dict] = []
        annots: list[dict] = []
        for dt in ed.index:
            d = str(dt.date())
            if d < chart_start or d > chart_end:
                continue
            # Snap to nearest trading day
            idx = bisect.bisect_left(dates, d)
            closest = dates[min(idx, len(dates) - 1)]
            shapes.append({
                "type": "line", "xref": "x", "yref": "paper",
                "x0": closest, "x1": closest,
                "y0": y0_paper, "y1": y1_paper,
                "line": {"color": _AMBER + "80", "width": 1, "dash": "dot"},
                "layer": "below",
            })
            annots.append({
                "x": closest, "y": y1_paper, "yref": "paper",
                "text": "E", "showarrow": False,
                "font": {"color": _AMBER, "size": 9},
                "xanchor": "center", "yanchor": "top",
            })
        return shapes, annots
    except Exception:
        return [], []


# ── Financial DataFrame row helper ─────────────────────────────────────────────

def _get_row(df: Any, *names: str) -> Any:
    """Return the first matching row from a yfinance financial DataFrame."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name].dropna()
    return None
