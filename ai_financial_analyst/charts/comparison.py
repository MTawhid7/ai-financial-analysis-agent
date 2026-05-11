"""Comparison charts: normalised return across multiple tickers."""
from __future__ import annotations

import logging
from typing import Any

from ._theme import _base_layout, _hline, _ZINC, _PALETTE
from ._data import _fetch_hist, _resolve_ticker

logger = logging.getLogger(__name__)


def generate_relative_performance_chart(
    tickers: list[str],
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> dict | None:
    """Normalised % return (base = 100) for up to 6 tickers.

    Accepts both raw tickers and common aliases (e.g. "Nasdaq", "S&P 500").
    """
    try:
        import yfinance as yf
        import pandas as pd

        frames: dict[str, Any] = {}
        for raw in tickers[:6]:
            resolved = _resolve_ticker(raw)
            hist, _ = _fetch_hist(resolved, period, start=start, end=end)
            if not hist.empty:
                frames[resolved] = hist["Close"].dropna()

        if len(frames) < 2:
            return None

        df = pd.DataFrame(frames).dropna(how="all")
        if len(df) < 5:
            return None

        normalized = df / df.iloc[0] * 100
        dates  = [str(d.date()) for d in df.index]
        traces = []
        for i, t in enumerate(frames):
            if t not in normalized.columns:
                continue
            vals  = normalized[t].ffill().round(2).tolist()
            delta = vals[-1] - 100 if vals else 0
            sign  = "+" if delta >= 0 else ""
            traces.append({"type": "scatter", "mode": "lines",
                           "x": dates, "y": vals,
                           "line": {"color": _PALETTE[i % len(_PALETTE)], "width": 2},
                           "name": f"{t} ({sign}{delta:.1f}%)",
                           "hovertemplate": f"{t}: %{{y:.1f}}<extra></extra>"})

        label  = f"{start} to {end or 'today'}" if start else period
        ticker_label = " vs ".join(list(frames.keys()))
        layout = _base_layout(f"Return Comparison — {ticker_label} ({label})", height=340)
        layout["yaxis"]["title"] = {"text": "Indexed (base = 100)", "font": {"size": 10}}
        layout["shapes"] = [_hline(100, _ZINC, "dot")]
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Relative perf chart: %s", exc)
        return None
