"""Technical indicator charts: RSI and MACD."""
from __future__ import annotations

import logging

from ._theme import _base_layout, _hline, _hrect, _BLUE, _GREEN, _RED, _AMBER, _ZINC, _GRID
from ._data import _fetch_hist

logger = logging.getLogger(__name__)


def generate_rsi_chart(
    ticker: str,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> dict | None:
    """RSI(14) with overbought (70) and oversold (30) shaded zones."""
    try:
        hist, resolved = _fetch_hist(ticker, period, interval="1d", start=start, end=end)
        if len(hist) < 20:
            return None
        close = hist["Close"].dropna()
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).round(2)
        dates = [str(d.date()) for d in close.index]
        label = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — RSI(14) ({label})")
        layout["yaxis"]["title"] = {"text": "RSI", "font": {"size": 10}}
        layout["yaxis"]["range"] = [0, 100]
        layout["shapes"] = [
            _hrect(70, 100, _RED), _hrect(0, 30, _GREEN),
            _hline(70, _RED), _hline(30, _GREEN), _hline(50, _ZINC),
        ]
        return {
            "data": [{"type": "scatter", "mode": "lines",
                      "x": dates, "y": rsi.fillna(50).tolist(),
                      "line": {"color": _BLUE, "width": 1.5}, "name": "RSI(14)",
                      "hovertemplate": "%{x}<br>RSI: %{y:.1f}<extra></extra>"}],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("RSI chart %s: %s", ticker, exc)
        return None


def generate_macd_chart(
    ticker: str,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> dict | None:
    """MACD(12,26,9): coloured histogram + MACD line + signal line."""
    try:
        hist, resolved = _fetch_hist(ticker, period, interval="1d", start=start, end=end)
        if len(hist) < 35:
            return None
        close  = hist["Close"].dropna()
        macd   = (close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()).round(4)
        signal = macd.ewm(span=9, adjust=False).mean().round(4)
        histo  = (macd - signal).round(4)
        dates  = [str(d.date()) for d in close.index]
        label  = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — MACD(12,26,9) ({label})")
        layout["yaxis"]["title"] = {"text": "MACD", "font": {"size": 10}}
        layout["shapes"] = [_hline(0, _ZINC)]
        return {
            "data": [
                {"type": "bar",
                 "x": dates, "y": histo.tolist(),
                 "marker": {"color": [_GREEN if v >= 0 else _RED for v in histo], "opacity": 0.7},
                 "name": "Histogram",
                 "hovertemplate": "Hist: %{y:.4f}<extra></extra>"},
                {"type": "scatter", "mode": "lines",
                 "x": dates, "y": macd.tolist(),
                 "line": {"color": _BLUE, "width": 1.5}, "name": "MACD",
                 "hovertemplate": "MACD: %{y:.4f}<extra></extra>"},
                {"type": "scatter", "mode": "lines",
                 "x": dates, "y": signal.tolist(),
                 "line": {"color": _AMBER, "width": 1.5}, "name": "Signal",
                 "hovertemplate": "Signal: %{y:.4f}<extra></extra>"},
            ],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("MACD chart %s: %s", ticker, exc)
        return None
