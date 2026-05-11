"""Combined multi-panel charts: price+RSI and price+MACD in a single figure."""
from __future__ import annotations

import logging

from ._theme import (
    _base_layout, _hline, _hrect, _xaxis_cfg, _yaxis_cfg,
    _BLUE, _GREEN, _RED, _AMBER, _PURPLE, _ZINC, _CYAN,
)
from ._data import _fetch_hist, _earnings_annotations

logger = logging.getLogger(__name__)


def _ohlcv_traces(hist, resolved: str) -> tuple[list[dict], list[str]]:
    """Build candlestick + SMA-50/200 + volume traces. Returns (traces, dates)."""
    dates   = [str(d.date()) for d in hist.index]
    opens   = hist["Open"].round(2).tolist()
    highs   = hist["High"].round(2).tolist()
    lows    = hist["Low"].round(2).tolist()
    closes  = hist["Close"].round(2).tolist()
    vols    = hist["Volume"].tolist()
    close_s = hist["Close"]
    sma50   = close_s.rolling(min(50,  len(close_s)), min_periods=1).mean().round(2).tolist()
    sma200  = close_s.rolling(min(200, len(close_s)), min_periods=1).mean().round(2).tolist()
    vc      = [_GREEN if c >= o else _RED for c, o in zip(closes, opens)]

    traces = [
        {"type": "candlestick", "xaxis": "x", "yaxis": "y",
         "x": dates, "open": opens, "high": highs, "low": lows, "close": closes,
         "increasing": {"line": {"color": _GREEN}, "fillcolor": _GREEN + "66"},
         "decreasing": {"line": {"color": _RED},   "fillcolor": _RED   + "66"},
         "name": resolved, "showlegend": False,
         "hovertemplate": "O $%{open:.2f}  H $%{high:.2f}<br>L $%{low:.2f}  C $%{close:.2f}<extra></extra>"},
        {"type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
         "x": dates, "y": sma50,
         "line": {"color": _AMBER, "width": 1.2}, "name": "SMA-50",
         "hovertemplate": "SMA-50: $%{y:.2f}<extra></extra>"},
        {"type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
         "x": dates, "y": sma200,
         "line": {"color": _PURPLE, "width": 1.2, "dash": "dash"}, "name": "SMA-200",
         "hovertemplate": "SMA-200: $%{y:.2f}<extra></extra>"},
        {"type": "bar", "xaxis": "x", "yaxis": "y2",
         "x": dates, "y": vols,
         "marker": {"color": vc, "opacity": 0.55}, "name": "Volume",
         "hovertemplate": "%{y:,.0f}<extra>Vol</extra>"},
    ]
    return traces, dates


def generate_price_rsi_chart(
    ticker: str,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
    overlays: list[str] | None = None,
) -> dict | None:
    """Two-panel: OHLCV candlestick (top 65%) + RSI-14 (bottom 30%)."""
    try:
        hist, resolved = _fetch_hist(ticker, period, start=start, end=end)
        if len(hist) < 20:
            return None

        traces, dates = _ohlcv_traces(hist, resolved)

        # RSI panel
        close = hist["Close"].dropna()
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).round(2).fillna(50)

        traces.append({
            "type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y3",
            "x": dates, "y": rsi.tolist(),
            "line": {"color": _BLUE, "width": 1.5}, "name": "RSI(14)",
            "hovertemplate": "%{x}<br>RSI: %{y:.1f}<extra></extra>",
        })

        label  = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — Price + RSI ({label})", height=520)
        layout.pop("xaxis", None); layout.pop("yaxis", None)
        layout.update({
            "xaxis":  {**_xaxis_cfg(), "anchor": "y3"},
            "yaxis":  _yaxis_cfg((0.36, 1.0), "Price ($)"),
            "yaxis2": _yaxis_cfg((0.22, 0.33), "Volume"),
            "yaxis3": {**_yaxis_cfg((0.0, 0.19), "RSI"), "range": [0, 100]},
        })
        layout["shapes"] = [
            _hrect(70, 100, _RED, "y3"), _hrect(0, 30, _GREEN, "y3"),
            _hline(70, _RED, "dot", "y3"), _hline(30, _GREEN, "dot", "y3"),
            _hline(50, _ZINC, "dot", "y3"),
        ]
        # Earnings — only in price panel (paper coords)
        e_shapes, e_annots = _earnings_annotations(resolved, dates, y0_paper=0.36, y1_paper=1.0)
        layout["shapes"] += e_shapes
        layout["annotations"] = e_annots
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Price+RSI chart %s: %s", ticker, exc)
        return None


def generate_price_macd_chart(
    ticker: str,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
    overlays: list[str] | None = None,
) -> dict | None:
    """Two-panel: OHLCV candlestick (top 62%) + MACD (bottom 32%)."""
    try:
        hist, resolved = _fetch_hist(ticker, period, start=start, end=end)
        if len(hist) < 35:
            return None

        traces, dates = _ohlcv_traces(hist, resolved)

        # MACD panel
        close  = hist["Close"].dropna()
        macd   = (close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()).round(4)
        signal = macd.ewm(span=9, adjust=False).mean().round(4)
        histo  = (macd - signal).round(4)
        bar_colors = [_GREEN if v >= 0 else _RED for v in histo.tolist()]

        traces += [
            {"type": "bar", "xaxis": "x", "yaxis": "y3",
             "x": dates, "y": histo.tolist(),
             "marker": {"color": bar_colors, "opacity": 0.7}, "name": "Histogram",
             "hovertemplate": "Hist: %{y:.4f}<extra></extra>"},
            {"type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y3",
             "x": dates, "y": macd.tolist(),
             "line": {"color": _BLUE, "width": 1.5}, "name": "MACD",
             "hovertemplate": "MACD: %{y:.4f}<extra></extra>"},
            {"type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y3",
             "x": dates, "y": signal.tolist(),
             "line": {"color": _AMBER, "width": 1.5}, "name": "Signal",
             "hovertemplate": "Signal: %{y:.4f}<extra></extra>"},
        ]

        label  = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — Price + MACD ({label})", height=520)
        layout.pop("xaxis", None); layout.pop("yaxis", None)
        layout.update({
            "xaxis":  {**_xaxis_cfg(), "anchor": "y3"},
            "yaxis":  _yaxis_cfg((0.40, 1.0), "Price ($)"),
            "yaxis2": _yaxis_cfg((0.26, 0.37), "Volume"),
            "yaxis3": _yaxis_cfg((0.0,  0.22), "MACD"),
        })
        layout["shapes"] = [_hline(0, _ZINC, "solid", "y3")]
        e_shapes, e_annots = _earnings_annotations(resolved, dates, y0_paper=0.40, y1_paper=1.0)
        layout["shapes"] += e_shapes
        layout["annotations"] = e_annots
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Price+MACD chart %s: %s", ticker, exc)
        return None
