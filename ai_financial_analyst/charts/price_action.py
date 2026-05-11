"""Price action charts: simple line and full candlestick with overlays."""
from __future__ import annotations

import logging
from typing import Any

from ._theme import (
    _base_layout, _hline, _xaxis_cfg, _yaxis_cfg,
    _BG, _GRID, _FONT, _BLUE, _GREEN, _RED, _AMBER, _PURPLE, _CYAN, _ZINC,
)
from ._data import _fetch_hist, _earnings_annotations

logger = logging.getLogger(__name__)


def generate_price_chart(
    ticker: str,
    raw_data: dict,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> dict | None:
    """Weekly/daily price line with 52-week high/low dashed bands."""
    try:
        hist, resolved = _fetch_hist(ticker, period, start=start, end=end)
        if hist.empty:
            return None
        dates  = [str(d.date()) for d in hist.index]
        prices = hist["Close"].round(2).tolist()
        shapes: list[dict] = []
        ph = (raw_data.get(resolved) or raw_data.get(ticker) or {}).get("price_history", {})
        for val, color in [(ph.get("52w_high"), _GREEN), (ph.get("52w_low"), _RED)]:
            if val:
                shapes.append(_hline(val, color))
        label  = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — Price ({label})")
        layout["shapes"] = shapes
        return {
            "data": [{"type": "scatter", "mode": "lines",
                      "x": dates, "y": prices,
                      "line": {"color": _BLUE, "width": 2}, "name": resolved,
                      "hovertemplate": "%{x}<br>$%{y:.2f}<extra></extra>"}],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Price chart %s: %s", ticker, exc)
        return None


def generate_candlestick_chart(
    ticker: str,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
    overlays: list[str] | None = None,
    show_earnings: bool = True,
) -> dict | None:
    """OHLCV candlestick + SMA-50/200 + volume.

    overlays: list of strings from {"bollinger", "bb", "ema", "ema_N"} where N is any integer.
    show_earnings: overlay vertical markers at quarterly earnings dates.
    """
    try:
        hist, resolved = _fetch_hist(ticker, period, start=start, end=end)
        if len(hist) < 10:
            return None

        dates   = [str(d.date()) for d in hist.index]
        opens   = hist["Open"].round(2).tolist()
        highs   = hist["High"].round(2).tolist()
        lows    = hist["Low"].round(2).tolist()
        closes  = hist["Close"].round(2).tolist()
        vols    = hist["Volume"].tolist()
        close_s = hist["Close"]

        sma50  = close_s.rolling(min(50,  len(close_s)), min_periods=1).mean().round(2).tolist()
        sma200 = close_s.rolling(min(200, len(close_s)), min_periods=1).mean().round(2).tolist()
        vol_colors = [_GREEN if c >= o else _RED for c, o in zip(closes, opens)]

        data: list[dict] = [
            {
                "type": "candlestick", "xaxis": "x", "yaxis": "y",
                "x": dates,
                "open": opens, "high": highs, "low": lows, "close": closes,
                "increasing": {"line": {"color": _GREEN}, "fillcolor": _GREEN + "66"},
                "decreasing": {"line": {"color": _RED},   "fillcolor": _RED   + "66"},
                "name": resolved, "showlegend": False,
                "hovertemplate": "O $%{open:.2f}  H $%{high:.2f}<br>L $%{low:.2f}  C $%{close:.2f}<extra></extra>",
            },
            {
                "type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
                "x": dates, "y": sma50,
                "line": {"color": _AMBER, "width": 1.2}, "name": "SMA-50",
                "hovertemplate": "SMA-50: $%{y:.2f}<extra></extra>",
            },
            {
                "type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
                "x": dates, "y": sma200,
                "line": {"color": _PURPLE, "width": 1.2, "dash": "dash"}, "name": "SMA-200",
                "hovertemplate": "SMA-200: $%{y:.2f}<extra></extra>",
            },
            {
                "type": "bar", "xaxis": "x", "yaxis": "y2",
                "x": dates, "y": vols,
                "marker": {"color": vol_colors, "opacity": 0.55}, "name": "Volume",
                "hovertemplate": "%{y:,.0f}<extra>Vol</extra>",
            },
        ]

        # Optional overlays
        ov_set = {o.lower().replace("-", "_").replace(" ", "_") for o in (overlays or [])}

        # ── Bollinger Bands ──────────────────────────────────────────────────
        if ov_set & {"bollinger", "bb", "bollinger_bands"}:
            bw = min(20, len(close_s) // 2)
            bb_ma  = close_s.rolling(bw, min_periods=1).mean()
            bb_std = close_s.rolling(bw, min_periods=1).std().fillna(0)
            upper  = (bb_ma + 2 * bb_std).round(2).tolist()
            middle = bb_ma.round(2).tolist()
            lower  = (bb_ma - 2 * bb_std).round(2).tolist()
            for vals, name, dash in [
                (upper,  f"BB Upper ({bw})",  "dot"),
                (middle, f"BB Mid ({bw})",    "dash"),
                (lower,  f"BB Lower ({bw})",  "dot"),
            ]:
                data.append({
                    "type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
                    "x": dates, "y": vals,
                    "line": {"color": _CYAN, "width": 1, "dash": dash},
                    "name": name, "opacity": 0.7,
                    "hovertemplate": f"{name}: $%{{y:.2f}}<extra></extra>",
                })

        # ── EMA overlays ─────────────────────────────────────────────────────
        ema_periods: list[int] = []
        for ov in (overlays or []):
            n = ov.lower().replace("-", "_").replace(" ", "_")
            if n in ("ema", "ema_default"):
                ema_periods = [20, 50]
            elif n.startswith("ema"):
                import re
                m = re.search(r"\d+", n)
                if m:
                    ema_periods.append(int(m.group()))
        ema_colors = [_CYAN, "#f472b6", "#34d399", "#fb923c"]
        for i, ep in enumerate(ema_periods):
            ema_v = close_s.ewm(span=ep, adjust=False).mean().round(2).tolist()
            data.append({
                "type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
                "x": dates, "y": ema_v,
                "line": {"color": ema_colors[i % len(ema_colors)], "width": 1.2},
                "name": f"EMA-{ep}",
                "hovertemplate": f"EMA-{ep}: $%{{y:.2f}}<extra></extra>",
            })

        label  = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — Candlestick ({label})", height=440)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        layout.update({
            "xaxis":  _xaxis_cfg(),
            "yaxis":  _yaxis_cfg((0.28, 1.0), "Price ($)"),
            "yaxis2": _yaxis_cfg((0.0,  0.24), "Volume"),
        })

        # ── Earnings annotations ─────────────────────────────────────────────
        if show_earnings:
            e_shapes, e_annots = _earnings_annotations(resolved, dates)
            layout["shapes"] = e_shapes
            layout["annotations"] = e_annots

        return {"data": data, "layout": layout}
    except Exception as exc:
        logger.warning("Candlestick %s: %s", ticker, exc)
        return None
