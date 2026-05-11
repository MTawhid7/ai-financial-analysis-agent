"""Risk charts: drawdown."""
from __future__ import annotations

import logging

from ._theme import _base_layout, _RED
from ._data import _fetch_hist

logger = logging.getLogger(__name__)


def generate_drawdown_chart(
    ticker: str,
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> dict | None:
    """Drawdown from rolling peak; filled area chart."""
    try:
        hist, resolved = _fetch_hist(ticker, period, start=start, end=end)
        if len(hist) < 20:
            return None
        close  = hist["Close"].dropna()
        dd_pct = ((close - close.cummax()) / close.cummax() * 100).round(2)
        dates  = [str(d.date()) for d in close.index]
        vals   = dd_pct.tolist()
        min_dd = min(vals)
        label  = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — Drawdown from Peak ({label})")
        layout["yaxis"]["title"]      = {"text": "Drawdown (%)", "font": {"size": 10}}
        layout["yaxis"]["ticksuffix"] = "%"
        layout["annotations"] = [{"x": dates[vals.index(min_dd)], "y": min_dd,
                                   "text": f"Max DD: {min_dd:.1f}%", "showarrow": True,
                                   "arrowhead": 2,
                                   "font": {"color": _RED, "size": 10}, "arrowcolor": _RED}]
        return {"data": [{"type": "scatter", "mode": "lines",
                          "x": dates, "y": vals,
                          "fill": "tozeroy", "fillcolor": _RED + "30",
                          "line": {"color": _RED, "width": 1.5}, "name": "Drawdown",
                          "hovertemplate": "%{x}<br>%{y:.1f}%<extra></extra>"}],
                "layout": layout}
    except Exception as exc:
        logger.warning("Drawdown chart %s: %s", ticker, exc)
        return None
