"""Volume profile (volume-at-price) chart.

Shows where volume traded across the price axis over a given period.
High-volume price levels (point of control ≥80% of max bin) are highlighted
in amber; all others use the standard blue.
"""
from __future__ import annotations

import logging

from ._data import _fetch_hist
from ._theme import _AMBER, _BLUE, _FONT, _base_layout

logger = logging.getLogger(__name__)

_DEFAULT_BINS = 30


def generate_volume_profile_chart(
    ticker: str,
    period: str = "1y",
    bins: int = _DEFAULT_BINS,
    start: str | None = None,
    end: str | None = None,
) -> dict | None:
    """Horizontal volume-at-price (VAP) histogram.

    Each bar represents the cumulative volume traded within a price range.
    Bars at or near the point of control (highest-volume level) are highlighted
    in amber to make support/resistance levels visually obvious.
    """
    try:
        hist, resolved = _fetch_hist(ticker, period=period, start=start, end=end)
        if hist is None or hist.empty:
            return None

        closes  = hist["Close"].values
        volumes = hist["Volume"].values.astype(float)

        price_min = float(closes.min())
        price_max = float(closes.max())
        if price_max <= price_min:
            return None

        bin_width = (price_max - price_min) / bins
        bin_edges = [price_min + bin_width * i for i in range(bins + 1)]
        bin_mids  = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]

        vap = [0.0] * bins
        for price, vol in zip(closes, volumes):
            idx = min(int((price - price_min) / bin_width), bins - 1)
            vap[idx] += vol

        max_vap = max(vap) if max(vap) > 0 else 1.0
        # Point of control: bins with ≥80% of the maximum volume
        colors = [_AMBER if v >= max_vap * 0.80 else _BLUE for v in vap]

        label  = f"{start} to {end or 'today'}" if start else period
        layout = _base_layout(f"{resolved} — Volume Profile ({label})")
        layout["xaxis"]["title"] = {"text": "Volume", "font": {"size": 10}}
        layout["yaxis"]["title"] = {"text": "Price ($)", "font": {"size": 10}}
        layout["annotations"] = [{
            "text": "🟡 Amber = point of control (highest-volume price levels)",
            "xref": "paper", "yref": "paper",
            "x": 0, "y": -0.12,
            "showarrow": False,
            "font": {"size": 9, "color": _FONT},
            "xanchor": "left",
        }]

        return {
            "data": [{
                "type": "bar",
                "x": [round(v) for v in vap],
                "y": [round(m, 2) for m in bin_mids],
                "orientation": "h",
                "marker": {"color": colors, "opacity": 0.8},
                "name": "Volume at Price",
                "hovertemplate": "Price: $%{y:.2f}<br>Volume: %{x:,.0f}<extra></extra>",
            }],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Volume profile chart %s: %s", ticker, exc)
        return None
