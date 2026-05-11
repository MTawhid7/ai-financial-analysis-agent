"""Shared visual theme for all Plotly charts."""
from __future__ import annotations

_BG       = "#18181b"
_PAPER_BG = "#27272a"
_FONT     = "#e4e4e7"
_GRID     = "#3f3f46"
_ZINC     = "#71717a"
_BLUE     = "#60a5fa"
_GREEN    = "#22c55e"
_RED      = "#ef4444"
_AMBER    = "#f59e0b"
_CYAN     = "#0891b2"
_PURPLE   = "#a855f7"
_ORANGE   = "#f97316"
_PALETTE  = [_BLUE, _GREEN, _AMBER, _RED, _PURPLE, _CYAN, _ORANGE]


def _base_layout(title: str, *, height: int = 320) -> dict:
    return {
        "title": {"text": title, "font": {"color": _FONT, "size": 13}},
        "paper_bgcolor": _PAPER_BG,
        "plot_bgcolor": _BG,
        "font": {"color": _FONT, "family": "Inter, system-ui, sans-serif", "size": 11},
        "margin": {"l": 60, "r": 20, "t": 48, "b": 48},
        "height": height,
        "legend": {
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1,
            "font": {"size": 10}, "bgcolor": "rgba(0,0,0,0)",
        },
        "xaxis": {"gridcolor": _GRID, "linecolor": _GRID, "zeroline": False},
        "yaxis": {"gridcolor": _GRID, "linecolor": _GRID, "zeroline": False},
    }


def _hline(y: float, color: str, dash: str = "dot", yref: str = "y") -> dict:
    """Horizontal reference line (paper x-coords, data y-coords)."""
    return {
        "type": "line", "xref": "paper", "yref": yref,
        "x0": 0, "x1": 1, "y0": y, "y1": y,
        "line": {"color": color, "width": 1, "dash": dash},
    }


def _hrect(y0: float, y1: float, color: str, yref: str = "y") -> dict:
    """Horizontal shaded band."""
    return {
        "type": "rect", "xref": "paper", "yref": yref,
        "x0": 0, "x1": 1, "y0": y0, "y1": y1,
        "fillcolor": color + "18", "line": {"width": 0}, "layer": "below",
    }


def _xaxis_cfg(nticks: int = 12) -> dict:
    """Standard category x-axis config for candlestick/multi-panel charts."""
    return {
        "rangeslider": {"visible": False},
        "gridcolor": _GRID, "linecolor": _GRID,
        "domain": [0, 1],
        "nticks": nticks, "tickangle": -30, "tickfont": {"size": 9},
    }


def _yaxis_cfg(domain: tuple[float, float], title: str = "", **extra) -> dict:
    base = {
        "domain": list(domain),
        "gridcolor": _GRID, "linecolor": _GRID,
        "zeroline": False,
    }
    if title:
        base["title"] = {"text": title, "font": {"size": 10}}
    base.update(extra)
    return base
