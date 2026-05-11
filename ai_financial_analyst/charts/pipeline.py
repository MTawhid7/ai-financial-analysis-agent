"""Pipeline charts — generated after full analysis (use cached AgentState data)."""
from __future__ import annotations

import logging

from ._theme import (
    _base_layout, _BLUE, _GREEN, _RED, _AMBER, _CYAN, _ZINC, _GRID, _BG, _FONT,
)

logger = logging.getLogger(__name__)


def generate_pe_chart(ticker: str, analysis: dict) -> dict | None:
    """Horizontal bar: company P/E vs sector average."""
    try:
        ta = analysis.get(ticker, {})
        co_pe, sec_pe = ta.get("company_pe"), ta.get("sector_pe_avg")
        if not co_pe or not sec_pe:
            return None
        sector  = ta.get("sector", "Sector")
        premium = ta.get("pe_vs_sector_premium_pct", 0)
        bar_col = _RED if premium > 20 else (_GREEN if premium < -10 else _BLUE)
        layout  = _base_layout(f"{ticker} — P/E vs {sector} Sector")
        layout["barmode"] = "group"
        layout["xaxis"]["title"] = {"text": "P/E Ratio", "font": {"size": 10}}
        return {
            "data": [
                {"type": "bar", "orientation": "h", "x": [round(co_pe,  1)], "y": [ticker],
                 "marker": {"color": bar_col}, "name": f"{ticker} P/E",
                 "hovertemplate": "P/E: %{x:.1f}x<extra></extra>"},
                {"type": "bar", "orientation": "h", "x": [round(sec_pe, 1)], "y": ["Sector avg"],
                 "marker": {"color": _ZINC}, "name": f"{sector} avg",
                 "hovertemplate": "Sector avg: %{x:.1f}x<extra></extra>"},
            ],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("P/E chart %s: %s", ticker, exc)
        return None


def generate_metrics_chart(ticker: str, raw_data: dict) -> dict | None:
    """Horizontal bar: Market Cap, Revenue TTM, Net Income."""
    try:
        fund = (raw_data.get(ticker) or {}).get("fundamentals", {})
        if not fund or isinstance(fund, str):
            return None
        rows = []
        for key, label, div in [
            ("market_cap",     "Market Cap",    1e12),
            ("revenue_ttm",    "Revenue (TTM)", 1e9),
            ("net_income_ttm", "Net Income",    1e9),
        ]:
            v = fund.get(key)
            if v and isinstance(v, (int, float)) and v > 0:
                rows.append((label, round(v / div, 2), "T" if div == 1e12 else "B"))
        if len(rows) < 2:
            return None
        layout = _base_layout(f"{ticker} — Key Financials")
        layout["xaxis"]["title"] = {"text": "USD (T = Trillion, B = Billion)", "font": {"size": 10}}
        return {
            "data": [{"type": "bar", "orientation": "h",
                      "x": [r[1] for r in rows], "y": [r[0] for r in rows],
                      "customdata": [r[2] for r in rows],
                      "marker": {"color": [_BLUE, _CYAN, _GREEN][:len(rows)]},
                      "hovertemplate": "%{y}: $%{x:.2f}%{customdata}<extra></extra>"}],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Metrics chart %s: %s", ticker, exc)
        return None


def generate_radar_chart(ticker: str, raw_data: dict, analysis: dict) -> dict | None:
    """Spider chart: Growth, Valuation, Profitability, Scale (0-10 scores)."""
    try:
        ta   = analysis.get(ticker, {})
        fund = (raw_data.get(ticker) or {}).get("fundamentals") or {}
        if isinstance(fund, str):
            import json as _j; fund = _j.loads(fund)
        scores: dict[str, float] = {}
        cagr    = ta.get("price_cagr_5y_pct")
        premium = ta.get("pe_vs_sector_premium_pct")
        margin  = fund.get("profit_margin")
        mcap    = fund.get("market_cap")
        if cagr    is not None: scores["Growth"]        = min(float(cagr) / 2, 10)
        if premium is not None: scores["Valuation"]     = max(0, 10 - float(premium) / 10)
        if margin  is not None: scores["Profitability"] = min(float(margin) * 25, 10)
        if mcap and isinstance(mcap, (int, float)):
                                scores["Scale"]         = min(float(mcap) / 1e11, 10)
        if len(scores) < 3:
            return None
        cats = list(scores.keys()) + [list(scores.keys())[0]]
        vals = list(scores.values()) + [list(scores.values())[0]]
        layout = _base_layout(f"{ticker} — Financial Profile")
        layout.pop("xaxis", None); layout.pop("yaxis", None)
        layout["polar"] = {
            "bgcolor": _BG,
            "radialaxis": {"visible": True, "range": [0, 10], "color": _GRID, "gridcolor": _GRID},
            "angularaxis": {"color": _FONT, "gridcolor": _GRID},
        }
        return {"data": [{"type": "scatterpolar", "r": vals, "theta": cats,
                          "fill": "toself", "fillcolor": _BLUE + "30",
                          "line": {"color": _BLUE, "width": 2}, "name": ticker,
                          "hovertemplate": "%{theta}: %{r:.1f}/10<extra></extra>"}],
                "layout": layout}
    except Exception as exc:
        logger.warning("Radar chart %s: %s", ticker, exc)
        return None
