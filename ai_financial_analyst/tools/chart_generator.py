"""Chart generator — produces Plotly JSON from pipeline analysis data.

All functions return a plain dict (serialisable Plotly figure) that the
frontend passes directly to react-plotly.js.  No image files are written.

The no-REPL invariant is preserved: chart data is constructed from a fixed
set of keys in AgentState — no arbitrary user-supplied code is evaluated.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Dark theme palette consistent with the React frontend
_BG = "#18181b"
_PAPER_BG = "#27272a"
_FONT_COLOR = "#e4e4e7"
_GRID_COLOR = "#3f3f46"
_ACCENT = "#7c3aed"
_ACCENT2 = "#2563eb"


def _base_layout(title: str) -> dict:
    return {
        "title": {"text": title, "font": {"color": _FONT_COLOR, "size": 14}},
        "paper_bgcolor": _PAPER_BG,
        "plot_bgcolor": _BG,
        "font": {"color": _FONT_COLOR, "family": "Inter, system-ui, sans-serif"},
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "xaxis": {"gridcolor": _GRID_COLOR, "linecolor": _GRID_COLOR},
        "yaxis": {"gridcolor": _GRID_COLOR, "linecolor": _GRID_COLOR},
    }


# ---------------------------------------------------------------------------
# Price history chart — line chart from yfinance (1-year weekly)
# ---------------------------------------------------------------------------


def generate_price_chart(ticker: str, raw_data: dict) -> dict | None:
    """Return a Plotly line chart for the ticker's 1-year price history.

    Fetches data directly from yfinance (will hit the 4-hour diskcache
    if the same ticker was already fetched during the pipeline run).
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y", interval="1wk")
        if hist.empty:
            return None

        dates = [str(d.date()) for d in hist.index]
        prices = [round(float(p), 2) for p in hist["Close"]]

        # Add 52-week high/low reference lines from raw_data if available
        shapes = []
        price_data = raw_data.get(ticker, {}).get("price_history", {})
        high_52w = price_data.get("52w_high")
        low_52w = price_data.get("52w_low")

        for val, color, label in [
            (high_52w, "#22c55e", "52w High"),
            (low_52w, "#ef4444", "52w Low"),
        ]:
            if val:
                shapes.append({
                    "type": "line", "xref": "paper", "yref": "y",
                    "x0": 0, "x1": 1, "y0": val, "y1": val,
                    "line": {"color": color, "width": 1, "dash": "dot"},
                })

        layout = _base_layout(f"{ticker} — Price (1 Year)")
        layout["shapes"] = shapes

        return {
            "data": [{
                "type": "scatter",
                "mode": "lines",
                "x": dates,
                "y": prices,
                "line": {"color": _ACCENT, "width": 2},
                "name": ticker,
                "hovertemplate": "%{x}<br>$%{y:.2f}<extra></extra>",
            }],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Could not generate price chart for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# P/E comparison chart — horizontal bar
# ---------------------------------------------------------------------------


def generate_pe_chart(ticker: str, analysis: dict) -> dict | None:
    """Return a Plotly horizontal bar chart comparing company P/E vs sector."""
    try:
        ticker_analysis = analysis.get(ticker, {})
        company_pe = ticker_analysis.get("company_pe")
        sector_pe = ticker_analysis.get("sector_pe_avg")
        sector = ticker_analysis.get("sector", "Sector")

        if not company_pe or not sector_pe:
            return None

        premium = ticker_analysis.get("pe_vs_sector_premium_pct", 0)
        bar_color = "#ef4444" if premium > 20 else "#22c55e" if premium < -10 else _ACCENT2

        layout = _base_layout(f"{ticker} — P/E vs {sector} Sector")
        layout["xaxis"]["title"] = {"text": "P/E Ratio", "font": {"color": _FONT_COLOR}}

        return {
            "data": [
                {
                    "type": "bar",
                    "orientation": "h",
                    "x": [round(company_pe, 1)],
                    "y": [ticker],
                    "marker": {"color": bar_color},
                    "name": f"{ticker} P/E",
                    "hovertemplate": "P/E: %{x:.1f}x<extra></extra>",
                },
                {
                    "type": "bar",
                    "orientation": "h",
                    "x": [round(sector_pe, 1)],
                    "y": ["Sector avg"],
                    "marker": {"color": _GRID_COLOR},
                    "name": f"{sector} avg",
                    "hovertemplate": "Sector avg: %{x:.1f}x<extra></extra>",
                },
            ],
            "layout": {**layout, "barmode": "group"},
        }
    except Exception as exc:
        logger.warning("Could not generate P/E chart for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Key metrics chart — horizontal bar for revenue, earnings, market cap
# ---------------------------------------------------------------------------


def generate_metrics_chart(ticker: str, raw_data: dict) -> dict | None:
    """Return a Plotly bar chart for key financial metrics."""
    try:
        fundamentals = raw_data.get(ticker, {}).get("fundamentals", {})
        if not fundamentals or isinstance(fundamentals, str):
            return None

        metrics = {}
        for key, label, divisor in [
            ("market_cap", "Market Cap", 1e12),
            ("revenue_ttm", "Revenue (TTM)", 1e9),
            ("net_income_ttm", "Net Income", 1e9),
        ]:
            val = fundamentals.get(key)
            if val and isinstance(val, (int, float)) and val > 0:
                metrics[label] = round(val / divisor, 2)

        if len(metrics) < 2:
            return None

        labels = list(metrics.keys())
        values = list(metrics.values())
        units = ["T" if k == "Market Cap" else "B" for k in labels]

        layout = _base_layout(f"{ticker} — Key Financials")
        layout["xaxis"]["title"] = {"text": "Value (T = Trillion, B = Billion)", "font": {"color": _FONT_COLOR}}

        return {
            "data": [{
                "type": "bar",
                "orientation": "h",
                "x": values,
                "y": labels,
                "marker": {"color": [_ACCENT, _ACCENT2, "#0891b2"][:len(labels)]},
                "hovertemplate": "%{y}: $%{x:.2f}%{customdata}<extra></extra>",
                "customdata": units,
            }],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Could not generate metrics chart for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Bull/Bear radar chart
# ---------------------------------------------------------------------------


def generate_radar_chart(ticker: str, raw_data: dict, analysis: dict) -> dict | None:
    """Return a Plotly radar chart scoring the stock across 5 financial dimensions."""
    try:
        ta = analysis.get(ticker, {})
        fund = (raw_data.get(ticker) or {}).get("fundamentals") or {}
        if isinstance(fund, str):
            import json as _j
            fund = _j.loads(fund)

        scores: dict[str, float] = {}

        # Growth: 5Y CAGR (score 0-10, 20%+ CAGR = 10)
        cagr = ta.get("price_cagr_5y_pct")
        if cagr is not None:
            scores["Growth"] = min(float(cagr) / 2, 10)

        # Valuation (lower P/E premium → better score)
        premium = ta.get("pe_vs_sector_premium_pct")
        if premium is not None:
            scores["Valuation"] = max(0, 10 - float(premium) / 10)

        # Profitability: profit margin (40%+ margin = 10)
        margin = fund.get("profit_margin")
        if margin is not None:
            scores["Profitability"] = min(float(margin) * 25, 10)

        # Size / Stability: market cap ($1T+ = 10)
        mcap = fund.get("market_cap")
        if mcap and isinstance(mcap, (int, float)):
            scores["Scale"] = min(float(mcap) / 1e11, 10)

        if len(scores) < 3:
            return None

        categories = list(scores.keys())
        values = [scores[c] for c in categories]
        # Close the radar polygon
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        layout = _base_layout(f"{ticker} — Financial Profile")
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        layout["polar"] = {
            "bgcolor": _BG,
            "radialaxis": {"visible": True, "range": [0, 10], "color": _GRID_COLOR, "gridcolor": _GRID_COLOR},
            "angularaxis": {"color": _FONT_COLOR, "gridcolor": _GRID_COLOR},
        }

        return {
            "data": [{
                "type": "scatterpolar",
                "r": values_closed,
                "theta": categories_closed,
                "fill": "toself",
                "fillcolor": f"{_ACCENT}33",
                "line": {"color": _ACCENT, "width": 2},
                "name": ticker,
                "hovertemplate": "%{theta}: %{r:.1f}/10<extra></extra>",
            }],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Could not generate radar chart for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# On-demand chart by type (called by Manager generate_chart tool)
# ---------------------------------------------------------------------------


def generate_on_demand_chart(ticker: str, chart_type: str, raw_data: dict, analysis: dict) -> dict | None:
    """Generate a specific chart type on user request."""
    ct = chart_type.lower().replace(" ", "_").replace("-", "_")
    if ct in ("price", "price_history", "price_chart"):
        return generate_price_chart(ticker, raw_data)
    if ct in ("pe", "pe_comparison", "valuation"):
        return generate_pe_chart(ticker, analysis)
    if ct in ("metrics", "financials", "key_financials", "revenue"):
        return generate_metrics_chart(ticker, raw_data)
    if ct in ("radar", "profile", "financial_profile"):
        return generate_radar_chart(ticker, raw_data, analysis)
    # Default: generate all and return first
    return generate_price_chart(ticker, raw_data)


# ---------------------------------------------------------------------------
# Main entry point — called after pipeline completes
# ---------------------------------------------------------------------------


def generate_all_charts(final_state: Any) -> list[dict]:
    """Generate all charts for a completed pipeline run.

    Returns a list of chart descriptors:
    [{"ticker": "AAPL", "chart_type": "price", "title": "...", "figure": {...}}]
    """
    charts: list[dict] = []
    raw_data: dict = final_state.get("raw_data", {}) if final_state else {}
    analysis: dict = final_state.get("analysis", {}) if final_state else {}

    for ticker in raw_data:
        price_fig = generate_price_chart(ticker, raw_data)
        if price_fig:
            charts.append({"ticker": ticker, "chart_type": "price",
                           "title": f"{ticker} — Price History (1Y)", "figure": price_fig})

        pe_fig = generate_pe_chart(ticker, analysis)
        if pe_fig:
            charts.append({"ticker": ticker, "chart_type": "pe",
                           "title": f"{ticker} — P/E vs Sector", "figure": pe_fig})

        metrics_fig = generate_metrics_chart(ticker, raw_data)
        if metrics_fig:
            charts.append({"ticker": ticker, "chart_type": "metrics",
                           "title": f"{ticker} — Key Financials", "figure": metrics_fig})

        radar_fig = generate_radar_chart(ticker, raw_data, analysis)
        if radar_fig:
            charts.append({"ticker": ticker, "chart_type": "radar",
                           "title": f"{ticker} — Financial Profile", "figure": radar_fig})

    return charts
