"""Fundamental trend charts: revenue, margins, cash flow, debt profile."""
from __future__ import annotations

import logging

from ._theme import _base_layout, _BLUE, _GREEN, _RED, _AMBER, _CYAN, _ORANGE, _ZINC
from ._data import _get_row

logger = logging.getLogger(__name__)


def generate_revenue_trend_chart(ticker: str) -> dict | None:
    """Annual Revenue vs Net Income grouped bar (last 5 years)."""
    try:
        import yfinance as yf
        fin = yf.Ticker(ticker.upper()).financials
        rev = _get_row(fin, "Total Revenue", "Revenue")
        ni  = _get_row(fin, "Net Income")
        if rev is None or ni is None:
            return None
        dates = sorted(rev.index.intersection(ni.index))[-5:]
        if len(dates) < 2:
            return None
        years  = [str(d.year) for d in dates]
        layout = _base_layout(f"{ticker.upper()} — Revenue vs Net Income (Annual, $B)")
        layout["barmode"] = "group"
        layout["yaxis"]["title"] = {"text": "USD Billions ($B)", "font": {"size": 10}}
        return {
            "data": [
                {"type": "bar", "name": "Revenue",
                 "x": years, "y": [round(float(rev[d]) / 1e9, 2) for d in dates],
                 "marker": {"color": _BLUE},
                 "hovertemplate": "Revenue: $%{y:.2f}B<extra></extra>"},
                {"type": "bar", "name": "Net Income",
                 "x": years, "y": [round(float(ni[d]) / 1e9, 2) for d in dates],
                 "marker": {"color": _GREEN},
                 "hovertemplate": "Net Income: $%{y:.2f}B<extra></extra>"},
            ],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Revenue trend %s: %s", ticker, exc)
        return None


def generate_margin_trend_chart(ticker: str) -> dict | None:
    """Gross / Operating / Net margin % trends (annual, last 5Y)."""
    try:
        import yfinance as yf
        fin = yf.Ticker(ticker.upper()).financials
        rev = _get_row(fin, "Total Revenue", "Revenue")
        if rev is None:
            return None
        dates = sorted(rev.index[rev > 0])[-5:]
        if len(dates) < 2:
            return None
        traces = []
        for row_names, label, color in [
            (("Gross Profit",),                         "Gross Margin",     _BLUE),
            (("Operating Income", "EBIT", "Ebit"),      "Operating Margin", _AMBER),
            (("Net Income",),                           "Net Margin",       _GREEN),
        ]:
            row = _get_row(fin, *row_names)
            if row is None:
                continue
            pts = [(d, round(float(row[d]) / float(rev[d]) * 100, 1)) for d in dates if d in row.index]
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            traces.append({"type": "scatter", "mode": "lines+markers",
                           "x": [str(d.year) for d in xs], "y": list(ys),
                           "line": {"color": color, "width": 2}, "marker": {"size": 6},
                           "name": label,
                           "hovertemplate": f"{label}: %{{y:.1f}}%<extra></extra>"})
        if not traces:
            return None
        layout = _base_layout(f"{ticker.upper()} — Margin Trends (%)")
        layout["yaxis"]["title"]      = {"text": "Margin (%)", "font": {"size": 10}}
        layout["yaxis"]["ticksuffix"] = "%"
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Margin trend %s: %s", ticker, exc)
        return None


def generate_cashflow_chart(ticker: str) -> dict | None:
    """Operating Cash Flow vs Free Cash Flow bars (annual, last 5Y)."""
    try:
        import yfinance as yf, pandas as pd
        cf    = yf.Ticker(ticker.upper()).cashflow
        ocf   = _get_row(cf, "Operating Cash Flow", "Total Cash From Operating Activities",
                         "Cash Flow From Continuing Operating Activities")
        fcf   = _get_row(cf, "Free Cash Flow")
        capex = _get_row(cf, "Capital Expenditure", "Purchase Of Plant", "Capital Expenditures")
        if ocf is None:
            return None
        if fcf is None and capex is not None:
            common = ocf.index.intersection(capex.index)
            fcf = pd.Series({d: float(ocf[d]) + float(capex[d]) for d in common})
        dates  = sorted(ocf.index)[-5:]
        years  = [str(d.year) for d in dates]
        traces = [{"type": "bar", "name": "Operating CF",
                   "x": years, "y": [round(float(ocf[d]) / 1e9, 2) for d in dates],
                   "marker": {"color": _BLUE},
                   "hovertemplate": "Oper. CF: $%{y:.2f}B<extra></extra>"}]
        if fcf is not None:
            fcf_pts = [(d, round(float(fcf[d]) / 1e9, 2)) for d in dates if d in fcf.index]
            if fcf_pts:
                xs, ys = zip(*fcf_pts)
                traces.append({"type": "bar", "name": "Free Cash Flow",
                               "x": [str(d.year) for d in xs], "y": list(ys),
                               "marker": {"color": [_GREEN if v >= 0 else _RED for v in ys]},
                               "hovertemplate": "FCF: $%{y:.2f}B<extra></extra>"})
        layout = _base_layout(f"{ticker.upper()} — Cash Flow (Annual, $B)")
        layout["barmode"] = "group"
        layout["yaxis"]["title"] = {"text": "USD Billions ($B)", "font": {"size": 10}}
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Cashflow chart %s: %s", ticker, exc)
        return None


def generate_debt_profile_chart(ticker: str) -> dict | None:
    """Short-term + long-term debt vs cash (annual, last 5Y)."""
    try:
        import yfinance as yf
        bs   = yf.Ticker(ticker.upper()).balance_sheet
        cash = _get_row(bs, "Cash And Cash Equivalents",
                        "Cash Cash Equivalents And Short Term Investments",
                        "Cash And Short Term Investments")
        ltd  = _get_row(bs, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
        std  = _get_row(bs, "Current Debt", "Short Long Term Debt", "Short Term Borrowings",
                        "Current Portion Of Long Term Debt")
        if cash is None and ltd is None:
            return None
        anchor = cash if cash is not None else ltd
        dates  = sorted(anchor.index)[-5:]
        traces = []
        for row, label, color in [
            (std,  "Short-term Debt",    _RED),
            (ltd,  "Long-term Debt",     _ORANGE),
            (cash, "Cash & Equivalents", _GREEN),
        ]:
            if row is None:
                continue
            pts = [(d, round(float(row[d]) / 1e9, 2)) for d in dates if d in row.index]
            if not pts:
                continue
            xs, ys = zip(*pts)
            traces.append({"type": "bar", "name": label,
                           "x": [str(d.year) for d in xs], "y": list(ys),
                           "marker": {"color": color},
                           "hovertemplate": f"{label}: $%{{y:.2f}}B<extra></extra>"})
        if not traces:
            return None
        layout = _base_layout(f"{ticker.upper()} — Debt & Cash Profile (Annual, $B)")
        layout["barmode"] = "group"
        layout["yaxis"]["title"] = {"text": "USD Billions ($B)", "font": {"size": 10}}
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Debt profile %s: %s", ticker, exc)
        return None
