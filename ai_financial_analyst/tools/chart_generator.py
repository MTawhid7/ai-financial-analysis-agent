"""Chart generator — Plotly JSON for all standard financial chart types.

Supports: price action (candlestick + volume + SMA), fundamental trends
(revenue/margins/cashflow/debt), technical indicators (RSI, MACD),
comparison (normalised return), and risk (drawdown).

On-demand charts fetch their own yfinance data so they work for any
ticker, with any period, at any time — no prior analysis required.

All functions return a plain serialisable Plotly dict or None.
No image files. No REPL — data from yfinance only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
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

# ── Period → yfinance interval mapping ───────────────────────────────────────
_PERIOD_INTERVAL: dict[str, str] = {
    "1d":  "1h",
    "5d":  "1h",
    "1mo": "1d",
    "3mo": "1d",
    "6mo": "1d",
    "1y":  "1d",
    "2y":  "1wk",
    "5y":  "1wk",
    "10y": "1mo",
    "ytd": "1d",
    "max": "1mo",
}

def _interval_for(period: str) -> str:
    return _PERIOD_INTERVAL.get(period, "1d")


# ── Layout / shape helpers ────────────────────────────────────────────────────

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


def _hline(y: float, color: str, dash: str = "dot") -> dict:
    return {
        "type": "line", "xref": "paper", "yref": "y",
        "x0": 0, "x1": 1, "y0": y, "y1": y,
        "line": {"color": color, "width": 1, "dash": dash},
    }


def _hrect(y0: float, y1: float, color: str) -> dict:
    return {
        "type": "rect", "xref": "paper", "yref": "y",
        "x0": 0, "x1": 1, "y0": y0, "y1": y1,
        "fillcolor": color + "18", "line": {"width": 0}, "layer": "below",
    }


def _get_row(df: Any, *names: str) -> Any:
    """Return first matching row from a yfinance financial DataFrame."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name].dropna()
    return None


# ── 1. Price line ─────────────────────────────────────────────────────────────

def generate_price_chart(ticker: str, raw_data: dict, period: str = "1y") -> dict | None:
    """Weekly/monthly price line with 52-week high/low dashed bands."""
    try:
        import yfinance as yf
        interval = _interval_for(period)
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist.empty:
            return None

        dates  = [str(d.date()) for d in hist.index]
        prices = hist["Close"].round(2).tolist()
        shapes: list[dict] = []

        ph = (raw_data.get(ticker) or {}).get("price_history", {})
        for val, color in [(ph.get("52w_high"), _GREEN), (ph.get("52w_low"), _RED)]:
            if val:
                shapes.append(_hline(val, color))

        layout = _base_layout(f"{ticker} — Price ({period})")
        layout["shapes"] = shapes
        return {
            "data": [{
                "type": "scatter", "mode": "lines",
                "x": dates, "y": prices,
                "line": {"color": _BLUE, "width": 2},
                "name": ticker,
                "hovertemplate": "%{x}<br>$%{y:.2f}<extra></extra>",
            }],
            "layout": layout,
        }
    except Exception as exc:
        logger.warning("Price chart %s: %s", ticker, exc)
        return None


# ── 2. Candlestick + Volume + SMA-50/200 ─────────────────────────────────────

def generate_candlestick_chart(ticker: str, period: str = "1y") -> dict | None:
    """OHLCV candlestick with SMA-50/200 overlays and volume subplot."""
    try:
        import yfinance as yf
        interval = _interval_for(period)
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if len(hist) < 10:
            return None

        dates  = [str(d.date()) for d in hist.index]
        opens  = hist["Open"].round(2).tolist()
        highs  = hist["High"].round(2).tolist()
        lows   = hist["Low"].round(2).tolist()
        closes = hist["Close"].round(2).tolist()
        vols   = hist["Volume"].tolist()

        close_s = hist["Close"]
        sma50  = close_s.rolling(min(50,  len(close_s)), min_periods=1).mean().round(2).tolist()
        sma200 = close_s.rolling(min(200, len(close_s)), min_periods=1).mean().round(2).tolist()
        vol_colors = [_GREEN if c >= o else _RED for c, o in zip(closes, opens)]

        data = [
            {
                "type": "candlestick", "xaxis": "x", "yaxis": "y",
                "x": dates,
                "open": opens, "high": highs, "low": lows, "close": closes,
                "increasing": {"line": {"color": _GREEN}, "fillcolor": _GREEN + "66"},
                "decreasing": {"line": {"color": _RED},   "fillcolor": _RED   + "66"},
                "name": ticker, "showlegend": False,
                "hovertemplate": "O $%{open:.2f}  H $%{high:.2f}<br>L $%{low:.2f}  C $%{close:.2f}<extra></extra>",
            },
            {
                "type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
                "x": dates, "y": sma50,
                "line": {"color": _AMBER, "width": 1.2},
                "name": "SMA-50",
                "hovertemplate": "SMA-50: $%{y:.2f}<extra></extra>",
            },
            {
                "type": "scatter", "mode": "lines", "xaxis": "x", "yaxis": "y",
                "x": dates, "y": sma200,
                "line": {"color": _PURPLE, "width": 1.2, "dash": "dash"},
                "name": "SMA-200",
                "hovertemplate": "SMA-200: $%{y:.2f}<extra></extra>",
            },
            {
                "type": "bar", "xaxis": "x", "yaxis": "y2",
                "x": dates, "y": vols,
                "marker": {"color": vol_colors, "opacity": 0.55},
                "name": "Volume",
                "hovertemplate": "%{y:,.0f}<extra>Vol</extra>",
            },
        ]

        layout = _base_layout(f"{ticker} — Candlestick ({period})", height=440)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        layout.update({
            "xaxis": {
                "rangeslider": {"visible": False},
                "gridcolor": _GRID, "linecolor": _GRID, "domain": [0, 1],
                "nticks": 12, "tickangle": -30, "tickfont": {"size": 9},
            },
            "yaxis":  {"domain": [0.28, 1.0], "gridcolor": _GRID, "linecolor": _GRID,
                       "title": {"text": "Price ($)", "font": {"size": 10}}},
            "yaxis2": {"domain": [0.0, 0.24], "gridcolor": _GRID, "linecolor": _GRID,
                       "title": {"text": "Volume", "font": {"size": 10}}},
        })
        return {"data": data, "layout": layout}
    except Exception as exc:
        logger.warning("Candlestick %s: %s", ticker, exc)
        return None


# ── 3. P/E comparison ─────────────────────────────────────────────────────────

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
        logger.warning("P/E chart %s: %s", ticker, exc); return None


# ── 4. Key metrics bar ────────────────────────────────────────────────────────

def generate_metrics_chart(ticker: str, raw_data: dict) -> dict | None:
    """Horizontal bar: Market Cap, Revenue TTM, Net Income."""
    try:
        fund = (raw_data.get(ticker) or {}).get("fundamentals", {})
        if not fund or isinstance(fund, str):
            return None
        rows = []
        for key, label, div in [("market_cap", "Market Cap", 1e12), ("revenue_ttm", "Revenue (TTM)", 1e9), ("net_income_ttm", "Net Income", 1e9)]:
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
        logger.warning("Metrics chart %s: %s", ticker, exc); return None


# ── 5. Radar ──────────────────────────────────────────────────────────────────

def generate_radar_chart(ticker: str, raw_data: dict, analysis: dict) -> dict | None:
    """Spider chart: Growth, Valuation, Profitability, Scale (0-10)."""
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
        logger.warning("Radar chart %s: %s", ticker, exc); return None


# ── 6. Revenue vs Net Income ──────────────────────────────────────────────────

def generate_revenue_trend_chart(ticker: str) -> dict | None:
    """Annual Revenue vs Net Income grouped bar (last 5 years)."""
    try:
        import yfinance as yf
        fin = yf.Ticker(ticker).financials
        rev = _get_row(fin, "Total Revenue", "Revenue")
        ni  = _get_row(fin, "Net Income")
        if rev is None or ni is None:
            return None
        dates = sorted(rev.index.intersection(ni.index))[-5:]
        if len(dates) < 2:
            return None
        years = [str(d.year) for d in dates]
        layout = _base_layout(f"{ticker} — Revenue vs Net Income (Annual, $B)")
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
        logger.warning("Revenue trend %s: %s", ticker, exc); return None


# ── 7. Margin trends ──────────────────────────────────────────────────────────

def generate_margin_trend_chart(ticker: str) -> dict | None:
    """Gross / Operating / Net margin % trends (annual, last 5Y)."""
    try:
        import yfinance as yf
        fin = yf.Ticker(ticker).financials
        rev = _get_row(fin, "Total Revenue", "Revenue")
        if rev is None:
            return None
        dates = sorted(rev.index[rev > 0])[-5:]
        if len(dates) < 2:
            return None
        traces = []
        for row_names, label, color in [
            (("Gross Profit",),                            "Gross Margin",     _BLUE),
            (("Operating Income", "EBIT", "Ebit"),         "Operating Margin", _AMBER),
            (("Net Income",),                              "Net Margin",       _GREEN),
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
        layout = _base_layout(f"{ticker} — Margin Trends (%)")
        layout["yaxis"]["title"] = {"text": "Margin (%)", "font": {"size": 10}}
        layout["yaxis"]["ticksuffix"] = "%"
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Margin trend %s: %s", ticker, exc); return None


# ── 8. Free Cash Flow ─────────────────────────────────────────────────────────

def generate_cashflow_chart(ticker: str) -> dict | None:
    """Operating Cash Flow vs Free Cash Flow (annual, last 5Y)."""
    try:
        import yfinance as yf
        cf  = yf.Ticker(ticker).cashflow
        ocf = _get_row(cf, "Operating Cash Flow", "Total Cash From Operating Activities",
                       "Cash Flow From Continuing Operating Activities")
        fcf   = _get_row(cf, "Free Cash Flow")
        capex = _get_row(cf, "Capital Expenditure", "Purchase Of Plant", "Capital Expenditures")
        if ocf is None:
            return None
        if fcf is None and capex is not None:
            import pandas as pd
            common = ocf.index.intersection(capex.index)
            fcf = pd.Series({d: float(ocf[d]) + float(capex[d]) for d in common})
        dates = sorted(ocf.index)[-5:]
        years = [str(d.year) for d in dates]
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
        layout = _base_layout(f"{ticker} — Cash Flow (Annual, $B)")
        layout["barmode"] = "group"
        layout["yaxis"]["title"] = {"text": "USD Billions ($B)", "font": {"size": 10}}
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Cashflow chart %s: %s", ticker, exc); return None


# ── 9. Debt profile ───────────────────────────────────────────────────────────

def generate_debt_profile_chart(ticker: str) -> dict | None:
    """Short-term + long-term debt vs cash (annual, last 5Y)."""
    try:
        import yfinance as yf
        bs   = yf.Ticker(ticker).balance_sheet
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
        for row, label, color in [(std, "Short-term Debt", _RED), (ltd, "Long-term Debt", _ORANGE), (cash, "Cash & Equivalents", _GREEN)]:
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
        layout = _base_layout(f"{ticker} — Debt & Cash Profile (Annual, $B)")
        layout["barmode"] = "group"
        layout["yaxis"]["title"] = {"text": "USD Billions ($B)", "font": {"size": 10}}
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Debt profile %s: %s", ticker, exc); return None


# ── 10. Normalised relative return ────────────────────────────────────────────

def generate_relative_performance_chart(tickers: list[str], period: str = "1y") -> dict | None:
    """Normalised % return (base = 100) for up to 6 tickers."""
    try:
        import yfinance as yf, pandas as pd
        interval = _interval_for(period)
        frames: dict[str, Any] = {}
        for t in tickers[:6]:
            hist = yf.Ticker(t.upper()).history(period=period, interval=interval)
            if not hist.empty:
                frames[t.upper()] = hist["Close"].dropna()
        if len(frames) < 2:
            return None
        df = pd.DataFrame(frames).dropna(how="all")
        if len(df) < 5:
            return None
        normalized = df / df.iloc[0] * 100
        dates = [str(d.date()) for d in df.index]
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
        label  = " vs ".join(list(frames.keys()))
        layout = _base_layout(f"Normalised Return — {label} ({period})", height=340)
        layout["yaxis"]["title"] = {"text": "Indexed (base = 100)", "font": {"size": 10}}
        layout["shapes"] = [_hline(100, _ZINC, "dot")]
        return {"data": traces, "layout": layout}
    except Exception as exc:
        logger.warning("Relative perf chart: %s", exc); return None


# ── 11. Drawdown ──────────────────────────────────────────────────────────────

def generate_drawdown_chart(ticker: str, period: str = "2y") -> dict | None:
    """Drawdown from rolling peak; period controls how far back to look."""
    try:
        import yfinance as yf
        interval = _interval_for(period)
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if len(hist) < 20:
            return None
        close  = hist["Close"].dropna()
        dd_pct = ((close - close.cummax()) / close.cummax() * 100).round(2)
        dates  = [str(d.date()) for d in close.index]
        vals   = dd_pct.tolist()
        min_dd = min(vals)
        layout = _base_layout(f"{ticker} — Drawdown from Peak ({period})")
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
        logger.warning("Drawdown chart %s: %s", ticker, exc); return None


# ── 12. RSI(14) ───────────────────────────────────────────────────────────────

def generate_rsi_chart(ticker: str, period: str = "1y") -> dict | None:
    """RSI(14) with overbought/oversold shaded zones."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        if len(hist) < 20:
            return None
        close = hist["Close"].dropna()
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).round(2)
        dates = [str(d.date()) for d in close.index]
        layout = _base_layout(f"{ticker} — RSI(14) ({period})")
        layout["yaxis"]["title"] = {"text": "RSI", "font": {"size": 10}}
        layout["yaxis"]["range"] = [0, 100]
        layout["shapes"] = [_hrect(70, 100, _RED), _hrect(0, 30, _GREEN),
                             _hline(70, _RED), _hline(30, _GREEN), _hline(50, _ZINC)]
        return {"data": [{"type": "scatter", "mode": "lines",
                          "x": dates, "y": rsi.fillna(50).tolist(),
                          "line": {"color": _BLUE, "width": 1.5}, "name": "RSI(14)",
                          "hovertemplate": "%{x}<br>RSI: %{y:.1f}<extra></extra>"}],
                "layout": layout}
    except Exception as exc:
        logger.warning("RSI chart %s: %s", ticker, exc); return None


# ── 13. MACD(12,26,9) ─────────────────────────────────────────────────────────

def generate_macd_chart(ticker: str, period: str = "1y") -> dict | None:
    """MACD histogram + MACD line + signal line."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        if len(hist) < 35:
            return None
        close  = hist["Close"].dropna()
        macd   = (close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()).round(4)
        signal = macd.ewm(span=9, adjust=False).mean().round(4)
        histo  = (macd - signal).round(4)
        dates  = [str(d.date()) for d in close.index]
        layout = _base_layout(f"{ticker} — MACD(12,26,9) ({period})")
        layout["yaxis"]["title"] = {"text": "MACD", "font": {"size": 10}}
        layout["shapes"] = [_hline(0, _ZINC)]
        return {
            "data": [
                {"type": "bar",
                 "x": dates, "y": histo.tolist(),
                 "marker": {"color": [_GREEN if v >= 0 else _RED for v in histo.tolist()], "opacity": 0.7},
                 "name": "Histogram", "hovertemplate": "Hist: %{y:.4f}<extra></extra>"},
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
        logger.warning("MACD chart %s: %s", ticker, exc); return None


# ── On-demand dispatcher ──────────────────────────────────────────────────────

def generate_on_demand_chart(
    ticker: str,
    chart_type: str,
    raw_data: dict,
    analysis: dict,
    period: str = "1y",
    compare_tickers: list[str] | None = None,
) -> dict | None:
    """Route a chart_type string to the correct generator, respecting period."""
    ct = chart_type.lower().replace(" ", "_").replace("-", "_")

    # Multi-ticker comparison
    if ct in ("relative_performance", "comparison", "compare", "normalized", "normalised",
              "relative_return", "peer_comparison"):
        tickers_list = compare_tickers or ([ticker] if ticker else [])
        return generate_relative_performance_chart(tickers_list, period)

    # Price action
    if ct in ("candlestick", "ohlc", "candle", "ohlcv"):
        return generate_candlestick_chart(ticker, period)
    if ct in ("price", "price_history", "price_chart", "line"):
        return generate_price_chart(ticker, raw_data, period)

    # Technical indicators
    if ct in ("rsi", "relative_strength", "relative_strength_index"):
        return generate_rsi_chart(ticker, period)
    if ct in ("macd", "momentum"):
        return generate_macd_chart(ticker, period)

    # Fundamental trends (always fetch fresh)
    if ct in ("revenue_trend", "revenue", "income", "revenue_vs_income", "revenue_vs_profit"):
        return generate_revenue_trend_chart(ticker)
    if ct in ("margin_trend", "margins", "margin", "profitability"):
        return generate_margin_trend_chart(ticker)
    if ct in ("cashflow", "cash_flow", "fcf", "free_cash_flow"):
        return generate_cashflow_chart(ticker)
    if ct in ("debt_profile", "debt", "debt_vs_cash", "balance_sheet_chart", "debt_chart"):
        return generate_debt_profile_chart(ticker)

    # Risk
    if ct in ("drawdown", "dd", "max_drawdown", "risk"):
        return generate_drawdown_chart(ticker, period)

    # Pipeline-integrated types
    if ct in ("pe", "pe_comparison", "valuation", "pe_ratio"):
        return generate_pe_chart(ticker, analysis)
    if ct in ("metrics", "financials", "key_financials"):
        return generate_metrics_chart(ticker, raw_data)
    if ct in ("radar", "profile", "financial_profile", "spider"):
        return generate_radar_chart(ticker, raw_data, analysis)

    # Default: candlestick with the requested period
    return generate_candlestick_chart(ticker, period) or generate_price_chart(ticker, raw_data, period)


# ── Pipeline entry point ──────────────────────────────────────────────────────

def generate_all_charts(final_state: Any) -> list[dict]:
    """Generate the standard post-analysis chart set (5 charts per ticker)."""
    charts: list[dict] = []
    raw_data: dict = final_state.get("raw_data", {}) if final_state else {}
    analysis: dict = final_state.get("analysis", {}) if final_state else {}

    for ticker in raw_data:
        for fn, ct, suffix in [
            (lambda t: generate_candlestick_chart(t, "1y"), "candlestick", "Price — 1Y Candlestick"),
            (lambda t: generate_pe_chart(t, analysis),      "pe",          "P/E vs Sector"),
            (lambda t: generate_revenue_trend_chart(t),     "revenue",     "Revenue vs Net Income"),
            (lambda t: generate_margin_trend_chart(t),      "margins",     "Margin Trends"),
            (lambda t: generate_radar_chart(t, raw_data, analysis), "radar", "Financial Profile"),
        ]:
            try:
                fig = fn(ticker)
                if fig:
                    charts.append({"ticker": ticker, "chart_type": ct,
                                   "title": f"{ticker} — {suffix}", "figure": fig})
            except Exception as exc:
                logger.warning("generate_all_charts %s/%s: %s", ticker, ct, exc)
    return charts
