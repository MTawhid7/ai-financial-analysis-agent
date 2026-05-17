"""Chart dispatcher — routes chart_type strings to the right generator."""
from __future__ import annotations

import logging
from typing import Any

from ._data import _resolve_ticker
from .price_action import generate_price_chart, generate_candlestick_chart
from .technical import generate_rsi_chart, generate_macd_chart
from .combined import generate_price_rsi_chart, generate_price_macd_chart
from .fundamentals import (
    generate_revenue_trend_chart, generate_margin_trend_chart,
    generate_cashflow_chart, generate_debt_profile_chart,
)
from .comparison import generate_relative_performance_chart
from .risk import generate_drawdown_chart
from .pipeline import generate_pe_chart, generate_metrics_chart, generate_radar_chart

logger = logging.getLogger(__name__)


def generate_on_demand_chart(
    ticker: str,
    chart_type: str,
    raw_data: dict,
    analysis: dict,
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
    overlays: list[str] | None = None,
    compare_tickers: list[str] | None = None,
    **kwargs: Any,
) -> dict | None:
    """Route a chart_type string to the correct generator.

    All parameters are optional/keyword so callers can add new ones without
    breaking existing call sites. Extra kwargs (e.g. interval="5m") are forwarded
    to chart generators that accept them.
    """
    ct       = chart_type.lower().replace(" ", "_").replace("-", "_")
    interval = kwargs.get("interval")

    # ── Comparison (multi-ticker) ─────────────────────────────────────────────
    if ct in ("relative_performance", "comparison", "compare", "normalized",
              "normalised", "relative_return", "peer_comparison"):
        tickers_list = compare_tickers or ([ticker] if ticker else [])
        return generate_relative_performance_chart(tickers_list, period, start, end)

    # ── Combined panels ───────────────────────────────────────────────────────
    if ct in ("price_rsi", "candlestick_rsi", "price_with_rsi"):
        return generate_price_rsi_chart(ticker, period, start, end, overlays)
    if ct in ("price_macd", "candlestick_macd", "price_with_macd"):
        return generate_price_macd_chart(ticker, period, start, end, overlays)

    # ── Price action ──────────────────────────────────────────────────────────
    if ct in ("candlestick", "ohlc", "candle", "ohlcv"):
        return generate_candlestick_chart(
            ticker, period, start, end, overlays, raw_data=raw_data, interval=interval
        )
    if ct in ("price", "price_history", "price_chart", "line"):
        return generate_price_chart(ticker, raw_data, period, start, end, interval=interval)

    # ── Technical indicators ──────────────────────────────────────────────────
    if ct in ("rsi", "relative_strength", "relative_strength_index"):
        return generate_rsi_chart(ticker, period, start, end)
    if ct in ("macd", "momentum"):
        return generate_macd_chart(ticker, period, start, end)

    # ── Volume profile ────────────────────────────────────────────────────────
    if ct in ("volume_profile", "vap", "volume_at_price", "price_volume"):
        from .volume import generate_volume_profile_chart
        return generate_volume_profile_chart(ticker, period=period, start=start, end=end)

    # ── Fundamental trends ────────────────────────────────────────────────────
    if ct in ("revenue_trend", "revenue", "income", "revenue_vs_income", "revenue_vs_profit"):
        return generate_revenue_trend_chart(ticker)
    if ct in ("margin_trend", "margins", "margin", "profitability"):
        return generate_margin_trend_chart(ticker)
    if ct in ("cashflow", "cash_flow", "fcf", "free_cash_flow"):
        return generate_cashflow_chart(ticker)
    if ct in ("debt_profile", "debt", "debt_vs_cash", "balance_sheet_chart", "debt_chart"):
        return generate_debt_profile_chart(ticker)

    # ── Risk ──────────────────────────────────────────────────────────────────
    if ct in ("drawdown", "dd", "max_drawdown", "risk"):
        return generate_drawdown_chart(ticker, period, start, end)

    # ── Pipeline-integrated (use cached analysis data) ────────────────────────
    if ct in ("pe", "pe_comparison", "valuation", "pe_ratio"):
        return generate_pe_chart(ticker, analysis)
    if ct in ("metrics", "financials", "key_financials"):
        return generate_metrics_chart(ticker, raw_data)
    if ct in ("radar", "profile", "financial_profile", "spider"):
        return generate_radar_chart(ticker, raw_data, analysis)

    # Default: candlestick with the requested period/range
    return (
        generate_candlestick_chart(
            ticker, period, start, end, overlays, raw_data=raw_data, interval=interval
        )
        or generate_price_chart(ticker, raw_data, period, start, end, interval=interval)
    )


def generate_all_charts(final_state: Any) -> list[dict]:
    """Generate the standard post-analysis chart set (5 charts per ticker)."""
    charts: list[dict] = []
    raw_data: dict = final_state.get("raw_data", {}) if final_state else {}
    analysis: dict = final_state.get("analysis", {}) if final_state else {}

    for ticker in raw_data:
        for fn, ct, suffix in [
            (lambda t: generate_candlestick_chart(t, raw_data=raw_data),   "candlestick", "Price — 1Y Candlestick"),
            (lambda t: generate_pe_chart(t, analysis),                     "pe",          "P/E vs Sector"),
            (lambda t: generate_revenue_trend_chart(t),                    "revenue",     "Revenue vs Net Income"),
            (lambda t: generate_margin_trend_chart(t),                     "margins",     "Margin Trends"),
            (lambda t: generate_radar_chart(t, raw_data, analysis),        "radar",       "Financial Profile"),
        ]:
            try:
                fig = fn(ticker)
                if fig:
                    charts.append({"ticker": ticker, "chart_type": ct,
                                   "title": f"{ticker} — {suffix}", "figure": fig})
            except Exception as exc:
                logger.warning("generate_all_charts %s/%s: %s", ticker, ct, exc)
    return charts
