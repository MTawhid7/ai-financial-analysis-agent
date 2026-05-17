"""Yahoo Finance — price_metrics data type (Sharpe, Sortino, beta, drawdown, CAGR)."""

from __future__ import annotations

import math

import yfinance as yf

from ..base import DataResult, assess_quality, null_result, safe_float, utc_now


def fetch(ticker: str) -> DataResult:
    import numpy as np
    from ..market.risk_free import get_risk_free_rate
    from ..market.sp500 import get_sp500_data

    stock = yf.Ticker(ticker)
    ts    = utc_now()

    hist = stock.history(period="5y", interval="1d", auto_adjust=True)
    if hist.empty:
        return null_result(ticker, "price_metrics", "Price history unavailable for risk metrics")

    prices  = hist["Close"].dropna()
    returns = prices.pct_change().dropna()

    if len(returns) < 60:
        return null_result(ticker, "price_metrics", "Insufficient history for risk metrics (< 60 days)")

    n_years = max(len(prices) / 252, 0.001)
    total_return_cagr = round(float((prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1) * 100, 2)
    volatility_annual = round(float(returns.std()) * math.sqrt(252) * 100, 2)

    rfr_annual = get_risk_free_rate()
    rfr_daily  = rfr_annual / 252
    excess     = returns - rfr_daily

    sharpe = sortino = None
    if excess.std() > 0:
        sharpe = round(float(excess.mean() / excess.std() * math.sqrt(252)), 3)
    downside = excess[excess < 0]
    if len(downside) > 5 and downside.std() > 0:
        sortino = round(float(excess.mean() * 252 / (downside.std() * math.sqrt(252))), 3)

    rolling_max  = prices.cummax()
    max_drawdown = round(float(((prices - rolling_max) / rolling_max).min()) * 100, 2)

    beta = relative_cagr = sp500_cagr_pct = None
    sp500 = get_sp500_data("5y")
    if sp500 and sp500.get("returns"):
        import pandas as pd
        sp_ret = pd.Series(sp500["returns"])
        n = min(len(returns), len(sp_ret))
        aligned_stock = returns.values[-n:]
        aligned_sp    = sp_ret.values[-n:]
        if len(aligned_stock) > 60:
            cov    = float(np.cov(aligned_stock, aligned_sp)[0, 1])
            sp_var = float(np.var(aligned_sp))
            if sp_var > 0:
                beta = round(cov / sp_var, 3)
            sp500_cagr_pct = round(sp500["cagr"] * 100, 2)
            relative_cagr  = round(total_return_cagr - sp500_cagr_pct, 2)

    quality, degradation = assess_quality(
        required={"sharpe_ratio": sharpe, "max_drawdown_pct": max_drawdown},
        optional={"beta_vs_sp500": beta},
    )

    return DataResult(
        ticker           = ticker,
        data_type        = "price_metrics",
        data_quality     = quality,
        data_timestamp   = ts,
        degradation_note = degradation,
        payload          = {
            "total_return_cagr_pct": total_return_cagr,
            "volatility_annual_pct": volatility_annual,
            "sharpe_ratio":          sharpe,
            "sortino_ratio":         sortino,
            "max_drawdown_pct":      max_drawdown,
            "beta_vs_sp500":         beta,
            "sp500_cagr_pct":        sp500_cagr_pct,
            "relative_cagr_pct":     relative_cagr,
            "risk_free_rate_used":   round(rfr_annual * 100, 3),
            "data_period_years":     round(n_years, 1),
        },
    )
