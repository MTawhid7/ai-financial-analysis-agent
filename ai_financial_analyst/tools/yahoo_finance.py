"""YahooFinanceTool — comprehensive financial data via yfinance (free tier).

Seven data types, each with the appropriate cache TTL:
  price_history      — adjusted OHLCV, true 52-week window, data freshness flag,
                       data_quality grade, corporate events (splits)
  fundamentals       — 25+ metrics + analyst recommendations (upgrade/downgrade history)
  balance_sheet      — assets, liabilities, equity, cash, debt ratios
  cash_flow          — OCF, FCF, capex, D&A, FCF yield, dividend payment history
  earnings           — next earnings date, trailing EPS surprises
  price_metrics      — Sharpe, Sortino, max drawdown, beta, volatility, relative CAGR
  financials_trend   — quarterly income statement + balance sheet trend (last 4 quarters)

Every response now includes a `data_quality` field:
  "FULL"        — all expected fields retrieved for the requested period
  "PARTIAL"     — some fields missing or history window shorter than requested
  "UNAVAILABLE" — no data returned (response is a null sentinel)

When data_quality is "PARTIAL", a `degradation_note` field explains the gap.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Literal

import yfinance as yf
from langchain_core.tools import tool
from pydantic import Field

from ..core.cache import (
    ResultCache,
    TTL_PRICE, TTL_FUNDAMENTALS, TTL_FINANCIALS,
)
from .base import StrictToolInput, ToolError, ErrorType, safe_tool_call

logger = logging.getLogger(__name__)

_cache = ResultCache()


class YahooFinanceInput(StrictToolInput):
    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL')")
    data_type: Literal[
        "price_history",
        "fundamentals",
        "balance_sheet",
        "cash_flow",
        "earnings",
        "price_metrics",
        "financials_trend",
    ] = Field(description="Type of financial data to retrieve")


# Map data_type → cache TTL
_TYPE_TTL = {
    "price_history":    TTL_PRICE,
    "fundamentals":     TTL_FUNDAMENTALS,
    "balance_sheet":    TTL_FINANCIALS,
    "cash_flow":        TTL_FINANCIALS,
    "earnings":         TTL_FUNDAMENTALS,
    "price_metrics":    TTL_FUNDAMENTALS,
    "financials_trend": TTL_FINANCIALS,
}


@tool("yahoo_finance", args_schema=YahooFinanceInput)
def yahoo_finance_tool(ticker: str, data_type: str) -> str:
    """Fetch financial data from Yahoo Finance for a given ticker.

    Returns structured JSON. Each data_type is cached at an appropriate TTL
    so stale prices are never returned while slow-moving financials can be
    cached longer. Returns an explicit null with a reason string when data
    is unavailable.
    """
    args = {"ticker": ticker.upper(), "data_type": data_type}
    ttl  = _TYPE_TTL.get(data_type, TTL_FUNDAMENTALS)

    def _fetch() -> str:
        return _fetch_data(ticker.upper(), data_type)

    result, hit = _cache.get_or_fetch("yahoo_finance", args, _fetch, ttl=ttl)
    if hit:
        logger.debug("yfinance cache HIT ticker=%s data_type=%s", ticker, data_type)
    return result


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------

def _fetch_data(ticker: str, data_type: str) -> str:
    def _run():
        stock = yf.Ticker(ticker)
        ts    = datetime.utcnow().isoformat() + "Z"

        if data_type == "price_history":
            return _price_history(stock, ticker, ts)
        if data_type == "fundamentals":
            return _fundamentals(stock, ticker, ts)
        if data_type == "balance_sheet":
            return _balance_sheet(stock, ticker, ts)
        if data_type == "cash_flow":
            return _cash_flow(stock, ticker, ts)
        if data_type == "earnings":
            return _earnings(stock, ticker, ts)
        if data_type == "price_metrics":
            return _price_metrics(stock, ticker, ts)
        if data_type == "financials_trend":
            return _financials_trend(stock, ticker, ts)

        return ToolError(
            error_type=ErrorType.TOOL_ERROR,
            tool="yahoo_finance",
            message=f"Unknown data_type: {data_type}",
            input={"ticker": ticker, "data_type": data_type},
        ).to_json()

    return safe_tool_call("yahoo_finance", _run, {"ticker": ticker, "data_type": data_type})


# ---------------------------------------------------------------------------
# 1. Price history — adjusted close, true 52-week window
# ---------------------------------------------------------------------------

def _price_history(stock: yf.Ticker, ticker: str, ts: str) -> str:
    # auto_adjust=True applies split/dividend adjustments to Close.
    # interval="1wk" keeps payload small while covering the full 5-year span.
    # Track which period actually returns data so we can surface degradation.
    hist = None
    period_received: str | None = None

    for period, interval in [("5y", "1wk"), ("2y", "1wk"), ("1y", "1d")]:
        h = stock.history(period=period, interval=interval, auto_adjust=True)
        if not h.empty:
            hist = h
            period_received = period
            break

    if hist is None or hist.empty:
        return _null(ticker, "price_history", "No price history available")

    closes = hist["Close"].dropna()
    highs  = hist["High"].dropna()
    lows   = hist["Low"].dropna()

    # True 52-week window: exactly 365 calendar days back from last data point
    cutoff           = hist.index[-1] - timedelta(days=365)
    last_year_closes = closes.loc[closes.index >= cutoff]
    last_year_highs  = highs.loc[highs.index  >= cutoff]
    last_year_lows   = lows.loc[lows.index   >= cutoff]

    current_price = round(float(closes.iloc[-1]), 2)
    price_5y_ago  = round(float(closes.iloc[0]),  2)

    # ── Data quality assessment ───────────────────────────────────────────────
    # Minimum expected data points per period (weekly bars)
    _MIN_POINTS = {"5y": 200, "2y": 80, "1y": 40}
    degradation_note: str | None = None

    if period_received != "5y":
        data_quality     = "PARTIAL"
        degradation_note = (
            f"Requested 5-year history; only {period_received} of data available. "
            "CAGR and long-term trend figures will be less reliable."
        )
    elif len(closes) < _MIN_POINTS.get(period_received, 40):
        data_quality     = "PARTIAL"
        degradation_note = (
            f"Fewer data points than expected ({len(closes)} bars). "
            "May indicate sparse trading or a recently listed stock."
        )
    else:
        data_quality = "FULL"

    # ── Freshness check ───────────────────────────────────────────────────────
    freshness_warning = None
    try:
        fi_price = getattr(stock.fast_info, "last_price", None)
        if fi_price and abs(fi_price - current_price) / max(current_price, 1) > 0.05:
            freshness_warning = (
                f"Price may be stale: cached close ${current_price:.2f} vs "
                f"live ~${fi_price:.2f}. Consider re-fetching."
            )
    except Exception:
        pass

    # ── Corporate events: stock splits (last 5 years) ────────────────────────
    corporate_events: list[dict] = []
    try:
        splits = stock.splits
        if splits is not None and not splits.empty:
            event_cutoff = hist.index[-1] - timedelta(days=365 * 5)
            recent_splits = splits[splits.index >= event_cutoff]
            for dt, ratio in recent_splits.items():
                if ratio and float(ratio) != 1.0:
                    r = float(ratio)
                    if r > 1:
                        description = f"{r:.0f}-for-1 stock split"
                    else:
                        description = f"1-for-{1/r:.0f} reverse split"
                    corporate_events.append({
                        "date":        str(dt)[:10],
                        "type":        "split",
                        "ratio":       round(r, 4),
                        "description": description,
                    })
    except Exception:
        pass

    return json.dumps({
        "ticker":            ticker,
        "data_type":         "price_history",
        "data_timestamp":    ts,
        "data_quality":      data_quality,
        "degradation_note":  degradation_note,
        "period_requested":  "5y",
        "period_received":   period_received,
        "current_price":     current_price,
        "price_5y_ago":      price_5y_ago,
        "52w_high":          _sf(last_year_highs.max()) if not last_year_highs.empty else None,
        "52w_low":           _sf(last_year_lows.min())  if not last_year_lows.empty  else None,
        "data_points":       len(closes),
        "price_adjusted":    True,
        "freshness_warning": freshness_warning,
        "corporate_events":  corporate_events,
    })


# ---------------------------------------------------------------------------
# 2. Fundamentals — 25+ fields from info + fast_info
# ---------------------------------------------------------------------------

def _fundamentals(stock: yf.Ticker, ticker: str, ts: str) -> str:
    info = stock.info

    # fast_info gives a more real-time current price than info dict
    current_price = None
    try:
        fi = stock.fast_info
        current_price = _sf(getattr(fi, "last_price", None))
    except Exception:
        pass
    if current_price is None:
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    has_price = current_price or info.get("previousClose")
    if not info or not has_price:
        return _null(ticker, "fundamentals", "Fundamentals unavailable")

    # Analyst price targets
    apt = {}
    try:
        apt = stock.analyst_price_targets or {}
    except Exception:
        pass

    # Data quality: FULL requires price + sector + at least one valuation multiple
    _core_fields = [current_price, info.get("sector"), info.get("marketCap")]
    _valuation   = [info.get("trailingPE"), info.get("forwardPE"), info.get("enterpriseToEbitda")]
    if all(_core_fields) and any(_valuation):
        fund_quality      = "FULL"
        fund_degradation  = None
    else:
        fund_quality      = "PARTIAL"
        missing = [k for k, v in {
            "price": current_price, "sector": info.get("sector"),
            "market_cap": info.get("marketCap"), "pe_ratio": info.get("trailingPE"),
        }.items() if not v]
        fund_degradation = f"Missing fields: {', '.join(missing)}"

    return json.dumps({
        "ticker":           ticker,
        "data_type":        "fundamentals",
        "data_timestamp":   ts,
        "data_quality":     fund_quality,
        "degradation_note": fund_degradation,
        # Price
        "current_price":    current_price,
        # Valuation multiples
        "pe_ratio":         _sf(info.get("trailingPE")),
        "forward_pe":       _sf(info.get("forwardPE")),
        "peg_ratio":        _sf(info.get("pegRatio")),
        "price_to_book":    _sf(info.get("priceToBook")),
        "price_to_sales":   _sf(info.get("priceToSalesTrailing12Months")),
        "ev_to_ebitda":     _sf(info.get("enterpriseToEbitda")),
        "ev_to_revenue":    _sf(info.get("enterpriseToRevenue")),
        "enterprise_value": info.get("enterpriseValue"),
        # Scale
        "market_cap":       info.get("marketCap"),
        "revenue_ttm":      info.get("totalRevenue"),
        "net_income_ttm":   info.get("netIncomeToCommon"),
        # Per-share
        "trailing_eps":     _sf(info.get("trailingEps")),
        "forward_eps":      _sf(info.get("forwardEps")),
        "book_value":       _sf(info.get("bookValue")),
        # Profitability
        "gross_margin":     _sf(info.get("grossMargins")),
        "operating_margin": _sf(info.get("operatingMargins")),
        "profit_margin":    _sf(info.get("profitMargins")),
        "return_on_equity": _sf(info.get("returnOnEquity")),
        "return_on_assets": _sf(info.get("returnOnAssets")),
        # Leverage & liquidity
        "debt_to_equity":   _sf(info.get("debtToEquity")),
        "current_ratio":    _sf(info.get("currentRatio")),
        "quick_ratio":      _sf(info.get("quickRatio")),
        # Growth
        "revenue_growth":   _sf(info.get("revenueGrowth")),
        "earnings_growth":  _sf(info.get("earningsGrowth")),
        # Market beta (from info — may differ from computed beta)
        "beta":             _sf(info.get("beta")),
        # Dividends
        "dividend_yield":   _sf(info.get("dividendYield")),
        "dividend_rate":    _sf(info.get("dividendRate")),
        "payout_ratio":     _sf(info.get("payoutRatio")),
        # Ownership & sentiment
        "short_ratio":              _sf(info.get("shortRatio")),
        "institutional_ownership":  _sf(info.get("heldPercentInstitutions")),
        "sp500_52w_change":         _sf(info.get("SandP52WeekChange")),
        "52w_change":               _sf(info.get("52WeekChange")),
        # Classification
        "sector":           info.get("sector"),
        "industry":         info.get("industry"),
        "country":          info.get("country"),
        "exchange":         info.get("exchange"),
        "company_name":     info.get("longName"),
        "employees":        info.get("fullTimeEmployees"),
        # Analyst consensus (aggregate — single-source, directional only)
        "analyst_target_mean":   _sf(apt.get("mean")),
        "analyst_target_high":   _sf(apt.get("high")),
        "analyst_target_low":    _sf(apt.get("low")),
        "analyst_target_median": _sf(apt.get("median")),
        # Analyst upgrade/downgrade history (last 10 actions)
        "analyst_recommendations": _analyst_recommendations(stock),
    })


# ---------------------------------------------------------------------------
# 3. Balance sheet — with derived ratios
# ---------------------------------------------------------------------------

def _balance_sheet(stock: yf.Ticker, ticker: str, ts: str) -> str:
    bs = stock.balance_sheet
    if bs is None or bs.empty:
        return _null(ticker, "balance_sheet", "Balance sheet unavailable")

    latest = bs.iloc[:, 0]  # most recent annual column
    total_assets  = _sf(latest.get("Total Assets"))
    total_liab    = _sf(latest.get("Total Liabilities Net Minority Interest"))
    equity        = _sf(latest.get("Stockholders Equity"))
    cash          = _sf(latest.get("Cash And Cash Equivalents"))
    lt_debt       = _sf(latest.get("Long Term Debt"))
    st_debt       = _sf(latest.get(
        "Current Debt",
        latest.get("Short Long Term Debt", latest.get("Current Portion Of Long Term Debt"))
    ))
    total_debt    = _sf(latest.get("Total Debt"))
    current_assets = _sf(latest.get("Current Assets"))
    current_liab   = _sf(latest.get("Current Liabilities"))
    inventory      = _sf(latest.get("Inventory"))

    # Derived ratios
    net_debt = None
    if total_debt is not None and cash is not None:
        net_debt = round(total_debt - cash, 2)

    current_ratio = None
    if current_assets and current_liab and current_liab > 0:
        current_ratio = round(current_assets / current_liab, 2)

    quick_ratio = None
    if current_assets and inventory is not None and current_liab and current_liab > 0:
        quick_ratio = round((current_assets - inventory) / current_liab, 2)

    _bs_core = [total_assets, total_liab, equity]
    bs_quality     = "FULL" if all(_bs_core) else "PARTIAL"
    bs_degradation = None if bs_quality == "FULL" else (
        "Missing: " + ", ".join(
            k for k, v in {"total_assets": total_assets,
                           "total_liabilities": total_liab,
                           "equity": equity}.items() if not v
        )
    )

    return json.dumps({
        "ticker":               ticker,
        "data_type":            "balance_sheet",
        "data_timestamp":       ts,
        "data_quality":         bs_quality,
        "degradation_note":     bs_degradation,
        "total_assets":         total_assets,
        "total_liabilities":    total_liab,
        "stockholders_equity":  equity,
        "cash_and_equivalents": cash,
        "long_term_debt":       lt_debt,
        "short_term_debt":      st_debt,
        "total_debt":           total_debt,
        "net_debt":             net_debt,
        "current_assets":       current_assets,
        "current_liabilities":  current_liab,
        "inventory":            inventory,
        "current_ratio_calc":   current_ratio,
        "quick_ratio_calc":     quick_ratio,
    })


# ---------------------------------------------------------------------------
# 4. Cash flow — OCF, FCF, capex, D&A, yields
# ---------------------------------------------------------------------------

def _cash_flow(stock: yf.Ticker, ticker: str, ts: str) -> str:
    cf = stock.cashflow
    if cf is None or cf.empty:
        # Try quarterly as fallback
        cf = stock.quarterly_cashflow
    if cf is None or cf.empty:
        return _null(ticker, "cash_flow", "Cash flow statement unavailable")

    latest = cf.iloc[:, 0]  # most recent period

    ocf   = _sf(latest.get("Operating Cash Flow",
                           latest.get("Total Cash From Operating Activities")))
    fcf   = _sf(latest.get("Free Cash Flow"))
    capex = _sf(latest.get("Capital Expenditure",
                           latest.get("Purchase Of Plant")))
    da    = _sf(latest.get("Depreciation And Amortization",
                           latest.get("Depreciation")))
    wc    = _sf(latest.get("Change In Working Capital"))

    # Compute FCF if not directly available: FCF = OCF + Capex (capex is negative)
    if fcf is None and ocf is not None and capex is not None:
        fcf = round(float(ocf) + float(capex), 2)

    # FCF yield = FCF / Market Cap
    fcf_yield = None
    try:
        mc = stock.fast_info.market_cap or stock.info.get("marketCap")
        if fcf is not None and mc and mc > 0:
            fcf_yield = round(float(fcf) / float(mc), 6)
    except Exception:
        pass

    # Cash conversion ratio = OCF / Net Income (>1 = good earnings quality)
    cash_conversion = None
    try:
        ni_row = _get_row(cf, "Net Income")
        if ni_row is not None and ocf is not None:
            ni = float(ni_row.iloc[0])
            if ni != 0:
                cash_conversion = round(float(ocf) / ni, 4)
    except Exception:
        pass

    # 4-year OCF trend
    ocf_trend: list[float | None] = []
    try:
        for col in cf.columns[:4]:
            v = cf[col].get("Operating Cash Flow",
                            cf[col].get("Total Cash From Operating Activities"))
            ocf_trend.append(_sf(v))
    except Exception:
        pass

    # Dividend payment history (date-indexed series of per-share amounts)
    dividend_history = None
    try:
        divs = stock.dividends
        if divs is not None and not divs.empty:
            cutoff = divs.index[-1] - timedelta(days=365 * 3)
            recent = divs[divs.index >= cutoff]

            # Annual totals per calendar year
            annual: dict[str, float] = {}
            for dt, amt in recent.items():
                yr = str(dt.year)
                annual[yr] = round(annual.get(yr, 0.0) + float(amt), 4)

            # Dividend CAGR across available annual totals
            div_cagr = None
            years = sorted(annual.keys())
            if len(years) >= 2:
                oldest, newest = annual[years[0]], annual[years[-1]]
                n = len(years) - 1
                if oldest > 0 and n > 0:
                    div_cagr = round(((newest / oldest) ** (1 / n) - 1) * 100, 2)

            dividend_history = {
                "recent_payments": [
                    {"date": str(dt)[:10], "amount": round(float(amt), 4)}
                    for dt, amt in recent.iloc[-8:].items()
                ],
                "annual_totals":         annual,
                "most_recent_date":      str(divs.index[-1])[:10],
                "most_recent_amount":    round(float(divs.iloc[-1]), 4),
                "dividend_cagr_3y_pct":  div_cagr,
            }
    except Exception:
        pass

    cf_quality     = "FULL" if (ocf is not None and fcf is not None) else "PARTIAL"
    cf_degradation = None if cf_quality == "FULL" else (
        "Missing: " + ", ".join(
            k for k, v in {"operating_cash_flow": ocf, "free_cash_flow": fcf}.items() if v is None
        )
    )

    return json.dumps({
        "ticker":                  ticker,
        "data_type":               "cash_flow",
        "data_timestamp":          ts,
        "data_quality":            cf_quality,
        "degradation_note":        cf_degradation,
        "operating_cash_flow":     ocf,
        "free_cash_flow":          fcf,
        "capital_expenditure":     capex,
        "depreciation_amortization": da,
        "change_in_working_capital": wc,
        "fcf_yield":               fcf_yield,
        "cash_conversion_ratio":   cash_conversion,
        "ocf_4yr_trend":           ocf_trend,
        "dividend_history":        dividend_history,
    })


# ---------------------------------------------------------------------------
# 5. Earnings — next date + trailing surprise history
# ---------------------------------------------------------------------------

def _earnings(stock: yf.Ticker, ticker: str, ts: str) -> str:
    next_date: str | None = None
    eps_estimate: float | None = None
    rev_estimate: float | None = None

    try:
        cal = stock.calendar
        if isinstance(cal, dict):
            nd = cal.get("Earnings Date")
            if nd is not None:
                if hasattr(nd, "__iter__") and not isinstance(nd, str):
                    nd = list(nd)[0]
                next_date = str(nd)[:10] if nd else None
            eps_estimate = _sf(cal.get("EPS Estimate"))
            rev_estimate = _sf(cal.get("Revenue Estimate"))
    except Exception:
        pass

    # Trailing earnings surprises
    surprises: list[dict] = []
    try:
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            # Filter to past dates (Reported EPS is available)
            reported = ed[ed.get("Reported EPS", ed.columns[0]).notna()] if "Reported EPS" in ed.columns else ed
            for dt, row in reported.head(8).iterrows():
                entry: dict = {"date": str(dt)[:10]}
                for col in ["EPS Estimate", "Reported EPS", "Surprise(%)"]:
                    if col in row:
                        entry[col.lower().replace(" ", "_").replace("(%)", "_pct")] = _sf(row[col])
                surprises.append(entry)
    except Exception:
        pass

    if next_date is None and not surprises:
        return _null(ticker, "earnings", "Earnings data unavailable")

    earn_quality     = "FULL" if (next_date is not None and len(surprises) >= 4) else "PARTIAL"
    earn_degradation = None if earn_quality == "FULL" else (
        ("Missing next earnings date. " if next_date is None else "") +
        (f"Only {len(surprises)} quarter(s) of surprise history (expect 4+)." if len(surprises) < 4 else "")
    ).strip() or None

    return json.dumps({
        "ticker":               ticker,
        "data_type":            "earnings",
        "data_timestamp":       ts,
        "data_quality":         earn_quality,
        "degradation_note":     earn_degradation,
        "next_earnings_date":   next_date,
        "eps_estimate":         eps_estimate,
        "revenue_estimate":     rev_estimate,
        "earnings_surprises":   surprises,   # list, most recent first
    })


# ---------------------------------------------------------------------------
# 6. Price metrics — Sharpe, Sortino, max drawdown, beta, relative CAGR
# ---------------------------------------------------------------------------

def _price_metrics(stock: yf.Ticker, ticker: str, ts: str) -> str:
    import numpy as np
    from .market_data import get_risk_free_rate, get_sp500_data

    # Fetch 5-year daily adjusted prices
    hist = stock.history(period="5y", interval="1d", auto_adjust=True)
    if hist.empty:
        return _null(ticker, "price_metrics", "Price history unavailable for risk metrics")

    prices  = hist["Close"].dropna()
    returns = prices.pct_change().dropna()

    if len(returns) < 60:  # need at least 60 trading days
        return _null(ticker, "price_metrics", "Insufficient history for risk metrics")

    n_years = max(len(prices) / 252, 0.001)

    # Total return CAGR (adjusted, includes dividends)
    total_return_cagr = round(
        float((prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1) * 100, 2
    )

    # Annualised volatility
    volatility_annual = round(float(returns.std()) * math.sqrt(252) * 100, 2)

    # Risk-free rate
    rfr_annual = get_risk_free_rate()    # e.g. 0.042
    rfr_daily  = rfr_annual / 252

    # Excess returns
    excess = returns - rfr_daily

    # Sharpe ratio
    sharpe = None
    if excess.std() > 0:
        sharpe = round(float(excess.mean() / excess.std() * math.sqrt(252)), 3)

    # Sortino ratio (uses downside deviation only)
    sortino = None
    downside = excess[excess < 0]
    if len(downside) > 5 and downside.std() > 0:
        sortino = round(float(excess.mean() * 252 / (downside.std() * math.sqrt(252))), 3)

    # Max drawdown
    rolling_max = prices.cummax()
    drawdown    = (prices - rolling_max) / rolling_max
    max_drawdown = round(float(drawdown.min()) * 100, 2)  # e.g. -34.2 (%)

    # Beta vs S&P 500 + relative CAGR
    beta = None
    relative_cagr = None
    sp500_cagr_pct = None

    sp500 = get_sp500_data("5y")
    if sp500 and sp500.get("returns"):
        import pandas as pd
        sp_ret = pd.Series(sp500["returns"])
        # Align by length (both are daily; may differ slightly)
        n = min(len(returns), len(sp_ret))
        aligned_stock = returns.values[-n:]
        aligned_sp    = sp_ret.values[-n:]
        if len(aligned_stock) > 60:
            cov      = float(np.cov(aligned_stock, aligned_sp)[0, 1])
            sp_var   = float(np.var(aligned_sp))
            if sp_var > 0:
                beta = round(cov / sp_var, 3)
            sp500_cagr_pct  = round(sp500["cagr"] * 100, 2)
            relative_cagr   = round(total_return_cagr - sp500_cagr_pct, 2)

    pm_core        = [sharpe, max_drawdown]
    pm_quality     = "FULL" if all(pm_core) and beta is not None else "PARTIAL"
    pm_degradation = None if pm_quality == "FULL" else (
        "Missing: " + ", ".join(
            k for k, v in {"sharpe_ratio": sharpe, "beta": beta,
                           "max_drawdown": max_drawdown}.items() if v is None
        )
    )

    return json.dumps({
        "ticker":               ticker,
        "data_type":            "price_metrics",
        "data_timestamp":       ts,
        "data_quality":         pm_quality,
        "degradation_note":     pm_degradation,
        "total_return_cagr_pct": total_return_cagr,
        "volatility_annual_pct": volatility_annual,
        "sharpe_ratio":         sharpe,
        "sortino_ratio":        sortino,
        "max_drawdown_pct":     max_drawdown,
        "beta_vs_sp500":        beta,
        "sp500_cagr_pct":       sp500_cagr_pct,
        "relative_cagr_pct":    relative_cagr,   # stock CAGR − S&P 500 CAGR
        "risk_free_rate_used":  round(rfr_annual * 100, 3),
        "data_period_years":    round(n_years, 1),
    })


# ---------------------------------------------------------------------------
# 7. Quarterly financial trends — income statement + balance sheet
# ---------------------------------------------------------------------------

def _financials_trend(stock: yf.Ticker, ticker: str, ts: str) -> str:
    """Last 4 quarters of revenue/earnings + balance sheet snapshots."""
    qf  = stock.quarterly_financials
    qbs = stock.quarterly_balance_sheet

    income_trend: list[dict] = []
    if qf is not None and not qf.empty:
        rev_row = _get_row(qf, "Total Revenue", "Revenue")
        ni_row  = _get_row(qf, "Net Income")
        gp_row  = _get_row(qf, "Gross Profit")

        cols = list(qf.columns[:5])   # up to 5 quarters — need 4 + 1 for YoY
        for i, col in enumerate(cols[:4]):
            entry: dict = {"quarter": str(col)[:10]}
            rev = _sf(rev_row[col]) if rev_row is not None and col in rev_row.index else None
            ni  = _sf(ni_row[col])  if ni_row  is not None and col in ni_row.index  else None
            gp  = _sf(gp_row[col])  if gp_row  is not None and col in gp_row.index  else None
            entry["revenue"]    = rev
            entry["net_income"] = ni

            # Gross margin
            if rev and gp and rev != 0:
                entry["gross_margin_pct"] = round(gp / rev * 100, 2)

            # Net margin
            if rev and ni and rev != 0:
                entry["net_margin_pct"] = round(ni / rev * 100, 2)

            # QoQ revenue growth (vs previous quarter)
            if i + 1 < len(cols) and rev_row is not None:
                prev_col = cols[i + 1]
                prev_rev = _sf(rev_row[prev_col]) if prev_col in rev_row.index else None
                if rev and prev_rev and prev_rev != 0:
                    entry["revenue_qoq_pct"] = round((rev - prev_rev) / abs(prev_rev) * 100, 2)

            # YoY revenue growth (vs same quarter prior year, index 4)
            yoy_col_idx = i + 4
            if yoy_col_idx < len(cols) and rev_row is not None:
                yoy_col = cols[yoy_col_idx]
                yoy_rev = _sf(rev_row[yoy_col]) if yoy_col in rev_row.index else None
                if rev and yoy_rev and yoy_rev != 0:
                    entry["revenue_yoy_pct"] = round((rev - yoy_rev) / abs(yoy_rev) * 100, 2)

            income_trend.append(entry)

    balance_trend: list[dict] = []
    if qbs is not None and not qbs.empty:
        cash_row  = _get_row(qbs, "Cash And Cash Equivalents",
                             "Cash Cash Equivalents And Short Term Investments")
        debt_row  = _get_row(qbs, "Total Debt", "Long Term Debt")
        eq_row    = _get_row(qbs, "Stockholders Equity")
        for col in list(qbs.columns[:4]):
            entry = {"quarter": str(col)[:10]}
            if cash_row is not None and col in cash_row.index:
                entry["cash"] = _sf(cash_row[col])
            if debt_row is not None and col in debt_row.index:
                entry["total_debt"] = _sf(debt_row[col])
            if eq_row is not None and col in eq_row.index:
                entry["equity"] = _sf(eq_row[col])
            balance_trend.append(entry)

    if not income_trend and not balance_trend:
        return _null(ticker, "financials_trend", "Quarterly financials unavailable")

    ft_quality     = "FULL" if len(income_trend) >= 4 else "PARTIAL"
    ft_degradation = None if ft_quality == "FULL" else (
        f"Only {len(income_trend)} quarter(s) of income data available (expect 4+)."
    )

    return json.dumps({
        "ticker":           ticker,
        "data_type":        "financials_trend",
        "data_timestamp":   ts,
        "data_quality":     ft_quality,
        "degradation_note": ft_degradation,
        "income_trend":     income_trend,    # newest quarter first
        "balance_trend":    balance_trend,   # newest quarter first
    })


# ---------------------------------------------------------------------------
# Analyst recommendations helper
# ---------------------------------------------------------------------------

def _analyst_recommendations(stock: yf.Ticker) -> dict | None:
    """Return last 10 analyst upgrade/downgrade actions with sentiment summary."""
    try:
        recs = stock.recommendations
        if recs is None or recs.empty:
            return None

        def _sentiment(grade: str) -> str:
            g = grade.lower()
            if any(k in g for k in ("buy", "overweight", "outperform", "strong buy",
                                    "positive", "add", "accumulate")):
                return "positive"
            if any(k in g for k in ("sell", "underweight", "underperform", "reduce",
                                    "negative", "strong sell")):
                return "negative"
            return "neutral"

        counts = {"positive": 0, "neutral": 0, "negative": 0}
        recent: list[dict] = []
        for dt, row in recs.tail(10).iterrows():
            to_grade   = str(row.get("To Grade",   row.get("toGrade",   "")))
            from_grade = str(row.get("From Grade", row.get("fromGrade", "")))
            firm       = str(row.get("Firm",       row.get("firm",      "")))
            action     = str(row.get("Action",     row.get("action",    "")))
            s = _sentiment(to_grade)
            counts[s] += 1
            recent.append({
                "date":       str(dt)[:10],
                "firm":       firm,
                "from_grade": from_grade,
                "to_grade":   to_grade,
                "action":     action,
                "sentiment":  s,
            })

        return {"recent": recent, "sentiment_counts": counts}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _null(ticker: str, data_type: str, reason: str) -> str:
    return json.dumps({
        "ticker":         ticker,
        "data_type":      data_type,
        "data_timestamp": datetime.utcnow().isoformat() + "Z",
        "data_quality":   "UNAVAILABLE",
        "result":         None,
        "reason":         reason,
    })


def _sf(val) -> float | None:
    """Safe float conversion, returning None on failure."""
    try:
        return round(float(val), 6) if val is not None and not math.isnan(float(val)) else None
    except (TypeError, ValueError):
        return None


def _get_row(df, *names):
    """First matching row from a financial DataFrame."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None
