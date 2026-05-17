"""Yahoo Finance — cash_flow data type (OCF, FCF, capex, dividend history)."""

from __future__ import annotations

from datetime import timedelta

import yfinance as yf

from ..base import DataResult, assess_quality, null_result, safe_float, utc_now, get_first_row


def fetch(ticker: str) -> DataResult:
    stock = yf.Ticker(ticker)
    ts    = utc_now()

    cf = stock.cashflow
    if cf is None or cf.empty:
        cf = stock.quarterly_cashflow
    if cf is None or cf.empty:
        return null_result(ticker, "cash_flow", "Cash flow statement unavailable")

    latest = cf.iloc[:, 0]

    ocf   = safe_float(latest.get("Operating Cash Flow", latest.get("Total Cash From Operating Activities")))
    fcf   = safe_float(latest.get("Free Cash Flow"))
    capex = safe_float(latest.get("Capital Expenditure", latest.get("Purchase Of Plant")))
    da    = safe_float(latest.get("Depreciation And Amortization", latest.get("Depreciation")))
    wc    = safe_float(latest.get("Change In Working Capital"))

    if fcf is None and ocf is not None and capex is not None:
        fcf = round(float(ocf) + float(capex), 2)

    fcf_yield = None
    try:
        mc = stock.fast_info.market_cap or stock.info.get("marketCap")
        if fcf is not None and mc and mc > 0:
            fcf_yield = round(float(fcf) / float(mc), 6)
    except Exception:
        pass

    cash_conversion = None
    try:
        ni_row = get_first_row(cf, "Net Income")
        if ni_row is not None and ocf is not None:
            ni = float(ni_row.iloc[0])
            if ni != 0:
                cash_conversion = round(float(ocf) / ni, 4)
    except Exception:
        pass

    ocf_trend: list[float | None] = []
    try:
        for col in cf.columns[:4]:
            v = cf[col].get("Operating Cash Flow", cf[col].get("Total Cash From Operating Activities"))
            ocf_trend.append(safe_float(v))
    except Exception:
        pass

    dividend_history = _dividend_history(stock)

    quality, degradation = assess_quality(
        required={"operating_cash_flow": ocf, "free_cash_flow": fcf},
    )

    return DataResult(
        ticker           = ticker,
        data_type        = "cash_flow",
        data_quality     = quality,
        data_timestamp   = ts,
        degradation_note = degradation,
        payload          = {
            "operating_cash_flow":       ocf,
            "free_cash_flow":            fcf,
            "capital_expenditure":       capex,
            "depreciation_amortization": da,
            "change_in_working_capital": wc,
            "fcf_yield":                 fcf_yield,
            "cash_conversion_ratio":     cash_conversion,
            "ocf_4yr_trend":             ocf_trend,
            "dividend_history":          dividend_history,
        },
    )


def _dividend_history(stock: yf.Ticker) -> dict | None:
    try:
        divs = stock.dividends
        if divs is None or divs.empty:
            return None
        cutoff = divs.index[-1] - timedelta(days=365 * 3)
        recent = divs[divs.index >= cutoff]
        annual: dict[str, float] = {}
        for dt, amt in recent.items():
            yr = str(dt.year)
            annual[yr] = round(annual.get(yr, 0.0) + float(amt), 4)
        div_cagr = None
        years = sorted(annual.keys())
        if len(years) >= 2:
            oldest, newest = annual[years[0]], annual[years[-1]]
            n = len(years) - 1
            if oldest > 0 and n > 0:
                div_cagr = round(((newest / oldest) ** (1 / n) - 1) * 100, 2)
        return {
            "recent_payments": [
                {"date": str(dt)[:10], "amount": round(float(amt), 4)}
                for dt, amt in recent.iloc[-8:].items()
            ],
            "annual_totals":        annual,
            "most_recent_date":     str(divs.index[-1])[:10],
            "most_recent_amount":   round(float(divs.iloc[-1]), 4),
            "dividend_cagr_3y_pct": div_cagr,
        }
    except Exception:
        return None
