"""Yahoo Finance — balance_sheet data type."""

from __future__ import annotations

import yfinance as yf

from ..base import DataResult, assess_quality, null_result, safe_float, utc_now, get_first_row


def fetch(ticker: str) -> DataResult:
    stock = yf.Ticker(ticker)
    ts    = utc_now()
    bs    = stock.balance_sheet

    if bs is None or bs.empty:
        return null_result(ticker, "balance_sheet", "Balance sheet unavailable")

    latest = bs.iloc[:, 0]

    total_assets   = safe_float(latest.get("Total Assets"))
    total_liab     = safe_float(latest.get("Total Liabilities Net Minority Interest"))
    equity         = safe_float(latest.get("Stockholders Equity"))
    cash           = safe_float(latest.get("Cash And Cash Equivalents"))
    lt_debt        = safe_float(latest.get("Long Term Debt"))
    st_debt        = safe_float(latest.get(
        "Current Debt",
        latest.get("Short Long Term Debt", latest.get("Current Portion Of Long Term Debt")),
    ))
    total_debt     = safe_float(latest.get("Total Debt"))
    current_assets = safe_float(latest.get("Current Assets"))
    current_liab   = safe_float(latest.get("Current Liabilities"))
    inventory      = safe_float(latest.get("Inventory"))

    net_debt = round(total_debt - cash, 2) if total_debt is not None and cash is not None else None
    current_ratio = round(current_assets / current_liab, 2) if current_assets and current_liab and current_liab > 0 else None
    quick_ratio   = round((current_assets - (inventory or 0)) / current_liab, 2) if current_assets and current_liab and current_liab > 0 else None

    quality, degradation = assess_quality(
        required={"total_assets": total_assets, "total_liabilities": total_liab, "equity": equity},
    )

    return DataResult(
        ticker           = ticker,
        data_type        = "balance_sheet",
        data_quality     = quality,
        data_timestamp   = ts,
        degradation_note = degradation,
        payload          = {
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
        },
    )
