"""Yahoo Finance — financials_trend data type (quarterly income + balance sheet)."""

from __future__ import annotations

import yfinance as yf

from ..base import DataResult, null_result, safe_float, utc_now, get_first_row


def fetch(ticker: str) -> DataResult:
    stock = yf.Ticker(ticker)
    ts    = utc_now()

    qf  = stock.quarterly_financials
    qbs = stock.quarterly_balance_sheet

    income_trend: list[dict] = []
    if qf is not None and not qf.empty:
        rev_row = get_first_row(qf, "Total Revenue", "Revenue")
        ni_row  = get_first_row(qf, "Net Income")
        gp_row  = get_first_row(qf, "Gross Profit")
        cols    = list(qf.columns[:5])

        for i, col in enumerate(cols[:4]):
            entry: dict = {"quarter": str(col)[:10]}
            rev = safe_float(rev_row[col]) if rev_row is not None and col in rev_row.index else None
            ni  = safe_float(ni_row[col])  if ni_row  is not None and col in ni_row.index  else None
            gp  = safe_float(gp_row[col])  if gp_row  is not None and col in gp_row.index  else None
            entry["revenue"] = rev
            entry["net_income"] = ni
            if rev and gp and rev != 0:
                entry["gross_margin_pct"] = round(gp / rev * 100, 2)
            if rev and ni and rev != 0:
                entry["net_margin_pct"] = round(ni / rev * 100, 2)
            if i + 1 < len(cols) and rev_row is not None:
                prev_rev = safe_float(rev_row[cols[i + 1]]) if cols[i + 1] in rev_row.index else None
                if rev and prev_rev and prev_rev != 0:
                    entry["revenue_qoq_pct"] = round((rev - prev_rev) / abs(prev_rev) * 100, 2)
            yoy_idx = i + 4
            if yoy_idx < len(cols) and rev_row is not None:
                yoy_rev = safe_float(rev_row[cols[yoy_idx]]) if cols[yoy_idx] in rev_row.index else None
                if rev and yoy_rev and yoy_rev != 0:
                    entry["revenue_yoy_pct"] = round((rev - yoy_rev) / abs(yoy_rev) * 100, 2)
            income_trend.append(entry)

    balance_trend: list[dict] = []
    if qbs is not None and not qbs.empty:
        cash_row = get_first_row(qbs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
        debt_row = get_first_row(qbs, "Total Debt", "Long Term Debt")
        eq_row   = get_first_row(qbs, "Stockholders Equity")
        for col in list(qbs.columns[:4]):
            entry = {"quarter": str(col)[:10]}
            if cash_row is not None and col in cash_row.index:
                entry["cash"] = safe_float(cash_row[col])
            if debt_row is not None and col in debt_row.index:
                entry["total_debt"] = safe_float(debt_row[col])
            if eq_row is not None and col in eq_row.index:
                entry["equity"] = safe_float(eq_row[col])
            balance_trend.append(entry)

    if not income_trend and not balance_trend:
        return null_result(ticker, "financials_trend", "Quarterly financials unavailable")

    quality     = "FULL" if len(income_trend) >= 4 else "PARTIAL"
    degradation = None if quality == "FULL" else f"Only {len(income_trend)} quarter(s) available (expect 4+)"

    return DataResult(
        ticker           = ticker,
        data_type        = "financials_trend",
        data_quality     = quality,
        data_timestamp   = ts,
        degradation_note = degradation,
        payload          = {
            "income_trend":  income_trend,
            "balance_trend": balance_trend,
        },
    )
