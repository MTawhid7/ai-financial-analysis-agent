"""Yahoo Finance — earnings data type (next date, EPS estimate, surprise history)."""

from __future__ import annotations

import yfinance as yf

from ..base import DataResult, assess_quality, null_result, safe_float, utc_now


def fetch(ticker: str) -> DataResult:
    stock = yf.Ticker(ticker)
    ts    = utc_now()

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
            eps_estimate = safe_float(cal.get("EPS Estimate"))
            rev_estimate = safe_float(cal.get("Revenue Estimate"))
    except Exception:
        pass

    surprises: list[dict] = []
    try:
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            reported = ed[ed.get("Reported EPS", ed.columns[0]).notna()] if "Reported EPS" in ed.columns else ed
            for dt, row in reported.head(8).iterrows():
                entry: dict = {"date": str(dt)[:10]}
                for col in ["EPS Estimate", "Reported EPS", "Surprise(%)"]:
                    if col in row:
                        entry[col.lower().replace(" ", "_").replace("(%)", "_pct")] = safe_float(row[col])
                surprises.append(entry)
    except Exception:
        pass

    if next_date is None and not surprises:
        return null_result(ticker, "earnings", "Earnings data unavailable")

    quality = "FULL" if (next_date is not None and len(surprises) >= 4) else "PARTIAL"
    degradation = None
    if quality == "PARTIAL":
        parts = []
        if next_date is None:
            parts.append("Missing next earnings date")
        if len(surprises) < 4:
            parts.append(f"Only {len(surprises)} quarter(s) of surprise history (expect 4+)")
        degradation = "; ".join(parts) or None

    return DataResult(
        ticker           = ticker,
        data_type        = "earnings",
        data_quality     = quality,
        data_timestamp   = ts,
        degradation_note = degradation,
        payload          = {
            "next_earnings_date": next_date,
            "eps_estimate":       eps_estimate,
            "revenue_estimate":   rev_estimate,
            "earnings_surprises": surprises,
        },
    )
