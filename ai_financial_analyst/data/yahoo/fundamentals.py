"""Yahoo Finance — fundamentals data type (25+ fields + analyst recommendations)."""

from __future__ import annotations

import yfinance as yf

from ..base import DataResult, assess_quality, null_result, safe_float, utc_now


def fetch(ticker: str) -> DataResult:
    stock = yf.Ticker(ticker)
    ts    = utc_now()
    info  = stock.info

    current_price = None
    try:
        current_price = safe_float(getattr(stock.fast_info, "last_price", None))
    except Exception:
        pass
    if current_price is None:
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    if not info or not (current_price or info.get("previousClose")):
        return null_result(ticker, "fundamentals", "Fundamentals unavailable")

    apt: dict = {}
    try:
        apt = stock.analyst_price_targets or {}
    except Exception:
        pass

    quality, degradation = assess_quality(
        required={"price": current_price, "sector": info.get("sector"), "market_cap": info.get("marketCap")},
        optional={"pe_ratio": info.get("trailingPE"), "ev_ebitda": info.get("enterpriseToEbitda")},
    )

    return DataResult(
        ticker           = ticker,
        data_type        = "fundamentals",
        data_quality     = quality,
        data_timestamp   = ts,
        degradation_note = degradation,
        payload          = {
            "current_price":    current_price,
            "pe_ratio":         safe_float(info.get("trailingPE")),
            "forward_pe":       safe_float(info.get("forwardPE")),
            "peg_ratio":        safe_float(info.get("pegRatio")),
            "price_to_book":    safe_float(info.get("priceToBook")),
            "price_to_sales":   safe_float(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda":     safe_float(info.get("enterpriseToEbitda")),
            "ev_to_revenue":    safe_float(info.get("enterpriseToRevenue")),
            "enterprise_value": info.get("enterpriseValue"),
            "market_cap":       info.get("marketCap"),
            "revenue_ttm":      info.get("totalRevenue"),
            "net_income_ttm":   info.get("netIncomeToCommon"),
            "trailing_eps":     safe_float(info.get("trailingEps")),
            "forward_eps":      safe_float(info.get("forwardEps")),
            "book_value":       safe_float(info.get("bookValue")),
            "gross_margin":     safe_float(info.get("grossMargins")),
            "operating_margin": safe_float(info.get("operatingMargins")),
            "profit_margin":    safe_float(info.get("profitMargins")),
            "return_on_equity": safe_float(info.get("returnOnEquity")),
            "return_on_assets": safe_float(info.get("returnOnAssets")),
            "debt_to_equity":   safe_float(info.get("debtToEquity")),
            "current_ratio":    safe_float(info.get("currentRatio")),
            "quick_ratio":      safe_float(info.get("quickRatio")),
            "revenue_growth":   safe_float(info.get("revenueGrowth")),
            "earnings_growth":  safe_float(info.get("earningsGrowth")),
            "beta":             safe_float(info.get("beta")),
            "dividend_yield":   safe_float(info.get("dividendYield")),
            "dividend_rate":    safe_float(info.get("dividendRate")),
            "payout_ratio":     safe_float(info.get("payoutRatio")),
            "short_ratio":              safe_float(info.get("shortRatio")),
            "institutional_ownership":  safe_float(info.get("heldPercentInstitutions")),
            "sp500_52w_change":         safe_float(info.get("SandP52WeekChange")),
            "52w_change":               safe_float(info.get("52WeekChange")),
            "sector":           info.get("sector"),
            "industry":         info.get("industry"),
            "country":          info.get("country"),
            "exchange":         info.get("exchange"),
            "company_name":     info.get("longName"),
            "employees":        info.get("fullTimeEmployees"),
            "analyst_target_mean":   safe_float(apt.get("mean")),
            "analyst_target_high":   safe_float(apt.get("high")),
            "analyst_target_low":    safe_float(apt.get("low")),
            "analyst_target_median": safe_float(apt.get("median")),
            "analyst_recommendations": _analyst_recommendations(stock),
        },
    )


def _analyst_recommendations(stock: yf.Ticker) -> dict | None:
    try:
        recs = stock.recommendations
        if recs is None or recs.empty:
            return None

        def _sentiment(grade: str) -> str:
            g = grade.lower()
            if any(k in g for k in ("buy", "overweight", "outperform", "strong buy", "positive", "add", "accumulate")):
                return "positive"
            if any(k in g for k in ("sell", "underweight", "underperform", "reduce", "negative", "strong sell")):
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
            recent.append({"date": str(dt)[:10], "firm": firm, "from_grade": from_grade,
                           "to_grade": to_grade, "action": action, "sentiment": s})
        return {"recent": recent, "sentiment_counts": counts}
    except Exception:
        return None
