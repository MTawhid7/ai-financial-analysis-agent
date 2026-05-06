"""YahooFinanceTool — fetches financial data via yfinance with diskcache."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Literal

import yfinance as yf
from langchain_core.tools import tool
from pydantic import Field

from ..core.cache import ResultCache
from .base import StrictToolInput, ToolError, ErrorType, safe_tool_call

logger = logging.getLogger(__name__)

_cache = ResultCache()
# yfinance 1.x requires curl_cffi (not requests) to manage its own session.
# Do NOT pass a session= argument — let yfinance handle it internally.


class YahooFinanceInput(StrictToolInput):
    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL')")
    data_type: Literal["price_history", "fundamentals", "balance_sheet"] = Field(
        description="Type of financial data to retrieve"
    )


@tool("yahoo_finance", args_schema=YahooFinanceInput)
def yahoo_finance_tool(ticker: str, data_type: str) -> str:
    """Fetch financial data from Yahoo Finance for a given ticker.

    Returns structured JSON with a data_timestamp field.
    Returns an explicit null with a reason string when data is unavailable.
    """
    args = {"ticker": ticker.upper(), "data_type": data_type}

    def _fetch() -> str:
        return _fetch_data(ticker.upper(), data_type)

    result, hit = _cache.get_or_fetch("yahoo_finance", args, _fetch)
    if hit:
        logger.debug("yfinance cache HIT ticker=%s data_type=%s", ticker, data_type)
    return result


def _fetch_data(ticker: str, data_type: str) -> str:
    def _run():
        stock = yf.Ticker(ticker)
        timestamp = datetime.utcnow().isoformat() + "Z"

        if data_type == "price_history":
            hist = stock.history(period="5y")
            if hist.empty:
                # Yahoo Finance sometimes blocks long-period requests; try shorter.
                hist = stock.history(period="2y")
            if hist.empty:
                hist = stock.history(period="1y")
            if hist.empty:
                return _null_result(ticker, data_type, "No price history available")
            prices = hist["Close"].dropna()
            return json.dumps({
                "ticker": ticker,
                "data_type": "price_history",
                "data_timestamp": timestamp,
                "current_price": round(float(prices.iloc[-1]), 2),
                "price_5y_ago": round(float(prices.iloc[0]), 2),
                "52w_high": round(float(prices[-252:].max()), 2),
                "52w_low": round(float(prices[-252:].min()), 2),
                "data_points": len(prices),
            })

        if data_type == "fundamentals":
            info = stock.info
            # regularMarketPrice was deprecated in yfinance 1.x; fall back to
            # currentPrice. Reject only if the dict is truly empty or has no
            # price-related key at all (avoids false nulls on valid stocks).
            has_price = any(info.get(k) for k in (
                "currentPrice", "regularMarketPrice", "previousClose"
            ))
            if not info or not has_price:
                return _null_result(ticker, data_type, "Fundamentals unavailable")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            return json.dumps({
                "ticker": ticker,
                "data_type": "fundamentals",
                "data_timestamp": timestamp,
                "current_price": current_price,
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "market_cap": info.get("marketCap"),
                "revenue_ttm": info.get("totalRevenue"),
                "net_income_ttm": info.get("netIncomeToCommon"),
                "profit_margin": info.get("profitMargins"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "company_name": info.get("longName"),
            })

        if data_type == "balance_sheet":
            bs = stock.balance_sheet
            if bs is None or bs.empty:
                return _null_result(ticker, data_type, "Balance sheet unavailable")
            latest = bs.iloc[:, 0]
            return json.dumps({
                "ticker": ticker,
                "data_type": "balance_sheet",
                "data_timestamp": timestamp,
                "total_assets": _safe_float(latest.get("Total Assets")),
                "total_liabilities": _safe_float(latest.get("Total Liabilities Net Minority Interest")),
                "stockholders_equity": _safe_float(latest.get("Stockholders Equity")),
                "cash_and_equivalents": _safe_float(latest.get("Cash And Cash Equivalents")),
                "long_term_debt": _safe_float(latest.get("Long Term Debt")),
            })

        return ToolError(
            error_type=ErrorType.TOOL_ERROR,
            tool="yahoo_finance",
            message=f"Unknown data_type: {data_type}",
            input={"ticker": ticker, "data_type": data_type},
        ).to_json()

    return safe_tool_call("yahoo_finance", _run, {"ticker": ticker, "data_type": data_type})


def _null_result(ticker: str, data_type: str, reason: str) -> str:
    return json.dumps({
        "ticker": ticker,
        "data_type": data_type,
        "data_timestamp": datetime.utcnow().isoformat() + "Z",
        "result": None,
        "reason": reason,
    })


def _safe_float(val) -> float | None:
    try:
        return round(float(val), 2) if val is not None else None
    except (TypeError, ValueError):
        return None
