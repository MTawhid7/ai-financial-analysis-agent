"""YahooFinanceTool — thin LangChain @tool wrapper over data/yahoo/.

All data-fetching logic lives in data/yahoo/{price,fundamentals,...}.py.
This file is responsible only for:
  1. Validating the input (StrictToolInput, Pydantic)
  2. Routing to the correct data module via fetch_single()
  3. Cache-aside with appropriate TTL per data type
  4. Serialising DataResult → JSON string

Seven data types, each with the appropriate cache TTL:
  price_history      — adjusted OHLCV, true 52-week window, data quality, splits
  fundamentals       — 25+ metrics + analyst recommendations
  balance_sheet      — assets, liabilities, equity, derived ratios
  cash_flow          — OCF, FCF, capex, D&A, FCF yield, dividend history
  earnings           — next earnings date, trailing EPS surprises
  price_metrics      — Sharpe, Sortino, max drawdown, beta, CAGR
  financials_trend   — quarterly income + balance sheet trend
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.tools import tool
from pydantic import Field

from ..config import settings
from ..core.cache import ResultCache
from ..data.yahoo import fetch_single
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


@tool("yahoo_finance", args_schema=YahooFinanceInput)
def yahoo_finance_tool(ticker: str, data_type: str) -> str:
    """Fetch financial data from Yahoo Finance for a given ticker.

    Returns structured JSON with a data_quality field ("FULL"|"PARTIAL"|"UNAVAILABLE").
    All data is cached at appropriate TTLs to prevent stale prices.
    Network errors and other exceptions are caught and returned as ToolError JSON.
    """
    args = {"ticker": ticker.upper(), "data_type": data_type}
    ttl  = settings.get_ttl(data_type)

    def _run() -> str:
        def _fetch() -> str:
            result = fetch_single(ticker.upper(), data_type)
            return json.dumps(result)

        result, hit = _cache.get_or_fetch("yahoo_finance", args, _fetch, ttl=ttl)
        if hit:
            logger.debug("yfinance cache HIT ticker=%s data_type=%s", ticker, data_type)
        return result

    return safe_tool_call("yahoo_finance", _run, args)
