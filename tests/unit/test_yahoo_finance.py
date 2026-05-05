"""Unit tests for YahooFinanceTool with mocked yfinance responses."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_financial_analyst.tools.yahoo_finance import yahoo_finance_tool


def _make_mock_ticker(price_history=True, fundamentals=True, balance_sheet=True):
    mock = MagicMock()

    if price_history:
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2020-01-01", periods=1260, freq="B")
        mock.history.return_value = pd.DataFrame(
            {"Close": np.linspace(100.0, 200.0, 1260)}, index=dates
        )
    else:
        mock.history.return_value = MagicMock(empty=True)

    if fundamentals:
        mock.info = {
            "regularMarketPrice": 200.0,
            "trailingPE": 28.5,
            "forwardPE": 24.0,
            "marketCap": 3_000_000_000_000,
            "totalRevenue": 400_000_000_000,
            "netIncomeToCommon": 100_000_000_000,
            "profitMargins": 0.25,
            "sector": "Information Technology",
            "industry": "Consumer Electronics",
            "longName": "Apple Inc.",
        }
    else:
        mock.info = {}

    if balance_sheet:
        import pandas as pd
        mock.balance_sheet = pd.DataFrame(
            {
                "2024-09-01": {
                    "Total Assets": 352_755_000_000,
                    "Total Liabilities Net Minority Interest": 308_030_000_000,
                    "Stockholders Equity": 56_950_000_000,
                    "Cash And Cash Equivalents": 29_965_000_000,
                    "Long Term Debt": 85_750_000_000,
                }
            }
        )
    else:
        mock.balance_sheet = MagicMock(empty=True)

    return mock


@patch("ai_financial_analyst.tools.yahoo_finance.yf.Ticker")
@patch("ai_financial_analyst.tools.yahoo_finance._cache")
class TestYahooFinanceTool:
    def test_price_history_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "price_history"})
        data = json.loads(result)

        assert data["ticker"] == "AAPL"
        assert data["data_type"] == "price_history"
        assert "data_timestamp" in data
        assert data["current_price"] > 0
        assert data["price_5y_ago"] > 0

    def test_fundamentals_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "fundamentals"})
        data = json.loads(result)

        assert data["sector"] == "Information Technology"
        assert data["pe_ratio"] == 28.5

    def test_balance_sheet_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "balance_sheet"})
        data = json.loads(result)

        assert data["total_assets"] is not None

    def test_null_result_when_no_data(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker(price_history=False)

        result = yahoo_finance_tool.invoke({"ticker": "FAKE", "data_type": "price_history"})
        data = json.loads(result)

        assert data["result"] is None
        assert "reason" in data

    def test_returns_tool_error_on_exception(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.side_effect = RuntimeError("Network error")

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "price_history"})
        data = json.loads(result)

        assert "error_type" in data

    def test_extra_field_rejected(self, mock_cache, mock_ticker_cls):
        with pytest.raises(Exception):
            yahoo_finance_tool.invoke(
                {"ticker": "AAPL", "data_type": "fundamentals", "evil": "payload"}
            )
