"""Unit tests for YahooFinanceTool with mocked yfinance responses."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from ai_financial_analyst.tools.yahoo_finance import yahoo_finance_tool


def _make_mock_ticker(
    price_history: bool = True,
    fundamentals:  bool = True,
    balance_sheet: bool = True,
    cash_flow:     bool = True,
    earnings:      bool = True,
):
    mock = MagicMock()

    if price_history:
        dates = pd.date_range("2020-01-01", periods=1260, freq="B")
        mock.history.return_value = pd.DataFrame(
            {
                "Open":   np.linspace(95.0,  195.0, 1260),
                "High":   np.linspace(105.0, 205.0, 1260),
                "Low":    np.linspace(90.0,  190.0, 1260),
                "Close":  np.linspace(100.0, 200.0, 1260),
                "Volume": np.ones(1260) * 50_000_000,
            },
            index=dates,
        )
    else:
        hist_mock = MagicMock()
        hist_mock.empty = True
        mock.history.return_value = hist_mock

    if fundamentals:
        mock.info = {
            "regularMarketPrice":           200.0,
            "currentPrice":                 200.0,
            "trailingPE":                   28.5,
            "forwardPE":                    24.0,
            "pegRatio":                     1.8,
            "priceToBook":                  7.1,
            "priceToSalesTrailing12Months": 6.2,
            "enterpriseToEbitda":           18.2,
            "enterpriseToRevenue":          5.8,
            "enterpriseValue":              2_800_000_000_000,
            "marketCap":                    3_000_000_000_000,
            "totalRevenue":                 400_000_000_000,
            "netIncomeToCommon":            100_000_000_000,
            "trailingEps":                  6.4,
            "forwardEps":                   7.2,
            "bookValue":                    4.0,
            "grossMargins":                 0.43,
            "operatingMargins":             0.30,
            "profitMargins":                0.25,
            "returnOnEquity":               1.47,
            "returnOnAssets":               0.22,
            "debtToEquity":                 1.5,
            "currentRatio":                 0.99,
            "quickRatio":                   0.85,
            "revenueGrowth":                0.06,
            "earningsGrowth":               0.11,
            "beta":                         1.24,
            "dividendYield":                0.005,
            "dividendRate":                 0.96,
            "payoutRatio":                  0.15,
            "shortRatio":                   0.8,
            "heldPercentInstitutions":      0.62,
            "SandP52WeekChange":            0.21,
            "52WeekChange":                 0.18,
            "sector":                       "Information Technology",
            "industry":                     "Consumer Electronics",
            "country":                      "United States",
            "exchange":                     "NMS",
            "longName":                     "Apple Inc.",
            "fullTimeEmployees":            164_000,
        }
        fi = MagicMock()
        fi.last_price  = 200.0
        fi.market_cap  = 3_000_000_000_000
        mock.fast_info = fi
        mock.analyst_price_targets = {"mean": 220.0, "high": 250.0, "low": 180.0, "median": 215.0}
    else:
        mock.info = {}
        fi = MagicMock()
        fi.last_price = None
        fi.market_cap = None
        mock.fast_info = fi
        mock.analyst_price_targets = {}

    if balance_sheet:
        mock.balance_sheet = pd.DataFrame(
            {
                "2024-09-01": {
                    "Total Assets":                           352_755_000_000,
                    "Total Liabilities Net Minority Interest": 308_030_000_000,
                    "Stockholders Equity":                      56_950_000_000,
                    "Cash And Cash Equivalents":                29_965_000_000,
                    "Long Term Debt":                           85_750_000_000,
                    "Current Assets":                          152_987_000_000,
                    "Current Liabilities":                     176_392_000_000,
                    "Inventory":                                 6_331_000_000,
                    "Total Debt":                              104_590_000_000,
                }
            }
        )
    else:
        mock.balance_sheet = MagicMock(empty=True)

    if cash_flow:
        mock.cashflow = pd.DataFrame(
            {
                "2024-09-01": {
                    "Operating Cash Flow":           110_543_000_000,
                    "Free Cash Flow":                101_565_000_000,
                    "Capital Expenditure":            -8_978_000_000,
                    "Depreciation And Amortization":  11_519_000_000,
                    "Net Income":                    100_000_000_000,
                }
            }
        )
    else:
        mock.cashflow           = MagicMock(empty=True)
        mock.quarterly_cashflow = MagicMock(empty=True)

    if earnings:
        mock.calendar = {
            "Earnings Date":  [datetime.now() + timedelta(days=45)],
            "EPS Estimate":   2.1,
            "Revenue Estimate": 95_000_000_000,
        }
        mock.earnings_dates = pd.DataFrame(
            {
                "EPS Estimate": [2.0, 1.9, 1.8],
                "Reported EPS": [2.1, 1.95, 1.85],
                "Surprise(%)":  [5.0, 2.6, 2.8],
            },
            index=pd.date_range("2024-10-01", periods=3, freq="-91D"),
        )
    else:
        mock.calendar       = {}
        mock.earnings_dates = pd.DataFrame()

    # Quarterly financials for financials_trend data type
    mock.quarterly_financials = pd.DataFrame(
        {
            "2024-09-01": {"Total Revenue": 94_930_000_000, "Net Income": 14_736_000_000, "Gross Profit": 43_881_000_000},
            "2024-06-01": {"Total Revenue": 85_777_000_000, "Net Income": 21_448_000_000, "Gross Profit": 39_673_000_000},
            "2024-03-01": {"Total Revenue": 90_753_000_000, "Net Income": 23_636_000_000, "Gross Profit": 42_270_000_000},
            "2023-12-01": {"Total Revenue": 119_575_000_000, "Net Income": 33_916_000_000, "Gross Profit": 56_995_000_000},
            "2023-09-01": {"Total Revenue": 89_498_000_000, "Net Income": 22_956_000_000, "Gross Profit": 40_428_000_000},
        }
    )
    mock.quarterly_balance_sheet = pd.DataFrame(
        {
            "2024-09-01": {"Cash And Cash Equivalents": 29_965_000_000, "Total Debt": 104_590_000_000, "Stockholders Equity": 56_950_000_000},
            "2024-06-01": {"Cash And Cash Equivalents": 32_695_000_000, "Total Debt": 101_304_000_000, "Stockholders Equity": 66_708_000_000},
        }
    )

    # Dividend history (pd already imported at top of file)
    mock.dividends = pd.Series(
        [0.24, 0.24, 0.25, 0.25, 0.25, 0.26, 0.26, 0.26],
        index=pd.date_range("2023-02-10", periods=8, freq="92D"),
    )

    # Analyst recommendations
    mock.recommendations = pd.DataFrame(
        {
            "Firm":       ["Goldman Sachs", "Morgan Stanley", "Barclays"],
            "To Grade":   ["Buy", "Overweight", "Equal Weight"],
            "From Grade": ["Neutral", "Equal Weight", "Equal Weight"],
            "Action":     ["up", "up", "main"],
        },
        index=pd.date_range("2024-09-01", periods=3, freq="-30D"),
    )

    return mock


@patch("ai_financial_analyst.tools.yahoo_finance.yf.Ticker")
@patch("ai_financial_analyst.tools.yahoo_finance._cache")
class TestYahooFinanceTool:

    def test_price_history_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "price_history"})
        data   = json.loads(result)

        assert data["ticker"]         == "AAPL"
        assert data["data_type"]      == "price_history"
        assert data["current_price"]  > 0
        assert data["price_5y_ago"]   > 0
        assert data["price_adjusted"] is True
        assert data["52w_high"]       is not None
        assert data["52w_low"]        is not None
        assert data["52w_high"] >= data["52w_low"]

    def test_price_history_null_when_no_data(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker(price_history=False)

        result = yahoo_finance_tool.invoke({"ticker": "FAKE", "data_type": "price_history"})
        data   = json.loads(result)
        assert data["result"] is None
        assert "reason" in data

    def test_fundamentals_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "fundamentals"})
        data   = json.loads(result)

        assert data["sector"]             == "Information Technology"
        assert data["pe_ratio"]           == 28.5
        assert data["ev_to_ebitda"]       is not None
        assert data["price_to_book"]      is not None
        assert data["return_on_equity"]   is not None
        assert data["operating_margin"]   is not None
        assert data["debt_to_equity"]     is not None
        assert data["analyst_target_mean"] == 220.0

    def test_fundamentals_uses_fast_info_price(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock = _make_mock_ticker()
        mock.fast_info.last_price = 210.5
        mock_ticker_cls.return_value = mock

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "fundamentals"})
        data   = json.loads(result)
        assert data["current_price"] == 210.5

    def test_balance_sheet_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "balance_sheet"})
        data   = json.loads(result)

        assert data["total_assets"]       is not None
        assert data["net_debt"]           is not None
        assert data["current_ratio_calc"] is not None

    def test_cash_flow_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "cash_flow"})
        data   = json.loads(result)

        assert data["data_type"]           == "cash_flow"
        assert data["operating_cash_flow"] is not None
        assert data["free_cash_flow"]      is not None
        assert data["fcf_yield"]           is not None

    def test_cash_flow_null_when_no_data(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker(cash_flow=False)

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "cash_flow"})
        data   = json.loads(result)
        assert data["result"] is None

    def test_earnings_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "earnings"})
        data   = json.loads(result)

        assert data["data_type"]          == "earnings"
        assert data["next_earnings_date"] is not None
        assert isinstance(data["earnings_surprises"], list)
        assert len(data["earnings_surprises"]) > 0

    def test_cash_flow_includes_dividend_history(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "cash_flow"})
        data   = json.loads(result)

        assert data["dividend_history"] is not None
        assert "recent_payments" in data["dividend_history"]
        assert len(data["dividend_history"]["recent_payments"]) > 0
        assert "annual_totals" in data["dividend_history"]
        assert "dividend_cagr_3y_pct" in data["dividend_history"]

    def test_fundamentals_includes_recommendations(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "fundamentals"})
        data   = json.loads(result)

        assert data["analyst_recommendations"] is not None
        recs = data["analyst_recommendations"]
        assert "recent" in recs
        assert "sentiment_counts" in recs
        sc = recs["sentiment_counts"]
        assert "positive" in sc and "neutral" in sc and "negative" in sc

    def test_financials_trend_success(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "financials_trend"})
        data   = json.loads(result)

        assert data["data_type"]    == "financials_trend"
        assert "income_trend"  in data
        assert "balance_trend" in data
        it = data["income_trend"]
        assert len(it) >= 2
        assert "quarter"    in it[0]
        assert "revenue"    in it[0]
        assert "net_income" in it[0]

    def test_financials_trend_has_yoy_growth(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.return_value = _make_mock_ticker()

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "financials_trend"})
        data   = json.loads(result)

        # The mock has 5 quarters, so YoY growth should be computable for Q0
        it = data["income_trend"]
        assert any("revenue_yoy_pct" in q for q in it), "YoY growth should be present when 5 quarters available"

    def test_returns_error_on_exception(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        mock_ticker_cls.side_effect = RuntimeError("Network error")

        result = yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "price_history"})
        data   = json.loads(result)
        assert "error_type" in data

    def test_extra_field_rejected(self, mock_cache, mock_ticker_cls):
        with pytest.raises(Exception):
            yahoo_finance_tool.invoke(
                {"ticker": "AAPL", "data_type": "fundamentals", "evil": "payload"}
            )

    def test_unknown_data_type_rejected(self, mock_cache, mock_ticker_cls):
        with pytest.raises(Exception):
            yahoo_finance_tool.invoke({"ticker": "AAPL", "data_type": "unknown_type"})
