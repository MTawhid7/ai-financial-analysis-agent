"""Unit tests for data/market/ — risk-free rate and S&P 500 data fetchers.

Uses mocked yfinance to avoid live network calls. Tests that the caching
layer is invoked correctly and that fallback defaults are applied on failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_financial_analyst.data.market.risk_free import get_risk_free_rate, _fetch as _fetch_rfr
from ai_financial_analyst.data.market.sp500 import get_sp500_data, _fetch as _fetch_sp500


class TestRiskFreeRate:
    @patch("ai_financial_analyst.data.market.risk_free.yf.Ticker")
    @patch("ai_financial_analyst.data.market.risk_free._cache")
    def test_returns_rate_from_fast_info(self, mock_cache, mock_ticker_cls):
        mock_cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
        fi = MagicMock()
        fi.last_price = 4.25  # TNX quoted as percentage
        mock_ticker_cls.return_value.fast_info = fi

        result = get_risk_free_rate()
        assert result == pytest.approx(0.0425, abs=1e-6)

    @patch("ai_financial_analyst.data.market.risk_free.yf.Ticker")
    def test_fallback_to_4_percent_on_failure(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = RuntimeError("network error")
        result = _fetch_rfr()
        assert result == pytest.approx(0.04)

    @patch("ai_financial_analyst.data.market.risk_free.yf.Ticker")
    def test_rejects_implausible_yield(self, mock_ticker_cls):
        """A TNX price of 0 or > 25 is implausible; should fall through to history."""
        fi = MagicMock()
        fi.last_price = 99.0  # implausible
        hist = MagicMock()
        hist.empty = True
        mock_ticker_cls.return_value.fast_info = fi
        mock_ticker_cls.return_value.history.return_value = hist

        result = _fetch_rfr()
        assert result == pytest.approx(0.04)  # fallback


class TestSP500Data:
    @patch("ai_financial_analyst.data.market.sp500.yf.Ticker")
    def test_returns_prices_returns_cagr(self, mock_ticker_cls):
        dates   = pd.date_range("2019-01-01", periods=252 * 5, freq="B")
        prices  = pd.Series(range(1000, 1000 + len(dates)), index=dates, dtype=float)
        hist    = pd.DataFrame({"Close": prices})
        mock_ticker_cls.return_value.history.return_value = hist

        result = _fetch_sp500("5y")
        assert result is not None
        assert "prices"  in result
        assert "returns" in result
        assert "cagr"    in result
        assert result["cagr"] > 0

    @patch("ai_financial_analyst.data.market.sp500.yf.Ticker")
    def test_returns_none_on_failure(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = RuntimeError("network error")
        result = _fetch_sp500("5y")
        assert result is None

    @patch("ai_financial_analyst.data.market.sp500.yf.Ticker")
    def test_returns_none_on_sparse_data(self, mock_ticker_cls):
        hist = pd.DataFrame({"Close": pd.Series([100.0] * 10)})
        mock_ticker_cls.return_value.history.return_value = hist
        result = _fetch_sp500("5y")
        assert result is None  # < 30 data points

    @patch("ai_financial_analyst.data.market.sp500._cache")
    @patch("ai_financial_analyst.data.market.sp500.yf.Ticker")
    def test_caches_result(self, mock_ticker_cls, mock_cache):
        mock_cache.get_or_fetch.return_value = ({"prices": [], "returns": [], "cagr": 0.1}, True)
        result = get_sp500_data("5y")
        assert result is not None
        mock_cache.get_or_fetch.assert_called_once()
