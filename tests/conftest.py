"""Shared pytest fixtures for the AI Financial Analyst Agent test suite.

Design principles:
- Every fixture produces an isolated instance — no shared mutable state.
- All I/O (files, network, LLM) is eliminated by default; tests opt in to real I/O.
- Fixtures follow the DI pattern of the production code — dependencies are injected,
  not grabbed from module globals.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_financial_analyst.config.settings import Settings
from ai_financial_analyst.core.llm import CircuitBreaker, RateLimitFallbackLLM
from ai_financial_analyst.memory.in_memory import InMemoryBackend


# ── Settings ──────────────────────────────────────────────────────────────────

@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Settings instance with temporary directories and test API keys.

    Prevents real file creation in the project root during tests.
    """
    return Settings(
        google_api_key        = "test-google-key",
        tavily_api_key        = "test-tavily-key",
        llm_primary_model     = "mock-primary-model",
        llm_fallback_model    = "mock-fallback-model",
        cache_dir             = str(tmp_path / "cache"),
        memory_db_path        = str(tmp_path / "memory.db"),
        checkpoint_db_path    = str(tmp_path / "checkpoints.db"),
        upload_dir            = str(tmp_path / "uploads"),
        artifacts_dir         = str(tmp_path / "artifacts"),
    )


# ── LLM mocks ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_response():
    """Default LLM response (AIMessage-like with .content attribute)."""
    resp = MagicMock()
    resp.content = '{"result": "mocked_response"}'
    return resp


@pytest.fixture
def mock_llm(mock_llm_response) -> MagicMock:
    """Injectable mock LLM that implements the LLMClient protocol.

    Returns the mock_llm_response by default.
    Override .ainvoke.return_value or .invoke.return_value in tests.
    """
    llm = MagicMock()
    llm.ainvoke               = AsyncMock(return_value=mock_llm_response)
    llm.invoke                = MagicMock(return_value=mock_llm_response)
    llm.bind_tools.return_value            = llm   # chaining works
    llm.with_structured_output.return_value = llm  # structured output works
    return llm


@pytest.fixture
def mock_circuit_breaker() -> CircuitBreaker:
    """A fresh CircuitBreaker in CLOSED state, isolated per test."""
    return CircuitBreaker(max_failures=5, window_s=60.0, half_open_delay_s=60.0)


@pytest.fixture
def mock_rate_limit_llm(mock_llm, mock_circuit_breaker) -> RateLimitFallbackLLM:
    """RateLimitFallbackLLM wrapping mock primary + mock fallback with isolated breaker."""
    fallback = MagicMock()
    fallback.ainvoke = AsyncMock(return_value=MagicMock(content="fallback_response"))
    fallback.invoke  = MagicMock(return_value=MagicMock(content="fallback_response"))
    return RateLimitFallbackLLM(
        primary         = mock_llm,
        fallback        = fallback,
        circuit_breaker = mock_circuit_breaker,
    )


# ── Cache mocks ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_cache() -> MagicMock:
    """Cache mock that always calls through to the fetch function (no caching)."""
    cache = MagicMock()
    cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: (fn(), False)
    cache.get.return_value         = None
    cache.set.return_value         = None
    return cache


@pytest.fixture
def mock_cache_hit() -> MagicMock:
    """Cache mock that always returns a cache hit with a fixed JSON string."""
    cache = MagicMock()
    cache.get_or_fetch.side_effect = lambda tool, args, fn, **kw: ('{"cached": true}', True)
    return cache


# ── Memory backend ────────────────────────────────────────────────────────────

@pytest.fixture
def in_memory_backend() -> InMemoryBackend:
    """Pure in-memory MemoryBackend — zero file I/O, isolated per test."""
    return InMemoryBackend(user_id="test-user")


@pytest.fixture
def db_path(tmp_path) -> str:
    """Path to a temporary SQLite file that is deleted after the test."""
    return str(tmp_path / "test_memory.db")


# ── AgentState factories ──────────────────────────────────────────────────────

@pytest.fixture
def minimal_agent_state() -> dict:
    """Minimal valid AgentState dict for orchestrator/agent tests."""
    from ai_financial_analyst.core.state import AgentState
    return AgentState(
        query         = "Analyse AAPL",
        tickers       = ["AAPL"],
        iteration_log = [],
        errors        = [],
        status        = "COMPLETE",
        run_id        = "fixture-run-001",
    )


@pytest.fixture
def agent_state_with_data(minimal_agent_state) -> dict:
    """AgentState with pre-populated raw_data suitable for quant_analyst tests."""
    state = dict(minimal_agent_state)
    state["raw_data"] = {
        "AAPL": {
            "price_history": {
                "ticker": "AAPL", "data_type": "price_history",
                "data_quality": "FULL", "current_price": 200.0, "price_5y_ago": 100.0,
                "52w_high": 210.0, "52w_low": 150.0, "data_points": 1260,
            },
            "fundamentals": {
                "ticker": "AAPL", "data_type": "fundamentals", "data_quality": "FULL",
                "pe_ratio": 28.5, "sector": "Information Technology",
                "market_cap": 3_000_000_000_000, "current_price": 200.0,
                "revenue_growth": 0.06, "return_on_equity": 1.47,
                "analyst_target_mean": 220.0, "analyst_target_low": 180.0,
                "analyst_target_high": 260.0, "forward_eps": 7.2,
                "country": "United States",
            },
            "price_metrics": {
                "ticker": "AAPL", "data_type": "price_metrics", "data_quality": "FULL",
                "sharpe_ratio": 1.42, "sortino_ratio": 1.85, "max_drawdown_pct": -34.2,
                "beta_vs_sp500": 1.24, "volatility_annual_pct": 28.4,
                "total_return_cagr_pct": 24.5, "sp500_cagr_pct": 14.2,
                "relative_cagr_pct": 10.3, "risk_free_rate_used": 4.2,
            },
        }
    }
    return state


# ── yfinance mock factory ─────────────────────────────────────────────────────

@pytest.fixture
def mock_yf_ticker():
    """Reusable factory for creating yfinance Ticker mocks with sane defaults.

    Usage in tests:
        @patch("yfinance.Ticker")
        def test_something(self, mock_cls, mock_yf_ticker):
            mock_cls.return_value = mock_yf_ticker(price_history=True)
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    def _make(
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
                {"Open": np.linspace(95, 195, 1260), "High": np.linspace(105, 205, 1260),
                 "Low": np.linspace(90, 190, 1260), "Close": np.linspace(100, 200, 1260),
                 "Volume": np.ones(1260) * 50_000_000},
                index=dates,
            )
        else:
            h = MagicMock()
            h.empty = True
            mock.history.return_value = h

        if fundamentals:
            mock.info = {
                "regularMarketPrice": 200.0, "currentPrice": 200.0,
                "trailingPE": 28.5, "forwardPE": 24.0, "pegRatio": 1.8,
                "priceToBook": 7.1, "priceToSalesTrailing12Months": 6.2,
                "enterpriseToEbitda": 18.2, "enterpriseToRevenue": 5.8,
                "enterpriseValue": 2_800_000_000_000, "marketCap": 3_000_000_000_000,
                "totalRevenue": 400_000_000_000, "netIncomeToCommon": 100_000_000_000,
                "trailingEps": 6.4, "forwardEps": 7.2, "bookValue": 4.0,
                "grossMargins": 0.43, "operatingMargins": 0.30, "profitMargins": 0.25,
                "returnOnEquity": 1.47, "returnOnAssets": 0.22, "debtToEquity": 1.5,
                "currentRatio": 0.99, "quickRatio": 0.85, "revenueGrowth": 0.06,
                "earningsGrowth": 0.11, "beta": 1.24, "dividendYield": 0.005,
                "dividendRate": 0.96, "payoutRatio": 0.15, "shortRatio": 0.8,
                "heldPercentInstitutions": 0.62, "SandP52WeekChange": 0.21,
                "52WeekChange": 0.18, "sector": "Information Technology",
                "industry": "Consumer Electronics", "country": "United States",
                "exchange": "NMS", "longName": "Apple Inc.", "fullTimeEmployees": 164_000,
            }
            fi = MagicMock()
            fi.last_price = 200.0
            fi.market_cap = 3_000_000_000_000
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
            mock.balance_sheet = pd.DataFrame({"2024-09-01": {
                "Total Assets": 352_755_000_000, "Total Liabilities Net Minority Interest": 308_030_000_000,
                "Stockholders Equity": 56_950_000_000, "Cash And Cash Equivalents": 29_965_000_000,
                "Long Term Debt": 85_750_000_000, "Current Assets": 152_987_000_000,
                "Current Liabilities": 176_392_000_000, "Inventory": 6_331_000_000,
                "Total Debt": 104_590_000_000,
            }})
        else:
            mock.balance_sheet = MagicMock(empty=True)

        if cash_flow:
            mock.cashflow = pd.DataFrame({"2024-09-01": {
                "Operating Cash Flow": 110_543_000_000, "Free Cash Flow": 101_565_000_000,
                "Capital Expenditure": -8_978_000_000, "Depreciation And Amortization": 11_519_000_000,
                "Net Income": 100_000_000_000,
            }})
        else:
            mock.cashflow = MagicMock(empty=True)
            mock.quarterly_cashflow = MagicMock(empty=True)

        if earnings:
            mock.calendar = {
                "Earnings Date": [datetime.now() + timedelta(days=45)],
                "EPS Estimate": 2.1, "Revenue Estimate": 95_000_000_000,
            }
            mock.earnings_dates = pd.DataFrame(
                {"EPS Estimate": [2.0, 1.9, 1.8], "Reported EPS": [2.1, 1.95, 1.85], "Surprise(%)": [5.0, 2.6, 2.8]},
                index=pd.date_range("2024-10-01", periods=3, freq="-91D"),
            )
        else:
            mock.calendar = {}
            mock.earnings_dates = pd.DataFrame()

        mock.quarterly_financials = pd.DataFrame({
            "2024-09-01": {"Total Revenue": 94_930_000_000, "Net Income": 14_736_000_000, "Gross Profit": 43_881_000_000},
            "2024-06-01": {"Total Revenue": 85_777_000_000, "Net Income": 21_448_000_000, "Gross Profit": 39_673_000_000},
            "2024-03-01": {"Total Revenue": 90_753_000_000, "Net Income": 23_636_000_000, "Gross Profit": 42_270_000_000},
            "2023-12-01": {"Total Revenue": 119_575_000_000, "Net Income": 33_916_000_000, "Gross Profit": 56_995_000_000},
            "2023-09-01": {"Total Revenue": 89_498_000_000, "Net Income": 22_956_000_000, "Gross Profit": 40_428_000_000},
        })
        mock.quarterly_balance_sheet = pd.DataFrame({
            "2024-09-01": {"Cash And Cash Equivalents": 29_965_000_000, "Total Debt": 104_590_000_000, "Stockholders Equity": 56_950_000_000},
            "2024-06-01": {"Cash And Cash Equivalents": 32_695_000_000, "Total Debt": 101_304_000_000, "Stockholders Equity": 66_708_000_000},
        })
        mock.dividends = pd.Series(
            [0.24, 0.24, 0.25, 0.25, 0.25, 0.26, 0.26, 0.26],
            index=pd.date_range("2023-02-10", periods=8, freq="92D"),
        )
        mock.recommendations = pd.DataFrame(
            {"Firm": ["Goldman Sachs", "Morgan Stanley", "Barclays"],
             "To Grade": ["Buy", "Overweight", "Equal Weight"],
             "From Grade": ["Neutral", "Equal Weight", "Equal Weight"],
             "Action": ["up", "up", "main"]},
            index=pd.date_range("2024-09-01", periods=3, freq="-30D"),
        )
        mock.splits = pd.Series([], dtype=float, name="Stock Splits")
        return mock

    return _make
