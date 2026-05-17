"""Integration tests for the Researcher agent node.

External data fetching is mocked at the data/ layer.
Tests verify state management, concurrent-fetch wiring, and error handling.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_financial_analyst.core.state import AgentState
from ai_financial_analyst.agents.researcher import researcher_node, MAX_ITERATIONS


def _make_state(tickers=None) -> AgentState:
    return AgentState(
        query        = "Analyse AAPL",
        tickers      = tickers or ["AAPL"],
        iteration_log= [],
        errors       = [],
        status       = "COMPLETE",
        run_id       = "test-run-001",
    )


def _ticker_data(ticker="AAPL") -> dict:
    """Minimal ticker_data dict as returned by fetch_ticker_data()."""
    return {
        "price_history": {
            "ticker": ticker, "data_type": "price_history",
            "data_quality": "FULL", "data_timestamp": "2024-01-01T00:00:00+00:00",
            "degradation_note": None,
            "current_price": 195.0, "price_5y_ago": 120.0,
            "52w_high": 200.0, "52w_low": 150.0,
            "data_points": 1260, "price_adjusted": True,
            "corporate_events": [],
        },
        "fundamentals": {
            "ticker": ticker, "data_type": "fundamentals",
            "data_quality": "FULL", "data_timestamp": "2024-01-01T00:00:00+00:00",
            "degradation_note": None,
            "pe_ratio": 28.5, "sector": "Information Technology",
            "company_name": "Apple Inc.", "market_cap": 3_000_000_000_000,
        },
        "balance_sheet": {
            "ticker": ticker, "data_type": "balance_sheet",
            "data_quality": "FULL", "data_timestamp": "2024-01-01T00:00:00+00:00",
            "degradation_note": None,
            "total_assets": 352_000_000_000, "total_liabilities": 280_000_000_000,
            "stockholders_equity": 72_000_000_000,
        },
    }


def _make_search_result():
    from ai_financial_analyst.data.search.tavily import SearchResult
    return SearchResult(
        headline    = "Apple posts record results",
        url         = "https://reuters.com/article/xyz",
        content     = "Apple Inc reported record quarterly revenue.",
        score       = 0.9,
        source_tier = 1,
    )


@pytest.mark.asyncio
async def test_researcher_populates_raw_data():
    """Researcher populates raw_data with ticker keys and data_coverage entries."""
    with (
        patch("ai_financial_analyst.agents.researcher.fetch_ticker_data",
              new=AsyncMock(return_value=_ticker_data())) as mock_fetch,
        patch("ai_financial_analyst.agents.researcher.TavilySearchClient") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.search.return_value = [_make_search_result()]
        mock_client_cls.return_value = mock_client

        state  = _make_state()
        result = await researcher_node(state, config={})

    assert "AAPL" in result["raw_data"]
    assert result["data_coverage"][0]["ticker"] == "AAPL"
    assert result["data_coverage"][0]["price_history"] is True
    assert result["data_coverage"][0]["fundamentals"]  is True
    assert len(result["iteration_log"]) > 0


@pytest.mark.asyncio
async def test_researcher_handles_empty_fetch_gracefully():
    """When fetch_ticker_data returns {} (all core types failed), researcher records gaps."""
    with (
        patch("ai_financial_analyst.agents.researcher.fetch_ticker_data",
              new=AsyncMock(return_value={})) as mock_fetch,
        patch("ai_financial_analyst.agents.researcher.TavilySearchClient") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_client_cls.return_value = mock_client

        state  = _make_state()
        result = await researcher_node(state, config={})

    # An empty ticker_data results in a gap entry
    assert len(result.get("researcher_gaps", [])) > 0 or result["raw_data"]["AAPL"] == {}


@pytest.mark.asyncio
async def test_researcher_records_data_quality_degradation():
    """PARTIAL data quality is surfaced as a gap in data_coverage."""
    partial_data = dict(_ticker_data())
    partial_data["price_history"]["data_quality"]     = "PARTIAL"
    partial_data["price_history"]["degradation_note"] = "Only 2yr data available"

    with (
        patch("ai_financial_analyst.agents.researcher.fetch_ticker_data",
              new=AsyncMock(return_value=partial_data)),
        patch("ai_financial_analyst.agents.researcher.TavilySearchClient") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_client_cls.return_value = mock_client

        state  = _make_state()
        result = await researcher_node(state, config={})

    gaps = result.get("researcher_gaps", [])
    assert any("2yr data" in g or "PARTIAL" in g or "price_history" in g for g in gaps)


@pytest.mark.asyncio
async def test_researcher_iteration_log_populated():
    """Iteration log has one entry per successfully fetched data type + news search."""
    with (
        patch("ai_financial_analyst.agents.researcher.fetch_ticker_data",
              new=AsyncMock(return_value=_ticker_data())),
        patch("ai_financial_analyst.agents.researcher.TavilySearchClient") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.search.return_value = [_make_search_result()]
        mock_client_cls.return_value = mock_client

        state  = _make_state()
        result = await researcher_node(state, config={})

    log = result.get("iteration_log", [])
    # 3 data types (price, fundamentals, balance_sheet) + 1 news search = 4 entries minimum
    assert len(log) >= 4
    for entry in log:
        assert "step"  in entry
        assert "tool"  in entry
        assert "agent" in entry
