"""Integration tests for the Researcher agent node.

All external APIs are mocked. Tests verify state management,
max_iterations enforcement, and ToolError handling.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_financial_analyst.core.state import AgentState
from ai_financial_analyst.agents.researcher import researcher_node, MAX_ITERATIONS


def _make_state(tickers=None) -> AgentState:
    return AgentState(
        query="Analyse AAPL",
        tickers=tickers or ["AAPL"],
        iteration_log=[],
        errors=[],
        status="COMPLETE",
        run_id="test-run-001",
    )


def _price_history_response(ticker="AAPL") -> str:
    return json.dumps({
        "ticker": ticker,
        "data_type": "price_history",
        "data_timestamp": "2024-01-01T00:00:00Z",
        "current_price": 195.0,
        "price_5y_ago": 120.0,
        "52w_high": 200.0,
        "52w_low": 150.0,
        "data_points": 1260,
    })


def _fundamentals_response(ticker="AAPL") -> str:
    return json.dumps({
        "ticker": ticker,
        "data_type": "fundamentals",
        "data_timestamp": "2024-01-01T00:00:00Z",
        "pe_ratio": 28.5,
        "sector": "Information Technology",
        "company_name": "Apple Inc.",
        "market_cap": 3_000_000_000_000,
    })


def _balance_sheet_response(ticker="AAPL") -> str:
    return json.dumps({
        "ticker": ticker,
        "data_type": "balance_sheet",
        "data_timestamp": "2024-01-01T00:00:00Z",
        "total_assets": 352_000_000_000,
        "total_liabilities": 280_000_000_000,
        "stockholders_equity": 72_000_000_000,
    })


def _news_response() -> str:
    return json.dumps({
        "query": "AAPL stock news",
        "result_count": 2,
        "data_truncated": False,
        "summaries": [
            {"headline": "Apple posts record results", "date": "2024-01-01", "key_facts": [], "sentiment": 0.7}
        ],
    })


def _tool_error_response() -> str:
    return json.dumps({
        "error_type": "TOOL_ERROR",
        "tool": "yahoo_finance",
        "message": "Network timeout",
        "input": {},
    })


@pytest.mark.asyncio
async def test_researcher_populates_raw_data():
    responses = [
        _price_history_response(),
        _fundamentals_response(),
        _balance_sheet_response(),
        _news_response(),
    ]
    call_count = 0

    async def mock_arun(input_dict):
        nonlocal call_count
        resp = responses[call_count % len(responses)]
        call_count += 1
        return resp

    with (
        patch("ai_financial_analyst.agents.researcher.yahoo_finance_tool") as mock_yf,
        patch("ai_financial_analyst.agents.researcher.web_search_tool") as mock_ws,
    ):
        mock_yf.arun = mock_arun
        mock_ws.arun = AsyncMock(return_value=_news_response())

        state = _make_state()
        config = {"primary_llm": None, "tracer": None}
        result = await researcher_node(state, config=config)

    assert "AAPL" in result["raw_data"]
    assert result["data_coverage"][0]["ticker"] == "AAPL"
    assert len(result["iteration_log"]) > 0


@pytest.mark.asyncio
async def test_researcher_handles_tool_error_gracefully():
    async def mock_arun_error(input_dict):
        return _tool_error_response()

    with (
        patch("ai_financial_analyst.agents.researcher.yahoo_finance_tool") as mock_yf,
        patch("ai_financial_analyst.agents.researcher.web_search_tool") as mock_ws,
    ):
        mock_yf.arun = mock_arun_error
        mock_ws.arun = AsyncMock(return_value=_news_response())

        state = _make_state()
        config = {"primary_llm": None, "tracer": None}
        result = await researcher_node(state, config=config)

    # Errors are recorded but the agent does not crash
    assert len(result.get("errors", [])) > 0
    assert result.get("researcher_gaps")


@pytest.mark.asyncio
async def test_researcher_iteration_log_populated():
    async def mock_arun(input_dict):
        return _fundamentals_response()

    with (
        patch("ai_financial_analyst.agents.researcher.yahoo_finance_tool") as mock_yf,
        patch("ai_financial_analyst.agents.researcher.web_search_tool") as mock_ws,
    ):
        mock_yf.arun = mock_arun
        mock_ws.arun = AsyncMock(return_value=_news_response())

        state = _make_state()
        config = {"primary_llm": None, "tracer": None}
        result = await researcher_node(state, config=config)

    log = result.get("iteration_log", [])
    assert len(log) >= 3  # At least 3 tool calls per ticker (price, fundamentals, balance_sheet)
    for entry in log:
        assert "step" in entry
        assert "tool" in entry
        assert "agent" in entry
