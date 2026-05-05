"""Integration tests for the Editor agent node.

Verifies SOP rubric enforcement, grounding check, and disclaimer injection.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_financial_analyst.core.state import AgentState


def _make_state_with_analysis(analysis: dict) -> AgentState:
    return AgentState(
        query="Analyse AAPL",
        tickers=["AAPL"],
        raw_data={"AAPL": {}},
        analysis=analysis,
        data_coverage=[
            {
                "ticker": "AAPL",
                "price_history": True,
                "fundamentals": True,
                "balance_sheet": True,
                "news_search": True,
                "data_gaps": [],
            }
        ],
        researcher_gaps=[],
        iteration_log=[
            {
                "step": 1,
                "agent": "quant_analyst",
                "tool": "calculator",
                "input": {"expression": "((200/100)**(1/5)-1)*100"},
                "output_tokens": 10,
                "cache_hit": False,
            }
        ],
        errors=[],
        status="COMPLETE",
        run_id="test-run-002",
    )


_COMPLETE_ANALYSIS = {
    "AAPL": {
        "ticker": "AAPL",
        "price_cagr_5y_pct": 14.87,
        "sector": "Information Technology",
        "sector_pe_avg": 28.5,
        "company_pe": 30.0,
        "pe_vs_sector_premium_pct": 5.3,
        "bull_case": ["Strong services growth", "AI integration"],
        "bear_case": ["Valuation premium", "China revenue risk"],
        "closest_peer": "MSFT",
        "citations": {
            "price_cagr_5y_pct": {"source_tool": "calculator", "observation_step": 1},
            "sector_pe_avg": {"source_tool": "benchmark_lookup", "observation_step": 2},
        },
    }
}

_INCOMPLETE_ANALYSIS = {
    "AAPL": {
        "ticker": "AAPL",
        # Missing price_cagr_5y_pct, sector_pe_avg
        "bull_case": ["Strong growth"],
        "bear_case": ["Valuation"],
        "citations": {},
    }
}


@pytest.mark.asyncio
async def test_editor_adds_disclaimer():
    from ai_financial_analyst.agents.editor import editor_node

    mock_report = "# Financial Report\n\n## Executive Summary\nApple is strong.\n"

    with patch("ai_financial_analyst.agents.editor.report_writer_tool") as mock_rw:
        mock_rw.arun = AsyncMock(return_value=mock_report)

        state = _make_state_with_analysis(_COMPLETE_ANALYSIS)
        config = {"primary_llm": None, "tracer": None}
        result = await editor_node(state, config=config)

    assert "This is not financial advice" in result["report_markdown"]


@pytest.mark.asyncio
async def test_editor_sop_checklist_populated():
    from ai_financial_analyst.agents.editor import editor_node

    mock_report = "# Report\n## Data Coverage Summary\n..."

    with patch("ai_financial_analyst.agents.editor.report_writer_tool") as mock_rw:
        mock_rw.arun = AsyncMock(return_value=mock_report)

        state = _make_state_with_analysis(_COMPLETE_ANALYSIS)
        config = {"primary_llm": None, "tracer": None}
        result = await editor_node(state, config=config)

    checklist = result.get("sop_checklist", {})
    assert "AAPL/price_cagr_5y_pct" in checklist


@pytest.mark.asyncio
async def test_editor_raises_on_missing_analysis():
    from ai_financial_analyst.core.state import PartialStateError
    from ai_financial_analyst.agents.editor import editor_node

    state = AgentState(
        query="Analyse AAPL",
        tickers=["AAPL"],
        iteration_log=[],
        errors=[],
        status="COMPLETE",
        run_id="test-run-003",
    )
    config = {"primary_llm": None, "tracer": None}

    with pytest.raises(PartialStateError):
        await editor_node(state, config=config)
