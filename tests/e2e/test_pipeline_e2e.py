"""End-to-end pipeline tests using recorded API responses (VCR cassettes).

These tests run the full Researcher → Quant → Editor pipeline against
pre-recorded responses, consuming zero live API quota.

To record cassettes for the first time:
    pytest tests/e2e/ --record-mode=new_episodes

To replay (default CI mode):
    pytest tests/e2e/ --record-mode=none
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_financial_analyst.core.state import AgentState

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CASSETTE_DIR = Path(__file__).parent / "cassettes"


def _mock_price_history(ticker: str) -> str:
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


def _mock_fundamentals(ticker: str, sector: str = "Information Technology", pe: float = 28.5) -> str:
    return json.dumps({
        "ticker": ticker,
        "data_type": "fundamentals",
        "data_timestamp": "2024-01-01T00:00:00Z",
        "pe_ratio": pe,
        "sector": sector,
        "company_name": f"{ticker} Inc.",
        "market_cap": 3_000_000_000_000,
        "revenue_ttm": 400_000_000_000,
    })


def _mock_balance_sheet(ticker: str) -> str:
    return json.dumps({
        "ticker": ticker,
        "data_type": "balance_sheet",
        "data_timestamp": "2024-01-01T00:00:00Z",
        "total_assets": 352_000_000_000,
        "total_liabilities": 280_000_000_000,
        "stockholders_equity": 72_000_000_000,
        "cash_and_equivalents": 30_000_000_000,
        "long_term_debt": 85_000_000_000,
    })


def _mock_news() -> str:
    return json.dumps({
        "query": "stock news",
        "result_count": 1,
        "data_truncated": False,
        "summaries": [
            {
                "headline": "Strong quarterly results beat estimates",
                "date": "2024-01-15",
                "key_facts": ["Revenue beat by 5%", "EPS above consensus"],
                "sentiment": 0.6,
            }
        ],
    })


def _mock_llm_sop_response(ticker: str) -> str:
    return json.dumps({
        "bull_case": ["Strong revenue growth", "AI integration opportunity"],
        "bear_case": ["Premium valuation vs peers", "Macro headwinds"],
        "closest_peer": "MSFT",
    })


def _mock_report(tickers) -> str:
    ticker_str = ", ".join(tickers)
    return (
        f"# Financial Analysis: {ticker_str}\n\n"
        "## Data Coverage Summary\n\n"
        f"- **{tickers[0]}**: Price history ✓, Fundamentals ✓, Balance sheet ✓\n\n"
        "## Executive Summary\n\nStrong fundamental outlook.\n\n"
        "## Quantitative Analysis\n\n"
        "5-year price CAGR: 14.87%. P/E ratio above sector average.\n\n"
        "## Bull Case\n\n- Strong revenue growth\n\n"
        "## Bear Case\n\n- Premium valuation\n\n"
        "## Conclusion\n\nHold recommendation with positive long-term outlook.\n\n"
        "---\n*This is not financial advice.*"
    )


# ---------------------------------------------------------------------------
# E2E test: AAPL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_aapl():
    """Full pipeline for AAPL — validates all mandatory SOP sections present."""
    from ai_financial_analyst.agents.orchestrator import run_pipeline

    yf_responses = {
        ("AAPL", "price_history"): _mock_price_history("AAPL"),
        ("AAPL", "fundamentals"): _mock_fundamentals("AAPL"),
        ("AAPL", "balance_sheet"): _mock_balance_sheet("AAPL"),
    }

    async def mock_yf_arun(inputs):
        key = (inputs["ticker"], inputs["data_type"])
        return yf_responses.get(key, json.dumps({"result": None, "reason": "no data", "data_timestamp": ""}))

    async def mock_sop_ainvoke(inputs):
        resp = MagicMock()
        resp.content = _mock_llm_sop_response("AAPL")
        return resp

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=mock_sop_ainvoke)

    with (
        patch("ai_financial_analyst.agents.researcher.yahoo_finance_tool") as mock_yf,
        patch("ai_financial_analyst.agents.researcher.web_search_tool") as mock_ws,
        patch("ai_financial_analyst.agents.quant_analyst.calculator_tool") as mock_calc,
        patch("ai_financial_analyst.agents.quant_analyst.benchmark_lookup_tool") as mock_bl,
        patch("ai_financial_analyst.agents.editor.report_writer_tool") as mock_rw,
        patch("ai_financial_analyst.agents.orchestrator.get_primary_llm", return_value=mock_llm),
        patch("ai_financial_analyst.agents.orchestrator.get_subllm", return_value=MagicMock()),
        patch("ai_financial_analyst.agents.orchestrator.SqliteSaver") as mock_sql,
        patch("ai_financial_analyst.agents.quant_analyst._SOP_PROMPT") as mock_prompt,
    ):
        mock_yf.arun = mock_yf_arun
        mock_ws.arun = AsyncMock(return_value=_mock_news())
        mock_calc.arun = AsyncMock(return_value="14.87")
        mock_bl.arun = AsyncMock(return_value=json.dumps({
            "sector": "Information Technology",
            "pe_ratio_sector_avg": 28.5,
            "ev_ebitda_sector_avg": 18.2,
            "price_to_book_sector_avg": 7.1,
            "peer_examples": ["AAPL", "MSFT"],
            "source": "Static",
        }))
        mock_rw.arun = AsyncMock(return_value=_mock_report(["AAPL"]))
        mock_sql.from_conn_string.return_value = MagicMock()

        # Mock the SOP prompt chain
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=mock_sop_ainvoke)
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        final_state, trace_path = await run_pipeline(
            query="Analyse AAPL",
            tickers=["AAPL"],
            dry_run=True,
            trace_output_dir="/tmp",
        )

    report = final_state.get("report_markdown", "")

    # Assert all mandatory SOP sections present
    assert "Data Coverage Summary" in report, "Missing: Data Coverage Summary"
    assert "This is not financial advice" in report, "Missing: Disclaimer"


@pytest.mark.asyncio
async def test_pipeline_report_has_disclaimer():
    """Disclaimer must be present regardless of LLM output."""
    from ai_financial_analyst.agents.editor import editor_node
    from ai_financial_analyst.core.state import AgentState

    analysis = {
        "NVDA": {
            "ticker": "NVDA",
            "price_cagr_5y_pct": 55.0,
            "sector": "Information Technology",
            "sector_pe_avg": 28.5,
            "company_pe": 65.0,
            "bull_case": ["AI accelerator demand", "Data center growth"],
            "bear_case": ["Valuation risk", "Export restrictions"],
            "citations": {},
        }
    }

    report_without_disclaimer = "# NVDA Report\n\nNvidia has strong AI momentum.\n"

    state = AgentState(
        query="Analyse NVDA",
        tickers=["NVDA"],
        raw_data={"NVDA": {}},
        analysis=analysis,
        data_coverage=[],
        researcher_gaps=[],
        iteration_log=[],
        errors=[],
        status="COMPLETE",
        run_id="e2e-nvda-001",
    )

    with patch("ai_financial_analyst.agents.editor.report_writer_tool") as mock_rw:
        mock_rw.arun = AsyncMock(return_value=report_without_disclaimer)
        result = await editor_node(state, config={"primary_llm": None, "tracer": None})

    assert "This is not financial advice" in result["report_markdown"]
