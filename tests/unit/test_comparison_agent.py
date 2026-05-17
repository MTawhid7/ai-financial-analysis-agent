"""Unit tests for the comparison agent helpers.

Tests the pure helper functions independently without running the pipeline
or making LLM calls.
"""

from __future__ import annotations

import pytest

from ai_financial_analyst.agents.comparison_agent import (
    _build_comparison_payload,
    _build_fallback_table,
    _extract_user_dimensions,
    _get_raw_dict,
    _validate_comparison_table,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_analysis() -> dict:
    return {
        "AAPL": {
            "price_cagr_5y_pct": 24.5,
            "company_pe": 28.3,
            "sector_pe_avg": 25.0,
            "pe_premium_pct": 13.2,
            "dcf_intrinsic_value": 185.0,
            "bull_case": "Strong iPhone cycle",
            "bear_case": "China headwinds",
        },
        "MSFT": {
            "price_cagr_5y_pct": 27.1,
            "company_pe": 31.5,
            "sector_pe_avg": 25.0,
            "pe_premium_pct": 26.0,
            "dcf_intrinsic_value": 390.0,
            "bull_case": "Azure cloud growth",
            "bear_case": "AI capex burn",
        },
    }


@pytest.fixture
def sample_raw_data() -> dict:
    return {
        "AAPL": {
            "fundamentals": {
                "current_price": 185.0,
                "market_cap": 2_900_000_000_000,
                "revenue_ttm": 385_000_000_000,
                "net_income_ttm": 97_000_000_000,
                "profit_margin": 0.252,
                "pe_ratio": 28.3,
                "sector": "Technology",
            },
            "cash_flow": {
                "free_cash_flow": 110_000_000_000,
                "ocf": 120_000_000_000,
                "dividend_yield_pct": 0.55,
                "annual_dividend_per_share": 0.96,
                "payout_ratio": 0.15,
            },
            "price_metrics": {
                "sharpe_ratio": 1.12,
                "max_drawdown_pct": -31.5,
                "beta": 1.21,
                "annualized_volatility_pct": 28.4,
            },
        },
        "MSFT": {
            "fundamentals": {
                "current_price": 415.0,
                "market_cap": 3_080_000_000_000,
                "revenue_ttm": 227_000_000_000,
                "net_income_ttm": 72_000_000_000,
                "profit_margin": 0.317,
                "pe_ratio": 31.5,
                "sector": "Technology",
            },
            "cash_flow": {
                "free_cash_flow": 70_000_000_000,
                "ocf": 87_000_000_000,
                "dividend_yield_pct": 0.72,
                "annual_dividend_per_share": 3.00,
                "payout_ratio": 0.24,
            },
        },
    }


# ── _get_raw_dict ─────────────────────────────────────────────────────────────

class TestGetRawDict:
    def test_returns_dict_for_known_ticker_and_type(self, sample_raw_data):
        result = _get_raw_dict(sample_raw_data, "AAPL", "fundamentals")
        assert result["current_price"] == 185.0

    def test_returns_empty_for_missing_ticker(self, sample_raw_data):
        result = _get_raw_dict(sample_raw_data, "FAKE", "fundamentals")
        assert result == {}

    def test_returns_empty_for_missing_data_type(self, sample_raw_data):
        result = _get_raw_dict(sample_raw_data, "AAPL", "nonexistent")
        assert result == {}

    def test_parses_json_string_value(self):
        raw_data = {"AAPL": {"fundamentals": '{"current_price": 200.0}'}}
        result = _get_raw_dict(raw_data, "AAPL", "fundamentals")
        assert result["current_price"] == 200.0

    def test_returns_empty_on_invalid_json_string(self):
        raw_data = {"AAPL": {"fundamentals": "not-json"}}
        result = _get_raw_dict(raw_data, "AAPL", "fundamentals")
        assert result == {}


# ── _build_comparison_payload ─────────────────────────────────────────────────

class TestBuildComparisonPayload:
    def test_extracts_fundamentals_for_each_ticker(self, sample_analysis, sample_raw_data):
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["AAPL", "MSFT"], [])
        assert payload["AAPL"]["fundamentals"]["current_price"] == 185.0
        assert payload["MSFT"]["fundamentals"]["market_cap"] == 3_080_000_000_000

    def test_extracts_analysis_fields(self, sample_analysis, sample_raw_data):
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["AAPL", "MSFT"], [])
        assert payload["AAPL"]["analysis"]["price_cagr_5y_pct"] == 24.5
        assert payload["MSFT"]["analysis"]["pe_premium_pct"] == 26.0

    def test_omits_none_values(self, sample_raw_data):
        analysis = {"AAPL": {"price_cagr_5y_pct": None, "company_pe": 28.3}}
        payload = _build_comparison_payload(analysis, sample_raw_data, ["AAPL"], [])
        assert "price_cagr_5y_pct" not in payload["AAPL"]["analysis"]
        assert payload["AAPL"]["analysis"]["company_pe"] == 28.3

    def test_no_truncation_with_large_field_values(self, sample_raw_data):
        long_text = "x" * 5000
        analysis = {"AAPL": {"bull_case": long_text, "price_cagr_5y_pct": 20.0}}
        payload = _build_comparison_payload(analysis, sample_raw_data, ["AAPL"], [])
        assert payload["AAPL"]["analysis"]["bull_case"] == long_text

    def test_extra_dimension_included_in_payload(self, sample_analysis, sample_raw_data):
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["AAPL", "MSFT"], ["dividend"])
        assert "dividend" in payload["AAPL"]
        assert payload["AAPL"]["dividend"]["dividend_yield_pct"] == 0.55

    def test_missing_extra_dimension_data_omitted(self, sample_analysis, sample_raw_data):
        # MSFT has no price_metrics in sample_raw_data → risk section absent
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["MSFT"], ["risk"])
        assert "risk" not in payload["MSFT"]

    def test_extraneous_fields_excluded(self, sample_analysis, sample_raw_data):
        # raw_data has "payout_ratio" in cash_flow but it is NOT in _FUNDAMENTALS_FIELDS
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["AAPL"], [])
        assert "payout_ratio" not in payload["AAPL"]["fundamentals"]


# ── _extract_user_dimensions ──────────────────────────────────────────────────

class TestExtractUserDimensions:
    def test_dividend_keyword_detected(self):
        dims = _extract_user_dimensions("Compare AAPL vs MSFT including dividend yield")
        assert "dividend" in dims

    def test_multiple_keywords(self):
        dims = _extract_user_dimensions("I want risk and cash flow in the comparison")
        assert "risk" in dims
        assert "cash flow" in dims

    def test_no_match_returns_empty(self):
        dims = _extract_user_dimensions("Compare AAPL vs MSFT")
        assert dims == []

    def test_case_insensitive(self):
        dims = _extract_user_dimensions("Show me DIVIDEND data")
        assert "dividend" in dims

    def test_valuation_keyword(self):
        dims = _extract_user_dimensions("Include valuation multiples")
        assert "valuation" in dims


# ── _validate_comparison_table ────────────────────────────────────────────────

class TestValidateComparisonTable:
    def test_all_tickers_present_returns_empty(self):
        md = "| Metric | AAPL | MSFT |\n|---|---|---|\n| Price | $185 | $415 |"
        missing = _validate_comparison_table(md, ["AAPL", "MSFT"])
        assert missing == []

    def test_missing_ticker_returned(self):
        md = "| Metric | AAPL |\n|---|---|\n| Price | $185 |"
        missing = _validate_comparison_table(md, ["AAPL", "MSFT"])
        assert "MSFT" in missing
        assert "AAPL" not in missing

    def test_case_insensitive_check(self):
        md = "aapl shows strong revenue. msft shows cloud growth."
        missing = _validate_comparison_table(md, ["AAPL", "MSFT"])
        assert missing == []

    def test_all_missing_returns_all(self):
        md = "Analysis not available."
        missing = _validate_comparison_table(md, ["AAPL", "MSFT", "GOOG"])
        assert set(missing) == {"AAPL", "MSFT", "GOOG"}


# ── _build_fallback_table ─────────────────────────────────────────────────────

class TestBuildFallbackTable:
    def test_contains_all_tickers_as_columns(self, sample_analysis, sample_raw_data):
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["AAPL", "MSFT"], [])
        table = _build_fallback_table(payload, ["AAPL", "MSFT"])
        assert "AAPL" in table
        assert "MSFT" in table

    def test_contains_required_metric_rows(self, sample_analysis, sample_raw_data):
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["AAPL", "MSFT"], [])
        table = _build_fallback_table(payload, ["AAPL", "MSFT"])
        assert "Current Price" in table
        assert "Market Cap" in table
        assert "P/E Ratio" in table
        assert "5Y Price CAGR" in table

    def test_na_for_missing_data(self):
        payload = {"AAPL": {}, "MSFT": {}}
        table = _build_fallback_table(payload, ["AAPL", "MSFT"])
        assert "N/A" in table

    def test_is_valid_markdown_table(self, sample_analysis, sample_raw_data):
        payload = _build_comparison_payload(sample_analysis, sample_raw_data, ["AAPL", "MSFT"], [])
        table = _build_fallback_table(payload, ["AAPL", "MSFT"])
        lines = [l for l in table.splitlines() if l.strip()]
        assert all("|" in line for line in lines)


# ── run_comparison — minimal contract test ────────────────────────────────────

class TestRunComparisonMinimum:
    @pytest.mark.asyncio
    async def test_single_ticker_returns_error_without_pipeline(self):
        from unittest.mock import AsyncMock, MagicMock

        from ai_financial_analyst.agents.comparison_agent import run_comparison

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()

        result, state = await run_comparison(
            message="Compare AAPL",
            tickers=["AAPL"],
            primary_llm=mock_llm,
        )

        assert "at least two" in result.lower()
        assert state is None
        mock_llm.ainvoke.assert_not_called()
