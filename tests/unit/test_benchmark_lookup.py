"""Unit tests for BenchmarkLookupTool — sector lookup, fuzzy matching, geographic context."""

import json
import pytest

from ai_financial_analyst.tools.benchmark_lookup import (
    benchmark_lookup_tool,
    _normalise_sector,
    _geographic_context,
)


class TestBenchmarkLookupTool:
    def test_known_sector(self):
        result = benchmark_lookup_tool.invoke({"gics_sector": "Information Technology"})
        data = json.loads(result)
        assert "pe_ratio_sector_avg" in data
        assert data["pe_ratio_sector_avg"] > 0
        assert "peer_examples" in data

    def test_case_insensitive(self):
        result = benchmark_lookup_tool.invoke({"gics_sector": "information technology"})
        data = json.loads(result)
        assert "pe_ratio_sector_avg" in data

    def test_unknown_sector_returns_tool_error(self):
        result = benchmark_lookup_tool.invoke({"gics_sector": "FakeSector XYZ"})
        data = json.loads(result)
        assert data["error_type"] == "TOOL_ERROR"

    def test_all_required_fields_present(self):
        result = benchmark_lookup_tool.invoke({"gics_sector": "Financials"})
        data = json.loads(result)
        for field in ("pe_ratio_sector_avg", "ev_ebitda_sector_avg", "price_to_book_sector_avg"):
            assert field in data, f"Missing field: {field}"

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            benchmark_lookup_tool.invoke(
                {"gics_sector": "Energy", "injected_field": "value"}
            )


class TestFuzzySectorMatching:
    """_normalise_sector now returns (canonical, original) tuple."""

    def test_exact_gics_match(self):
        sector, matched_from = _normalise_sector("Information Technology")
        assert sector == "Information Technology"

    def test_yfinance_technology_alias(self):
        sector, matched_from = _normalise_sector("Technology")
        assert sector == "Information Technology"
        assert matched_from == "Technology"

    def test_yfinance_healthcare_alias(self):
        sector, matched_from = _normalise_sector("Healthcare")
        assert sector == "Health Care"

    def test_yfinance_consumer_cyclical_alias(self):
        sector, matched_from = _normalise_sector("Consumer Cyclical")
        assert sector == "Consumer Discretionary"

    def test_yfinance_financial_services_alias(self):
        sector, matched_from = _normalise_sector("Financial Services")
        assert sector == "Financials"

    def test_yfinance_consumer_defensive_alias(self):
        sector, matched_from = _normalise_sector("Consumer Defensive")
        assert sector == "Consumer Staples"

    def test_fuzzy_match_health_care_variant(self):
        # "Health Care" vs "Healthcare" handled by alias; test a true fuzzy case
        sector, matched_from = _normalise_sector("Industrials")
        assert sector == "Industrials"

    def test_unknown_sector_returns_none(self):
        sector, matched_from = _normalise_sector("FakeSector XYZ 999")
        assert sector is None
        assert matched_from is None

    def test_tool_accepts_yfinance_sector_name(self):
        """End-to-end: yfinance 'Technology' maps through to valid benchmark data."""
        result = benchmark_lookup_tool.invoke({"gics_sector": "Technology"})
        data = json.loads(result)
        assert "error_type" not in data
        assert "pe_ratio_sector_avg" in data
        # Provenance fields present when alias was used
        assert data.get("sector_matched_from") == "Technology"

    def test_tool_accepts_basic_materials(self):
        result = benchmark_lookup_tool.invoke({"gics_sector": "Basic Materials"})
        data = json.loads(result)
        assert "error_type" not in data
        assert data.get("sector_matched_from") == "Basic Materials"


class TestGeographicContext:
    def test_us_company_no_context(self):
        assert _geographic_context("United States") is None
        assert _geographic_context("USA") is None
        assert _geographic_context(None) is None

    def test_emerging_market_china(self):
        ctx = _geographic_context("China")
        assert ctx is not None
        assert ctx["geographic_scope"] == "emerging_market"
        assert ctx["typical_pe_discount_vs_us"] < 0
        assert "Emerging Market" in ctx["benchmark_note"]

    def test_emerging_market_india(self):
        ctx = _geographic_context("India")
        assert ctx["geographic_scope"] == "emerging_market"

    def test_developed_ex_us_uk(self):
        ctx = _geographic_context("United Kingdom")
        assert ctx["geographic_scope"] == "developed_ex_us"
        assert ctx["typical_pe_discount_vs_us"] == -12

    def test_developed_ex_us_germany(self):
        ctx = _geographic_context("Germany")
        assert ctx["geographic_scope"] == "developed_ex_us"

    def test_unclassified_country(self):
        ctx = _geographic_context("Narnia")
        assert ctx["geographic_scope"] == "non_us_unclassified"
        assert ctx["typical_pe_discount_vs_us"] is None

    def test_tool_includes_geo_context_for_non_us(self):
        result = benchmark_lookup_tool.invoke({
            "gics_sector": "Information Technology",
            "country": "China",
        })
        data = json.loads(result)
        assert "error_type" not in data
        assert "geographic_context" in data
        assert data["geographic_context"]["geographic_scope"] == "emerging_market"

    def test_tool_no_geo_context_for_us(self):
        result = benchmark_lookup_tool.invoke({
            "gics_sector": "Information Technology",
            "country": "United States",
        })
        data = json.loads(result)
        assert "geographic_context" not in data
