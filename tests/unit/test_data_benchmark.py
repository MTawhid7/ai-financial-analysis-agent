"""Unit tests for data/benchmark/ — static lazy-load, normalizer, lookup."""

from __future__ import annotations

import pytest

from ai_financial_analyst.data.benchmark.static import load, sector_names, get_static
from ai_financial_analyst.data.benchmark.normalizer import normalise_sector
from ai_financial_analyst.data.benchmark.lookup import _geographic_context


class TestStaticBenchmark:
    def test_load_returns_dict(self):
        data = load()
        assert isinstance(data, dict)

    def test_sector_names_non_empty(self):
        names = sector_names()
        assert len(names) >= 11  # all 11 GICS sectors

    def test_get_static_includes_required_fields(self):
        data = get_static("Information Technology")
        assert "pe_ratio_sector_avg" in data
        assert "ev_ebitda_sector_avg" in data
        assert "source" in data

    def test_get_static_missing_sector_returns_empty(self):
        data = get_static("NonExistentSector")
        assert data["pe_ratio_sector_avg"] is None


class TestSectorNormalizer:
    def test_exact_gics_match(self):
        sector, _ = normalise_sector("Information Technology")
        assert sector == "Information Technology"

    def test_yfinance_technology_alias(self):
        sector, matched = normalise_sector("Technology")
        assert sector == "Information Technology"
        assert matched == "Technology"

    def test_yfinance_healthcare_alias(self):
        sector, _ = normalise_sector("Healthcare")
        assert sector == "Health Care"

    def test_consumer_cyclical_alias(self):
        sector, _ = normalise_sector("Consumer Cyclical")
        assert sector == "Consumer Discretionary"

    def test_financial_services_alias(self):
        sector, _ = normalise_sector("Financial Services")
        assert sector == "Financials"

    def test_consumer_defensive_alias(self):
        sector, _ = normalise_sector("Consumer Defensive")
        assert sector == "Consumer Staples"

    def test_unknown_returns_none(self):
        sector, matched = normalise_sector("FakeSector XYZ 999")
        assert sector is None
        assert matched is None

    def test_case_insensitive_exact(self):
        sector, _ = normalise_sector("information technology")
        assert sector == "Information Technology"


class TestGeographicContext:
    def test_us_returns_none(self):
        assert _geographic_context("United States") is None
        assert _geographic_context("USA") is None

    def test_em_china(self):
        ctx = _geographic_context("China")
        assert ctx["geographic_scope"] == "emerging_market"
        assert ctx["typical_pe_discount_vs_us"] == -30

    def test_em_india(self):
        ctx = _geographic_context("India")
        assert ctx["geographic_scope"] == "emerging_market"

    def test_developed_germany(self):
        ctx = _geographic_context("Germany")
        assert ctx["geographic_scope"] == "developed_ex_us"
        assert ctx["typical_pe_discount_vs_us"] == -12

    def test_unclassified_country(self):
        ctx = _geographic_context("Narnia")
        assert ctx["geographic_scope"] == "non_us_unclassified"
        assert ctx["typical_pe_discount_vs_us"] is None
