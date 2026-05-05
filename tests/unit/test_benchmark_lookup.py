"""Unit tests for BenchmarkLookupTool."""

import json
import pytest

from ai_financial_analyst.tools.benchmark_lookup import benchmark_lookup_tool


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
        assert "FakeSector XYZ" in data["message"]

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
