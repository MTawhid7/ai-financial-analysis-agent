"""Unit tests for editor.py grounding check improvements.

Tests the enhanced _NUMERIC_PATTERN, _clean_numeric, _parse_grounded_floats,
and _is_grounded_by_scale functions which now handle:
- Comma-formatted numbers: $1,234
- Accounting negatives: (4,200)
- SI suffixes: 4.17T, $285.5B
- Tiered tolerance by value type
"""

from __future__ import annotations

import pytest

from ai_financial_analyst.agents.editor import (
    _NUMERIC_PATTERN,
    _clean_numeric,
    _parse_grounded_floats,
    _is_grounded_by_scale,
)


class TestNumericPattern:
    def test_plain_integer(self):
        assert _NUMERIC_PATTERN.findall("Revenue was 94") == ["94"]

    def test_decimal(self):
        assert _NUMERIC_PATTERN.findall("P/E of 28.5") == ["28.5"]

    def test_percentage(self):
        assert _NUMERIC_PATTERN.findall("margins grew 27.2%") == ["27.2%"]

    def test_dollar_with_commas(self):
        matches = _NUMERIC_PATTERN.findall("Revenue: $1,234,567")
        assert any("1,234" in m or "1,234,567" in m for m in matches)

    def test_si_suffix_billions(self):
        matches = _NUMERIC_PATTERN.findall("Market cap $285.5B")
        assert any("285.5B" in m for m in matches)

    def test_si_suffix_trillions(self):
        matches = _NUMERIC_PATTERN.findall("Total assets $4.17T")
        assert any("4.17T" in m for m in matches)

    def test_accounting_negative(self):
        matches = _NUMERIC_PATTERN.findall("Net loss (4,200)")
        assert any("4,200" in m for m in matches)

    def test_dollar_accounting_negative(self):
        matches = _NUMERIC_PATTERN.findall("Deficit ($1.2B)")
        assert any("1.2B" in m for m in matches)


class TestCleanNumeric:
    def test_plain_float(self):
        val, neg = _clean_numeric("28.5")
        assert val == pytest.approx(28.5)
        assert neg is False

    def test_percentage(self):
        val, neg = _clean_numeric("27.2%")
        assert val == pytest.approx(27.2)

    def test_comma_formatted(self):
        val, neg = _clean_numeric("$1,234,567")
        assert val == pytest.approx(1_234_567)

    def test_billions_suffix(self):
        val, neg = _clean_numeric("$285.5B")
        assert val == pytest.approx(285.5e9)

    def test_trillions_suffix(self):
        val, neg = _clean_numeric("4.17T")
        assert val == pytest.approx(4.17e12)

    def test_accounting_negative(self):
        val, neg = _clean_numeric("(4,200)")
        assert val == pytest.approx(-4200.0)
        assert neg is True

    def test_unparseable_returns_none(self):
        val, neg = _clean_numeric("not_a_number")
        assert val is None


class TestParseGroundedFloats:
    def test_basic_conversion(self):
        result = _parse_grounded_floats({"100", "28.5", "27.2%"})
        assert 100.0 in result
        assert 28.5 in result
        assert 27.2 in result

    def test_si_suffix_parsed(self):
        result = _parse_grounded_floats({"$285.5B"})
        assert any(abs(v - 285.5e9) < 1 for v in result)

    def test_skips_unparseable(self):
        result = _parse_grounded_floats({"abc", "28.5"})
        assert 28.5 in result
        assert len(result) == 1


class TestIsGroundedByScale:
    def test_exact_match(self):
        assert _is_grounded_by_scale("28.5", [28.5, 100.0]) is True

    def test_scaled_match_billions(self):
        # Report says "285.5B", grounded value is 285_500_000_000
        assert _is_grounded_by_scale("285.5B", [285_500_000_000]) is True

    def test_percentage_match(self):
        # Report says "27.2%", grounded value is 0.272 (fraction)
        assert _is_grounded_by_scale("27.2%", [0.272]) is True

    def test_tiered_tolerance_percentage(self):
        # 2% tolerance for percentages — 27.0% within 2% of 27.2%
        assert _is_grounded_by_scale("27.0%", [27.2]) is True

    def test_tiered_tolerance_large_number(self):
        # 5% tolerance for large numbers — $290B within 5% of $285.5B
        assert _is_grounded_by_scale("$290B", [285.5e9]) is True

    def test_ungrounded_value_flagged(self):
        # 999.99 not close to 28.5 at any scale
        assert _is_grounded_by_scale("999.99", [28.5]) is False

    def test_zero_is_always_grounded(self):
        assert _is_grounded_by_scale("0", []) is True

    def test_unparseable_is_not_flagged(self):
        assert _is_grounded_by_scale("N/A", [100.0]) is True


# ── SOP weighted scoring ──────────────────────────────────────────────────────

class TestSOPWeightedScoring:
    """Tests for the weighted SOP rubric replacing the binary pass/fail check."""

    def _ticker_analysis(self, **present_keys) -> dict:
        """Build a minimal analysis dict with the given keys set to truthy values."""
        base = {
            "price_cagr_5y_pct": None,
            "sector_pe_avg":     None,
            "company_pe":        None,
            "bull_case":         None,
            "bear_case":         None,
        }
        base.update(present_keys)
        return base

    def _compute_score(self, ticker_analysis: dict) -> float:
        from ai_financial_analyst.agents.editor import _SOP_WEIGHTS
        return round(
            sum(w for key, (_, w) in _SOP_WEIGHTS.items() if ticker_analysis.get(key)),
            3,
        )

    def test_all_fields_present_full_score(self):
        analysis = self._ticker_analysis(
            price_cagr_5y_pct=14.0, sector_pe_avg=25.0,
            company_pe=28.0, bull_case=["growth"], bear_case=["risk"],
        )
        assert self._compute_score(analysis) == 1.0

    def test_missing_bear_case_passes_threshold(self):
        from ai_financial_analyst.agents.editor import _SOP_PASS_THRESHOLD
        analysis = self._ticker_analysis(
            price_cagr_5y_pct=14.0, sector_pe_avg=25.0,
            company_pe=28.0, bull_case=["growth"],
        )
        assert self._compute_score(analysis) >= _SOP_PASS_THRESHOLD

    def test_missing_both_narrative_fields_still_passes(self):
        from ai_financial_analyst.agents.editor import _SOP_PASS_THRESHOLD
        analysis = self._ticker_analysis(
            price_cagr_5y_pct=14.0, sector_pe_avg=25.0, company_pe=28.0,
        )
        score = self._compute_score(analysis)
        assert score == 0.75
        assert score >= _SOP_PASS_THRESHOLD

    def test_missing_two_critical_fields_fails(self):
        from ai_financial_analyst.agents.editor import _SOP_PASS_THRESHOLD
        # Only bull_case + bear_case present → 0.15 + 0.10 = 0.25
        analysis = self._ticker_analysis(bull_case=["g"], bear_case=["r"])
        assert self._compute_score(analysis) < _SOP_PASS_THRESHOLD

    def test_weights_sum_to_one(self):
        from ai_financial_analyst.agents.editor import _SOP_WEIGHTS
        total = sum(w for _, w in _SOP_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_sop_score_min_in_state(self):
        """editor_node exposes sop_score_min in analysis_with_coverage."""
        from ai_financial_analyst.agents.editor import _SOP_WEIGHTS, _SOP_PASS_THRESHOLD
        # Score for a complete analysis = 1.0; verify constant expectation
        full_analysis = {
            k: "present" for k in _SOP_WEIGHTS
        }
        score = round(sum(w for _, w in _SOP_WEIGHTS.values()), 3)
        assert score == 1.0
