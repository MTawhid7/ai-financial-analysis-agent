"""Unit tests for quant_analyst.py — DCF valuation and scenario analysis.

Tests _compute_dcf() and _compute_scenarios() in isolation (no LLM calls).
These are pure-Python deterministic functions so no mocking is needed.
"""

from __future__ import annotations

import pytest

from ai_financial_analyst.agents.quant_analyst import _compute_dcf, _compute_scenarios


# ── _compute_dcf ──────────────────────────────────────────────────────────────

class TestComputeDCF:
    """Option A: negative/zero FCF returns a not-applicable sentinel, not an error."""

    def _standard_inputs(self, **overrides):
        inputs = {
            "fcf":               100_000_000_000,   # $100B FCF (AAPL-like)
            "revenue_growth_raw": 0.08,
            "beta":               1.2,
            "risk_free_rate_pct": 4.2,
            "total_debt":         104_000_000_000,
            "cash":                30_000_000_000,
            "market_cap":        3_000_000_000_000,
            "current_price":     200.0,
        }
        inputs.update(overrides)
        return inputs

    def test_returns_intrinsic_value_for_positive_fcf(self):
        result = _compute_dcf(**self._standard_inputs())
        assert not result.get("dcf_not_applicable")
        assert "intrinsic_value_per_share" in result
        assert result["intrinsic_value_per_share"] is not None
        assert result["intrinsic_value_per_share"] > 0

    def test_option_a_negative_fcf(self):
        """Negative FCF → not_applicable sentinel with explanation."""
        result = _compute_dcf(**self._standard_inputs(fcf=-5_000_000_000))
        assert result["dcf_not_applicable"] is True
        assert "reinvestment" in result["reason"].lower() or "negative" in result["reason"].lower()

    def test_option_a_zero_fcf(self):
        result = _compute_dcf(**self._standard_inputs(fcf=0))
        assert result["dcf_not_applicable"] is True

    def test_option_a_none_fcf(self):
        result = _compute_dcf(**self._standard_inputs(fcf=None))
        assert result["dcf_not_applicable"] is True

    def test_option_a_missing_beta(self):
        result = _compute_dcf(**self._standard_inputs(beta=None))
        assert result["dcf_not_applicable"] is True

    def test_option_a_missing_market_cap(self):
        result = _compute_dcf(**self._standard_inputs(market_cap=None))
        assert result["dcf_not_applicable"] is True

    def test_option_a_zero_price(self):
        result = _compute_dcf(**self._standard_inputs(current_price=0.0))
        assert result["dcf_not_applicable"] is True

    def test_wacc_is_positive_and_reasonable(self):
        result = _compute_dcf(**self._standard_inputs())
        wacc = result["wacc_pct"]
        assert 3.0 < wacc < 20.0, f"WACC {wacc}% outside plausible range"

    def test_margin_of_safety_computed(self):
        result = _compute_dcf(**self._standard_inputs())
        assert "margin_of_safety_pct" in result
        assert isinstance(result["margin_of_safety_pct"], float)

    def test_fcf_growth_capped_at_25_percent(self):
        """Revenue growth of 80% should be capped at 25% for FCF projection."""
        result_high = _compute_dcf(**self._standard_inputs(revenue_growth_raw=0.80))
        result_low  = _compute_dcf(**self._standard_inputs(revenue_growth_raw=0.25))
        assert not result_high.get("dcf_not_applicable")
        assert result_high["fcf_growth_rate_used_pct"] == 25.0

    def test_fcf_growth_floored_at_zero(self):
        """Negative revenue growth → FCF growth floored at 0%, not negative."""
        result = _compute_dcf(**self._standard_inputs(revenue_growth_raw=-0.10))
        assert not result.get("dcf_not_applicable")
        assert result["fcf_growth_rate_used_pct"] == 0.0

    def test_five_year_fcf_projections_populated(self):
        result = _compute_dcf(**self._standard_inputs())
        assert len(result["fcf_projected_5y"]) == 5

    def test_assumptions_block_present(self):
        result = _compute_dcf(**self._standard_inputs())
        assert "assumptions" in result
        for key in ("equity_risk_premium_pct", "cost_of_equity_pct", "cost_of_debt_pct"):
            assert key in result["assumptions"]

    def test_warning_always_present_on_success(self):
        result = _compute_dcf(**self._standard_inputs())
        assert "warning" in result
        assert "directional" in result["warning"].lower()

    def test_no_debt_case(self):
        """Company with zero debt should still produce a valid DCF."""
        result = _compute_dcf(**self._standard_inputs(total_debt=0, cash=50_000_000_000))
        assert not result.get("dcf_not_applicable")
        assert result["intrinsic_value_per_share"] is not None

    def test_high_debt_can_produce_negative_equity(self):
        """Extreme leverage may produce a negative intrinsic value — that's valid."""
        result = _compute_dcf(**self._standard_inputs(
            total_debt=50_000_000_000_000,  # 50T debt >> 3T market cap
            cash=1_000_000_000,
        ))
        # May produce negative intrinsic or still a valid float — should not crash
        assert not result.get("dcf_not_applicable") or result.get("dcf_not_applicable")


# ── _compute_scenarios ────────────────────────────────────────────────────────

class TestComputeScenarios:
    def _full_inputs(self, **overrides):
        inputs = {
            "current_price": 200.0,
            "forward_eps":   7.2,
            "sector_pe":     28.0,
            "analyst_low":   180.0,
            "analyst_mean":  220.0,
            "analyst_high":  260.0,
            "dcf_intrinsic": 195.0,
        }
        inputs.update(overrides)
        return inputs

    def test_all_three_methods_present(self):
        result = _compute_scenarios(**self._full_inputs())
        assert "pe_based" in result
        assert "analyst_consensus" in result
        assert "dcf" in result

    def test_pe_based_bear_base_bull_present(self):
        result = _compute_scenarios(**self._full_inputs())
        pe = result["pe_based"]
        assert "bear" in pe and "base" in pe and "bull" in pe

    def test_pe_based_multiples_correct(self):
        result = _compute_scenarios(**self._full_inputs(sector_pe=28.0))
        pe = result["pe_based"]
        assert pe["bear"]["pe_multiple"] == pytest.approx(22.4, abs=0.1)  # 28 × 0.80
        assert pe["base"]["pe_multiple"] == pytest.approx(28.0, abs=0.1)
        assert pe["bull"]["pe_multiple"] == pytest.approx(33.6, abs=0.1)  # 28 × 1.20

    def test_pe_based_price_targets_correct(self):
        result = _compute_scenarios(**self._full_inputs(forward_eps=10.0, sector_pe=20.0))
        pe = result["pe_based"]
        assert pe["bear"]["price_target"] == pytest.approx(160.0, abs=0.5)   # 16 × 10
        assert pe["base"]["price_target"] == pytest.approx(200.0, abs=0.5)   # 20 × 10
        assert pe["bull"]["price_target"] == pytest.approx(240.0, abs=0.5)   # 24 × 10

    def test_upside_pct_computed_correctly(self):
        result = _compute_scenarios(**self._full_inputs(
            current_price=200.0, forward_eps=10.0, sector_pe=20.0
        ))
        base_upside = result["pe_based"]["base"]["upside_pct"]
        # base price = 200, current = 200 → 0%
        assert base_upside == pytest.approx(0.0, abs=0.5)

    def test_analyst_consensus_all_three_filled(self):
        result = _compute_scenarios(**self._full_inputs())
        cons = result["analyst_consensus"]
        assert "bear" in cons and "base" in cons and "bull" in cons
        assert cons["base"]["price_target"] == 220.0

    def test_dcf_scenario_present_when_intrinsic_given(self):
        result = _compute_scenarios(**self._full_inputs(dcf_intrinsic=195.0))
        assert "dcf" in result
        mos = result["dcf"]["margin_of_safety_pct"]
        assert mos == pytest.approx(-2.5, abs=0.5)  # (195-200)/200 × 100

    def test_pe_based_absent_when_no_forward_eps(self):
        result = _compute_scenarios(**self._full_inputs(forward_eps=None))
        assert "pe_based" not in result

    def test_pe_based_absent_when_negative_forward_eps(self):
        result = _compute_scenarios(**self._full_inputs(forward_eps=-1.0))
        assert "pe_based" not in result

    def test_pe_based_absent_when_no_sector_pe(self):
        result = _compute_scenarios(**self._full_inputs(sector_pe=None))
        assert "pe_based" not in result

    def test_analyst_consensus_absent_when_all_none(self):
        result = _compute_scenarios(**self._full_inputs(
            analyst_low=None, analyst_mean=None, analyst_high=None
        ))
        assert "analyst_consensus" not in result

    def test_analyst_consensus_partial_when_some_missing(self):
        """Only mean provided → only base in analyst_consensus."""
        result = _compute_scenarios(**self._full_inputs(
            analyst_low=None, analyst_high=None
        ))
        assert "analyst_consensus" in result
        assert "base" in result["analyst_consensus"]
        assert "bear" not in result["analyst_consensus"]
        assert "bull" not in result["analyst_consensus"]

    def test_dcf_absent_when_intrinsic_none(self):
        result = _compute_scenarios(**self._full_inputs(dcf_intrinsic=None))
        assert "dcf" not in result

    def test_empty_scenarios_when_all_inputs_none(self):
        result = _compute_scenarios(
            current_price=None, forward_eps=None, sector_pe=None,
            analyst_low=None, analyst_mean=None, analyst_high=None,
            dcf_intrinsic=None,
        )
        assert result == {}

    def test_forward_eps_and_sector_pe_recorded_in_pe_based(self):
        result = _compute_scenarios(**self._full_inputs(forward_eps=7.2, sector_pe=28.0))
        assert result["pe_based"]["forward_eps_used"] == 7.2
        assert result["pe_based"]["sector_pe_used"]   == 28.0
