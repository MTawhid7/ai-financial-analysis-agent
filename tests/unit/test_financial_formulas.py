"""Unit tests for FinancialFormulas tool and individual function implementations.

Tests the function registry pattern: each implementation is tested independently
(pure-Python, no mocking), then the tool-level dispatch is tested end-to-end.
"""

from __future__ import annotations

import json
import pytest

from ai_financial_analyst.tools.financial_formulas import (
    financial_formulas_tool,
    _npv, _irr, _pv, _fv, _cagr, _wacc, _payback_period, _roi,
    _domain_warning,
)


# ── NPV ───────────────────────────────────────────────────────────────────────

class TestNPV:
    def test_standard_case_below_irr(self):
        # IRR of [-1000, 300, 400, 500] ≈ 8.9%; at 5% (< IRR) NPV is positive
        result = _npv(0.05, [-1000, 300, 400, 500])
        assert result == pytest.approx(80.44, abs=0.5)

    def test_standard_case_above_irr(self):
        # At 9% (> IRR ≈ 8.9%) NPV is negative
        result = _npv(0.09, [-1000, 300, 400, 500])
        assert result == pytest.approx(-2.01, abs=0.5)

    def test_negative_npv(self):
        # High discount rate → NPV negative
        result = _npv(0.50, [-1000, 300, 400, 500])
        assert result < 0

    def test_zero_rate_is_sum(self):
        # At 0% discount rate NPV = simple sum
        result = _npv(0.0, [-1000, 300, 300, 400])
        assert result == pytest.approx(0.0, abs=0.01)

    def test_single_future_cashflow(self):
        # PV of 110 in 1yr at 10% = 100 → NPV of [-100, 110] = 0
        result = _npv(0.10, [-100, 110])
        assert result == pytest.approx(0.0, abs=0.01)

    def test_rate_below_minus_one_raises(self):
        with pytest.raises(ValueError):
            _npv(-1.1, [-100, 50])


# ── IRR ───────────────────────────────────────────────────────────────────────

class TestIRR:
    def test_simple_case(self):
        # -100 at t=0, 110 at t=1 → IRR = 10%
        result = _irr([-100, 110])
        assert result == pytest.approx(10.0, abs=0.01)

    def test_multi_period_case(self):
        # IRR of [-1000, 300, 400, 500] ≈ 8.9% (rate where NPV crosses zero)
        result = _irr([-1000, 300, 400, 500])
        assert 7.0 < result < 11.0

    def test_known_irr_20_percent(self):
        # -100 at t=0, +120 at t=1 → IRR exactly 20%
        result = _irr([-100, 120])
        assert result == pytest.approx(20.0, abs=0.01)

    def test_known_irr_22_percent(self):
        # -1000 at t=0, 0 at t=1, +1500 at t=2 → IRR = sqrt(1.5)-1 ≈ 22.47%
        result = _irr([-1000, 0, 1500])
        assert result == pytest.approx(22.47, abs=0.05)

    def test_requires_sign_change(self):
        with pytest.raises(ValueError, match="negative and one positive"):
            _irr([100, 200, 300])

    def test_too_few_cashflows(self):
        with pytest.raises(ValueError):
            _irr([-100])

    def test_returns_percent(self):
        # Result must be in % not decimal (i.e. > 1 for a 10%+ IRR)
        result = _irr([-100, 110])
        assert result > 1.0   # 10%, not 0.10


# ── PV ────────────────────────────────────────────────────────────────────────

class TestPV:
    def test_basic_annuity(self):
        # 5% rate, 10 periods, $1000 pmt → PV ≈ -$7722
        result = _pv(0.05, 10, 1000)
        assert result == pytest.approx(-7721.73, abs=1.0)

    def test_zero_rate(self):
        # 0% rate → PV = -(pmt × nper)
        result = _pv(0.0, 5, 100)
        assert result == pytest.approx(-500.0, abs=0.01)

    def test_with_future_value(self):
        # Single lump sum: pmt=0, fv_amount=1000 at 10% in 1yr → PV ≈ -909.09
        result = _pv(0.10, 1, 0, fv_amount=1000)
        assert result == pytest.approx(-909.09, abs=0.5)


# ── FV ────────────────────────────────────────────────────────────────────────

class TestFV:
    def test_basic_annuity(self):
        # 5% rate, 10 periods, $1000 pmt → FV ≈ -$12578
        result = _fv(0.05, 10, 1000)
        assert result == pytest.approx(-12577.89, abs=1.0)

    def test_zero_rate(self):
        result = _fv(0.0, 5, 100)
        assert result == pytest.approx(-500.0, abs=0.01)

    def test_pv_fv_relationship(self):
        # Sign convention: PV of positive PMT is negative (you pay), so pv_amount
        # is negative → FV comes out positive (you'd receive the accumulated value).
        pv = _pv(0.05, 5, 100)        # ≈ -432.95
        fv = _fv(0.05, 5, 0, pv_amount=pv)
        assert fv == pytest.approx(552.56, abs=1.0)


# ── CAGR ──────────────────────────────────────────────────────────────────────

class TestCAGR:
    def test_standard_case(self):
        # 100 → 200 in 5yr ≈ 14.87%
        result = _cagr(100, 200, 5)
        assert result == pytest.approx(14.87, abs=0.05)

    def test_negative_growth(self):
        # 200 → 100 in 5yr: negative CAGR
        result = _cagr(200, 100, 5)
        assert result < 0

    def test_zero_growth(self):
        result = _cagr(100, 100, 5)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_invalid_start_value_zero(self):
        with pytest.raises(ValueError, match="positive"):
            _cagr(0, 200, 5)

    def test_invalid_years_zero(self):
        with pytest.raises(ValueError, match="positive"):
            _cagr(100, 200, 0)

    def test_returns_percent(self):
        result = _cagr(100, 200, 5)
        assert result > 1.0   # 14.87%, not 0.1487


# ── WACC ──────────────────────────────────────────────────────────────────────

class TestWACC:
    def test_standard_case(self):
        # ke=12%, kd=5%, E=3T, D=1T, tax=21%
        # we=0.75, wd=0.25 → WACC = 0.75×12 + 0.25×5×(1-0.21) = 9+0.9875 = 9.9875%
        result = _wacc(0.12, 0.05, 3e12, 1e12, 0.21)
        assert result == pytest.approx(9.99, abs=0.05)

    def test_all_equity_no_debt(self):
        # All equity → WACC = ke
        result = _wacc(0.12, 0.05, 1.0, 0.0, 0.21)
        assert result == pytest.approx(12.0, abs=0.01)

    def test_invalid_zero_total(self):
        with pytest.raises(ValueError, match="positive"):
            _wacc(0.12, 0.05, 0.0, 0.0, 0.21)

    def test_returns_percent(self):
        result = _wacc(0.12, 0.05, 3e12, 1e12, 0.21)
        assert result > 1.0   # ~9.87%, not 0.0987


# ── Payback Period ────────────────────────────────────────────────────────────

class TestPaybackPeriod:
    def test_exact_recovery(self):
        # 200 investment, [100, 100] cashflows → 2 years
        result = _payback_period(200, [100, 100])
        assert result == pytest.approx(2.0, abs=0.01)

    def test_fractional_recovery(self):
        # 150 investment, [100, 100] → recovered midway through year 2
        result = _payback_period(150, [100, 100])
        assert result == pytest.approx(1.5, abs=0.01)

    def test_never_recovered_returns_none(self):
        result = _payback_period(1000, [100, 100])
        assert result is None

    def test_invalid_initial_investment_zero(self):
        with pytest.raises(ValueError):
            _payback_period(0, [100, 100])


# ── ROI ───────────────────────────────────────────────────────────────────────

class TestROI:
    def test_positive_roi(self):
        result = _roi(50, 100)
        assert result == pytest.approx(50.0, abs=0.01)

    def test_negative_roi(self):
        result = _roi(-20, 100)
        assert result == pytest.approx(-20.0, abs=0.01)

    def test_zero_cost_raises(self):
        with pytest.raises(ValueError, match="zero"):
            _roi(50, 0)

    def test_returns_percent(self):
        result = _roi(10, 100)
        assert result == pytest.approx(10.0, abs=0.01)


# ── Tool-level dispatch (end-to-end) ──────────────────────────────────────────

class TestFinancialFormulasTool:
    def test_npv_via_tool(self):
        # At 5% (below IRR ≈ 8.9%), NPV is positive ≈ 80.44
        result = financial_formulas_tool.invoke({
            "function":  "npv",
            "rate":      0.05,
            "cashflows": [-1000, 300, 400, 500],
        })
        data = json.loads(result)
        assert "error_type" not in data
        assert data["function"] == "npv"
        assert data["result_raw"] == pytest.approx(80.44, abs=0.5)
        assert data["unit"] == "currency"
        assert "$" in data["result_formatted"]

    def test_irr_via_tool(self):
        # -100 at t=0, +120 at t=1 → IRR exactly 20%
        result = financial_formulas_tool.invoke({
            "function":  "irr",
            "cashflows": [-100, 120],
        })
        data = json.loads(result)
        assert "error_type" not in data
        assert data["result_raw"] == pytest.approx(20.0, abs=0.01)
        assert "%" in data["result_formatted"]

    def test_cagr_via_tool(self):
        result = financial_formulas_tool.invoke({
            "function":    "cagr",
            "start_value": 100.0,
            "end_value":   200.0,
            "years":       5.0,
        })
        data = json.loads(result)
        assert data["result_raw"] == pytest.approx(14.87, abs=0.05)
        assert "%" in data["result_formatted"]
        assert data["unit"] == "percent"

    def test_wacc_via_tool(self):
        result = financial_formulas_tool.invoke({
            "function":  "wacc",
            "ke":        0.12,
            "kd":        0.05,
            "equity":    3e12,
            "debt":      1e12,
            "tax_rate":  0.21,
        })
        data = json.loads(result)
        assert data["result_raw"] == pytest.approx(9.99, abs=0.05)

    def test_pv_via_tool(self):
        result = financial_formulas_tool.invoke({
            "function": "pv",
            "rate":     0.05,
            "nper":     10,
            "pmt":      1000.0,
        })
        data = json.loads(result)
        assert data["result_raw"] == pytest.approx(-7721.73, abs=1.0)

    def test_roi_via_tool(self):
        result = financial_formulas_tool.invoke({
            "function": "roi",
            "gain":     50.0,
            "cost":     100.0,
        })
        data = json.loads(result)
        assert data["result_raw"] == pytest.approx(50.0, abs=0.01)

    def test_payback_period_never_recovered(self):
        result = financial_formulas_tool.invoke({
            "function":           "payback_period",
            "initial_investment": 1000.0,
            "cashflows":          [100.0, 100.0],
        })
        data = json.loads(result)
        assert data["result_raw"] is None
        assert "not recovered" in data["note"].lower()

    def test_missing_required_params_returns_error(self):
        result = financial_formulas_tool.invoke({
            "function": "npv",
            # rate and cashflows are missing
        })
        data = json.loads(result)
        assert data["error_type"] == "TOOL_ERROR"
        assert "Missing" in data["message"]

    def test_unknown_function_returns_error(self):
        with pytest.raises(Exception):   # Pydantic rejects unknown Literal
            financial_formulas_tool.invoke({"function": "not_a_function"})

    def test_inputs_echoed_in_response(self):
        result = financial_formulas_tool.invoke({
            "function":    "cagr",
            "start_value": 100.0,
            "end_value":   200.0,
            "years":       5.0,
        })
        data = json.loads(result)
        assert "inputs_used" in data
        assert data["inputs_used"]["start_value"] == 100.0

    def test_irr_domain_warning_high_return(self):
        # IRR > 500% triggers a domain warning
        warning = _domain_warning("irr", 550.0)
        assert warning is not None
        assert "unusually high" in warning.lower()

    def test_cagr_domain_warning_extreme(self):
        warning = _domain_warning("cagr", 200.0)
        assert warning is not None

    def test_no_warning_for_normal_irr(self):
        warning = _domain_warning("irr", 15.0)
        assert warning is None
