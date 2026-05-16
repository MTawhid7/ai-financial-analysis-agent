"""Unit tests for CalculatorTool — AST whitelist, numexpr evaluation, validation, formatting."""

import json
import pytest

from ai_financial_analyst.tools.calculator import (
    calculator_tool,
    _validate_ast,
    _validate_context_keys,
    _validate_result,
    _format_result,
)


class TestASTValidation:
    def test_valid_arithmetic(self):
        _validate_ast("(100 / 50) * 2 - 1")

    def test_valid_power(self):
        _validate_ast("((1.25 ** 5) - 1) * 100")

    def test_valid_cagr_expression(self):
        _validate_ast("((150.0 / 100.0) ** (1/5) - 1) * 100")

    def test_rejects_import(self):
        # Outer Call has func=Attribute (not a Name) → fn_name=None → rejected.
        with pytest.raises(ValueError):
            _validate_ast("__import__('os').system('rm -rf /')")

    def test_rejects_dangerous_name(self):
        # Inner Call has func=Name('__import__') → not in _SAFE_FUNCTIONS → rejected.
        with pytest.raises(ValueError, match="Disallowed"):
            _validate_ast("__import__('os')")

    def test_rejects_assignment(self):
        with pytest.raises((ValueError, SyntaxError)):
            _validate_ast("x = 1 + 2")

    def test_rejects_function_def(self):
        with pytest.raises((ValueError, SyntaxError)):
            _validate_ast("def f(): return 1")

    def test_rejects_string_ops(self):
        # String literals are non-numeric Constants — must be blocked.
        with pytest.raises(ValueError, match="Disallowed constant type"):
            _validate_ast("'hello' + 'world'")

    def test_syntax_error(self):
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            _validate_ast("(1 + ")


class TestCalculatorTool:
    def test_basic_arithmetic(self):
        result = calculator_tool.invoke({"expression": "2 + 2"})
        assert result == "4.0" or float(result) == pytest.approx(4.0)

    def test_cagr_calculation(self):
        # ((200 / 100) ** (1/5) - 1) * 100 ≈ 14.87%
        result = calculator_tool.invoke(
            {"expression": "((200.0 / 100.0) ** (1.0/5.0) - 1) * 100"}
        )
        assert float(result) == pytest.approx(14.87, abs=0.1)

    def test_percentage_change(self):
        result = calculator_tool.invoke({"expression": "(150 - 100) / 100 * 100"})
        assert float(result) == pytest.approx(50.0)

    def test_injection_attempt_returns_tool_error(self):
        result = calculator_tool.invoke({"expression": "__import__('os')"})
        data = json.loads(result)
        assert data["error_type"] == "TOOL_ERROR"

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            calculator_tool.invoke({"expression": "1+1", "evil_field": "hack"})


class TestResultValidation:
    def test_division_by_zero_returns_tool_error(self):
        result = calculator_tool.invoke({"expression": "1 / 0"})
        data   = json.loads(result)
        assert data["error_type"] == "TOOL_ERROR"
        assert "infinity" in data["message"].lower() or "zero" in data["message"].lower()

    def test_nan_returns_tool_error(self):
        # sqrt of a negative number produces nan in numexpr
        result = calculator_tool.invoke({"expression": "sqrt(-1.0)"})
        data   = json.loads(result)
        # numexpr returns nan for sqrt(-1); validate_result should catch it
        assert data["error_type"] == "TOOL_ERROR" or "nan" in str(data).lower()

    def test_validate_result_inf_returns_none(self):
        import math
        val, msg = _validate_result(math.inf)
        assert val is None
        assert "infinity" in msg.lower()

    def test_validate_result_nan_returns_none(self):
        import math
        val, msg = _validate_result(math.nan)
        assert val is None
        assert "nan" in msg.lower() or "undefined" in msg.lower()

    def test_validate_result_large_magnitude_warns(self):
        val, warning = _validate_result(2e16)
        assert val == 2e16            # value returned
        assert warning is not None    # but with a warning
        assert "large" in warning.lower()

    def test_validate_result_normal_no_warning(self):
        val, warning = _validate_result(42.5)
        assert val     == 42.5
        assert warning is None


class TestContextVariables:
    def test_named_variable_in_expression(self):
        result = calculator_tool.invoke({
            "expression": "market_cap_usd / revenue_usd",
            "context":    {"market_cap_usd": 3e12, "revenue_usd": 4e11},
        })
        assert float(result) == pytest.approx(7.5, abs=0.01)

    def test_context_with_multiple_vars(self):
        result = calculator_tool.invoke({
            "expression": "fcf_usd / market_cap_usd * 100",
            "context":    {"fcf_usd": 1e11, "market_cap_usd": 3e12},
        })
        assert float(result) == pytest.approx(3.333, abs=0.01)

    def test_invalid_context_key_starting_with_underscore(self):
        result = calculator_tool.invoke({
            "expression": "1 + 1",
            "context":    {"_bad_key": 1.0},
        })
        data = json.loads(result)
        assert data["error_type"] == "TOOL_ERROR"

    def test_validate_context_keys_rejects_underscore_prefix(self):
        with pytest.raises(ValueError, match="not start with"):
            _validate_context_keys({"_private": 1.0})

    def test_validate_context_keys_allows_valid_identifier(self):
        _validate_context_keys({"market_cap_usd": 1e12, "revenue_usd": 4e11})  # no error

    def test_context_var_in_ast_allowed(self):
        # 'revenue_usd' is not in _SAFE_NAMES but is in context_keys — must be allowed
        _validate_ast("revenue_usd * 2", context_keys={"revenue_usd"})

    def test_context_var_not_in_context_blocked(self):
        # name not provided in context → disallowed
        with pytest.raises(ValueError, match="Disallowed name"):
            _validate_ast("revenue_usd * 2", context_keys=set())


class TestFormatResult:
    def test_percent_format(self):
        formatted, applied = _format_result(14.22, "percent")
        assert formatted == "14.22%"
        assert applied   == "percent"

    def test_currency_trillions(self):
        formatted, applied = _format_result(4_170_000_000_000, "currency")
        assert "T" in formatted
        assert applied == "currency_trillions"

    def test_currency_billions(self):
        formatted, applied = _format_result(285_500_000_000, "currency")
        assert "B" in formatted
        assert applied == "currency_billions"

    def test_currency_millions(self):
        formatted, applied = _format_result(94_000_000, "currency")
        assert "M" in formatted

    def test_ratio_format(self):
        formatted, applied = _format_result(28.5, "ratio")
        assert "28.5×" == formatted

    def test_integer_format(self):
        formatted, applied = _format_result(15_726_000_000, "integer")
        assert "," in formatted    # thousands separator
        assert "." not in formatted

    def test_format_auto_large(self):
        formatted, applied = _format_result(3e12, "auto")
        assert "T" in formatted

    def test_format_returns_json_when_specified(self):
        result = calculator_tool.invoke({
            "expression": "14.22",
            "format":     "percent",
        })
        data = json.loads(result)
        assert data["result_raw"]       == pytest.approx(14.22)
        assert data["result_formatted"] == "14.22%"
        assert data["format_applied"]   == "percent"

    def test_no_format_returns_raw_string(self):
        result = calculator_tool.invoke({"expression": "2 + 2"})
        # No format → raw string, backward compatible
        assert float(result) == pytest.approx(4.0)
