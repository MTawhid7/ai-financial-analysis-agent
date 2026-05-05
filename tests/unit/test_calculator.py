"""Unit tests for CalculatorTool — AST whitelist and numexpr evaluation."""

import json
import pytest

from ai_financial_analyst.tools.calculator import calculator_tool, _validate_ast


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
