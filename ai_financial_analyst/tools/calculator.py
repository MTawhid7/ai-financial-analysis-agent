"""CalculatorTool — constrained numexpr evaluator with AST whitelist validation.

The LLM-generated expression is treated as untrusted input.
Only numeric arithmetic is permitted. Validated at three levels:
  1. Node-type whitelist — rejects any non-arithmetic AST construct.
  2. Name allowlist — only known numexpr math constants (pi, e, …).
  3. Function allowlist — only known safe math functions (sqrt, log, …).
  4. Constant type check — string/bytes literals are rejected.
"""

from __future__ import annotations

import ast
import logging

import numexpr
from langchain_core.tools import tool
from pydantic import Field

from .base import StrictToolInput, safe_tool_call

logger = logging.getLogger(__name__)

# Structural node types that are safe.
_SAFE_AST_NODES = frozenset({
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Constant,
    ast.Name,
    ast.Call,
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BoolOp, ast.And, ast.Or,
    ast.IfExp,
    ast.Load,
    ast.Num,   # Python <3.8 compat
})

# Only these Name identifiers are allowed (numexpr mathematical constants).
_SAFE_NAMES = frozenset({
    "e", "pi", "inf", "nan", "True", "False",
})

# Only these function names are allowed (numexpr built-in math functions).
_SAFE_FUNCTIONS = frozenset({
    "abs", "sqrt", "log", "log2", "log10", "exp",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh",
    "floor", "ceil", "round",
    "min", "max", "where",
    "real", "imag", "conj",
    "sum", "prod",
})


class CalculatorInput(StrictToolInput):
    expression: str = Field(
        description=(
            "A single arithmetic expression to evaluate. "
            "Examples: '((1.25 ** 5) - 1) * 100', '(150 - 100) / 100 * 100'. "
            "No imports, no assignments, no string operations."
        )
    )


@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """Evaluate a safe arithmetic expression using numexpr.

    Validates the expression against an AST whitelist before evaluation.
    Returns the numeric result as a string, or a ToolError JSON on failure.
    """
    def _run():
        _validate_ast(expression)
        result = numexpr.evaluate(expression)
        scalar = float(result) if hasattr(result, "__float__") else result
        logger.debug("Calculator: %s = %s", expression, scalar)
        return str(round(scalar, 6))

    return safe_tool_call("calculator", _run, {"expression": expression})


def _validate_ast(expression: str) -> None:
    """Raise ValueError if the expression is not safe to evaluate.

    Three checks beyond the node-type whitelist:
    - Name nodes: only identifiers in _SAFE_NAMES (blocks __import__, etc.)
    - Call nodes: only function names in _SAFE_FUNCTIONS (blocks arbitrary calls)
    - Constant nodes: only numeric types (blocks string/bytes literals)
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    for node in ast.walk(tree):
        node_type = type(node)

        if node_type not in _SAFE_AST_NODES:
            raise ValueError(
                f"Disallowed AST node '{node_type.__name__}' in expression. "
                "Only arithmetic operations are permitted."
            )

        # Block dangerous names like __import__, __builtins__, os, sys.
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            raise ValueError(
                f"Disallowed name '{node.id}'. "
                f"Allowed identifiers: {sorted(_SAFE_NAMES)}."
            )

        # Block arbitrary function calls — only math functions permitted.
        if isinstance(node, ast.Call):
            func = node.func
            fn_name = func.id if isinstance(func, ast.Name) else None
            if fn_name is None or fn_name not in _SAFE_FUNCTIONS:
                raise ValueError(
                    f"Disallowed function call '{fn_name}'. "
                    f"Allowed functions: {sorted(_SAFE_FUNCTIONS)}."
                )

        # Block string / bytes / other non-numeric literals.
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float, complex, bool)
        ):
            raise ValueError(
                f"Disallowed constant type '{type(node.value).__name__}'. "
                "Only numeric literals are permitted."
            )
