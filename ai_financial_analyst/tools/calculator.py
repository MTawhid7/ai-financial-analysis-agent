"""CalculatorTool — constrained numexpr evaluator with AST whitelist validation.

The LLM-generated expression is treated as untrusted input.
Validated at three levels:
  1. Node-type whitelist — rejects any non-arithmetic AST construct.
  2. Name allowlist — only known math constants + user-supplied context keys.
  3. Function allowlist — only known safe math functions (sqrt, log, …).
  4. Constant type check — string/bytes literals are rejected.

Improvements over original:
  - context dict: named variables the expression can reference (unit-context pattern)
  - format hint: per-type result formatting (percent, currency, ratio, integer)
  - result validation: inf/nan detection + large-magnitude warnings returned as metadata
  - structured JSON return when format is requested; raw string otherwise (backward compat)
"""

from __future__ import annotations

import ast
import json
import logging
import math
from typing import Literal

import numexpr
from langchain_core.tools import tool
from pydantic import Field

from .base import StrictToolInput, safe_tool_call

logger = logging.getLogger(__name__)

# ── AST whitelist ─────────────────────────────────────────────────────────────

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

# Intrinsic numexpr mathematical constants.
_SAFE_NAMES = frozenset({
    "e", "pi", "inf", "nan", "True", "False",
})

# Safe numexpr built-in math functions.
_SAFE_FUNCTIONS = frozenset({
    "abs", "sqrt", "log", "log2", "log10", "exp",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh",
    "floor", "ceil", "round",
    "min", "max", "where",
    "real", "imag", "conj",
    "sum", "prod",
})


# ── Input schema ─────────────────────────────────────────────────────────────

class CalculatorInput(StrictToolInput):
    expression: str = Field(
        description=(
            "A single arithmetic expression to evaluate. "
            "May reference names from 'context' (e.g. 'market_cap_usd / revenue_usd'). "
            "Examples: '((1.25 ** 5) - 1) * 100', 'sqrt(variance) * 252 ** 0.5'. "
            "No imports, no assignments, no string operations."
        )
    )
    context: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional named variables the expression can reference. "
            "Use descriptive names with unit suffixes to document intent: "
            "{'market_cap_usd': 3e12, 'revenue_usd': 4e11} → expression 'market_cap_usd / revenue_usd'. "
            "All values must be floats. Keys must be valid identifiers not starting with '_'."
        ),
    )
    format: Literal["auto", "percent", "currency", "ratio", "integer"] | None = Field(
        default=None,
        description=(
            "Optional formatting hint for the result. "
            "'percent' → '14.22%'; "
            "'currency' → '$4.17T'; "
            "'ratio' → '28.5×'; "
            "'integer' → '15,726,000,000'; "
            "'auto' → smart detection by magnitude. "
            "When omitted the raw number is returned (backward-compatible)."
        ),
    )


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(
    expression: str,
    context: dict[str, float] | None = None,
    format: str | None = None,
) -> str:
    """Evaluate a safe arithmetic expression using numexpr.

    Validates against an AST whitelist before evaluation.
    Named variables can be supplied via 'context' (unit-context pattern).
    Result is validated for inf/nan; optional formatting applied on request.
    Returns a raw number string (no format) or JSON (with format) or ToolError JSON on failure.
    """
    def _run():
        ctx = context or {}
        _validate_context_keys(ctx)
        _validate_ast(expression, context_keys=set(ctx.keys()))

        result = numexpr.evaluate(expression, local_dict=ctx)
        scalar  = float(result) if hasattr(result, "__float__") else float(result)

        scalar, warning = _validate_result(scalar)
        if scalar is None:
            raise ValueError(warning)

        logger.debug("Calculator: %s = %s  (context=%s)", expression, scalar, list(ctx.keys()))

        if format:
            formatted, applied = _format_result(scalar, format)
            payload: dict = {
                "result_raw":       scalar,
                "result_formatted": formatted,
                "format_applied":   applied,
            }
            if warning:
                payload["warning"] = warning
            return json.dumps(payload)

        # Backward-compatible raw string return
        if warning:
            logger.warning("Calculator warning: %s — %s", scalar, warning)
        return str(round(scalar, 6))

    return safe_tool_call("calculator", _run, {"expression": expression})


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_ast(expression: str, context_keys: set[str] | None = None) -> None:
    """Raise ValueError if the expression is not safe to evaluate.

    Checks:
    - Node-type whitelist — structural node types must be arithmetic only.
    - Name nodes — allowed identifiers: math constants + context keys.
    - Call nodes — only whitelisted math functions.
    - Constant nodes — only numeric literals.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    allowed_names = _SAFE_NAMES | (context_keys or set())

    for node in ast.walk(tree):
        node_type = type(node)

        if node_type not in _SAFE_AST_NODES:
            raise ValueError(
                f"Disallowed AST node '{node_type.__name__}' in expression. "
                "Only arithmetic operations are permitted."
            )

        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(
                f"Disallowed name '{node.id}'. "
                f"Allowed: math constants {sorted(_SAFE_NAMES)} + context keys {sorted(context_keys or [])}."
            )

        if isinstance(node, ast.Call):
            func    = node.func
            fn_name = func.id if isinstance(func, ast.Name) else None
            if fn_name is None or fn_name not in _SAFE_FUNCTIONS:
                raise ValueError(
                    f"Disallowed function '{fn_name}'. "
                    f"Allowed functions: {sorted(_SAFE_FUNCTIONS)}."
                )

        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float, complex, bool)
        ):
            raise ValueError(
                f"Disallowed constant type '{type(node.value).__name__}'. "
                "Only numeric literals are permitted."
            )


def _validate_context_keys(context: dict[str, float]) -> None:
    """Ensure context keys are safe Python identifiers not starting with '_'."""
    for key in context:
        if not isinstance(key, str) or not key.isidentifier() or key.startswith("_"):
            raise ValueError(
                f"Invalid context key '{key}': must be a valid Python identifier "
                "and must not start with '_'."
            )


def _validate_result(value: float) -> tuple[float | None, str | None]:
    """Return (value, warning) — or (None, error_message) for unusable results.

    Hard errors (return None):  inf, -inf, nan.
    Soft warnings (return value with message): magnitude > 1e15 or near zero.
    """
    if math.isinf(value):
        sign = "positive" if value > 0 else "negative"
        return None, f"Result is {sign} infinity — check for division by zero or overflow."
    if math.isnan(value):
        return None, "Result is undefined (NaN) — check for 0÷0 or invalid operation."

    abs_v   = abs(value)
    warning = None
    if abs_v > 1e15:
        warning = f"Result magnitude {abs_v:.2e} is very large — verify units are consistent."
    elif 0 < abs_v < 1e-10:
        warning = f"Result {value:.2e} is near zero — verify there is no denominator error."

    return value, warning


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_result(value: float, hint: str) -> tuple[str, str]:
    """Return (formatted_string, format_applied_label).

    'auto'     → infer from magnitude
    'percent'  → value is already in % scale (e.g. 14.22 → "14.22%")
    'currency' → prefix $ and apply B/T/M suffix
    'ratio'    → 1dp with × suffix (P/E, EV/EBITDA)
    'integer'  → round to 0dp with thousands separator
    """
    abs_v = abs(value)

    if hint == "percent":
        return f"{value:.2f}%", "percent"

    if hint == "currency":
        if abs_v >= 1e12:
            return f"${value / 1e12:.3f}T", "currency_trillions"
        if abs_v >= 1e9:
            return f"${value / 1e9:.3f}B", "currency_billions"
        if abs_v >= 1e6:
            return f"${value / 1e6:.3f}M", "currency_millions"
        return f"${value:,.2f}", "currency"

    if hint == "ratio":
        return f"{value:.1f}×", "ratio"

    if hint == "integer":
        return f"{int(round(value)):,}", "integer"

    # "auto" — infer from magnitude
    if abs_v >= 1e12:
        return f"{value / 1e12:.3f}T", "auto_trillions"
    if abs_v >= 1e9:
        return f"{value / 1e9:.3f}B", "auto_billions"
    if abs_v >= 1e6:
        return f"{value / 1e6:.3f}M", "auto_millions"
    if abs_v > 0 and abs_v < 0.01:
        return f"{value:.6f}", "auto_small"

    return str(round(value, 4)), "auto"
