"""FinancialFormulas tool — named financial functions with validated inputs.

Design: function registry pattern with flat optional fields on one Pydantic schema.
  - Adding a new function requires only: add name to Literal, add any new fields,
    register implementation in _IMPLEMENTATIONS.
  - Each implementation is a standalone pure-Python function (easily unit-tested).
  - Dispatcher validates required parameters at runtime and returns a structured
    JSON result with raw value, formatted value, unit label, and input echo.

Available functions:
  npv(rate, cashflows)                      → Net Present Value
  irr(cashflows)                            → Internal Rate of Return (%)
  pv(rate, nper, pmt, fv_amount)            → Present Value
  fv(rate, nper, pmt, pv_amount)            → Future Value
  cagr(start_value, end_value, years)       → Compound Annual Growth Rate (%)
  wacc(ke, kd, equity, debt, tax_rate)      → Weighted Average Cost of Capital (%)
  payback_period(initial_investment, cashflows) → Simple payback period (years)
  roi(gain, cost)                           → Return on Investment (%)

All rate/ratio parameters are decimals (e.g. 0.09 for 9%).
Functions that return percentages multiply by 100 — e.g. cagr returns 14.22 for 14.22%.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Callable, Literal

from langchain_core.tools import tool
from pydantic import Field

from .base import StrictToolInput, safe_tool_call

logger = logging.getLogger(__name__)

# ── Input schema — flat optional fields, one schema for all functions ─────────
# Adding a new function: append its name to the Literal + add any new fields here.

class FinancialFormulasInput(StrictToolInput):
    function: Literal[
        "npv", "irr", "pv", "fv", "cagr", "wacc", "payback_period", "roi"
    ] = Field(
        description=(
            "Financial function to compute:\n"
            "  npv(rate, cashflows) — Net Present Value of a cash-flow series\n"
            "  irr(cashflows) — Internal Rate of Return (%)\n"
            "  pv(rate, nper, pmt [, fv_amount]) — Present Value of an annuity/bond\n"
            "  fv(rate, nper, pmt [, pv_amount]) — Future Value of an annuity\n"
            "  cagr(start_value, end_value, years) — Compound Annual Growth Rate (%)\n"
            "  wacc(ke, kd, equity, debt, tax_rate) — Weighted Average Cost of Capital (%)\n"
            "  payback_period(initial_investment, cashflows) — Years to break even\n"
            "  roi(gain, cost) — Return on Investment (%)"
        )
    )
    # Shared rate / period parameters
    rate: float | None = Field(
        None,
        description="Discount or interest rate as a decimal (e.g. 0.09 for 9%). Used by: npv, pv, fv."
    )
    nper: int | None = Field(
        None,
        description="Number of periods (years, months, etc.). Used by: pv, fv."
    )
    pmt: float | None = Field(
        None,
        description="Payment per period (negative for cash out, positive for cash in). Used by: pv, fv."
    )
    fv_amount: float | None = Field(
        None,
        description="Future value at end of periods (default 0). Used by: pv."
    )
    pv_amount: float | None = Field(
        None,
        description="Present value / initial balance (default 0). Used by: fv."
    )
    # Cash flow series
    cashflows: list[float] | None = Field(
        None,
        description=(
            "Ordered list of cash flows. First element is typically the initial "
            "investment (negative). E.g. [-1000, 300, 400, 500]. Used by: npv, irr, payback_period."
        )
    )
    # CAGR parameters
    start_value: float | None = Field(
        None,
        description="Starting value (must be positive). Used by: cagr."
    )
    end_value: float | None = Field(
        None,
        description="Ending value. Used by: cagr."
    )
    years: float | None = Field(
        None,
        description="Number of years (must be positive). Used by: cagr."
    )
    # WACC parameters
    ke: float | None = Field(
        None,
        description="Cost of equity as a decimal (e.g. 0.12 for 12%). Used by: wacc."
    )
    kd: float | None = Field(
        None,
        description="Cost of debt as a decimal (e.g. 0.05 for 5%). Used by: wacc."
    )
    equity: float | None = Field(
        None,
        description="Market value of equity. Used by: wacc."
    )
    debt: float | None = Field(
        None,
        description="Market value of debt. Used by: wacc."
    )
    tax_rate: float | None = Field(
        None,
        description="Corporate tax rate as a decimal (e.g. 0.21 for 21%). Used by: wacc."
    )
    # Payback / ROI parameters
    initial_investment: float | None = Field(
        None,
        description="Initial investment amount (positive number). Used by: payback_period."
    )
    gain: float | None = Field(
        None,
        description="Gain or net profit. Used by: roi."
    )
    cost: float | None = Field(
        None,
        description="Cost or initial investment. Used by: roi."
    )


# ── Pure-Python function implementations ─────────────────────────────────────
# Each function validates its own required inputs and raises ValueError on failure.
# Result units: rate-functions return percentages (already × 100); value-functions
# return the same currency unit as the inputs.

def _npv(rate: float, cashflows: list[float]) -> float:
    """Net Present Value: sum of discounted cash flows."""
    if rate <= -1:
        raise ValueError("rate must be > -1 (i.e. > -100%)")
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def _irr(cashflows: list[float]) -> float:
    """Internal Rate of Return via Newton-Raphson (returns %).

    Requires at least one sign change in cashflows (otherwise IRR is undefined).
    """
    if len(cashflows) < 2:
        raise ValueError("irr requires at least 2 cash flows")

    signs = {1 if cf > 0 else (-1 if cf < 0 else 0) for cf in cashflows if cf != 0}
    if len(signs) < 2:
        raise ValueError(
            "irr requires at least one negative and one positive cash flow "
            "(otherwise IRR is undefined)"
        )

    rate = 0.1   # initial guess
    for iteration in range(1000):
        try:
            npv  = sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
            dnpv = -sum(
                t * cf / (1 + rate) ** (t + 1)
                for t, cf in enumerate(cashflows)
                if t > 0
            )
        except (ZeroDivisionError, OverflowError):
            raise ValueError("IRR computation overflow — check cashflow magnitudes")

        if abs(dnpv) < 1e-14:
            raise ValueError(
                "IRR Newton-Raphson stalled — derivative near zero. "
                "Try different initial cashflows."
            )

        new_rate = rate - npv / dnpv

        if new_rate <= -1:
            raise ValueError(
                "IRR diverged below -100% — verify cashflow signs "
                "(initial investment should be negative)."
            )

        if abs(new_rate - rate) < 1e-10:
            return new_rate * 100   # return as %

        rate = new_rate

    raise ValueError(
        "IRR did not converge after 1000 iterations — "
        "check that cashflows produce a realistic return."
    )


def _pv(
    rate: float, nper: int, pmt: float, fv_amount: float = 0.0
) -> float:
    """Present Value of an annuity (Excel PV, ordinary annuity type=0).

    Negative result = cash outflow (money you pay today).
    """
    if rate == 0:
        return -(pmt * nper + fv_amount)
    factor = (1 + rate) ** (-nper)
    return -(pmt * (1 - factor) / rate + fv_amount * factor)


def _fv(
    rate: float, nper: int, pmt: float, pv_amount: float = 0.0
) -> float:
    """Future Value of an annuity (Excel FV, ordinary annuity type=0).

    Negative result = what you owe / pay at end.
    """
    if rate == 0:
        return -(pmt * nper + pv_amount)
    factor = (1 + rate) ** nper
    return -(pmt * (factor - 1) / rate + pv_amount * factor)


def _cagr(start_value: float, end_value: float, years: float) -> float:
    """Compound Annual Growth Rate (returns %).

    Example: start=100, end=200, years=5 → 14.87%
    """
    if start_value <= 0:
        raise ValueError("cagr: start_value must be positive")
    if years <= 0:
        raise ValueError("cagr: years must be positive")
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100


def _wacc(
    ke: float, kd: float,
    equity: float, debt: float, tax_rate: float
) -> float:
    """Weighted Average Cost of Capital (returns %).

    ke, kd, tax_rate: decimal (e.g. 0.12 for 12%)
    equity, debt: market values in any consistent currency unit
    """
    total = equity + debt
    if total <= 0:
        raise ValueError("wacc: equity + debt must be positive")
    we = equity / total
    wd = debt   / total
    return (we * ke + wd * kd * (1.0 - tax_rate)) * 100


def _payback_period(
    initial_investment: float, cashflows: list[float]
) -> float | None:
    """Simple (undiscounted) payback period in years.

    Returns None if the investment is never fully recovered.
    For fractional-year precision, interpolates within the recovery period.
    """
    if initial_investment <= 0:
        raise ValueError("payback_period: initial_investment must be positive")
    cumulative = 0.0
    for i, cf in enumerate(cashflows):
        if cf <= 0:
            continue  # skip non-positive periods
        cumulative += cf
        if cumulative >= initial_investment:
            overshoot = cumulative - initial_investment
            return i + 1 - (overshoot / cf if cf != 0 else 0)
    return None   # never recovered within provided cashflows


def _roi(gain: float, cost: float) -> float:
    """Return on Investment (returns %).

    roi = (gain / cost) × 100
    """
    if cost == 0:
        raise ValueError("roi: cost cannot be zero")
    return (gain / cost) * 100


# ── Function registry: name → (implementation, required_params) ───────────────
# Extend here to add new functions — no other code changes needed.

_IMPLEMENTATIONS: dict[str, tuple[Callable, list[str]]] = {
    "npv":            (_npv,            ["rate", "cashflows"]),
    "irr":            (_irr,            ["cashflows"]),
    "pv":             (_pv,             ["rate", "nper", "pmt"]),
    "fv":             (_fv,             ["rate", "nper", "pmt"]),
    "cagr":           (_cagr,           ["start_value", "end_value", "years"]),
    "wacc":           (_wacc,           ["ke", "kd", "equity", "debt", "tax_rate"]),
    "payback_period": (_payback_period, ["initial_investment", "cashflows"]),
    "roi":            (_roi,            ["gain", "cost"]),
}

# Result unit labels and formatting per function
_RESULT_META: dict[str, dict[str, str]] = {
    "npv":            {"unit": "currency",  "format": "currency"},
    "irr":            {"unit": "percent",   "format": "percent"},
    "pv":             {"unit": "currency",  "format": "currency"},
    "fv":             {"unit": "currency",  "format": "currency"},
    "cagr":           {"unit": "percent",   "format": "percent"},
    "wacc":           {"unit": "percent",   "format": "percent"},
    "payback_period": {"unit": "years",     "format": "decimal"},
    "roi":            {"unit": "percent",   "format": "percent"},
}


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool("financial_formulas", args_schema=FinancialFormulasInput)
def financial_formulas_tool(
    function: str,
    rate: float | None = None,
    nper: int | None = None,
    pmt: float | None = None,
    fv_amount: float | None = None,
    pv_amount: float | None = None,
    cashflows: list[float] | None = None,
    start_value: float | None = None,
    end_value: float | None = None,
    years: float | None = None,
    ke: float | None = None,
    kd: float | None = None,
    equity: float | None = None,
    debt: float | None = None,
    tax_rate: float | None = None,
    initial_investment: float | None = None,
    gain: float | None = None,
    cost: float | None = None,
) -> str:
    """Compute standard financial functions: NPV, IRR, PV, FV, CAGR, WACC, payback, ROI.

    All rate parameters are decimals (0.09 = 9%). Functions returning percentages
    already multiply by 100 (cagr returns 14.22 for 14.22%, not 0.1422).

    Returns JSON with result_raw, result_formatted, unit, inputs_used, and optional warning.
    """
    def _run() -> str:
        if function not in _IMPLEMENTATIONS:
            raise ValueError(f"Unknown function '{function}'. Available: {list(_IMPLEMENTATIONS)}")

        impl, required = _IMPLEMENTATIONS[function]
        meta           = _RESULT_META[function]

        # Build kwargs from available params and validate required ones are present
        all_params: dict[str, Any] = {
            "rate":               rate,
            "nper":               nper,
            "pmt":                pmt,
            "fv_amount":          fv_amount,
            "pv_amount":          pv_amount,
            "cashflows":          cashflows,
            "start_value":        start_value,
            "end_value":          end_value,
            "years":              years,
            "ke":                 ke,
            "kd":                 kd,
            "equity":             equity,
            "debt":               debt,
            "tax_rate":           tax_rate,
            "initial_investment": initial_investment,
            "gain":               gain,
            "cost":               cost,
        }

        missing = [p for p in required if all_params.get(p) is None]
        if missing:
            raise ValueError(
                f"{function}() requires: {required}. "
                f"Missing: {missing}."
            )

        # Call implementation with only the required (+ optional) params it accepts
        import inspect
        sig = inspect.signature(impl)
        kwargs = {
            k: all_params[k]
            for k in sig.parameters
            if k in all_params and all_params[k] is not None
        }
        raw = impl(**kwargs)

        # Handle payback_period returning None (never recovered)
        if raw is None:
            return json.dumps({
                "function":     function,
                "result_raw":   None,
                "result_formatted": "Not recovered",
                "unit":         "years",
                "note":         "Investment not recovered within the provided cashflow period.",
                "inputs_used":  {k: v for k, v in all_params.items() if v is not None},
            })

        # Validate result
        raw_float = float(raw)
        if math.isinf(raw_float):
            raise ValueError(f"{function}() returned infinity — check inputs")
        if math.isnan(raw_float):
            raise ValueError(f"{function}() returned NaN — check inputs")

        # Domain-specific range warnings
        warning = _domain_warning(function, raw_float)

        # Format result
        formatted = _format_financial(raw_float, meta["format"])

        return json.dumps({
            "function":         function,
            "result_raw":       raw_float,
            "result_formatted": formatted,
            "unit":             meta["unit"],
            **({"warning": warning} if warning else {}),
            "inputs_used":      {k: v for k, v in all_params.items() if v is not None},
        })

    return safe_tool_call("financial_formulas", _run, {"function": function})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _domain_warning(function: str, result: float) -> str | None:
    """Return a warning string for results that are technically valid but unusual."""
    if function == "irr":
        if result > 500:
            return f"IRR of {result:.1f}% is unusually high — verify cashflow signs and magnitudes."
        if result < -99:
            return f"IRR of {result:.1f}% indicates near-total loss — verify initial investment sign."
    if function == "cagr":
        if abs(result) > 100:
            return f"CAGR of {result:.1f}% is very high — verify start_value, end_value, and years."
    if function in ("pv", "fv", "npv"):
        if abs(result) > 1e15:
            return f"Result magnitude {result:.2e} is very large — verify input units."
    return None


def _format_financial(value: float, fmt: str) -> str:
    """Format a financial result for display."""
    if fmt == "percent":
        return f"{value:.2f}%"
    if fmt == "currency":
        abs_v = abs(value)
        if abs_v >= 1e12:
            return f"${value / 1e12:.3f}T"
        if abs_v >= 1e9:
            return f"${value / 1e9:.3f}B"
        if abs_v >= 1e6:
            return f"${value / 1e6:.3f}M"
        return f"${value:,.2f}"
    if fmt == "decimal":
        return f"{value:.2f}"
    return str(round(value, 6))
