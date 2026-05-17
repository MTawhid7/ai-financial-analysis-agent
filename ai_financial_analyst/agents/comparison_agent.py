"""Comparison agent — side-by-side analysis of two or more stocks.

Runs the full pipeline for all requested tickers in a single pass
(the pipeline already handles multi-ticker runs), then generates a
focused comparison table from the analysis data using Flash primary LLM.

Key design decisions:
- Structured field extraction (not raw JSON dump) avoids data truncation
  that would silently cut off the last ticker in large comparisons.
- Output is validated for ticker presence; a fallback table is appended
  if the LLM omits any ticker from the generated table.
- User-specified dimensions (dividend, risk, growth, etc.) are detected
  from the user's message and added to both the payload and the prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm import CircuitBreakerError, content_to_str
from ..core.state import PartialStateError

logger = logging.getLogger(__name__)

_COMPARISON_SYSTEM = (
    "You are a senior financial analyst. Produce clear, concise, data-driven comparisons. "
    "Use ONLY numbers that appear in the provided analysis JSON. "
    "Format numbers professionally: $4.2T, 27.2%, 34.4x P/E."
)

_COMPARISON_PROMPT = """\
Generate a side-by-side comparison of the following stocks based on this analysis data.

REQUIRED SECTIONS:
1. A Markdown table with these columns:
   Metric | {tickers_header}
   Rows: Current Price, Market Cap, Revenue (TTM), Net Income, Profit Margin,
         P/E Ratio, Sector P/E Avg, P/E Premium %, 5Y Price CAGR, Closest Peer
{extra_rows_instruction}
2. ## Key Differentiators  (3–4 bullet points: what sets each stock apart)

3. ## Verdict  (1–2 sentences: which stock looks more attractive and why, citing specific metrics)

Analysis data (JSON):
{comparison_json}

Generate the comparison report now. Use only figures from the data above.
"""

# ── Field manifests ───────────────────────────────────────────────────────────

# Quant-analyst output fields that map directly to comparison table rows.
_ANALYSIS_FIELDS = (
    "price_cagr_5y_pct",
    "total_return_cagr_pct",
    "company_pe",
    "sector_pe_avg",
    "pe_premium_pct",
    "dcf_intrinsic_value",
    "margin_of_safety_pct",
    "bull_case",
    "bear_case",
)

# raw_data["fundamentals"] fields needed for the standard comparison table rows.
_FUNDAMENTALS_FIELDS = (
    "current_price",
    "market_cap",
    "revenue_ttm",
    "net_income_ttm",
    "profit_margin",
    "pe_ratio",
    "forward_pe",
    "ev_to_ebitda",
    "price_to_book",
    "price_to_sales",
    "sector",
    "analyst_target_mean",
)

# Extra field sets keyed by dimension keyword found in the user's message.
_DIMENSION_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "dividend":  ("cash_flow",        ("dividend_yield_pct", "annual_dividend_per_share", "payout_ratio")),
    "cash flow": ("cash_flow",        ("free_cash_flow", "ocf", "fcf_yield_pct")),
    "debt":      ("balance_sheet",    ("net_debt", "debt_to_equity", "current_ratio")),
    "risk":      ("price_metrics",    ("sharpe_ratio", "max_drawdown_pct", "beta", "annualized_volatility_pct")),
    "growth":    ("financials_trend", ("revenue_qoq_growth_pct", "revenue_yoy_growth_pct")),
    "valuation": ("fundamentals",     ("ev_to_ebitda", "price_to_book", "price_to_sales")),
    "earnings":  ("earnings",         ("next_earnings_date", "eps_estimate", "avg_surprise_pct")),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_raw_dict(raw_data: dict, ticker: str, data_type: str) -> dict:
    """Safely retrieve any data-type dict for a ticker; handles JSON-string values."""
    val = (raw_data.get(ticker) or {}).get(data_type) or {}
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            val = {}
    return val


def _build_comparison_payload(
    analysis: dict,
    raw_data: dict,
    tickers: list[str],
    extra_dims: list[str],
) -> dict:
    """Build a compact per-ticker comparison dict — no truncation, no extraneous fields.

    Extracts only the fields that map to comparison table rows plus any
    user-specified extra dimensions. None values are omitted.
    """
    payload: dict = {}
    for ticker in tickers:
        t_analysis = analysis.get(ticker, {})
        fundamentals = _get_raw_dict(raw_data, ticker, "fundamentals")

        ticker_entry: dict = {
            "analysis": {
                k: t_analysis[k] for k in _ANALYSIS_FIELDS
                if t_analysis.get(k) is not None
            },
            "fundamentals": {
                k: fundamentals[k] for k in _FUNDAMENTALS_FIELDS
                if fundamentals.get(k) is not None
            },
        }

        for dim in extra_dims:
            data_type, fields = _DIMENSION_FIELDS[dim]
            source = _get_raw_dict(raw_data, ticker, data_type)
            dim_data = {k: source[k] for k in fields if source.get(k) is not None}
            if dim_data:
                ticker_entry[dim] = dim_data

        payload[ticker] = ticker_entry
    return payload


def _extract_user_dimensions(message: str) -> list[str]:
    """Return dimension keys from _DIMENSION_FIELDS found in the user's message."""
    msg_lower = message.lower()
    return [dim for dim in _DIMENSION_FIELDS if dim in msg_lower]


def _extra_rows_text(extra_dims: list[str]) -> str:
    """Build the optional extra-rows instruction line for the prompt."""
    if not extra_dims:
        return ""
    labels = ", ".join(d.title() for d in extra_dims)
    return f"   Also include user-requested dimensions: {labels}\n"


def _validate_comparison_table(md: str, tickers: list[str]) -> list[str]:
    """Return list of tickers absent from the generated comparison markdown."""
    return [t for t in tickers if t.upper() not in md.upper()]


def _build_fallback_table(payload: dict, tickers: list[str]) -> str:
    """Build a minimal structured comparison table from raw payload data.

    Used when the LLM omits one or more tickers from the generated table.
    """
    header = "| Metric | " + " | ".join(tickers) + " |"
    sep    = "|" + "|".join("---" for _ in range(len(tickers) + 1)) + "|"
    rows   = [header, sep]

    for label, (section, field) in [
        ("Current Price", ("fundamentals", "current_price")),
        ("Market Cap",    ("fundamentals", "market_cap")),
        ("Revenue (TTM)", ("fundamentals", "revenue_ttm")),
        ("Profit Margin", ("fundamentals", "profit_margin")),
        ("P/E Ratio",     ("fundamentals", "pe_ratio")),
        ("5Y Price CAGR", ("analysis",     "price_cagr_5y_pct")),
    ]:
        values = [str(payload.get(t, {}).get(section, {}).get(field, "N/A")) for t in tickers]
        rows.append(f"| {label} | " + " | ".join(values) + " |")

    return "\n".join(rows)


# ── Public API ────────────────────────────────────────────────────────────────

async def run_comparison(
    message: str,
    tickers: list[str],
    primary_llm: Any,
    step_callback: Callable[[dict], None] | None = None,
) -> tuple[str, Any]:
    """Run multi-ticker pipeline and produce a comparison report.

    Returns:
        (comparison_markdown, final_state)
    """
    if len(tickers) < 2:
        return (
            f"Comparison requires at least two tickers. I only found one: **{tickers[0] if tickers else '?'}**. "
            "Please specify a second stock — e.g. *\"Compare AAPL vs MSFT\"*.",
            None,
        )

    from .orchestrator import run_pipeline

    ticker_str = ", ".join(tickers)
    logger.info("Running comparison for %s", ticker_str)

    try:
        final_state, _trace, _artifacts = await run_pipeline(
            query=f"Compare {ticker_str}",
            tickers=tickers,
            step_callback=step_callback,
        )
    except CircuitBreakerError:
        logger.warning("Comparison aborted — primary model rate-limited for %s", ticker_str)
        return (
            f"**Rate limit reached** — the Gemini API is recovering from heavy use. "
            f"Please try the **{ticker_str}** comparison again in about a minute.",
            None,
        )
    except PartialStateError as exc:
        logger.warning("Comparison incomplete for %s: %s", ticker_str, exc)
        return (
            f"**Analysis incomplete** for **{ticker_str}** — some data could not be "
            f"retrieved (likely a yfinance or network issue). Partial results may be "
            f"available if you retry.",
            None,
        )
    except Exception as exc:
        logger.exception("Comparison pipeline failed for %s", ticker_str)
        return (
            f"**Analysis failed** for **{ticker_str}** due to an unexpected error. "
            f"Please try again or check the application logs for details.",
            None,
        )

    analysis = final_state.get("analysis", {})
    raw_data  = final_state.get("raw_data", {})

    # Detect user-specified comparison dimensions from the original message.
    extra_dims = _extract_user_dimensions(message)

    # Build structured payload — no truncation, only comparison-relevant fields.
    comparison_payload = _build_comparison_payload(analysis, raw_data, tickers, extra_dims)

    tickers_header = " | ".join(tickers)
    prompt = _COMPARISON_PROMPT.format(
        tickers_header=tickers_header,
        extra_rows_instruction=_extra_rows_text(extra_dims),
        comparison_json=json.dumps(comparison_payload, indent=2, default=str),
    )

    try:
        response = await primary_llm.ainvoke([
            SystemMessage(content=_COMPARISON_SYSTEM),
            HumanMessage(content=prompt),
        ])
        comparison_md = content_to_str(
            response.content if hasattr(response, "content") else response
        ).strip()
    except Exception as exc:
        logger.error("Comparison LLM call failed: %s", exc)
        comparison_md = (
            f"The analysis completed for **{ticker_str}** but I couldn't generate the "
            f"comparison table: `{exc}`"
        )

    # Validate that all tickers appear in the output; append fallback if any are missing.
    missing = _validate_comparison_table(comparison_md, tickers)
    if missing:
        logger.warning("Comparison table missing tickers %s — appending fallback", missing)
        fallback = _build_fallback_table(comparison_payload, tickers)
        comparison_md += (
            f"\n\n---\n*Note: data for **{', '.join(missing)}** was missing from the "
            f"table above. Raw data summary:*\n\n{fallback}"
        )

    header = f"Here is a side-by-side comparison of **{ticker_str}**:\n\n"
    return header + comparison_md, final_state
