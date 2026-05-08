"""Comparison agent — side-by-side analysis of two or more stocks.

Runs the full pipeline for all requested tickers in a single pass
(the pipeline already handles multi-ticker runs), then generates a
focused comparison table from the analysis data using Flash primary LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm import content_to_str

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

2. ## Key Differentiators  (3–4 bullet points: what sets each stock apart)

3. ## Verdict  (1–2 sentences: which stock looks more attractive and why, citing specific metrics)

Analysis data (JSON):
{analysis_json}

Raw fundamentals summary:
{fundamentals_json}

Generate the comparison report now. Use only figures from the data above.
"""


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
    except Exception as exc:
        logger.exception("Comparison pipeline failed for %s", ticker_str)
        return (
            f"I ran into an error while analysing **{ticker_str}**: `{exc}`\n\n"
            "Please try again.",
            None,
        )

    analysis = final_state.get("analysis", {})
    raw_data = final_state.get("raw_data", {})

    # Build a compact fundamentals summary (market cap, PE, etc.)
    import json
    fundamentals_summary: dict = {}
    for ticker in tickers:
        fund = (raw_data.get(ticker) or {}).get("fundamentals") or {}
        if isinstance(fund, str):
            try:
                fund = json.loads(fund)
            except Exception:
                fund = {}
        fundamentals_summary[ticker] = {
            k: v for k, v in fund.items()
            if k in ("current_price", "market_cap", "revenue_ttm", "net_income_ttm",
                     "profit_margin", "pe_ratio", "forward_pe", "sector")
        }

    tickers_header = " | ".join(tickers)

    try:
        chain_input = [
            SystemMessage(content=_COMPARISON_SYSTEM),
            HumanMessage(content=_COMPARISON_PROMPT.format(
                tickers_header=tickers_header,
                analysis_json=json.dumps(analysis, indent=2, default=str)[:4000],
                fundamentals_json=json.dumps(fundamentals_summary, indent=2, default=str)[:2000],
            )),
        ]
        response = await primary_llm.ainvoke(chain_input)
        comparison_md = content_to_str(
            response.content if hasattr(response, "content") else response
        ).strip()
    except Exception as exc:
        logger.error("Comparison LLM call failed: %s", exc)
        comparison_md = (
            f"The analysis completed for **{ticker_str}** but I couldn't generate the "
            f"comparison table: `{exc}`"
        )

    header = f"Here is a side-by-side comparison of **{ticker_str}**:\n\n"
    return header + comparison_md, final_state
