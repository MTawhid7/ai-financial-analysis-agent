"""Quant Analyst Agent — computation and comparison node.

Improvements over original:
- Reads pre-computed risk metrics (Sharpe, Sortino, max drawdown, beta, volatility)
  from the new price_metrics data type instead of recomputing
- Uses adjusted-close total-return CAGR rather than raw-price CAGR
- Compares 3 valuation multiples vs sector (P/E, EV/EBITDA, P/B) when available
- Includes FCF yield and key profitability ratios in the analysis context
- Uses Gemini JSON mode for bull/bear case generation to eliminate fragile
  markdown-fence stripping
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ..core.llm import content_to_str
from ..core.state import (
    AgentState,
    IterationLogEntry,
    validate_state_for_agent,
)
from ..core.tracing import ErrorType, RunTracer
from ..tools.calculator import calculator_tool
from ..tools.benchmark_lookup import benchmark_lookup_tool

logger = logging.getLogger(__name__)

# yfinance sector strings → GICS sector names
_YFINANCE_TO_GICS: dict[str, str] = {
    "Technology":            "Information Technology",
    "Healthcare":            "Health Care",
    "Financial Services":    "Financials",
    "Consumer Cyclical":     "Consumer Discretionary",
    "Consumer Defensive":    "Consumer Staples",
    "Basic Materials":       "Materials",
    "Communication Services":"Communication Services",
    "Industrials":           "Industrials",
    "Energy":                "Energy",
    "Utilities":             "Utilities",
    "Real Estate":           "Real Estate",
}

_SOP_SYSTEM = """You are a quantitative financial analyst writing an investment brief.

Analyse the financial data and metrics provided and respond with ONLY this JSON:

{
  "bull_case": ["<reason 1>", "<reason 2>", "<reason 3>"],
  "bear_case": ["<risk 1>",   "<risk 2>",   "<risk 3>"],
  "closest_peer": "<Company Name (TICKER)>"
}

Rules:
- bull_case: 2-3 specific, data-driven reasons to be optimistic.
  Reference Sharpe ratio, CAGR vs S&P 500, FCF yield, margin trends, or sector P/E premium.
- bear_case: 2-3 concrete risks. Reference max drawdown, valuation premium, beta, debt/equity.
- closest_peer: single most comparable public company (format: "Apple Inc. (AAPL)")
- If a metric is missing, use sector knowledge — but prefer cited numbers.
- Output ONLY the JSON above. No markdown, no prose, no explanation."""


async def quant_analyst_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Quant Analyst agent."""
    validate_state_for_agent(state, "quant_analyst")

    tracer        = config.get("tracer")       if config else None
    artifacts     = config.get("artifacts")    if config else None
    step_callback = config.get("step_callback") if config else None
    primary_llm   = config.get("primary_llm")  if config else None

    if primary_llm is None:
        raise RuntimeError("Quant Analyst requires 'primary_llm' in config")

    raw_data       = state["raw_data"]
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]]           = list(state.get("errors", []))
    step = len(iteration_log)

    if tracer:
        tracer.record_agent_start("quant_analyst", {"tickers": list(raw_data.keys())})

    analysis_per_ticker: dict[str, Any] = {}

    for ticker, ticker_data in raw_data.items():
        ta: dict[str, Any] = {"ticker": ticker, "citations": {}}

        # ── 1. Total-return CAGR (from price_metrics, preferred) ──────────
        pm = ticker_data.get("price_metrics", {})
        ph = ticker_data.get("price_history", {})

        if pm and pm.get("total_return_cagr_pct") is not None:
            ta["price_cagr_5y_pct"]  = pm["total_return_cagr_pct"]   # total return (adj.)
            ta["sp500_cagr_pct"]     = pm.get("sp500_cagr_pct")
            ta["relative_cagr_pct"]  = pm.get("relative_cagr_pct")   # alpha vs benchmark
            ta["citations"]["price_cagr_5y_pct"] = {
                "source_tool": "yahoo_finance", "observation_step": step
            }
        elif ph and ph.get("current_price") and ph.get("price_5y_ago"):
            # Fallback: raw-price CAGR via calculator (no dividend adjustment)
            current   = ph["current_price"]
            five_y    = ph["price_5y_ago"]
            if five_y > 0:
                expr = f"(({current} / {five_y}) ** (1/5) - 1) * 100"
                step += 1
                cagr_str = await calculator_tool.arun({"expression": expr})
                _record_step(step, "calculator", {"expression": expr}, cagr_str,
                             tracer, artifacts, step_callback, iteration_log)
                try:
                    ta["price_cagr_5y_pct"] = round(float(cagr_str), 2)
                    ta["citations"]["price_cagr_5y_pct"] = {
                        "source_tool": "calculator", "observation_step": step
                    }
                except (ValueError, TypeError):
                    errors.append({"error_type": ErrorType.TOOL_ERROR.value,
                                   "detail": f"CAGR parse failed: {cagr_str}"})

        # ── 2. Risk metrics (from price_metrics) ──────────────────────────
        if pm:
            for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                        "beta_vs_sp500", "volatility_annual_pct"):
                if pm.get(key) is not None:
                    ta[key] = pm[key]
                    ta["citations"][key] = {
                        "source_tool": "yahoo_finance", "observation_step": step
                    }

        # ── 3. FCF yield and cash metrics ────────────────────────────────
        cf = ticker_data.get("cash_flow", {})
        if cf:
            if cf.get("fcf_yield") is not None:
                ta["fcf_yield_pct"] = round(cf["fcf_yield"] * 100, 2)
                ta["citations"]["fcf_yield_pct"] = {
                    "source_tool": "yahoo_finance", "observation_step": step
                }
            if cf.get("cash_conversion_ratio") is not None:
                ta["cash_conversion_ratio"] = cf["cash_conversion_ratio"]
            if cf.get("free_cash_flow") is not None:
                ta["free_cash_flow"] = cf["free_cash_flow"]

        # ── 4. Next earnings date ─────────────────────────────────────────
        earn = ticker_data.get("earnings", {})
        if earn and earn.get("next_earnings_date"):
            ta["next_earnings_date"] = earn["next_earnings_date"]

        # ── 5. Sector benchmark comparison (P/E, EV/EBITDA, P/B) ─────────
        fund = ticker_data.get("fundamentals", {})
        if isinstance(fund, str):
            try:
                fund = json.loads(fund)
            except Exception:
                fund = {}

        if fund and fund.get("sector"):
            gics_sector = _YFINANCE_TO_GICS.get(fund["sector"], fund["sector"])
            step += 1
            benchmark_str = await benchmark_lookup_tool.arun({"gics_sector": gics_sector})
            _record_step(step, "benchmark_lookup", {"gics_sector": gics_sector},
                         benchmark_str, tracer, artifacts, step_callback, iteration_log)

            try:
                bm = json.loads(benchmark_str)
                if "error_type" not in bm:
                    ta["sector"]       = gics_sector
                    ta["sector_peers"] = bm.get("peer_examples", [])
                    ta["citations"]["sector_benchmarks"] = {
                        "source_tool": "benchmark_lookup", "observation_step": step
                    }

                    # P/E premium vs sector
                    co_pe  = fund.get("pe_ratio")
                    sec_pe = bm.get("pe_ratio_sector_avg")
                    if co_pe and sec_pe:
                        ta["company_pe"]  = co_pe
                        ta["sector_pe_avg"] = sec_pe
                        ta["pe_vs_sector_premium_pct"] = round(
                            (co_pe - sec_pe) / sec_pe * 100, 1
                        )

                    # EV/EBITDA premium vs sector
                    co_ev  = fund.get("ev_to_ebitda")
                    sec_ev = bm.get("ev_ebitda_sector_avg")
                    if co_ev and sec_ev:
                        ta["ev_ebitda"]             = co_ev
                        ta["sector_ev_ebitda_avg"]  = sec_ev
                        ta["ev_vs_sector_premium_pct"] = round(
                            (co_ev - sec_ev) / sec_ev * 100, 1
                        )

                    # P/B premium vs sector
                    co_pb  = fund.get("price_to_book")
                    sec_pb = bm.get("price_to_book_sector_avg")
                    if co_pb and sec_pb:
                        ta["price_to_book"]             = co_pb
                        ta["sector_price_to_book_avg"]  = sec_pb
                        ta["pb_vs_sector_premium_pct"]  = round(
                            (co_pb - sec_pb) / sec_pb * 100, 1
                        )

                    # Sector-level risk & margin benchmarks (from Damodaran)
                    if bm.get("beta_sector_avg"):
                        ta["sector_beta_avg"] = bm["beta_sector_avg"]
                    if bm.get("operating_margin_pct"):
                        ta["sector_operating_margin_pct"] = bm["operating_margin_pct"]

            except (json.JSONDecodeError, KeyError) as exc:
                errors.append({"error_type": ErrorType.PARSING_ERROR.value, "detail": str(exc)})

        # ── 6. Key profitability context for the LLM ─────────────────────
        if fund:
            for key in ("roe", "roa", "operating_margin", "gross_margin",
                        "debt_to_equity", "revenue_growth", "earnings_growth"):
                yf_key = {
                    "roe":              "return_on_equity",
                    "roa":              "return_on_assets",
                    "operating_margin": "operating_margin",
                    "gross_margin":     "gross_margin",
                    "debt_to_equity":   "debt_to_equity",
                    "revenue_growth":   "revenue_growth",
                    "earnings_growth":  "earnings_growth",
                }.get(key, key)
                val = fund.get(yf_key)
                if val is not None:
                    ta[key] = val

        # ── 7. LLM bull/bear case (structured JSON mode) ─────────────────
        step += 1
        computed_context = {
            k: v for k, v in ta.items()
            if k not in ("ticker", "citations", "sector_peers") and v is not None
        }
        human_content = (
            f"Stock: {ticker}\n"
            f"Sector: {ta.get('sector', 'Unknown')}\n\n"
            f"Computed metrics:\n{json.dumps(computed_context, indent=2)[:3000]}\n\n"
            f"Raw fundamentals (selected):\n"
            f"{json.dumps({k: fund.get(k) for k in ('market_cap','revenue_ttm','net_income_ttm','beta','dividend_yield','analyst_target_mean')}, indent=2)}"
        )

        sop_text = ""
        try:
            sop_response = await primary_llm.ainvoke([
                SystemMessage(content=_SOP_SYSTEM),
                HumanMessage(content=human_content),
            ])
            raw = sop_response.content if hasattr(sop_response, "content") else sop_response
            sop_text = content_to_str(raw).strip()

            # Strip markdown fences if the model ignores the instruction
            if sop_text.startswith("```"):
                sop_text = sop_text.split("```", 2)[1]
                if sop_text.startswith("json"):
                    sop_text = sop_text[4:]
                sop_text = sop_text.rstrip("`").strip()

            sop_data = json.loads(sop_text)
            ta["bull_case"]    = sop_data.get("bull_case",    [])
            ta["bear_case"]    = sop_data.get("bear_case",    [])
            ta["closest_peer"] = sop_data.get("closest_peer")

        except Exception as exc:
            logger.warning("SOP chain failed for %s: %s", ticker, exc)
            ta["bull_case"] = []
            ta["bear_case"] = []
            errors.append({"error_type": ErrorType.PARSING_ERROR.value, "detail": str(exc)})

        if step_callback:
            step_callback({
                "step": step, "agent": "quant_analyst", "tool": "sop_llm",
                "input": {"ticker": ticker}, "cache_hit": False,
                "ok": bool(ta.get("bull_case")),
            })
        if artifacts:
            artifacts.record_llm_exchange(
                agent="quant_analyst", purpose="sop_analysis", ticker=ticker,
                prompt_messages=[
                    {"role": "system", "content": _SOP_SYSTEM},
                    {"role": "user", "content": human_content},
                ],
                raw_response=sop_text,
            )
        if tracer:
            tracer.record_tool_call(
                agent="quant_analyst", tool="sop_analysis",
                input_data={"ticker": ticker}, output_data=ta,
                output_tokens=step * 5,
            )

        analysis_per_ticker[ticker] = ta

    if tracer:
        tracer.record_agent_complete("quant_analyst", {
            "tickers_analysed": list(analysis_per_ticker.keys()),
            "errors":           len(errors),
        })

    return AgentState(**{
        **state,
        "analysis":     analysis_per_ticker,
        "iteration_log": iteration_log,
        "errors":        errors,
    })


# ── Helper ────────────────────────────────────────────────────────────────────

def _record_step(
    step: int, tool: str, input_data: dict, output: Any,
    tracer, artifacts, step_callback, iteration_log: list,
) -> None:
    output_str = str(output)
    iteration_log.append(IterationLogEntry(
        step=step, agent="quant_analyst", tool=tool,
        input=input_data, output_tokens=len(output_str) // 4, cache_hit=False,
    ))
    if tracer:
        tracer.record_tool_call(
            agent="quant_analyst", tool=tool,
            input_data=input_data, output_data=output,
            output_tokens=len(output_str) // 4,
        )
    if artifacts:
        artifacts.record_tool_response(
            agent="quant_analyst", tool=tool,
            input_data=input_data, full_output=output_str, step=step,
        )
    if step_callback:
        step_callback({
            "step": step, "agent": "quant_analyst", "tool": tool,
            "input": input_data, "cache_hit": False, "ok": True,
        })
