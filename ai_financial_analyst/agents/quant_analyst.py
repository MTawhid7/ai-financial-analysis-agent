"""Quant Analyst Agent — computation and comparison node.

Responsibilities:
- Validate Researcher output before proceeding.
- Compute 5-year CAGR for revenue and price.
- Compare P/E ratios against sector median from BenchmarkLookupTool.
- Identify closest sector peer.
- Formulate bull and bear cases.
- Cite source_tool + observation_index for every numerical value.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

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

# yfinance sector strings → GICS sector names used in benchmarks.json
_YFINANCE_TO_GICS: dict[str, str] = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}

_SOP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a quantitative financial analyst. Analyse the financial data provided
and respond with ONLY this exact JSON structure — no other keys, no markdown, no prose:

{{
  "bull_case": ["<reason 1>", "<reason 2>", "<reason 3>"],
  "bear_case": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "closest_peer": "<Company Name (TICKER)>"
}}

Rules:
- bull_case: exactly 2-3 concise reasons to be optimistic about this stock. Reference specific computed metrics (CAGR, P/E premium, sector benchmark) where available.
- bear_case: exactly 2-3 concise risks or concerns. Reference the valuation premium vs. sector if available.
- closest_peer: the single most comparable publicly traded peer company
- If data is missing, use general sector knowledge to provide reasonable analysis
- Output ONLY the JSON object above — nothing else""",
        ),
        (
            "human",
            "Stock: {raw_data_json}\n\nComputed metrics so far:\n{computed_metrics_json}\n\nPrevious tool observations:\n{iteration_log_json}",
        ),
    ]
)


async def quant_analyst_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Quant Analyst agent."""
    validate_state_for_agent(state, "quant_analyst")

    tracer: RunTracer | None = config.get("tracer") if config else None
    artifacts = config.get("artifacts") if config else None
    step_callback = config.get("step_callback") if config else None
    primary_llm = config.get("primary_llm") if config else None

    if primary_llm is None:
        raise RuntimeError("Quant Analyst node requires 'primary_llm' in config")

    raw_data = state["raw_data"]
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]] = list(state.get("errors", []))
    step = len(iteration_log)

    if tracer:
        tracer.record_agent_start("quant_analyst", {"tickers": list(raw_data.keys())})

    analysis_per_ticker: dict[str, Any] = {}

    for ticker, ticker_data in raw_data.items():
        ticker_analysis: dict[str, Any] = {"ticker": ticker, "citations": {}}

        # --- 5-year price CAGR ---
        price_data = ticker_data.get("price_history")
        if price_data and price_data.get("current_price") and price_data.get("price_5y_ago"):
            current = price_data["current_price"]
            five_y_ago = price_data["price_5y_ago"]
            if five_y_ago > 0:
                cagr_expr = f"(({current} / {five_y_ago}) ** (1/5) - 1) * 100"
                step += 1
                cagr_str = await calculator_tool.arun({"expression": cagr_expr})

                if tracer:
                    tracer.record_tool_call(
                        agent="quant_analyst",
                        tool="calculator",
                        input_data={"expression": cagr_expr},
                        output_data=cagr_str,
                        output_tokens=10,
                    )
                if artifacts:
                    artifacts.record_tool_response(
                        agent="quant_analyst",
                        tool="calculator",
                        input_data={"expression": cagr_expr},
                        full_output=str(cagr_str),
                        step=step,
                    )
                if step_callback:
                    step_callback({
                        "step": step, "agent": "quant_analyst", "tool": "calculator",
                        "input": {"expression": cagr_expr}, "cache_hit": False, "ok": True,
                    })

                iteration_log.append(
                    IterationLogEntry(
                        step=step,
                        agent="quant_analyst",
                        tool="calculator",
                        input={"expression": cagr_expr},
                        output_tokens=10,
                        cache_hit=False,
                    )
                )

                try:
                    ticker_analysis["price_cagr_5y_pct"] = round(float(cagr_str), 2)
                    ticker_analysis["citations"]["price_cagr_5y_pct"] = {
                        "source_tool": "calculator",
                        "observation_step": step,
                    }
                except ValueError:
                    errors.append({"error_type": ErrorType.TOOL_ERROR.value, "detail": f"CAGR parse failed: {cagr_str}"})
            else:
                ticker_analysis["price_cagr_5y_pct"] = None
        else:
            ticker_analysis["price_cagr_5y_pct"] = None

        # --- Sector benchmark comparison ---
        fundamentals = ticker_data.get("fundamentals")
        logger.debug(
            "Fundamentals for %s: type=%s, sector=%s",
            ticker,
            type(fundamentals).__name__,
            fundamentals.get("sector") if isinstance(fundamentals, dict) else "NOT A DICT",
        )
        if isinstance(fundamentals, str):
            # Checkpoint deserialised as JSON string — parse it back.
            try:
                import json as _json
                fundamentals = _json.loads(fundamentals)
            except Exception:
                fundamentals = None
        if fundamentals and fundamentals.get("sector"):
            sector = _YFINANCE_TO_GICS.get(fundamentals["sector"], fundamentals["sector"])
            step += 1
            benchmark_str = await benchmark_lookup_tool.arun({"gics_sector": sector})

            if tracer:
                tracer.record_tool_call(
                    agent="quant_analyst",
                    tool="benchmark_lookup",
                    input_data={"gics_sector": sector},
                    output_data=benchmark_str,
                    output_tokens=len(benchmark_str) // 4,
                )
            if artifacts:
                artifacts.record_tool_response(
                    agent="quant_analyst",
                    tool="benchmark_lookup",
                    input_data={"gics_sector": sector},
                    full_output=benchmark_str,
                    step=step,
                )
            if step_callback:
                step_callback({
                    "step": step, "agent": "quant_analyst", "tool": "benchmark_lookup",
                    "input": {"gics_sector": sector}, "cache_hit": False,
                    "ok": "error_type" not in (json.loads(benchmark_str) if benchmark_str.startswith("{") else {}),
                })

            iteration_log.append(
                IterationLogEntry(
                    step=step,
                    agent="quant_analyst",
                    tool="benchmark_lookup",
                    input={"gics_sector": sector},
                    output_tokens=len(benchmark_str) // 4,
                    cache_hit=False,
                )
            )

            try:
                benchmark = json.loads(benchmark_str)
                if "error_type" not in benchmark:
                    ticker_analysis["sector"] = sector
                    ticker_analysis["sector_pe_avg"] = benchmark["pe_ratio_sector_avg"]
                    ticker_analysis["sector_peers"] = benchmark["peer_examples"]
                    ticker_analysis["citations"]["sector_pe_avg"] = {
                        "source_tool": "benchmark_lookup",
                        "observation_step": step,
                    }

                    company_pe = fundamentals.get("pe_ratio")
                    if company_pe and benchmark["pe_ratio_sector_avg"]:
                        pe_premium_pct = (
                            (company_pe - benchmark["pe_ratio_sector_avg"])
                            / benchmark["pe_ratio_sector_avg"]
                            * 100
                        )
                        ticker_analysis["company_pe"] = company_pe
                        ticker_analysis["pe_vs_sector_premium_pct"] = round(pe_premium_pct, 1)
                        ticker_analysis["citations"]["pe_vs_sector_premium_pct"] = {
                            "source_tool": "benchmark_lookup",
                            "observation_step": step,
                        }
            except (json.JSONDecodeError, KeyError) as exc:
                errors.append({"error_type": ErrorType.PARSING_ERROR.value, "detail": str(exc)})

        # --- LLM-generated bull/bear cases via SOP prompt ---
        step += 1
        sop_chain = _SOP_PROMPT | primary_llm
        # Include already-computed metrics so the LLM can reference sector P/E premium
        # and CAGR directly rather than re-deriving them from raw numbers.
        computed_so_far = {
            k: v for k, v in ticker_analysis.items()
            if k not in ("ticker", "citations") and v is not None
        }
        sop_input = {
            "raw_data_json": json.dumps({ticker: ticker_data}, indent=2)[:2500],
            "computed_metrics_json": json.dumps(computed_so_far, indent=2),
            "iteration_log_json": json.dumps(iteration_log[-10:], indent=2),
        }

        sop_text = ""
        try:
            sop_response = await sop_chain.ainvoke(sop_input)
            raw_content = sop_response.content if hasattr(sop_response, "content") else sop_response
            sop_text = content_to_str(raw_content)
            # Strip markdown code fences if the model wraps the JSON
            sop_text = sop_text.strip()
            if sop_text.startswith("```"):
                sop_text = sop_text.split("```", 2)[1]
                if sop_text.startswith("json"):
                    sop_text = sop_text[4:]
                sop_text = sop_text.rstrip("`").strip()
            logger.debug("SOP chain response for %s: %s", ticker, sop_text[:300])
            sop_data = json.loads(sop_text)
            # Support both flat {"bull_case": [...]} and nested {"analysis": {"investment_thesis": {...}}}
            thesis = (
                sop_data.get("analysis", {}).get("investment_thesis", sop_data)
                if "analysis" in sop_data
                else sop_data
            )
            ticker_analysis["bull_case"] = thesis.get("bull_case", [])
            ticker_analysis["bear_case"] = thesis.get("bear_case", [])
            ticker_analysis["closest_peer"] = (
                thesis.get("closest_peer")
                or sop_data.get("analysis", {}).get("most_likely_peer")
            )
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("SOP chain failed for %s: %s", ticker, exc)
            ticker_analysis["bull_case"] = []
            ticker_analysis["bear_case"] = []
            errors.append({"error_type": ErrorType.PARSING_ERROR.value, "detail": str(exc)})

        if step_callback:
            step_callback({
                "step": step, "agent": "quant_analyst", "tool": "sop_llm",
                "input": {"ticker": ticker}, "cache_hit": False, "ok": bool(ticker_analysis.get("bull_case")),
            })

        if artifacts:
            try:
                formatted = _SOP_PROMPT.format_messages(**sop_input)
                prompt_msgs = [{"role": m.type, "content": m.content} for m in formatted]
            except Exception:
                prompt_msgs = [{"role": "user", "content": str(sop_input)}]
            artifacts.record_llm_exchange(
                agent="quant_analyst",
                purpose="sop_analysis",
                ticker=ticker,
                prompt_messages=prompt_msgs,
                raw_response=sop_text,
            )

        if tracer:
            tracer.record_tool_call(
                agent="quant_analyst",
                tool="sop_analysis",
                input_data={"ticker": ticker},
                output_data=ticker_analysis,
                output_tokens=step * 5,
            )

        analysis_per_ticker[ticker] = ticker_analysis

    if tracer:
        tracer.record_agent_complete("quant_analyst", {
            "tickers_analysed": list(analysis_per_ticker.keys()),
            "errors": len(errors),
        })

    return AgentState(**{
        **state,
        "analysis": analysis_per_ticker,
        "iteration_log": iteration_log,
        "errors": errors,
    })
