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

from ..core.state import (
    AgentState,
    IterationLogEntry,
    validate_state_for_agent,
)
from ..core.tracing import ErrorType, RunTracer
from ..tools.calculator import calculator_tool
from ..tools.benchmark_lookup import benchmark_lookup_tool

logger = logging.getLogger(__name__)

_SOP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a quantitative financial analyst. You receive structured financial data
and must produce a rigorous JSON analysis following the SOP below.

MANDATORY SOP — you MUST complete ALL of the following:
(a) Compute 5-year price CAGR using: ((current_price / price_5y_ago) ^ (1/5) - 1) * 100
(b) Identify the company sector and call benchmark_lookup for sector averages.
(c) Compare the company P/E ratio to the sector average P/E. State whether it is
    at a premium or discount and by what percentage.
(d) Name the most likely sector peer from benchmark peer_examples.
(e) State a Bull Case (2-3 reasons to be optimistic).
(f) State a Bear Case (2-3 reasons to be cautious).

For EVERY numerical value you include, you MUST cite:
  "source_tool": "<tool_name>",
  "observation_step": <step_number_from_iteration_log>

Output ONLY a JSON object — no markdown, no prose outside the JSON.""",
        ),
        (
            "human",
            "Raw financial data:\n{raw_data_json}\n\nIteration log:\n{iteration_log_json}",
        ),
    ]
)


async def quant_analyst_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Quant Analyst agent."""
    validate_state_for_agent(state, "quant_analyst")

    tracer: RunTracer | None = config.get("tracer") if config else None
    primary_llm = config.get("primary_llm") if config else None

    if primary_llm is None:
        raise RuntimeError("Quant Analyst node requires 'primary_llm' in config")

    raw_data = state["raw_data"]
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]] = list(state.get("errors", []))
    step = len(iteration_log)

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
        if fundamentals and fundamentals.get("sector"):
            sector = fundamentals["sector"]
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
        sop_input = {
            "raw_data_json": json.dumps({ticker: ticker_data}, indent=2)[:3000],
            "iteration_log_json": json.dumps(iteration_log[-10:], indent=2),
        }

        try:
            sop_response = await sop_chain.ainvoke(sop_input)
            sop_text = sop_response.content if hasattr(sop_response, "content") else str(sop_response)
            sop_data = json.loads(sop_text)
            ticker_analysis["bull_case"] = sop_data.get("bull_case", [])
            ticker_analysis["bear_case"] = sop_data.get("bear_case", [])
            ticker_analysis["closest_peer"] = sop_data.get("closest_peer")
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("SOP chain failed for %s: %s", ticker, exc)
            ticker_analysis["bull_case"] = []
            ticker_analysis["bear_case"] = []
            errors.append({"error_type": ErrorType.PARSING_ERROR.value, "detail": str(exc)})

        if tracer:
            tracer.record_tool_call(
                agent="quant_analyst",
                tool="sop_analysis",
                input_data={"ticker": ticker},
                output_data=ticker_analysis,
                output_tokens=step * 5,
            )

        analysis_per_ticker[ticker] = ticker_analysis

    return AgentState(**{
        **state,
        "analysis": analysis_per_ticker,
        "iteration_log": iteration_log,
        "errors": errors,
    })
