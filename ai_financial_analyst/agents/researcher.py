"""Researcher Agent — data acquisition node for the LangGraph pipeline.

Responsibilities:
- Fetch price history, fundamentals, and balance sheet for each ticker.
- Search for recent news and analyst commentary.
- Enforce max_iterations=5 to prevent infinite loops.
- Produce a structured fallback with explicit data_gaps on loop exit.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..core.state import (
    AgentState,
    DataCoverage,
    IterationLogEntry,
    RESEARCHER_REQUIRED,
    PartialStateError,
)
from ..core.tracing import ErrorType, RunTracer
from ..tools.yahoo_finance import yahoo_finance_tool
from ..tools.web_search import web_search_tool

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5

_SYSTEM_PROMPT = """You are a financial data researcher. Your job is to gather comprehensive
financial data for the requested stock tickers.

For EACH ticker you MUST:
1. Call yahoo_finance with data_type="price_history"
2. Call yahoo_finance with data_type="fundamentals"
3. Call yahoo_finance with data_type="balance_sheet"
4. Call web_search with a query like "{TICKER} recent news analyst outlook 2024"

Rules:
- Do NOT skip any data type for any ticker.
- If a tool returns a ToolError JSON, record it and move on — do NOT retry the same call.
- If a tool returns null data, note the gap explicitly in your final summary.
- Your final response MUST be a JSON object with keys:
  "raw_data": {{ ticker -> {{ data_type -> result }} }},
  "data_coverage": [ {{ ticker, price_history, fundamentals, balance_sheet, news_search, data_gaps }} ],
  "researcher_gaps": [ "description of each missing data point" ]
"""


async def researcher_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Researcher agent."""
    tracer: RunTracer | None = config.get("tracer") if config else None
    artifacts = config.get("artifacts") if config else None
    step_callback = config.get("step_callback") if config else None
    # primary_llm is not used by the Researcher — it calls tools directly.

    tickers = state.get("tickers", [])
    query = state.get("query", "")
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]] = list(state.get("errors", []))

    if tracer:
        tracer.record_agent_start("researcher", {"tickers": tickers, "query": query})

    # Build tool map
    tools = {
        "yahoo_finance": yahoo_finance_tool,
        "web_search": web_search_tool,
    }

    raw_data: dict[str, Any] = {}
    data_coverage: list[DataCoverage] = []
    data_gaps: list[str] = []
    step = len(iteration_log)

    for ticker in tickers:
        ticker_data: dict[str, Any] = {}
        coverage = DataCoverage(
            ticker=ticker,
            price_history=False,
            fundamentals=False,
            balance_sheet=False,
            news_search=False,
            data_gaps=[],
        )
        calls_made = 0

        # Fetch the three required data types
        for data_type in ["price_history", "fundamentals", "balance_sheet"]:
            if calls_made >= MAX_ITERATIONS:
                logger.warning("max_iterations=%d reached for ticker %s", MAX_ITERATIONS, ticker)
                break

            step += 1
            calls_made += 1
            tool_input = {"ticker": ticker, "data_type": data_type}

            try:
                result_str = await yahoo_finance_tool.arun(tool_input)
                result = json.loads(result_str)
            except Exception as exc:
                result = {"error_type": "TOOL_ERROR", "message": str(exc)}
                result_str = json.dumps(result)

            cache_hit = result.get("cache_hit", False)
            iteration_log.append(
                IterationLogEntry(
                    step=step,
                    agent="researcher",
                    tool="yahoo_finance",
                    input=tool_input,
                    output_tokens=len(result_str) // 4,
                    cache_hit=cache_hit,
                )
            )

            if tracer:
                tracer.record_tool_call(
                    agent="researcher",
                    tool="yahoo_finance",
                    input_data=tool_input,
                    output_data=result,
                    output_tokens=len(result_str) // 4,
                    cache_hit=cache_hit,
                )
            if artifacts:
                artifacts.record_tool_response(
                    agent="researcher",
                    tool="yahoo_finance",
                    input_data=tool_input,
                    full_output=result_str,
                    step=step,
                    cache_hit=cache_hit,
                )
            if step_callback:
                step_callback({
                    "step": step, "agent": "researcher", "tool": "yahoo_finance",
                    "input": tool_input, "cache_hit": cache_hit,
                    "ok": "error_type" not in result,
                })

            if "error_type" in result:
                gap = f"{ticker}/{data_type}: {result.get('message', 'unknown error')}"
                coverage["data_gaps"].append(gap)
                data_gaps.append(gap)
                errors.append({"error_type": ErrorType.TOOL_ERROR.value, "detail": gap})
            elif "reason" in result and result.get("result") is None:
                # Explicit null returned by the tool: {"result": None, "reason": "..."}
                # NOTE: dict.get("result") returns None for both missing keys AND explicit
                # None values, so we check for "reason" to distinguish null from success.
                gap = f"{ticker}/{data_type}: {result['reason']}"
                coverage["data_gaps"].append(gap)
                data_gaps.append(gap)
            else:
                ticker_data[data_type] = result
                coverage[data_type.replace("price_history", "price_history")] = True  # type: ignore[literal-required]
                # Explicit boolean flags
                if data_type == "price_history":
                    coverage["price_history"] = True
                elif data_type == "fundamentals":
                    coverage["fundamentals"] = True
                elif data_type == "balance_sheet":
                    coverage["balance_sheet"] = True

        # News search
        if calls_made < MAX_ITERATIONS:
            step += 1
            calls_made += 1
            news_query = f"{ticker} stock news analyst outlook financial results 2024"
            news_input = {"query": news_query, "max_results": 3}

            try:
                news_str = await web_search_tool.arun(news_input)
                news_result = json.loads(news_str)
            except Exception as exc:
                news_result = {"error_type": "TOOL_ERROR", "message": str(exc)}
                news_str = json.dumps(news_result)

            iteration_log.append(
                IterationLogEntry(
                    step=step,
                    agent="researcher",
                    tool="web_search",
                    input=news_input,
                    output_tokens=len(news_str) // 4,
                    cache_hit=False,
                )
            )

            if tracer:
                tracer.record_tool_call(
                    agent="researcher",
                    tool="web_search",
                    input_data=news_input,
                    output_data=news_result,
                    output_tokens=len(news_str) // 4,
                )
            if artifacts:
                artifacts.record_tool_response(
                    agent="researcher",
                    tool="web_search",
                    input_data=news_input,
                    full_output=news_str,
                    step=step,
                )
            if step_callback:
                step_callback({
                    "step": step, "agent": "researcher", "tool": "web_search",
                    "input": news_input, "cache_hit": False,
                    "ok": "error_type" not in news_result,
                })

            if "error_type" not in news_result:
                ticker_data["news"] = news_result
                coverage["news_search"] = True
            else:
                gap = f"{ticker}/news: {news_result.get('message', 'search failed')}"
                coverage["data_gaps"].append(gap)
                data_gaps.append(gap)

        raw_data[ticker] = ticker_data
        data_coverage.append(coverage)

    # Validate output before yielding to the next agent.
    if not raw_data:
        raise PartialStateError(
            "researcher",
            ["raw_data — no data retrieved for any ticker"],
        )

    if tracer:
        tracer.record_agent_complete("researcher", {
            "tickers_fetched": list(raw_data.keys()),
            "total_gaps": len(data_gaps),
            "gaps": data_gaps[:5],
        })

    return AgentState(**{
        **state,
        "raw_data": raw_data,
        "data_coverage": data_coverage,
        "researcher_gaps": data_gaps,
        "iteration_log": iteration_log,
        "errors": errors,
    })
