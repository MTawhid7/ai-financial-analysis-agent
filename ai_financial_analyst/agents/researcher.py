"""Researcher Agent — concurrent data acquisition node for the LangGraph pipeline.

Architecture change from sequential to concurrent:
  Old: 7 data types × N tickers fetched one-by-one in a serial loop (~60–120s/ticker)
  New: Phase-1 (3 core types parallel) + Phase-2 (4 extended types parallel) per ticker
       with adaptive early exit when all core types fail (~6–10s/ticker)

yfinance HTTP calls consume zero Gemini RPM quota — concurrency is safe.
The asyncio.Semaphore in data/yahoo/__init__.py limits to 3 concurrent threads.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from ..config import settings
from ..core.state import (
    AgentState,
    DataCoverage,
    IterationLogEntry,
    RESEARCHER_REQUIRED,
    PartialStateError,
)
from ..core.tracing import ErrorType, RunTracer
from ..core.utils import estimate_tokens
from ..data.yahoo import fetch_ticker_data
from ..data.search.tavily import TavilySearchClient

logger = logging.getLogger(__name__)

# Retained for backward compatibility with existing tests and callers.
# The concurrent fetcher no longer uses a hard iteration counter — adaptive
# early-exit logic in data/yahoo/__init__.py replaces it.
MAX_ITERATIONS = 10


def _current_year() -> str:
    return str(datetime.utcnow().year)


async def researcher_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Researcher agent (concurrent data fetching)."""
    tracer:        RunTracer | None = config.get("tracer")        if config else None
    artifacts                      = config.get("artifacts")      if config else None
    step_callback                  = config.get("step_callback")  if config else None

    tickers        = state.get("tickers", [])
    query          = state.get("query", "")
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]]           = list(state.get("errors", []))

    if tracer:
        tracer.record_agent_start("researcher", {"tickers": tickers, "query": query})

    raw_data:      dict[str, Any]     = {}
    data_coverage: list[DataCoverage] = []
    data_gaps:     list[str]          = []
    step = len(iteration_log)

    for ticker in tickers:
        step += 1

        # ── Phase 1+2: Concurrent yfinance fetch (adaptive) ──────────────────
        if step_callback:
            step_callback({
                "step": step, "agent": "researcher", "tool": "yahoo_finance",
                "input": {"ticker": ticker, "mode": "concurrent"}, "cache_hit": False, "ok": True,
            })

        ticker_data: dict[str, Any] = await fetch_ticker_data(ticker)

        # Build iteration log entries and coverage from results
        coverage = DataCoverage(
            ticker        = ticker,
            price_history = "price_history" in ticker_data,
            fundamentals  = "fundamentals"  in ticker_data,
            balance_sheet = "balance_sheet" in ticker_data,
            news_search   = False,
            data_gaps     = [],
        )

        # Surface data quality / degradation from each fetched type
        for data_type, result in ticker_data.items():
            result_str = json.dumps(result)
            iteration_log.append(IterationLogEntry(
                step          = step,
                agent         = "researcher",
                tool          = "yahoo_finance",
                input         = {"ticker": ticker, "data_type": data_type},
                output_tokens = estimate_tokens(result_str),
                cache_hit     = False,
            ))
            if tracer:
                tracer.record_tool_call(
                    agent        = "researcher",
                    tool         = "yahoo_finance",
                    input_data   = {"ticker": ticker, "data_type": data_type},
                    output_data  = result,
                    output_tokens= estimate_tokens(result_str),
                )
            if artifacts:
                artifacts.record_tool_response(
                    agent      = "researcher",
                    tool       = "yahoo_finance",
                    input_data = {"ticker": ticker, "data_type": data_type},
                    full_output= result_str,
                    step       = step,
                )
            # Collect quality degradation notes as data gaps
            note = result.get("degradation_note")
            if result.get("data_quality") == "PARTIAL" and note:
                gap = f"{ticker}/{data_type}: {note}"
                coverage["data_gaps"].append(gap)
                data_gaps.append(gap)
            elif result.get("data_quality") == "UNAVAILABLE":
                reason = result.get("reason", "unavailable")
                gap    = f"{ticker}/{data_type}: {reason}"
                coverage["data_gaps"].append(gap)
                data_gaps.append(gap)
                errors.append({"error_type": ErrorType.TOOL_ERROR.value, "detail": gap})

        if not ticker_data:
            gap = f"{ticker}: all data types failed — skipping"
            data_gaps.append(gap)
            errors.append({"error_type": ErrorType.TOOL_ERROR.value, "detail": gap})

        # ── Web news search ───────────────────────────────────────────────────
        step += 1
        news_query = (
            f"{ticker} stock news analyst outlook earnings results {_current_year()}"
        )
        news_result: dict = {}
        try:
            client     = TavilySearchClient()
            results    = client.search(news_query)
            news_result = {
                "query":        news_query,
                "result_count": len(results),
                "summaries": [
                    {
                        "headline":    r.headline,
                        "url":         r.url,
                        "content":     r.content,
                        "score":       r.score,
                        "source_tier": r.source_tier,
                    }
                    for r in results if r.content
                ],
            }
            ticker_data["news"] = news_result
            coverage["news_search"] = True
        except Exception as exc:
            gap = f"{ticker}/news: {exc}"
            coverage["data_gaps"].append(gap)
            data_gaps.append(gap)
            logger.warning("Web search failed for %s: %s", ticker, exc)

        news_str = json.dumps(news_result)
        iteration_log.append(IterationLogEntry(
            step          = step,
            agent         = "researcher",
            tool          = "web_search",
            input         = {"query": news_query},
            output_tokens = estimate_tokens(news_str),
            cache_hit     = False,
        ))
        if tracer:
            tracer.record_tool_call(
                agent         = "researcher",
                tool          = "web_search",
                input_data    = {"query": news_query},
                output_data   = news_result,
                output_tokens = estimate_tokens(news_str),
            )
        if step_callback:
            step_callback({
                "step": step, "agent": "researcher", "tool": "web_search",
                "input": {"query": news_query}, "cache_hit": False,
                "ok": coverage["news_search"],
            })

        raw_data[ticker] = ticker_data
        data_coverage.append(coverage)

    if not raw_data:
        raise PartialStateError("researcher", ["raw_data — no data retrieved for any ticker"])

    if tracer:
        tracer.record_agent_complete("researcher", {
            "tickers_fetched": list(raw_data.keys()),
            "total_gaps":      len(data_gaps),
            "gaps":            data_gaps[:5],
        })

    return AgentState(**{
        **state,
        "raw_data":        raw_data,
        "data_coverage":   data_coverage,
        "researcher_gaps": data_gaps,
        "iteration_log":   iteration_log,
        "errors":          errors,
    })
