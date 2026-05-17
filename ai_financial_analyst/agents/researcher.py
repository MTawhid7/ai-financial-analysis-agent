"""Researcher Agent — concurrent data acquisition node for the LangGraph pipeline.

Architecture change from sequential to concurrent:
  Old: 7 data types × N tickers fetched one-by-one in a serial loop (~60–120s/ticker)
  New: Phase-1 (3 core types parallel) + Phase-2 (4 extended types parallel) per ticker
       with adaptive early exit when all core types fail (~6–10s/ticker)

yfinance HTTP calls consume zero Gemini RPM quota — concurrency is safe.
The asyncio.Semaphore in data/yahoo/__init__.py limits to 3 concurrent threads.
"""

from __future__ import annotations

import asyncio
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


# Return type for _fetch_single_ticker
_TickerResult = tuple[
    str,                      # ticker
    dict[str, Any],           # ticker_data (raw)
    "DataCoverage",           # coverage
    list[str],                # data_gaps
    list[dict],               # errors
    list["IterationLogEntry"],# iteration_log_entries
]


async def _fetch_single_ticker(
    ticker: str,
    step_base: int,
    step_callback: Any,
    tracer: RunTracer | None,
    artifacts: Any,
    semaphore: asyncio.Semaphore,
) -> _TickerResult:
    """Fetch all data for a single ticker: yfinance (concurrent) + Tavily news.

    Wrapped by the caller with asyncio.gather for inter-ticker parallelism.
    The semaphore gates how many tickers can be fetched simultaneously.
    yfinance HTTP calls consume zero Gemini RPM — concurrency is safe.
    """
    async with semaphore:
        step = step_base
        ticker_data: dict[str, Any] = {}
        gaps:   list[str]           = []
        errs:   list[dict]          = []
        log_entries: list[IterationLogEntry] = []

        # ── Phase 1+2: Concurrent yfinance fetch (adaptive) ──────────────────
        if step_callback:
            step_callback({
                "step": step, "agent": "researcher", "tool": "yahoo_finance",
                "input": {"ticker": ticker, "mode": "concurrent"}, "cache_hit": False, "ok": True,
            })

        ticker_data = await fetch_ticker_data(ticker)

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
            log_entries.append(IterationLogEntry(
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
            note = result.get("degradation_note")
            if result.get("data_quality") == "PARTIAL" and note:
                gap = f"{ticker}/{data_type}: {note}"
                coverage["data_gaps"].append(gap)
                gaps.append(gap)
            elif result.get("data_quality") == "UNAVAILABLE":
                reason = result.get("reason", "unavailable")
                gap    = f"{ticker}/{data_type}: {reason}"
                coverage["data_gaps"].append(gap)
                gaps.append(gap)
                errs.append({"error_type": ErrorType.TOOL_ERROR.value, "detail": gap})

        if not ticker_data:
            gap = f"{ticker}: all data types failed — skipping"
            gaps.append(gap)
            errs.append({"error_type": ErrorType.TOOL_ERROR.value, "detail": gap})

        # ── Web news search ───────────────────────────────────────────────────
        step += 1
        news_query  = f"{ticker} stock news analyst outlook earnings results {_current_year()}"
        news_result: dict = {}
        try:
            client      = TavilySearchClient()
            results     = client.search(news_query)
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
            gaps.append(gap)
            logger.warning("Web search failed for %s: %s", ticker, exc)

        news_str = json.dumps(news_result)
        log_entries.append(IterationLogEntry(
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

        return ticker, ticker_data, coverage, gaps, errs, log_entries


async def researcher_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Researcher agent — fetches all tickers concurrently.

    Each ticker's yfinance + Tavily fetch runs in parallel using asyncio.gather.
    Concurrency is capped by settings.researcher_ticker_concurrency (default 3).
    One ticker failing does not abort the others.
    """
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

    # Semaphore caps simultaneous ticker fetches (yfinance threads + Tavily calls)
    semaphore = asyncio.Semaphore(settings.researcher_ticker_concurrency)
    initial_step = len(iteration_log)

    # Dispatch all tickers concurrently; return_exceptions=True prevents one
    # ticker failure from aborting the rest.
    tasks = [
        _fetch_single_ticker(
            ticker,
            step_base    = initial_step + idx * 2 + 1,
            step_callback= step_callback,
            tracer       = tracer,
            artifacts    = artifacts,
            semaphore    = semaphore,
        )
        for idx, ticker in enumerate(tickers)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for ticker, outcome in zip(tickers, results):
        if isinstance(outcome, Exception):
            gap = f"{ticker}: fetch failed — {outcome}"
            data_gaps.append(gap)
            errors.append({"error_type": ErrorType.TOOL_ERROR.value, "detail": gap})
            logger.warning("Ticker fetch raised exception for %s: %s", ticker, outcome)
            continue

        _ticker, ticker_data, coverage, gaps, errs, log_entries = outcome
        raw_data[ticker]       = ticker_data
        data_coverage.append(coverage)
        data_gaps.extend(gaps)
        errors.extend(errs)
        iteration_log.extend(log_entries)

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
