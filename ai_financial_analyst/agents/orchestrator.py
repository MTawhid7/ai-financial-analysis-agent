"""LangGraph orchestrator — wires the three-agent pipeline with checkpointing.

Pipeline: START → researcher → quant_analyst → editor → END

Each node is wrapped to catch PartialStateError and CircuitBreakerError,
producing a graceful partial output rather than a hard crash.
State is checkpointed after each node via SQLite.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
# SqliteSaver lives in the separate langgraph-checkpoint-sqlite package (v3+).
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from ..core.budget_tracker import RequestBudgetTracker
from ..core.llm import CircuitBreakerError, get_primary_llm, get_subllm
from ..core.state import AgentState, PartialStateError
from ..core.tracing import RunStatus, RunTracer
from ..tools import web_search as web_search_module
from ..tools import report_writer as report_writer_module
from .editor import editor_node
from .quant_analyst import quant_analyst_node
from .researcher import researcher_node

load_dotenv()

logger = logging.getLogger(__name__)

_CHECKPOINT_PATH = os.getenv("CHECKPOINT_DB_PATH", ".checkpoints/state.db")


def build_pipeline(dry_run: bool = False):
    """Construct and return the compiled LangGraph pipeline.

    Args:
        dry_run: Use MemorySaver (no SQLite, no API calls) for demo replay.

    Returns:
        Compiled LangGraph app ready for ainvoke().
    """
    budget = RequestBudgetTracker()
    primary_llm = get_primary_llm(budget_tracker=budget)
    subllm = get_subllm(budget_tracker=budget)

    # Inject sub-LLM into tools that need it.
    web_search_module.configure(subllm=subllm)
    report_writer_module.configure(primary_llm=primary_llm)

    tracer = RunTracer()

    # Shared config injected into every node via LangGraph's config parameter.
    node_config = {
        "primary_llm": primary_llm,
        "tracer": tracer,
        "budget": budget,
    }

    async def _researcher(state: AgentState) -> AgentState:
        return await _safe_node("researcher", researcher_node, state, node_config, tracer)

    async def _quant_analyst(state: AgentState) -> AgentState:
        return await _safe_node("quant_analyst", quant_analyst_node, state, node_config, tracer)

    async def _editor(state: AgentState) -> AgentState:
        return await _safe_node("editor", editor_node, state, node_config, tracer)

    graph = StateGraph(AgentState)
    graph.add_node("researcher", _researcher)
    graph.add_node("quant_analyst", _quant_analyst)
    graph.add_node("editor", _editor)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "quant_analyst")
    graph.add_edge("quant_analyst", "editor")
    graph.add_edge("editor", END)

    if dry_run:
        checkpointer = MemorySaver()
    else:
        db_path = Path(_CHECKPOINT_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(str(db_path))

    app = graph.compile(checkpointer=checkpointer)

    # Attach metadata so the UI can access tracer/budget after the run.
    app._tracer = tracer  # type: ignore[attr-defined]
    app._budget = budget  # type: ignore[attr-defined]

    return app


async def _safe_node(
    name: str,
    node_fn,
    state: AgentState,
    config: dict,
    tracer: RunTracer,
) -> AgentState:
    """Wrap a node to catch known failure modes and emit partial output."""
    try:
        return await node_fn(state, config=config)
    except PartialStateError as exc:
        logger.error("PartialStateError in %s: %s", name, exc)
        tracer.set_status(RunStatus.PARTIAL)
        errors = list(state.get("errors", []))
        errors.append(
            {
                "error_type": "STATE_VALIDATION_ERROR",
                "agent": name,
                "missing_fields": exc.missing_fields,
            }
        )
        return AgentState(**state, errors=errors, status="PARTIAL")
    except CircuitBreakerError as exc:
        logger.critical("Circuit breaker tripped in %s: %s", name, exc)
        tracer.set_status(RunStatus.RATE_LIMITED)
        errors = list(state.get("errors", []))
        errors.append({"error_type": "RATE_LIMIT", "agent": name, "detail": str(exc)})
        report = state.get("report_markdown") or _partial_report(state, name)
        return AgentState(**state, errors=errors, status="RATE_LIMITED", report_markdown=report)
    except Exception as exc:
        logger.exception("Unexpected error in node %s", name)
        tracer.set_status(RunStatus.FAILED)
        errors = list(state.get("errors", []))
        errors.append({"error_type": "UNKNOWN", "agent": name, "detail": str(exc)})
        return AgentState(**state, errors=errors, status="FAILED")


def _partial_report(state: AgentState, failed_at: str) -> str:
    """Build a minimal fallback report when the pipeline is halted."""
    tickers = state.get("tickers", [])
    gaps = state.get("researcher_gaps", [])
    analysis = state.get("analysis", {})

    lines = [
        "# Partial Report — Pipeline Halted",
        "",
        f"> **Status:** Pipeline halted at `{failed_at}` due to rate limiting.",
        "",
        "## Data Coverage Summary",
        "",
    ]
    for ticker in tickers:
        status = "Analysed (partial)" if ticker in analysis else "No data retrieved"
        lines.append(f"- **{ticker}**: {status}")

    if gaps:
        lines += ["", "## Data Gaps", ""]
        lines += [f"- {g}" for g in gaps]

    lines += [
        "",
        "---",
        "*DISCLAIMER: This report was generated by an AI system. "
        "All figures should be independently verified before making any "
        "investment decisions. This is not financial advice.*",
    ]
    return "\n".join(lines)


async def run_pipeline(
    query: str,
    tickers: list[str],
    dry_run: bool = False,
    trace_output_dir: str = ".",
) -> tuple[AgentState, str]:
    """High-level entry point: run the full pipeline and export the trace.

    Returns:
        (final_state, trace_path)
    """
    app = build_pipeline(dry_run=dry_run)
    run_id = str(uuid.uuid4())

    initial_state = AgentState(
        query=query,
        tickers=[t.strip().upper() for t in tickers],
        iteration_log=[],
        errors=[],
        status="COMPLETE",
        run_id=run_id,
    )

    thread_config = {"configurable": {"thread_id": run_id}}
    final_state = await app.ainvoke(initial_state, config=thread_config)

    tracer: RunTracer = app._tracer  # type: ignore[attr-defined]
    budget: RequestBudgetTracker = app._budget  # type: ignore[attr-defined]
    trace_path = tracer.export(output_dir=trace_output_dir)
    tracer.build(budget_stats=budget.get_stats())

    return final_state, str(trace_path)
