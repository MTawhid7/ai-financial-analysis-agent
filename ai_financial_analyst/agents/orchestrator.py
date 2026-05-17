"""LangGraph orchestrator — wires the three-agent pipeline with checkpointing.

Pipeline: START → researcher → [conditional] → quant_analyst → [conditional] → editor → END
                                     ↘ early_exit ↗                   ↘ early_exit ↗

Conditional routing skips downstream nodes when:
- researcher retrieved no usable data (all errors / empty raw_data)
- status is RATE_LIMITED or FAILED (preserve quota)
- quant_analyst produced no analysis dict

Each node is wrapped to catch PartialStateError and CircuitBreakerError,
producing a graceful partial output rather than a hard crash.
State is checkpointed after each node via AsyncSqliteSaver (required for
async pipelines; SqliteSaver only supports synchronous methods).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from ..config import settings
from ..core.artifacts import RunArtifacts
from ..core.budget_tracker import RequestBudgetTracker
from ..core.llm import CircuitBreakerError, get_primary_llm_with_fallback, get_subllm
from ..core.sanitizer import ContentSanitizer, SanitizationAlert
from ..core.state import AgentState, PartialStateError
from ..core.tracing import RunStatus, RunTracer
from ..tools import web_search as web_search_module
from .editor import editor_node
from .quant_analyst import quant_analyst_node
from .researcher import researcher_node

logger = logging.getLogger(__name__)

_CHECKPOINT_PATH = settings.checkpoint_db_path
_ARTIFACTS_DIR   = settings.artifacts_dir


def _clear_old_artifacts(output_dir: str) -> None:
    """Delete previous run files so only the latest run is kept in the folder."""
    folder = Path(output_dir)
    if not folder.exists():
        return
    for pattern in ("run_trace_*.json", "run_artifacts_*.json"):
        for f in folder.glob(pattern):
            try:
                f.unlink()
                logger.debug("Removed old artifact: %s", f.name)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", f, exc)


def _route_after_researcher(state: AgentState) -> str:
    """Route to quant_analyst or early_exit after the researcher runs."""
    raw_data = state.get("raw_data", {})
    status = state.get("status", "COMPLETE")
    if not raw_data or status in ("RATE_LIMITED", "FAILED"):
        return "early_exit"
    # Check whether any ticker has at least one non-error key
    any_has_data = any(
        any(k not in ("error_type", "message", "reason") for k in td)
        for td in raw_data.values()
    )
    return "early_exit" if not any_has_data else "quant_analyst"


def _route_after_quant(state: AgentState) -> str:
    """Route to editor or early_exit after the quant analyst runs."""
    status = state.get("status", "COMPLETE")
    if not state.get("analysis") or status in ("RATE_LIMITED", "FAILED"):
        return "early_exit"
    return "editor"


# Fields that contain free-form LLM-generated text per node.
# researcher is excluded: its output is structured API data already sanitized by
# ContentSanitizer before reaching any LLM — no LLM narratives in state.
_LLM_NARRATIVE_FIELDS: dict[str, tuple[str, ...]] = {
    "quant_analyst": ("analysis",),
    "editor":        ("report_markdown",),
}


def _check_intermediate_canary(node_name: str, state: AgentState) -> None:
    """Check for canary token in LLM-generated state fields after each node.

    The canary is random per process and never present in legitimate outputs.
    Its appearance signals that injected content was echoed by the LLM.
    Raises SanitizationAlert if the canary is found.
    """
    fields = _LLM_NARRATIVE_FIELDS.get(node_name, ())
    if not fields:
        return
    sanitizer = ContentSanitizer()
    for field in fields:
        value = state.get(field)
        if value is None:
            continue
        text = json.dumps(value, default=str) if not isinstance(value, str) else value
        sanitizer.check_canary(text)


def _build_graph(node_config: dict) -> StateGraph:
    """Build the StateGraph with nodes and conditional routing (not yet compiled).

    Compilation is deferred to run_pipeline() so the async checkpointer
    context manager can wrap the full ainvoke() call.
    """
    tracer: RunTracer = node_config["tracer"]

    async def _researcher(state: AgentState) -> AgentState:
        return await _safe_node("researcher", researcher_node, state, node_config, tracer)

    async def _quant_analyst(state: AgentState) -> AgentState:
        return await _safe_node("quant_analyst", quant_analyst_node, state, node_config, tracer)

    async def _editor(state: AgentState) -> AgentState:
        return await _safe_node("editor", editor_node, state, node_config, tracer)

    async def _early_exit(state: AgentState) -> AgentState:
        """Ensure a partial report is generated before ending early."""
        if not state.get("report_markdown"):
            return AgentState(**{**state, "report_markdown": _partial_report(state, "early_exit")})
        return state

    graph = StateGraph(AgentState)
    graph.add_node("researcher", _researcher)
    graph.add_node("quant_analyst", _quant_analyst)
    graph.add_node("editor", _editor)
    graph.add_node("early_exit", _early_exit)

    graph.add_edge(START, "researcher")
    graph.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {"quant_analyst": "quant_analyst", "early_exit": "early_exit"},
    )
    graph.add_conditional_edges(
        "quant_analyst",
        _route_after_quant,
        {"editor": "editor", "early_exit": "early_exit"},
    )
    graph.add_edge("editor", END)
    graph.add_edge("early_exit", END)
    return graph


async def _safe_node(
    name: str,
    node_fn,
    state: AgentState,
    config: dict,
    tracer: RunTracer,
) -> AgentState:
    """Wrap a node to catch known failure modes and emit partial output.

    Transient errors (generic Exception) are retried up to
    settings.pipeline_node_max_retries times with linear backoff.
    PartialStateError, CircuitBreakerError, and SanitizationAlert are never
    retried — they represent permanent or rate-limit failures.
    """
    max_retries = settings.pipeline_node_max_retries
    retry_delay = settings.pipeline_node_retry_delay_s

    for attempt in range(max_retries + 1):
        try:
            result = await node_fn(state, config=config)
            _check_intermediate_canary(name, result)
            return result
        except SanitizationAlert as exc:
            logger.critical("SECURITY ALERT in node %s: %s", name, exc)
            tracer.set_status(RunStatus.FAILED)
            errors = list(state.get("errors", []))
            errors.append({"error_type": "SECURITY", "agent": name, "detail": str(exc)})
            report = state.get("report_markdown") or _partial_report(state, name)
            return AgentState(**{**state, "errors": errors, "status": "FAILED", "report_markdown": report})
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
            return AgentState(**{**state, "errors": errors, "status": "PARTIAL"})
        except CircuitBreakerError as exc:
            logger.critical("Circuit breaker tripped in %s: %s", name, exc)
            tracer.set_status(RunStatus.RATE_LIMITED)
            errors = list(state.get("errors", []))
            errors.append({"error_type": "RATE_LIMIT", "agent": name, "detail": str(exc)})
            report = state.get("report_markdown") or _partial_report(state, name)
            return AgentState(**{**state, "errors": errors, "status": "RATE_LIMITED", "report_markdown": report})
        except Exception as exc:
            if attempt < max_retries:
                delay = retry_delay * (attempt + 1)
                logger.warning(
                    "Transient error in %s (attempt %d/%d), retrying in %.1fs: %s",
                    name, attempt + 1, max_retries + 1, delay, exc,
                )
                await asyncio.sleep(delay)
                continue
            logger.exception(
                "Unexpected error in node %s (failed after %d attempt(s))", name, attempt + 1
            )
            tracer.set_status(RunStatus.FAILED)
            errors = list(state.get("errors", []))
            errors.append({"error_type": "UNKNOWN", "agent": name, "detail": str(exc)})
            return AgentState(**{**state, "errors": errors, "status": "FAILED"})


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


def _get_checkpointer_ctx(sqlite_path: str):
    """Return the appropriate async checkpointer context manager.

    Uses AsyncPostgresSaver when DATABASE_URL is set and the optional package
    is installed (pip install langgraph-checkpoint-postgres).
    Falls back to AsyncSqliteSaver with a warning if the package is missing.
    """
    if settings.database_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore
            return AsyncPostgresSaver.from_conn_string(settings.database_url)
        except ImportError:
            logger.warning(
                "DATABASE_URL is set but 'langgraph-checkpoint-postgres' is not installed. "
                "Falling back to SQLite checkpointer. "
                "Install with: pip install langgraph-checkpoint-postgres"
            )
    return AsyncSqliteSaver.from_conn_string(sqlite_path)


def run_pipeline_from_tool(
    query: str,
    tickers: list[str],
    step_callback: Any | None = None,
) -> tuple[AgentState, str, str]:
    """Synchronous wrapper around run_pipeline for use inside LangChain @tool functions.

    Calls asyncio.run(), so it must NOT be invoked from within an already-running
    event loop. Use `await run_pipeline(...)` directly from async contexts instead.
    """
    return asyncio.run(run_pipeline(query=query, tickers=tickers, step_callback=step_callback))


async def run_pipeline(
    query: str,
    tickers: list[str],
    dry_run: bool = False,
    trace_output_dir: str = _ARTIFACTS_DIR,
    step_callback: Any | None = None,
) -> tuple[AgentState, str, str]:
    """High-level entry point: run the full pipeline and export the trace.

    AsyncSqliteSaver must wrap the ainvoke() call because its context manager
    owns the aiosqlite connection lifetime.

    Returns:
        (final_state, trace_path)
    """
    budget = RequestBudgetTracker()
    primary_llm = get_primary_llm_with_fallback(budget_tracker=budget)
    subllm = get_subllm(budget_tracker=budget)

    web_search_module.configure(subllm=subllm)

    tracer = RunTracer()
    run_id = str(uuid.uuid4())
    artifacts = RunArtifacts(run_id=run_id, tickers=[t.strip().upper() for t in tickers])
    node_config = {
        "primary_llm": primary_llm,
        "subllm": subllm,
        "tracer": tracer,
        "budget": budget,
        "artifacts": artifacts,
        "step_callback": step_callback,
    }

    graph = _build_graph(node_config)

    initial_state = AgentState(
        query=query,
        tickers=[t.strip().upper() for t in tickers],
        iteration_log=[],
        errors=[],
        status="COMPLETE",
        run_id=run_id,
    )
    thread_config = {"configurable": {"thread_id": run_id}}

    if dry_run:
        app = graph.compile(checkpointer=MemorySaver())
        final_state = await app.ainvoke(initial_state, config=thread_config)
    else:
        db_path = Path(_CHECKPOINT_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        async with _get_checkpointer_ctx(str(db_path)) as checkpointer:
            app = graph.compile(checkpointer=checkpointer)
            final_state = await app.ainvoke(initial_state, config=thread_config)

    _clear_old_artifacts(trace_output_dir)
    trace_path = tracer.export(output_dir=trace_output_dir)
    tracer.build(budget_stats=budget.get_stats())

    artifacts.set_report(final_state.get("report_markdown", ""))
    artifacts_path = artifacts.save(output_dir=trace_output_dir)

    return final_state, str(trace_path), str(artifacts_path)
