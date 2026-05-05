"""Shared state contract for the LangGraph multi-agent pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class IterationLogEntry(TypedDict):
    step: int
    agent: str
    tool: str
    input: dict[str, Any]
    output_tokens: int
    cache_hit: bool


class DataCoverage(TypedDict):
    ticker: str
    price_history: bool
    fundamentals: bool
    balance_sheet: bool
    news_search: bool
    data_gaps: list[str]


class AgentState(TypedDict, total=False):
    # --- Input ---
    query: str                          # populated_by: user
    tickers: list[str]                  # populated_by: orchestrator

    # --- Researcher output ---
    raw_data: dict[str, Any]            # populated_by: researcher
    data_coverage: list[DataCoverage]   # populated_by: researcher
    researcher_gaps: list[str]          # populated_by: researcher

    # --- Quant Analyst output ---
    analysis: dict[str, Any]            # populated_by: quant_analyst

    # --- Editor output ---
    report_markdown: str                # populated_by: editor
    sop_checklist: dict[str, bool]      # populated_by: editor

    # --- Shared diagnostics ---
    iteration_log: list[IterationLogEntry]   # appended by all agents
    errors: list[dict[str, Any]]             # appended by all agents
    status: str                              # COMPLETE | PARTIAL | RATE_LIMITED | FAILED
    run_id: str                              # populated_by: orchestrator


# Required fields each agent must receive before it can run.
RESEARCHER_REQUIRED: tuple[str, ...] = ("query", "tickers")
QUANT_REQUIRED: tuple[str, ...] = ("raw_data",)
EDITOR_REQUIRED: tuple[str, ...] = ("analysis",)


class PartialStateError(RuntimeError):
    """Raised when an agent boundary validation check fails.

    Carries the names of the missing fields so the orchestrator can build
    a structured PartialStateError entry in the run trace.
    """

    def __init__(self, agent: str, missing_fields: list[str]) -> None:
        self.agent = agent
        self.missing_fields = missing_fields
        super().__init__(
            f"{agent} cannot proceed: missing required state fields: {missing_fields}"
        )


def validate_state_for_agent(state: AgentState, agent: str) -> None:
    """Raise PartialStateError if required fields are absent or empty."""
    required_map = {
        "quant_analyst": QUANT_REQUIRED,
        "editor": EDITOR_REQUIRED,
    }
    required = required_map.get(agent, ())
    missing = [f for f in required if not state.get(f)]
    if missing:
        raise PartialStateError(agent, missing)
