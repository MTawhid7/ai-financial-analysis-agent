"""Unit tests for orchestrator conditional routing functions.

Tests _route_after_researcher and _route_after_quant to verify that:
- Empty raw_data routes to early_exit
- Error-only raw_data routes to early_exit
- Valid raw_data routes to quant_analyst
- RATE_LIMITED / FAILED status always routes to early_exit
- Missing analysis routes to early_exit
"""

from __future__ import annotations

import pytest

from ai_financial_analyst.agents.orchestrator import (
    _route_after_researcher,
    _route_after_quant,
)
from ai_financial_analyst.core.state import AgentState


def _base_state(**overrides) -> AgentState:
    base = {
        "query": "Analyse AAPL",
        "tickers": ["AAPL"],
        "raw_data": {},
        "data_coverage": [],
        "researcher_gaps": [],
        "analysis": {},
        "report_markdown": None,
        "sop_checklist": {},
        "iteration_log": [],
        "errors": [],
        "status": "COMPLETE",
        "run_id": "test-run",
    }
    base.update(overrides)
    return AgentState(**base)


class TestRouteAfterResearcher:
    def test_empty_raw_data_routes_to_early_exit(self):
        state = _base_state(raw_data={})
        assert _route_after_researcher(state) == "early_exit"

    def test_rate_limited_routes_to_early_exit(self):
        state = _base_state(
            raw_data={"AAPL": {"price_history": {"current_price": 200}}},
            status="RATE_LIMITED",
        )
        assert _route_after_researcher(state) == "early_exit"

    def test_failed_routes_to_early_exit(self):
        state = _base_state(
            raw_data={"AAPL": {"price_history": {"current_price": 200}}},
            status="FAILED",
        )
        assert _route_after_researcher(state) == "early_exit"

    def test_error_only_raw_data_routes_to_early_exit(self):
        """Tickers with only error/message/reason keys have no usable data."""
        state = _base_state(raw_data={
            "AAPL": {"error_type": "TOOL_ERROR", "message": "network failure"}
        })
        assert _route_after_researcher(state) == "early_exit"

    def test_valid_data_routes_to_quant_analyst(self):
        state = _base_state(raw_data={
            "AAPL": {"price_history": {"current_price": 200.0}, "fundamentals": {"pe_ratio": 28.5}}
        })
        assert _route_after_researcher(state) == "quant_analyst"

    def test_partial_data_with_some_valid_routes_to_quant(self):
        """If even one ticker has valid data, proceed to quant."""
        state = _base_state(raw_data={
            "AAPL": {"price_history": {"current_price": 200.0}},
            "FAKE": {"error_type": "TOOL_ERROR", "message": "not found"},
        })
        assert _route_after_researcher(state) == "quant_analyst"


class TestRouteAfterQuant:
    def test_empty_analysis_routes_to_early_exit(self):
        state = _base_state(analysis={})
        assert _route_after_quant(state) == "early_exit"

    def test_none_analysis_routes_to_early_exit(self):
        state = _base_state(analysis=None)
        assert _route_after_quant(state) == "early_exit"

    def test_rate_limited_routes_to_early_exit(self):
        state = _base_state(
            analysis={"AAPL": {"bull_case": ["strong"], "bear_case": ["risk"]}},
            status="RATE_LIMITED",
        )
        assert _route_after_quant(state) == "early_exit"

    def test_valid_analysis_routes_to_editor(self):
        state = _base_state(analysis={
            "AAPL": {"bull_case": ["strong revenue"], "bear_case": ["competition"]}
        })
        assert _route_after_quant(state) == "editor"
