"""Unit tests for orchestrator conditional routing functions.

Tests _route_after_researcher and _route_after_quant to verify that:
- Empty raw_data routes to early_exit
- Error-only raw_data routes to early_exit
- Valid raw_data routes to quant_analyst
- RATE_LIMITED / FAILED status always routes to early_exit
- Missing analysis routes to early_exit

Also tests _check_intermediate_canary:
- Canary in quant_analyst or editor output raises SanitizationAlert
- Researcher output is NOT canary-checked (structured API data)
- Clean outputs pass silently
"""

from __future__ import annotations

import pytest

from unittest.mock import AsyncMock, MagicMock, patch

from ai_financial_analyst.agents.orchestrator import (
    _check_intermediate_canary,
    _get_checkpointer_ctx,
    _route_after_quant,
    _route_after_researcher,
    _safe_node,
)
from ai_financial_analyst.core.llm import CircuitBreakerError
from ai_financial_analyst.core.sanitizer import CANARY_TOKEN, SanitizationAlert
from ai_financial_analyst.core.state import AgentState, PartialStateError
from ai_financial_analyst.core.tracing import RunTracer


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


class TestIntermediateCanaryCheck:
    """_check_intermediate_canary guards LLM-generated narrative fields per node."""

    def test_canary_in_quant_analysis_raises(self):
        """Canary appearing in quant_analyst analysis dict should raise SanitizationAlert."""
        state = _base_state(analysis={
            "AAPL": {"bull_case": f"Strong growth {CANARY_TOKEN}", "bear_case": "Competition risk"}
        })
        with pytest.raises(SanitizationAlert):
            _check_intermediate_canary("quant_analyst", state)

    def test_canary_in_report_markdown_raises(self):
        """Canary appearing in editor report_markdown should raise SanitizationAlert."""
        state = _base_state(report_markdown=f"# AAPL Report\n\n{CANARY_TOKEN}\n\nThis is not financial advice.")
        with pytest.raises(SanitizationAlert):
            _check_intermediate_canary("editor", state)

    def test_clean_quant_output_passes(self):
        """Normal analysis dict without the canary should pass silently."""
        state = _base_state(analysis={
            "AAPL": {"bull_case": "Revenue growth strong", "bear_case": "Margin pressure"}
        })
        _check_intermediate_canary("quant_analyst", state)  # must not raise

    def test_researcher_output_not_canary_checked(self):
        """Researcher state is excluded — structured API data, not LLM narratives.
        Even if raw_data somehow contained the canary string, no alert is raised."""
        state = _base_state(raw_data={
            "AAPL": {"fundamentals": {"company_description": f"Tech company {CANARY_TOKEN}"}}
        })
        _check_intermediate_canary("researcher", state)  # must not raise

    def test_none_field_skipped(self):
        """None analysis field should not raise."""
        state = _base_state(analysis=None)
        _check_intermediate_canary("quant_analyst", state)  # must not raise

    def test_unknown_node_is_noop(self):
        """Nodes not in _LLM_NARRATIVE_FIELDS are silently skipped."""
        state = _base_state(report_markdown=f"Contains {CANARY_TOKEN}")
        _check_intermediate_canary("early_exit", state)  # must not raise


# ── Node retry ────────────────────────────────────────────────────────────────


class TestNodeRetry:
    """_safe_node retries transient errors; never retries structural/security errors."""

    def _make_tracer(self):
        tracer = MagicMock(spec=RunTracer)
        tracer.set_status = MagicMock()
        return tracer

    @pytest.mark.asyncio
    async def test_node_retries_on_transient_error_then_succeeds(self):
        """node_fn fails once then succeeds → result is not FAILED."""
        good_state = _base_state(status="COMPLETE")
        calls = []

        async def flaky_node(state, config=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient network error")
            return good_state

        with patch("ai_financial_analyst.agents.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            result = await _safe_node("researcher", flaky_node, _base_state(), {}, self._make_tracer())

        assert result["status"] == "COMPLETE"
        assert len(calls) == 2  # one failure + one success

    @pytest.mark.asyncio
    async def test_node_fails_after_all_retries_exhausted(self):
        """node_fn always fails → returns FAILED state after max_retries+1 attempts."""
        async def always_fails(state, config=None):
            raise RuntimeError("persistent error")

        with patch("ai_financial_analyst.agents.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            with patch("ai_financial_analyst.agents.orchestrator.settings") as mock_settings:
                mock_settings.pipeline_node_max_retries   = 2
                mock_settings.pipeline_node_retry_delay_s = 0.0
                result = await _safe_node("researcher", always_fails, _base_state(), {}, self._make_tracer())

        assert result["status"] == "FAILED"
        assert any(e["error_type"] == "UNKNOWN" for e in result.get("errors", []))

    @pytest.mark.asyncio
    async def test_partial_state_error_not_retried(self):
        """PartialStateError is a structural failure — never retried."""
        call_count = 0

        async def raises_partial(state, config=None):
            nonlocal call_count
            call_count += 1
            raise PartialStateError("quant_analyst", ["analysis"])

        result = await _safe_node("quant_analyst", raises_partial, _base_state(), {}, self._make_tracer())

        assert call_count == 1
        assert result["status"] == "PARTIAL"

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_not_retried(self):
        """CircuitBreakerError is a rate-limit failure — never retried."""
        call_count = 0

        async def raises_cb(state, config=None):
            nonlocal call_count
            call_count += 1
            raise CircuitBreakerError("rate limited")

        result = await _safe_node("researcher", raises_cb, _base_state(), {}, self._make_tracer())

        assert call_count == 1
        assert result["status"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_max_retries_zero_disables_retry(self):
        """PIPELINE_NODE_MAX_RETRIES=0 → single attempt, no retry on failure."""
        call_count = 0

        async def always_fails(state, config=None):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("error")

        with patch("ai_financial_analyst.agents.orchestrator.settings") as mock_settings:
            mock_settings.pipeline_node_max_retries   = 0
            mock_settings.pipeline_node_retry_delay_s = 0.0
            result = await _safe_node("editor", always_fails, _base_state(), {}, self._make_tracer())

        assert call_count == 1
        assert result["status"] == "FAILED"


# ── Checkpointer selection ────────────────────────────────────────────────────


class TestCheckpointerSelection:
    """_get_checkpointer_ctx delegates to the right checkpointer based on settings."""

    def test_uses_sqlite_when_no_database_url(self, tmp_path):
        """Without DATABASE_URL, AsyncSqliteSaver.from_conn_string is called."""
        with patch("ai_financial_analyst.agents.orchestrator.settings") as mock_settings:
            mock_settings.database_url = None
            with patch(
                "ai_financial_analyst.agents.orchestrator.AsyncSqliteSaver.from_conn_string"
            ) as mock_sqlite:
                _get_checkpointer_ctx(str(tmp_path / "test.db"))
                mock_sqlite.assert_called_once()

    def test_falls_back_to_sqlite_when_postgres_package_missing(self, tmp_path, caplog):
        """ImportError on postgres package → falls back to SQLite with warning logged."""
        import logging

        with patch("ai_financial_analyst.agents.orchestrator.settings") as mock_settings:
            mock_settings.database_url = "postgresql://user:pass@localhost/db"
            # Simulate langgraph-checkpoint-postgres not installed
            with patch.dict("sys.modules", {"langgraph.checkpoint.postgres.aio": None}):
                with patch(
                    "ai_financial_analyst.agents.orchestrator.AsyncSqliteSaver.from_conn_string"
                ) as mock_sqlite:
                    with caplog.at_level(logging.WARNING):
                        _get_checkpointer_ctx(str(tmp_path / "test.db"))
                    mock_sqlite.assert_called_once()

        assert "langgraph-checkpoint-postgres" in caplog.text
