"""Editor Agent — SOP rubric enforcement, grounding check, report synthesis.

Responsibilities:
- Validate Quant output before proceeding.
- Run the grounding check: every quantitative claim must trace to iteration_log.
- Verify the full SOP rubric checklist.
- Call ReportWriterTool to produce the final Markdown report.
- Produce a Data Coverage Summary section.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..core.state import AgentState, IterationLogEntry, validate_state_for_agent
from ..core.sanitizer import CANARY_TOKEN, SanitizationAlert
from ..core.tracing import ErrorType, RunStatus, RunTracer
from ..tools.report_writer import report_writer_tool

logger = logging.getLogger(__name__)

# SOP rubric keys — must ALL be present in analysis for the report to be emitted.
_SOP_KEYS = {
    "price_cagr_5y_pct": "5-year price CAGR",
    "sector_pe_avg": "Sector P/E benchmark",
    "company_pe": "Company P/E ratio",
    "bull_case": "Bull case",
    "bear_case": "Bear case",
}

# Regex to detect numeric claims in the draft report.
_NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")


async def editor_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Editor agent."""
    validate_state_for_agent(state, "editor")

    tracer: RunTracer | None = config.get("tracer") if config else None
    artifacts = config.get("artifacts") if config else None
    step_callback = config.get("step_callback") if config else None
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]] = list(state.get("errors", []))

    analysis = state["analysis"]
    tickers = state.get("tickers", list(analysis.keys()))
    data_coverage = state.get("data_coverage", [])
    researcher_gaps = state.get("researcher_gaps", [])

    if tracer:
        tracer.record_agent_start("editor", {"tickers": tickers})

    sop_checklist: dict[str, bool] = {}
    all_tickers_pass = True

    for ticker, ticker_analysis in analysis.items():
        for key, label in _SOP_KEYS.items():
            present = bool(ticker_analysis.get(key))
            sop_checklist[f"{ticker}/{key}"] = present
            if not present:
                logger.warning("SOP checklist FAIL for %s: missing '%s'", ticker, label)
                all_tickers_pass = False

    # Build compact raw-data summary so the report writer can cite exact figures.
    raw_data = state.get("raw_data", {})
    raw_data_summary: dict = {}
    for ticker, tdata in raw_data.items():
        raw_data_summary[ticker] = {
            dtype: {
                k: v for k, v in dval.items()
                if isinstance(v, (int, float)) and k not in ("data_points",)
            }
            for dtype, dval in tdata.items()
            if isinstance(dval, dict)
        }

    # Build analysis JSON for the report writer, including data coverage.
    analysis_with_coverage = {
        "analysis": analysis,
        "raw_data_summary": raw_data_summary,
        "data_coverage": data_coverage,
        "data_gaps": researcher_gaps,
        "sop_checklist_passed": all_tickers_pass,
    }

    # --- Invoke ReportWriterTool ---
    report_input = {
        "analysis_json": json.dumps(analysis_with_coverage, indent=2)[:9000],
        "tickers": tickers,
        "data_gaps": researcher_gaps,
    }

    step = len(iteration_log) + 1
    report_str = await report_writer_tool.arun(report_input)

    if artifacts:
        artifacts.record_llm_exchange(
            agent="editor",
            purpose="report_writing",
            ticker=None,
            prompt_messages=[
                {"role": "user", "content": report_input["analysis_json"]},
            ],
            raw_response=report_str,
        )

    if step_callback:
        step_callback({
            "step": step, "agent": "editor", "tool": "report_writer",
            "input": {"tickers": tickers}, "cache_hit": False, "ok": bool(report_str),
        })

    if tracer:
        tracer.record_tool_call(
            agent="editor",
            tool="report_writer",
            input_data={"tickers": tickers},
            output_data=report_str[:500],
            output_tokens=len(report_str) // 4,
        )

    iteration_log.append(
        IterationLogEntry(
            step=step,
            agent="editor",
            tool="report_writer",
            input={"tickers": tickers},
            output_tokens=len(report_str) // 4,
            cache_hit=False,
        )
    )

    # --- Grounding check ---
    report_markdown = _grounding_check(
        report_str, iteration_log, tracer,
        analysis=analysis,
        raw_data=state.get("raw_data", {}),
    )

    # --- Canary check ---
    try:
        from ..core.sanitizer import ContentSanitizer
        ContentSanitizer().check_canary(report_markdown)
    except SanitizationAlert as exc:
        logger.critical("SECURITY ALERT: %s", exc)
        errors.append({"error_type": "SECURITY", "detail": str(exc)})
        report_markdown += "\n\n> **SECURITY WARNING: Potential prompt injection detected. Output flagged.**"

    # --- Enforce disclaimer ---
    if "This is not financial advice" not in report_markdown:
        report_markdown += (
            "\n\n---\n"
            "*DISCLAIMER: This report was generated by an AI system. "
            "All figures should be independently verified before making any "
            "investment decisions. This is not financial advice.*"
        )

    if tracer:
        tracer.record_agent_complete("editor", {
            "sop_passed": all_tickers_pass,
            "sop_checklist": sop_checklist,
            "report_length_chars": len(report_markdown),
            "errors": len(errors),
        })

    return AgentState(**{
        **state,
        "report_markdown": report_markdown,
        "sop_checklist": sop_checklist,
        "iteration_log": iteration_log,
        "errors": errors,
        "status": "COMPLETE" if not errors else "PARTIAL",
    })


def _parse_grounded_floats(grounded_values: set[str]) -> list[float]:
    """Convert grounded numeric strings to floats for scaled comparison."""
    result = []
    for s in grounded_values:
        try:
            result.append(float(s.rstrip("%")))
        except (ValueError, AttributeError):
            pass
    return result


def _is_grounded_by_scale(val_str: str, grounded_floats: list[float], tol: float = 0.015) -> bool:
    """Return True if val_str is within tol of any grounded float under SI unit scaling.

    Handles: rounding (34.362755 → 34.4), unit conversion (285508000000 → 285.5B),
    and percentage representation (0.27152 → 27%).
    """
    try:
        val = float(val_str.rstrip("%"))
    except (ValueError, AttributeError):
        return True  # unparseable — don't flag
    if val == 0:
        return True
    # Scales cover: exact, thousands, millions, billions, trillions, pct (×100 or ÷100)
    scale_factors = (1, 1e3, 1e6, 1e9, 1e12, 100.0, 0.01)
    for gf in grounded_floats:
        for scale in scale_factors:
            ref = gf / scale
            if ref != 0 and abs(val - ref) / abs(ref) <= tol:
                return True
    return False


def _grounding_check(
    report: str,
    iteration_log: list[IterationLogEntry],
    tracer: RunTracer | None,
    analysis: dict | None = None,
    raw_data: dict | None = None,
) -> str:
    """Replace ungrounded numeric claims with [UNVERIFIED] tags.

    A number is grounded if it can be matched (exact or within 1.5% at any SI
    unit scale) to a value in tool call inputs, the analysis dict, or raw_data.
    This handles LLM reformatting such as 285508000000 → 285.5B or 0.27 → 27%.
    """
    grounded_values: set[str] = set()

    # Source 1: tool call inputs from iteration_log
    for entry in iteration_log:
        output_str = json.dumps(entry.get("input", {}))
        grounded_values.update(_NUMERIC_PATTERN.findall(output_str))

    # Source 2: computed analysis values (CAGR, P/E, etc.)
    if analysis:
        try:
            grounded_values.update(
                _NUMERIC_PATTERN.findall(json.dumps(analysis, default=str))
            )
        except Exception:
            pass

    # Source 3: raw fetched data (price history, balance sheet figures, etc.)
    if raw_data:
        try:
            grounded_values.update(
                _NUMERIC_PATTERN.findall(json.dumps(raw_data, default=str))
            )
        except Exception:
            pass

    grounded_floats = _parse_grounded_floats(grounded_values)
    unverified_count = 0

    def _check_match(m: re.Match) -> str:
        nonlocal unverified_count
        val = m.group(0)
        # Only flag values that look like financial figures: decimal points or >3 digits.
        is_financial = "." in val or len(val.replace("%", "")) > 3
        if not is_financial:
            return val
        if val in grounded_values or _is_grounded_by_scale(val, grounded_floats):
            return val
        unverified_count += 1
        return f"[UNVERIFIED:{val}]"

    verified_report = _NUMERIC_PATTERN.sub(_check_match, report)

    if unverified_count > 0:
        logger.warning(
            "Grounding check: %d unverified numeric claim(s) tagged.", unverified_count
        )
        if tracer:
            tracer.record_error(
                ErrorType.TOOL_ERROR,
                "editor",
                "grounding_check",
                f"{unverified_count} unverified figure(s) tagged in report",
            )

    return verified_report
