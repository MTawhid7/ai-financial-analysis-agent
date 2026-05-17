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
from ..tools.report_writer import _DISCLAIMER, write_report

logger = logging.getLogger(__name__)

# SOP rubric weights — values are (label, weight); weights sum to 1.0.
# All-or-nothing binary scoring overly penalised partial-but-useful analyses
# (e.g. missing the narrative bear_case fails a report that has all metrics).
# A weighted score >= _SOP_PASS_THRESHOLD is now required for a report to pass.
_SOP_WEIGHTS: dict[str, tuple[str, float]] = {
    "price_cagr_5y_pct": ("5-year price CAGR",    0.30),
    "sector_pe_avg":     ("Sector P/E benchmark", 0.25),
    "company_pe":        ("Company P/E ratio",     0.20),
    "bull_case":         ("Bull case",             0.15),
    "bear_case":         ("Bear case",             0.10),
}
_SOP_PASS_THRESHOLD = 0.60

# Regex to detect numeric claims in the draft report.
# Handles: $1,234 (commas), (4,200) (accounting negatives),
# $4.17T (SI suffixes K/M/B/T), plain integers and decimals with %.
_NUMERIC_PATTERN = re.compile(
    r"""
    (?:
        \(\s*\$?[\d,]+(?:\.\d+)?\s*[KMBTkmbt]?\s*\)   # (4,200) or ($1.2B) — accounting negative
        |
        \$?[\d,]+(?:\.\d+)?[KMBTkmbt]?%?               # $1,234 or 4.17T or 27.2%
    )
    """,
    re.VERBOSE,
)

_SI_SUFFIX: dict[str, float] = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


async def editor_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Editor agent."""
    validate_state_for_agent(state, "editor")

    tracer: RunTracer | None = config.get("tracer") if config else None
    artifacts = config.get("artifacts") if config else None
    step_callback = config.get("step_callback") if config else None
    primary_llm = config.get("primary_llm") if config else None
    subllm = config.get("subllm") if config else None
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]] = list(state.get("errors", []))

    analysis = state["analysis"]
    tickers = state.get("tickers", list(analysis.keys()))
    data_coverage = state.get("data_coverage", [])
    researcher_gaps = state.get("researcher_gaps", [])

    if tracer:
        tracer.record_agent_start("editor", {"tickers": tickers})

    sop_checklist: dict[str, bool] = {}
    ticker_scores: dict[str, float] = {}

    for ticker, ticker_analysis in analysis.items():
        score = 0.0
        for key, (label, weight) in _SOP_WEIGHTS.items():
            present = bool(ticker_analysis.get(key))
            sop_checklist[f"{ticker}/{key}"] = present
            if present:
                score += weight
            else:
                logger.warning(
                    "SOP miss for %s: '%s' absent (weight %.2f)", ticker, label, weight
                )
        ticker_scores[ticker] = round(score, 3)
        logger.info(
            "SOP score for %s: %.2f (threshold %.2f)", ticker, score, _SOP_PASS_THRESHOLD
        )

    all_tickers_pass = all(s >= _SOP_PASS_THRESHOLD for s in ticker_scores.values())
    sop_score_min = min(ticker_scores.values(), default=0.0)

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
        "sop_score_min": sop_score_min,
    }

    # --- Invoke ReportWriterTool ---
    # No truncation: Gemini Flash has 1M token context; truncating loses later tickers.
    report_input = {
        "analysis_json": json.dumps(analysis_with_coverage, indent=2),
        "tickers": tickers,
        "data_gaps": researcher_gaps,
    }

    step = len(iteration_log) + 1
    report_str = await write_report(
        analysis_json=report_input["analysis_json"],
        tickers=report_input["tickers"],
        data_gaps=report_input["data_gaps"],
        primary_llm=primary_llm,
        subllm=subllm,
    )

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

    # --- Enforce disclaimer (secondary guard — report_writer already appends it) ---
    if "This is not financial advice" not in report_markdown:
        report_markdown += _DISCLAIMER

    if tracer:
        tracer.record_agent_complete("editor", {
            "sop_passed": all_tickers_pass,
            "sop_score_min": sop_score_min,
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


def _clean_numeric(s: str) -> tuple[float | None, bool]:
    """Parse a financial numeric string to (float_value, is_negative).

    Handles: $1,234 commas, (4,200) accounting negatives, SI suffixes K/M/B/T,
    dollar signs, percent signs.  Returns (None, False) if unparseable.
    """
    orig = s.strip()
    is_neg = orig.startswith("(") and orig.endswith(")")
    # Strip outer parens, $, leading/trailing whitespace
    cleaned = orig.replace(",", "").lstrip("($").rstrip("%)").strip()
    # Detect SI suffix
    suffix: float | None = None
    if cleaned and cleaned[-1].lower() in _SI_SUFFIX:
        suffix = _SI_SUFFIX[cleaned[-1].lower()]
        cleaned = cleaned[:-1]
    # Strip trailing % for conversion
    cleaned = cleaned.rstrip("%")
    try:
        val = float(cleaned)
        if suffix is not None:
            val *= suffix
        if is_neg:
            val = -val
        return val, is_neg
    except (ValueError, AttributeError):
        return None, False


def _parse_grounded_floats(grounded_values: set[str]) -> list[float]:
    """Convert grounded numeric strings to floats for scaled comparison."""
    result = []
    for s in grounded_values:
        val, _ = _clean_numeric(s)
        if val is not None:
            result.append(val)
    return result


def _is_grounded_by_scale(val_str: str, grounded_floats: list[float]) -> bool:
    """Return True if val_str is within tolerance of any grounded float under SI scaling.

    Tolerance is tiered:
    - Percentage values (%) : 2% tolerance (rounding common in analyst copy)
    - Large absolute values (>1e6): 5% tolerance (T/B rounding)
    - Everything else: 1.5% tolerance
    """
    val, _ = _clean_numeric(val_str)
    if val is None:
        return True  # unparseable — don't flag
    if val == 0:
        return True

    is_pct = "%" in val_str
    tol = 0.02 if is_pct else (0.05 if abs(val) > 1e6 else 0.015)

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
        # Only flag values that look like financial figures: have $ or decimal, or >3 digits,
        # or use SI suffix, or are wrapped in parentheses.
        stripped = val.replace(",", "").replace("$", "").replace("%", "")
        stripped = stripped.strip("()").rstrip("KMBTkmbt")
        is_financial = (
            "$" in val
            or "." in val
            or len(stripped) > 3
            or val.endswith(tuple("KMBTkmbt"))
            or (val.startswith("(") and val.endswith(")"))
        )
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
