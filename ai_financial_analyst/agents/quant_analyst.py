"""Quant Analyst Agent — computation, valuation, and scenario analysis node.

Computes all quantitative metrics deterministically in Python before invoking
the LLM. The LLM's role is narrowly scoped to narrative commentary on the
computed numbers — it does not invent price targets or scenarios.

Pipeline per ticker:
  1. Total-return CAGR (from price_metrics)
  2. Risk metrics: Sharpe, Sortino, max drawdown, beta, volatility
  3. FCF yield and cash metrics
  4. Next earnings date
  5. Sector benchmark comparison (P/E, EV/EBITDA, P/B) — raw yfinance sector
     string passed directly; benchmark_lookup resolves via alias + fuzzy match
  6. Key profitability metrics
  7. Simplified DCF valuation (pure Python — Option A for negative FCF)
  8. Price-target scenario analysis: P/E × EPS + analyst consensus + DCF
  9. LLM narration: writes qualitative commentary on the computed scenarios
     using with_structured_output (Gemini JSON mode enforces schema)
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from ..core.state import (
    AgentState,
    IterationLogEntry,
    validate_state_for_agent,
)
from ..core.tracing import ErrorType, RunTracer
from ..tools.calculator import calculator_tool
from ..tools.benchmark_lookup import benchmark_lookup_tool

logger = logging.getLogger(__name__)

# ── DCF constants ─────────────────────────────────────────────────────────────
_ERP             = 0.055   # Equity Risk Premium — Damodaran 2024 US estimate
_CREDIT_SPREAD   = 0.020   # Cost-of-debt premium over risk-free (rough IG proxy)
_TAX_RATE        = 0.21    # US federal corporate tax rate
_TERMINAL_GROWTH = 0.025   # Perpetual FCF growth — conservative GDP rate
_DCF_YEARS       = 5       # Projection horizon
_MAX_FCF_GROWTH  = 0.25    # Cap on FCF growth assumption (25%)


# ── Structured output schema for LLM narration ────────────────────────────────

class _SOPOutput(BaseModel):
    """Schema enforced by Gemini JSON mode for the narrative commentary step."""
    bull_case:    list[str] = Field(description="2-3 data-anchored bull-case drivers")
    bear_case:    list[str] = Field(description="2-3 data-anchored risk factors")
    closest_peer: str | None = Field(None, description='Most comparable public company e.g. "Apple Inc. (AAPL)"')


_SOP_SYSTEM = """You are a quantitative financial analyst writing narrative commentary for a data-driven investment brief.

Price-target scenarios, a DCF estimate, and financial metrics have already been computed from real market data.
Your task: write concise qualitative commentary explaining the drivers and risks behind the computed targets.

Rules:
- bull_case: 2-3 bullets. Each must cite a specific computed metric (e.g. Sharpe 1.4, FCF yield 4.2%, 30% above sector P/E) and name the qualitative CATALYST that could push the stock toward its bull-case target.
- bear_case: 2-3 bullets. Each must reference a specific risk signal (e.g. max drawdown -42%, beta 1.8, D/E 1.5, 20% overvalued by DCF) and explain what could drive the stock toward its bear-case target.
- closest_peer: the single most comparable publicly-traded company.
- Do NOT invent price targets — they are already computed and shown below.
- Do NOT repeat the scenario numbers verbatim; explain the qualitative logic behind them.
- Output only the JSON matching the schema. No markdown, no prose outside the JSON."""


# ── DCF computation (pure Python) ─────────────────────────────────────────────

def _compute_dcf(
    fcf: float | None,
    revenue_growth_raw: float | None,
    beta: float | None,
    risk_free_rate_pct: float | None,
    total_debt: float | None,
    cash: float | None,
    market_cap: float | None,
    current_price: float | None,
) -> dict:
    """Simplified 5-year DCF valuation using data already fetched by the pipeline.

    Option A: returns {dcf_not_applicable, reason} for negative/zero FCF
    so the agent correctly labels the company as being in reinvestment phase.

    Returns a dict with intrinsic_value_per_share, wacc_pct, margin_of_safety_pct,
    projected FCFs, and explicit assumptions — or a not-applicable sentinel.
    """
    # ── Guard: negative FCF (Option A) ───────────────────────────────────────
    if fcf is None or fcf <= 0:
        return {
            "dcf_not_applicable": True,
            "reason": (
                "FCF is negative or unavailable — company appears to be in "
                "reinvestment phase. DCF requires positive free cash flow."
            ),
        }

    # ── Guard: required WACC inputs ──────────────────────────────────────────
    if not all([beta, market_cap, current_price]):
        return {
            "dcf_not_applicable": True,
            "reason": "Insufficient data for WACC (missing beta, market cap, or price).",
        }
    if current_price <= 0 or market_cap <= 0:
        return {
            "dcf_not_applicable": True,
            "reason": "Invalid current price or market cap.",
        }

    # ── WACC ─────────────────────────────────────────────────────────────────
    rf   = (risk_free_rate_pct or 4.2) / 100          # default 4.2% if unavailable
    ke   = rf + float(beta) * _ERP                     # CAPM cost of equity
    kd   = rf + _CREDIT_SPREAD                         # cost of debt (IG proxy)
    debt = float(total_debt or 0.0)
    cash_amt = float(cash or 0.0)

    total_capital = float(market_cap) + debt
    we   = float(market_cap) / total_capital
    wd   = debt / total_capital
    wacc = we * ke + wd * kd * (1 - _TAX_RATE)

    if wacc <= _TERMINAL_GROWTH:
        return {
            "dcf_not_applicable": True,
            "reason": (
                f"WACC ({wacc:.1%}) ≤ terminal growth rate ({_TERMINAL_GROWTH:.1%}) — "
                "terminal value formula undefined."
            ),
        }

    # ── FCF growth rate (clamp to plausible range) ────────────────────────────
    raw_g    = float(revenue_growth_raw) if revenue_growth_raw is not None else 0.05
    fcf_g    = max(0.0, min(_MAX_FCF_GROWTH, raw_g))

    # ── 5-year FCF projection ─────────────────────────────────────────────────
    projected: list[float] = []
    fcf_t = float(fcf)
    for i in range(1, _DCF_YEARS + 1):
        g     = fcf_g if i <= 3 else fcf_g * 0.5   # mean-revert in years 4-5
        fcf_t = fcf_t * (1 + g)
        projected.append(fcf_t)

    # ── Discount to present value ─────────────────────────────────────────────
    pv_fcfs = sum(f / (1 + wacc) ** t for t, f in enumerate(projected, 1))
    tv      = projected[-1] * (1 + _TERMINAL_GROWTH) / (wacc - _TERMINAL_GROWTH)
    pv_tv   = tv / (1 + wacc) ** _DCF_YEARS

    # ── Enterprise → Equity → Per-share ──────────────────────────────────────
    ev          = pv_fcfs + pv_tv
    equity_val  = ev + cash_amt - debt
    shares      = float(market_cap) / float(current_price)
    intrinsic   = equity_val / shares if shares > 0 else None

    mos = (
        round((intrinsic - current_price) / current_price * 100, 1)
        if intrinsic is not None else None
    )

    return {
        "intrinsic_value_per_share": round(intrinsic, 2) if intrinsic is not None else None,
        "current_price":             round(current_price, 2),
        "margin_of_safety_pct":      mos,
        "wacc_pct":                  round(wacc * 100, 2),
        "fcf_growth_rate_used_pct":  round(fcf_g * 100, 2),
        "terminal_growth_pct":       round(_TERMINAL_GROWTH * 100, 1),
        "fcf_projected_5y":          [round(f) for f in projected],
        "pv_projected_fcfs":         round(pv_fcfs),
        "pv_terminal_value":         round(pv_tv),
        "assumptions": {
            "equity_risk_premium_pct": round(_ERP * 100, 1),
            "cost_of_equity_pct":      round(ke * 100, 2),
            "cost_of_debt_pct":        round(kd * 100, 2),
            "tax_rate_pct":            int(_TAX_RATE * 100),
            "risk_free_rate_pct":      round(rf * 100, 2),
        },
        "warning": (
            "Simplified back-of-envelope DCF with estimated WACC. "
            "Directional reference only — not a precise valuation."
        ),
    }


# ── Scenario analysis (pure Python) ───────────────────────────────────────────

def _compute_scenarios(
    current_price: float | None,
    forward_eps: float | None,
    sector_pe: float | None,
    analyst_low: float | None,
    analyst_mean: float | None,
    analyst_high: float | None,
    dcf_intrinsic: float | None,
) -> dict:
    """Compute bear / base / bull price-target scenarios from three independent methods.

    Returns a dict with 'pe_based', 'analyst_consensus', and 'dcf' sub-dicts.
    Each price target includes an upside_pct vs current price.
    """

    def _upside(target: float | None, price: float | None) -> float | None:
        if price and price > 0 and target is not None:
            return round((target - price) / price * 100, 1)
        return None

    scenarios: dict = {}

    # ── Method 1: Sector P/E × Forward EPS ───────────────────────────────────
    if forward_eps and sector_pe and forward_eps > 0 and sector_pe > 0:
        bear_pe = round(sector_pe * 0.80, 1)
        base_pe = round(sector_pe,        1)
        bull_pe = round(sector_pe * 1.20, 1)

        bear_pt = round(bear_pe * forward_eps, 2)
        base_pt = round(base_pe * forward_eps, 2)
        bull_pt = round(bull_pe * forward_eps, 2)

        scenarios["pe_based"] = {
            "bear": {
                "pe_multiple":  bear_pe,
                "price_target": bear_pt,
                "upside_pct":   _upside(bear_pt, current_price),
                "assumption":   f"Stock de-rates to 20% below sector avg P/E ({bear_pe}×)",
            },
            "base": {
                "pe_multiple":  base_pe,
                "price_target": base_pt,
                "upside_pct":   _upside(base_pt, current_price),
                "assumption":   f"Stock trades at sector average P/E ({base_pe}×)",
            },
            "bull": {
                "pe_multiple":  bull_pe,
                "price_target": bull_pt,
                "upside_pct":   _upside(bull_pt, current_price),
                "assumption":   f"Stock commands 20% premium to sector P/E ({bull_pe}×)",
            },
            "forward_eps_used": forward_eps,
            "sector_pe_used":   sector_pe,
        }

    # ── Method 2: Analyst consensus range ────────────────────────────────────
    if any(v is not None for v in (analyst_low, analyst_mean, analyst_high)):
        cons: dict = {}
        if analyst_low is not None:
            cons["bear"] = {
                "price_target": analyst_low,
                "upside_pct":   _upside(analyst_low, current_price),
                "label":        "Analyst consensus low (most pessimistic estimate)",
            }
        if analyst_mean is not None:
            cons["base"] = {
                "price_target": analyst_mean,
                "upside_pct":   _upside(analyst_mean, current_price),
                "label":        "Analyst consensus mean",
            }
        if analyst_high is not None:
            cons["bull"] = {
                "price_target": analyst_high,
                "upside_pct":   _upside(analyst_high, current_price),
                "label":        "Analyst consensus high (most optimistic estimate)",
            }
        if cons:
            scenarios["analyst_consensus"] = cons

    # ── Method 3: DCF intrinsic value ────────────────────────────────────────
    if dcf_intrinsic is not None and current_price and current_price > 0:
        scenarios["dcf"] = {
            "intrinsic_value_per_share": dcf_intrinsic,
            "margin_of_safety_pct":      _upside(dcf_intrinsic, current_price),
            "label":                     "DCF intrinsic value (simplified — directional only)",
        }

    return scenarios


# ── LangGraph node ─────────────────────────────────────────────────────────────

async def quant_analyst_node(state: AgentState, config: dict | None = None) -> AgentState:
    """LangGraph node: Quant Analyst agent."""
    validate_state_for_agent(state, "quant_analyst")

    tracer        = config.get("tracer")        if config else None
    artifacts     = config.get("artifacts")     if config else None
    step_callback = config.get("step_callback") if config else None
    primary_llm   = config.get("primary_llm")   if config else None

    if primary_llm is None:
        raise RuntimeError("Quant Analyst requires 'primary_llm' in config")

    raw_data       = state["raw_data"]
    iteration_log: list[IterationLogEntry] = list(state.get("iteration_log", []))
    errors: list[dict[str, Any]]           = list(state.get("errors", []))
    step = len(iteration_log)

    if tracer:
        tracer.record_agent_start("quant_analyst", {"tickers": list(raw_data.keys())})

    analysis_per_ticker: dict[str, Any] = {}

    for ticker, ticker_data in raw_data.items():
        ta: dict[str, Any] = {"ticker": ticker, "citations": {}}

        # ── 1. Total-return CAGR ─────────────────────────────────────────────
        pm = ticker_data.get("price_metrics", {})
        ph = ticker_data.get("price_history", {})

        if pm and pm.get("total_return_cagr_pct") is not None:
            ta["price_cagr_5y_pct"] = pm["total_return_cagr_pct"]
            ta["sp500_cagr_pct"]    = pm.get("sp500_cagr_pct")
            ta["relative_cagr_pct"] = pm.get("relative_cagr_pct")
            ta["citations"]["price_cagr_5y_pct"] = {
                "source_tool": "yahoo_finance", "observation_step": step
            }
        elif ph and ph.get("current_price") and ph.get("price_5y_ago"):
            current = ph["current_price"]
            five_y  = ph["price_5y_ago"]
            if five_y > 0:
                expr     = f"(({current} / {five_y}) ** (1/5) - 1) * 100"
                step    += 1
                cagr_str = await calculator_tool.arun({"expression": expr})
                _record_step(step, "calculator", {"expression": expr}, cagr_str,
                             tracer, artifacts, step_callback, iteration_log)
                try:
                    ta["price_cagr_5y_pct"] = round(float(cagr_str), 2)
                    ta["citations"]["price_cagr_5y_pct"] = {
                        "source_tool": "calculator", "observation_step": step
                    }
                except (ValueError, TypeError):
                    errors.append({"error_type": ErrorType.TOOL_ERROR.value,
                                   "detail": f"CAGR parse failed: {cagr_str}"})

        # ── 2. Risk metrics ──────────────────────────────────────────────────
        if pm:
            for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                        "beta_vs_sp500", "volatility_annual_pct"):
                if pm.get(key) is not None:
                    ta[key] = pm[key]
                    ta["citations"][key] = {
                        "source_tool": "yahoo_finance", "observation_step": step
                    }

        # ── 3. FCF yield and cash metrics ────────────────────────────────────
        cf = ticker_data.get("cash_flow", {})
        if cf:
            if cf.get("fcf_yield") is not None:
                ta["fcf_yield_pct"] = round(cf["fcf_yield"] * 100, 2)
                ta["citations"]["fcf_yield_pct"] = {
                    "source_tool": "yahoo_finance", "observation_step": step
                }
            if cf.get("cash_conversion_ratio") is not None:
                ta["cash_conversion_ratio"] = cf["cash_conversion_ratio"]
            if cf.get("free_cash_flow") is not None:
                ta["free_cash_flow"] = cf["free_cash_flow"]

        # ── 4. Next earnings date ─────────────────────────────────────────────
        earn = ticker_data.get("earnings", {})
        if earn and earn.get("next_earnings_date"):
            ta["next_earnings_date"] = earn["next_earnings_date"]

        # ── 5. Sector benchmark (raw sector string → benchmark_lookup resolves)
        fund = ticker_data.get("fundamentals", {})
        if isinstance(fund, str):
            try:
                fund = json.loads(fund)
            except Exception:
                fund = {}

        bs = ticker_data.get("balance_sheet", {})

        bm: dict = {}
        if fund and fund.get("sector"):
            raw_sector = fund["sector"]   # pass as-is; benchmark_lookup resolves
            country    = fund.get("country")
            step      += 1
            bm_input   = {"gics_sector": raw_sector}
            if country:
                bm_input["country"] = country
            benchmark_str = await benchmark_lookup_tool.arun(bm_input)
            _record_step(step, "benchmark_lookup", bm_input,
                         benchmark_str, tracer, artifacts, step_callback, iteration_log)

            try:
                bm = json.loads(benchmark_str)
                if "error_type" not in bm:
                    ta["sector"]       = bm.get("sector", raw_sector)
                    ta["sector_peers"] = bm.get("peer_examples", [])
                    if bm.get("geographic_context"):
                        ta["geographic_context"] = bm["geographic_context"]
                    ta["citations"]["sector_benchmarks"] = {
                        "source_tool": "benchmark_lookup", "observation_step": step
                    }

                    co_pe  = fund.get("pe_ratio")
                    sec_pe = bm.get("pe_ratio_sector_avg")
                    if co_pe and sec_pe:
                        ta["company_pe"]            = co_pe
                        ta["sector_pe_avg"]         = sec_pe
                        ta["pe_vs_sector_premium_pct"] = round(
                            (co_pe - sec_pe) / sec_pe * 100, 1
                        )

                    co_ev  = fund.get("ev_to_ebitda")
                    sec_ev = bm.get("ev_ebitda_sector_avg")
                    if co_ev and sec_ev:
                        ta["ev_ebitda"]            = co_ev
                        ta["sector_ev_ebitda_avg"] = sec_ev
                        ta["ev_vs_sector_premium_pct"] = round(
                            (co_ev - sec_ev) / sec_ev * 100, 1
                        )

                    co_pb  = fund.get("price_to_book")
                    sec_pb = bm.get("price_to_book_sector_avg")
                    if co_pb and sec_pb:
                        ta["price_to_book"]           = co_pb
                        ta["sector_price_to_book_avg"] = sec_pb
                        ta["pb_vs_sector_premium_pct"] = round(
                            (co_pb - sec_pb) / sec_pb * 100, 1
                        )

                    if bm.get("beta_sector_avg"):
                        ta["sector_beta_avg"] = bm["beta_sector_avg"]
                    if bm.get("operating_margin_pct"):
                        ta["sector_operating_margin_pct"] = bm["operating_margin_pct"]
                else:
                    bm = {}
            except (json.JSONDecodeError, KeyError) as exc:
                errors.append({"error_type": ErrorType.PARSING_ERROR.value, "detail": str(exc)})
                bm = {}

        # ── 6. Key profitability context ─────────────────────────────────────
        if fund:
            for key in ("roe", "roa", "operating_margin", "gross_margin",
                        "debt_to_equity", "revenue_growth", "earnings_growth"):
                yf_key = {
                    "roe":              "return_on_equity",
                    "roa":              "return_on_assets",
                    "operating_margin": "operating_margin",
                    "gross_margin":     "gross_margin",
                    "debt_to_equity":   "debt_to_equity",
                    "revenue_growth":   "revenue_growth",
                    "earnings_growth":  "earnings_growth",
                }.get(key, key)
                val = fund.get(yf_key)
                if val is not None:
                    ta[key] = val

        # ── 7. Simplified DCF valuation ───────────────────────────────────────
        dcf_result = _compute_dcf(
            fcf               = cf.get("free_cash_flow") if cf else None,
            revenue_growth_raw= fund.get("revenue_growth") if fund else None,
            beta              = (pm.get("beta_vs_sp500") or fund.get("beta")) if (pm or fund) else None,
            risk_free_rate_pct= pm.get("risk_free_rate_used") if pm else None,
            total_debt        = bs.get("total_debt") if bs else None,
            cash              = bs.get("cash_and_equivalents") if bs else None,
            market_cap        = fund.get("market_cap") if fund else None,
            current_price     = (fund.get("current_price") or
                                 (ph.get("current_price") if ph else None)),
        )
        ta["dcf"] = dcf_result
        if not dcf_result.get("dcf_not_applicable"):
            ta["citations"]["dcf"] = {
                "source_tool": "quant_analyst_dcf", "observation_step": step
            }

        # ── 8. Scenario analysis ──────────────────────────────────────────────
        dcf_intrinsic = (
            dcf_result.get("intrinsic_value_per_share")
            if not dcf_result.get("dcf_not_applicable") else None
        )
        current_price_for_scenarios = (
            fund.get("current_price") or (ph.get("current_price") if ph else None)
        )
        scenarios = _compute_scenarios(
            current_price = current_price_for_scenarios,
            forward_eps   = fund.get("forward_eps") if fund else None,
            sector_pe     = bm.get("pe_ratio_sector_avg") if bm else None,
            analyst_low   = fund.get("analyst_target_low") if fund else None,
            analyst_mean  = fund.get("analyst_target_mean") if fund else None,
            analyst_high  = fund.get("analyst_target_high") if fund else None,
            dcf_intrinsic = dcf_intrinsic,
        )
        ta["scenarios"] = scenarios

        # ── 9. LLM narrative — narrates computed scenarios, does NOT invent them
        step += 1
        computed_context = {
            k: v for k, v in ta.items()
            if k not in ("ticker", "citations", "sector_peers", "dcf", "scenarios")
            and v is not None
        }

        # Quarterly revenue trend for momentum context
        qt_summary = ""
        ft = ticker_data.get("financials_trend", {})
        if ft and ft.get("income_trend"):
            qt_rows = [
                f"  {q.get('quarter','?')}: rev=${q.get('revenue',0) or 0:,.0f}"
                f"  YoY={q.get('revenue_yoy_pct','N/A')}%"
                f"  net_margin={q.get('net_margin_pct','N/A')}%"
                for q in ft["income_trend"][:4]
            ]
            qt_summary = "\n\nQuarterly trend (newest first):\n" + "\n".join(qt_rows)

        # Analyst sentiment summary
        rec_summary = ""
        recs_data = fund.get("analyst_recommendations") if isinstance(fund, dict) else None
        if isinstance(recs_data, dict) and recs_data.get("sentiment_counts"):
            sc = recs_data["sentiment_counts"]
            rec_summary = (
                f"\n\nAnalyst sentiment (last 10): "
                f"{sc.get('positive',0)} positive / "
                f"{sc.get('neutral',0)} neutral / "
                f"{sc.get('negative',0)} negative"
            )

        # DCF note
        dcf_summary = ""
        if dcf_result.get("dcf_not_applicable"):
            dcf_summary = f"\n\nDCF: Not applicable — {dcf_result.get('reason', '')}"
        else:
            iv   = dcf_result.get("intrinsic_value_per_share")
            mos  = dcf_result.get("margin_of_safety_pct")
            wacc = dcf_result.get("wacc_pct")
            if iv is not None:
                dcf_summary = (
                    f"\n\nDCF intrinsic value: ${iv:.2f} "
                    f"(margin of safety: {mos:+.1f}%, WACC: {wacc:.1f}%)"
                )

        human_content = (
            f"Stock: {ticker}\n"
            f"Sector: {ta.get('sector', 'Unknown')}\n\n"
            f"COMPUTED SCENARIOS:\n{json.dumps(scenarios, indent=2)[:1500]}"
            f"{dcf_summary}\n\n"
            f"COMPUTED METRICS:\n{json.dumps(computed_context, indent=2)[:2000]}"
            f"{qt_summary}"
            f"{rec_summary}"
        )

        sop_result: dict = {}
        try:
            structured_llm = primary_llm.with_structured_output(_SOPOutput)
            sop: _SOPOutput = await structured_llm.ainvoke([
                SystemMessage(content=_SOP_SYSTEM),
                HumanMessage(content=human_content),
            ])
            ta["bull_case"]    = sop.bull_case
            ta["bear_case"]    = sop.bear_case
            ta["closest_peer"] = sop.closest_peer
            sop_result = {
                "bull_case":    sop.bull_case,
                "bear_case":    sop.bear_case,
                "closest_peer": sop.closest_peer,
            }
        except Exception as exc:
            logger.warning("SOP structured output failed for %s: %s", ticker, exc)
            ta["bull_case"] = []
            ta["bear_case"] = []
            errors.append({"error_type": ErrorType.PARSING_ERROR.value, "detail": str(exc)})

        if step_callback:
            step_callback({
                "step": step, "agent": "quant_analyst", "tool": "sop_llm",
                "input": {"ticker": ticker}, "cache_hit": False,
                "ok": bool(ta.get("bull_case")),
            })
        if artifacts:
            artifacts.record_llm_exchange(
                agent="quant_analyst", purpose="sop_narration", ticker=ticker,
                prompt_messages=[
                    {"role": "system", "content": _SOP_SYSTEM},
                    {"role": "user",   "content": human_content},
                ],
                raw_response=json.dumps(sop_result),
            )
        if tracer:
            tracer.record_tool_call(
                agent="quant_analyst", tool="sop_narration",
                input_data={"ticker": ticker}, output_data=ta,
                output_tokens=step * 5,
            )

        analysis_per_ticker[ticker] = ta

    if tracer:
        tracer.record_agent_complete("quant_analyst", {
            "tickers_analysed": list(analysis_per_ticker.keys()),
            "errors":           len(errors),
        })

    return AgentState(**{
        **state,
        "analysis":      analysis_per_ticker,
        "iteration_log": iteration_log,
        "errors":        errors,
    })


# ── Helper ─────────────────────────────────────────────────────────────────────

def _record_step(
    step: int, tool: str, input_data: dict, output: Any,
    tracer, artifacts, step_callback, iteration_log: list,
) -> None:
    output_str = str(output)
    iteration_log.append(IterationLogEntry(
        step=step, agent="quant_analyst", tool=tool,
        input=input_data, output_tokens=len(output_str) // 4, cache_hit=False,
    ))
    if tracer:
        tracer.record_tool_call(
            agent="quant_analyst", tool=tool,
            input_data=input_data, output_data=output,
            output_tokens=len(output_str) // 4,
        )
    if artifacts:
        artifacts.record_tool_response(
            agent="quant_analyst", tool=tool,
            input_data=input_data, full_output=output_str, step=step,
        )
    if step_callback:
        step_callback({
            "step": step, "agent": "quant_analyst", "tool": tool,
            "input": input_data, "cache_hit": False, "ok": True,
        })
