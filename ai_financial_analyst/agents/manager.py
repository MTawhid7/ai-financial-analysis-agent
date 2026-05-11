"""Manager LLM — tool-use orchestrator replacing the hardcoded intent classifier.

The Manager uses LangChain's bind_tools / function-calling to decide which
tool(s) to invoke, in what order, for any user message.  It handles:

  - Single-step requests: "Analyse AAPL"
  - Multi-step requests: "Analyse AAPL and MSFT, then compare them"
  - Ambiguous requests resolved by memory context
  - Compound requests: "What's the CAGR we found for AAPL, and is that better than MSFT's?"
  - Clarification: asks the user when genuinely unclear

Adding new capabilities requires only registering a new @tool function — no
classifier changes, no new intent enum values.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ..core.conversation_state import ConversationState
from ..core.llm import content_to_str
from ..memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 5  # prevent infinite tool loops

# Accepts any reasonable time-period string and normalises it to a yfinance period.
_VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
_PERIOD_ALIASES: dict[str, str] = {
    # months
    "1month": "1mo", "1 month": "1mo", "one month": "1mo", "30d": "1mo", "30days": "1mo",
    "3months": "3mo", "3 months": "3mo", "three months": "3mo", "quarter": "3mo",
    "6months": "6mo", "6 months": "6mo", "six months": "6mo", "half year": "6mo",
    # years
    "1year": "1y", "1 year": "1y", "one year": "1y", "12months": "1y", "12 months": "1y",
    "2years": "2y", "2 years": "2y", "two years": "2y",
    "3years": "3y", "3 years": "3y", "three years": "5y",  # yfinance has no 3y
    "5years": "5y", "5 years": "5y", "five years": "5y",
    "10years": "10y", "10 years": "10y", "ten years": "10y", "decade": "10y",
    # special
    "ytd": "ytd", "year to date": "ytd", "this year": "ytd",
    "all time": "max", "alltime": "max", "all": "max", "max": "max",
    "since ipo": "max", "inception": "max",
}


def _normalise_period(raw: str) -> str:
    """Convert any period string the LLM might produce to a valid yfinance value."""
    if not raw:
        return "1y"
    s = raw.strip().lower().rstrip(".")
    if s in _VALID_PERIODS:
        return s
    if s in _PERIOD_ALIASES:
        return _PERIOD_ALIASES[s]
    import re
    m = re.match(r"^(\d+)\s*(y(?:r|ear)?s?|mo(?:nth)?s?|d(?:ay)?s?)$", s)
    if m:
        n, unit = m.group(1), m.group(2)
        if unit.startswith("y"):
            return f"{n}y" if n in {"1","2","5","10"} else ("5y" if int(n) > 5 else "1y")
        if unit.startswith("mo") or unit == "m":
            return f"{n}mo" if n in {"1","3","6"} else ("6mo" if int(n) > 3 else "1mo")
        if unit.startswith("d"):
            return "5d" if int(n) <= 7 else "1mo"
    return "1y"


def _normalise_date(raw: str) -> str | None:
    """Convert any date expression to YYYY-MM-DD string, or None if not parseable."""
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    import re
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^(19|20)\d{2}$", s.strip()):
        return f"{s.strip()}-01-01"
    try:
        from dateutil import parser as dp
        from datetime import datetime
        dt = dp.parse(s, fuzzy=True, dayfirst=False)
        if 1900 <= dt.year <= datetime.now().year + 1:
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})", s.lower())
    if m:
        mm = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
              "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
        return f"{m.group(2)}-{mm[m.group(1)[:3]]:02d}-01"
    return None


# Known historical events → (start, end | None) date ranges
_NAMED_EVENTS: dict[str, tuple[str, str | None]] = {
    "covid crash": ("2020-02-15", "2020-06-30"),
    "covid":        ("2020-01-01", "2021-12-31"),
    "pandemic":     ("2020-01-01", "2021-12-31"),
    "2008 financial crisis": ("2008-01-01", "2009-06-30"),
    "financial crisis":      ("2008-01-01", "2009-06-30"),
    "gfc":           ("2008-01-01", "2009-06-30"),
    "dot com":       ("2000-01-01", "2002-12-31"),
    "dotcom":        ("2000-01-01", "2002-12-31"),
    "dot-com bubble":("2000-01-01", "2002-12-31"),
    "2022 bear market":  ("2022-01-01", "2022-12-31"),
    "2022 correction":   ("2022-01-01", "2022-12-31"),
    "ai rally":  ("2023-01-01", None),
    "ai boom":   ("2023-01-01", None),
    "great recession": ("2007-12-01", "2009-06-30"),
    "flash crash":     ("2010-05-01", "2010-06-30"),
    "taper tantrum":   ("2013-05-01", "2013-12-31"),
    "rate hike cycle": ("2022-03-01", "2023-12-31"),
}


def _event_date_range(query: str) -> tuple[str | None, str | None]:
    """Return (start, end) if query mentions a named historical event, else (None, None)."""
    q = query.lower()
    for event, (start, end) in _NAMED_EVENTS.items():
        if event in q:
            return start, end
    return None, None

_MANAGER_SYSTEM = """\
You are an expert AI Financial Analyst assistant. You help users with:
- Stock analysis and financial research
- Comparing companies side-by-side
- Recalling and refining past analyses
- Answering financial/investment questions
- Processing uploaded financial documents

You have access to tools. Use them to fulfill the user's request.
For compound requests ("analyse X then compare with Y"), call tools in sequence.
For off-topic requests (weather, cooking, etc.), use the `reject_request` tool.
If you need clarification, use `ask_clarification` and wait for the user's answer.

Always synthesise a clear, professional response after tool results are returned.
Never invent financial figures — use only data from tool outputs.
"""


def build_tools(
    agent: Any,  # ConversationalAgent instance, used for pipeline access
    step_callback: Callable[[dict], None] | None,
) -> list:
    """Build the tool list, capturing agent + callback in closures."""

    @tool
    async def run_financial_analysis(tickers: list[str]) -> str:
        """Run a full financial analysis pipeline for the given stock tickers.

        Use this when the user wants an in-depth analysis of one or more stocks.
        Returns a complete Markdown report.

        Args:
            tickers: List of stock ticker symbols (e.g. ["AAPL", "MSFT"])
        """
        from .orchestrator import run_pipeline
        ticker_str = ", ".join(tickers)
        logger.info("Manager: run_financial_analysis for %s", ticker_str)

        try:
            final_state, _trace, _artifacts = await run_pipeline(
                query=f"Analyse {ticker_str}",
                tickers=tickers,
                step_callback=step_callback,
            )
        except Exception as exc:
            return f"Error running analysis for {ticker_str}: {exc}"

        agent.last_analysis_state = final_state

        # Save analysis summary to memory
        report = final_state.get("report_markdown", "")
        if report:
            try:
                session_id = agent._current_session_id or ""
                await agent._memory.maybe_save_analysis_summary(
                    session_id=session_id,
                    tickers=tickers,
                    report_markdown=report,
                    run_id=final_state.get("run_id", ""),
                )
            except Exception:
                pass

        return report or f"Analysis completed for {ticker_str} but no report was generated."

    @tool
    async def compare_stocks(tickers: list[str]) -> str:
        """Compare two or more stocks side-by-side with a structured comparison table.

        Use when the user explicitly wants a comparison (e.g. "AAPL vs MSFT",
        "compare Tesla and Ford"). Runs the pipeline for all tickers and generates
        a focused comparison table.

        Args:
            tickers: List of stock tickers to compare (minimum 2)
        """
        from .comparison_agent import run_comparison
        logger.info("Manager: compare_stocks for %s", tickers)

        response, final_state = await run_comparison(
            message=f"Compare {', '.join(tickers)}",
            tickers=tickers,
            primary_llm=agent._primary_llm,
            step_callback=step_callback,
        )
        if final_state:
            agent.last_analysis_state = final_state
        return response

    @tool
    async def recall_past_analysis(query: str, tickers: list[str] | None = None) -> str:
        """Retrieve stored summaries from previous analyses.

        Use when the user asks about past research, previous results, or says
        things like "what did we find about AAPL", "remind me", "earlier".

        Args:
            query: The user's question or keywords to search
            tickers: Optional list of tickers to narrow the search
        """
        tickers = tickers or []
        summaries: list[dict] = []

        if tickers:
            seen: set[str] = set()
            for ticker in tickers:
                for r in await agent._memory._lt.search_summaries(ticker, limit=2):
                    if r["summary"] not in seen:
                        summaries.append(r)
                        seen.add(r["summary"])

        if not summaries:
            summaries = await agent._memory._lt.search_summaries(query, limit=3)

        if not summaries:
            offer = f" Would you like me to run a fresh analysis?" if tickers else ""
            return f"I don't have stored analyses matching your query.{offer}"

        label = summaries[0]["tickers"]
        lines = [f"From a previous analysis of **{label}**:\n"]
        for s in summaries[:3]:
            lines.append(s["summary"])
        lines.append("\n*Would you like me to run a fresh analysis for the latest data?*")
        return "\n\n".join(lines)

    @tool
    async def edit_report_section(instruction: str) -> str:
        """Edit or refine a section of the most recently generated report.

        Use when the user asks to modify, update, or improve a previous analysis:
        "make the bear case more pessimistic", "add a risks section",
        "assume 15% revenue growth", "rewrite the conclusion".

        Args:
            instruction: The user's specific editing instruction
        """
        conv_id = getattr(agent, "_current_conversation_id", None)
        if not conv_id:
            return "No stored report to edit. Please run an analysis first."

        from .refinement_handler import refine_analysis
        from ..core.llm import _DEFAULT_DB_PATH as db_path_const
        import os
        db_path = os.getenv("MEMORY_DB_PATH", ".memory/memory.db")

        return await refine_analysis(
            message=instruction,
            conversation_id=conv_id,
            user_id=agent.user_id,
            primary_llm=agent._primary_llm,
            db_path=db_path,
        )

    @tool
    async def answer_finance_question(question: str) -> str:
        """Answer a general financial or investment question directly.

        Use for educational/conceptual questions that don't require running the
        full analysis pipeline: "What is a P/E ratio?", "How does CAGR work?",
        "What is the difference between ETF and mutual fund?"

        Args:
            question: The financial question to answer
        """
        logger.info("Manager: answer_finance_question: %s…", question[:60])
        try:
            response = await agent._primary_llm.ainvoke([
                SystemMessage(content=(
                    "You are an expert financial educator. Answer the question clearly "
                    "and concisely. Use examples where helpful."
                )),
                HumanMessage(content=question),
            ])
            return content_to_str(response.content if hasattr(response, "content") else response)
        except Exception as exc:
            return f"Error answering question: {exc}"

    @tool
    def reject_request(reason: str) -> str:
        """Politely decline a request that is not related to finance or investing.

        Use when the user asks about topics completely unrelated to finance, stocks,
        investing, economics, or financial documents.

        Args:
            reason: Brief description of why the request is out of scope
        """
        return (
            "I'm specialised as an AI Financial Analyst. I can help with:\n\n"
            "- **Stock analysis** — e.g. *\"Analyse AAPL\"*\n"
            "- **Comparisons** — e.g. *\"Compare AAPL vs MSFT\"*\n"
            "- **Financial questions** — e.g. *\"What is a P/E ratio?\"*\n"
            "- **Past research** — e.g. *\"What did we find about NVDA?\"*\n\n"
            "That request falls outside my area. Is there a financial topic I can help with?"
        )

    @tool
    async def generate_chart(
        chart_type: str,
        ticker: str = "",
        period: str = "1y",
        start_date: str = "",
        end_date: str = "",
        overlays: list[str] | None = None,
        compare_tickers: list[str] | None = None,
    ) -> str:
        """Generate an interactive financial chart — always use this for any chart request.

        CHART TYPES:
          Price action:
            candlestick  — OHLCV bars + volume + SMA-50/200 (best default for price)
            price        — simple close-price line
            price_rsi    — candlestick (top) + RSI-14 panel (bottom)
            price_macd   — candlestick (top) + MACD panel (bottom)
            rsi          — RSI(14) standalone
            macd         — MACD(12,26,9) standalone

          Fundamental trends (annual; period/dates not needed):
            revenue_trend  — Revenue vs Net Income bars
            margin_trend   — Gross / Operating / Net margins
            cashflow       — Operating CF vs Free Cash Flow
            debt_profile   — Debt structure vs Cash

          Comparison (requires compare_tickers):
            relative_performance — normalised % return across tickers

          Risk:
            drawdown — max drawdown from rolling peak

          Pipeline data: pe, metrics, radar

        PERIOD — always extract from the user's message:
          "1mo" "3mo" "6mo" "1y" "2y" "5y" "10y" "ytd" "max"
          "last 10 years"/"decade" → "10y"  |  "year to date" → "ytd"
          "all time"/"since IPO"   → "max"  |  "last quarter" → "3mo"

        DATE RANGES — use when the user specifies specific dates or a named event:
          start_date / end_date: "YYYY-MM-DD", "March 2020", "Jan 2021", or "2020"
          Named events → dates (pass start_date and end_date, leave period as default):
            "COVID crash"          → start_date="2020-02-15", end_date="2020-06-30"
            "2008 financial crisis"→ start_date="2008-01-01", end_date="2009-06-30"
            "dot-com bubble"       → start_date="2000-01-01", end_date="2002-12-31"
            "2022 bear market"     → start_date="2022-01-01", end_date="2022-12-31"
            "AI rally"             → start_date="2023-01-01" (no end_date)
          "from January 2020 to December 2021" → start_date="2020-01-01", end_date="2021-12-31"

        OVERLAYS — add to candlestick / price charts:
          "bollinger" or "bb" — Bollinger Bands (20-day, ±2σ)
          "ema" — EMA-20 and EMA-50
          "ema_20", "ema_50", "ema_200" — specific EMA period

        TICKER ALIASES — these are auto-resolved, no need to convert manually:
          "Nasdaq" → QQQ  |  "S&P 500" → SPY  |  "Dow" → DIA
          "semiconductors" → SOXX  |  "gold" → GLD  |  "bitcoin" → BTC-USD
          "VIX" / "volatility" → ^VIX

        EXAMPLES:
          "AAPL with Bollinger Bands"              → chart_type="candlestick", overlays=["bollinger"]
          "MSFT with EMA-20"                       → chart_type="candlestick", overlays=["ema_20"]
          "NVDA price with RSI"                    → chart_type="price_rsi"
          "TSLA from March 2020 to Dec 2021"       → start_date="2020-03-01", end_date="2021-12-31"
          "AAPL during COVID crash"                → start_date="2020-02-15", end_date="2020-06-30"
          "Nasdaq vs S&P 500 this year"            → chart_type="relative_performance", compare_tickers=["QQQ","SPY"], period="ytd"

        Args:
            chart_type: Chart type (see above)
            ticker: Ticker symbol or alias (e.g. "AAPL", "Nasdaq")
            period: Time period string — always extract from the user's message
            start_date: Start date for custom date range (overrides period)
            end_date: End date for custom date range (optional)
            overlays: Indicator overlays e.g. ["bollinger", "ema_20"]
            compare_tickers: For relative_performance — list of tickers/aliases to compare
        """
        from ai_financial_analyst.charts import generate_on_demand_chart

        # Normalise period and dates
        p     = _normalise_period(period)
        start = _normalise_date(start_date) if start_date else None
        end   = _normalise_date(end_date)   if end_date   else None

        # For comparison charts, pass tickers directly — no raw_data needed
        if chart_type.lower().replace("-", "_").replace(" ", "_") in (
            "relative_performance", "comparison", "compare", "normalized", "normalised",
            "relative_return", "peer_comparison",
        ):
            fig = generate_on_demand_chart(
                ticker=ticker, chart_type=chart_type,
                raw_data={}, analysis={},
                period=p, start=start, end=end,
                compare_tickers=compare_tickers,
            )
            if fig is None:
                return "Could not generate comparison chart. Ensure at least 2 valid tickers are provided."
            tickers_label = ", ".join(compare_tickers or [ticker])
            range_label   = f"{start} to {end or 'today'}" if start else p
            if not hasattr(agent, "_pending_charts"):
                agent._pending_charts = []
            agent._pending_charts.append({
                "ticker": tickers_label,
                "chart_type": chart_type,
                "title": f"Return Comparison ({range_label}) — {tickers_label}",
                "figure": fig,
            })
            range_label = f"{start} to {end or 'today'}" if start else p
            return f"Comparison chart for **{tickers_label}** ({range_label}) is displayed below."

        # Single-ticker charts
        final_state = agent.last_analysis_state or {}
        raw_data = final_state.get("raw_data", {})
        analysis = final_state.get("analysis", {})

        # For charts that always fetch fresh data, raw_data not needed — pass empty
        fresh_chart_types = {
            "candlestick", "price", "rsi", "macd",
            "revenue_trend", "margin_trend", "cashflow", "debt_profile", "drawdown",
        }
        ct_norm = chart_type.lower().replace("-", "_").replace(" ", "_")
        if ct_norm in fresh_chart_types or ticker not in raw_data:
            raw_data = {}  # let chart generator fetch directly

        fig = generate_on_demand_chart(
            ticker=ticker.upper(), chart_type=chart_type,
            raw_data=raw_data, analysis=analysis,
            period=p, start=start, end=end,
            overlays=overlays, compare_tickers=compare_tickers,
        )
        if fig is None:
            return f"Could not generate the {chart_type} chart for {ticker}. Try a different period or run an analysis first."

        range_label = f"{start} to {end or 'today'}" if start else p
        if not hasattr(agent, "_pending_charts"):
            agent._pending_charts = []
        agent._pending_charts.append({
            "ticker": ticker.upper(),
            "chart_type": chart_type,
            "title": f"{ticker.upper()} — {chart_type.replace('_', ' ').title()} ({range_label})",
            "figure": fig,
        })
        return f"Here is the **{chart_type.replace('_', ' ')}** chart for **{ticker.upper()}** ({range_label})."

    @tool
    def ask_clarification(question: str) -> str:
        """Ask the user a clarifying question when the request is ambiguous.

        Use when the user's intent is unclear and you need more information before
        selecting the right tool.

        Args:
            question: The clarifying question to ask the user
        """
        return question

    return [
        run_financial_analysis,
        compare_stocks,
        recall_past_analysis,
        edit_report_section,
        answer_finance_question,
        generate_chart,
        reject_request,
        ask_clarification,
    ]


class ManagerAgent:
    """LLM manager that autonomously selects and chains tools to fulfil any request."""

    def __init__(self, primary_llm: Any, agent: Any) -> None:
        self._primary_llm = primary_llm
        self._agent = agent

    async def run(
        self,
        message: str,
        state: ConversationState,
        step_callback: Callable[[dict], None] | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """Process a user message through the tool-use loop.

        Returns the final text response after all tool calls are resolved.
        """
        # Store conversation_id so tools can access it via agent
        self._agent._current_session_id = state.get("session_id", "")
        self._agent._current_conversation_id = conversation_id

        tools = build_tools(self._agent, step_callback)
        llm_with_tools = self._primary_llm.bind_tools(tools)

        # Build message history from ConversationState
        history = []
        for msg in ShortTermMemory.get_windowed_messages(state.get("messages", []), max_tokens=4000):
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))

        # Inject memory context into system prompt
        try:
            memory_ctx = await self._agent._memory.build_memory_context(
                messages=state.get("messages", []),
                query=message,
            )
        except Exception:
            memory_ctx = ""

        system_content = _MANAGER_SYSTEM
        if memory_ctx:
            system_content = f"{_MANAGER_SYSTEM}\n\n---\nUser context:\n{memory_ctx}"

        messages: list = [SystemMessage(content=system_content)] + history + [HumanMessage(content=message)]

        # Tool-use loop
        tool_map = {t.name: t for t in tools}

        for _round in range(_MAX_TOOL_ROUNDS):
            try:
                response = await llm_with_tools.ainvoke(messages)
            except Exception as exc:
                logger.error("Manager LLM call failed: %s", exc)
                return f"I encountered an error: `{exc}`. Please try again."

            if not getattr(response, "tool_calls", None):
                # No more tool calls — this is the final text response
                return content_to_str(response.content if hasattr(response, "content") else response)

            # Execute each tool call
            messages.append(response)
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                t = tool_map.get(tool_name)
                if t is None:
                    result = f"Unknown tool: {tool_name}"
                else:
                    try:
                        logger.info("Manager: calling %s(%s)", tool_name, list(tool_args.keys()))
                        raw = await t.ainvoke(tool_args) if hasattr(t, "ainvoke") else t.invoke(tool_args)
                        result = str(raw)
                    except Exception as exc:
                        logger.error("Tool %s failed: %s", tool_name, exc)
                        result = f"Tool error ({tool_name}): {exc}"

                messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        # Exceeded max rounds — return what we have
        logger.warning("Manager exceeded max tool rounds (%d)", _MAX_TOOL_ROUNDS)
        return "I was unable to complete the request within the allowed steps. Please try a more specific question."
