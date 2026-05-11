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
    async def generate_chart(ticker: str, chart_type: str = "price") -> str:
        """Generate an interactive Plotly chart for a specific ticker on demand.

        Use when the user explicitly asks for a chart or visualization:
        "show me a chart of AAPL's price", "create a P/E comparison chart",
        "visualise MSFT's financial profile".

        The chart is automatically displayed in the UI alongside this response.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            chart_type: One of: price, pe, metrics, radar (default: price)
        """
        from ai_financial_analyst.tools.chart_generator import generate_on_demand_chart

        # Try to use last analysis state if available
        final_state = agent.last_analysis_state or {}
        raw_data = final_state.get("raw_data", {})
        analysis = final_state.get("analysis", {})

        if ticker not in raw_data:
            # Fetch fresh data for this ticker
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                hist = t.history(period="1y", interval="1wk")
                raw_data = {ticker: {"price_history": {
                    "current_price": float(hist["Close"].iloc[-1]) if not hist.empty else None,
                    "price_5y_ago": None,
                    "52w_high": float(hist["High"].max()) if not hist.empty else None,
                    "52w_low": float(hist["Low"].min()) if not hist.empty else None,
                }}}
            except Exception:
                return f"Could not fetch data for {ticker} to generate the chart."

        fig = generate_on_demand_chart(ticker, chart_type, raw_data, analysis)
        if fig is None:
            return f"Could not generate a {chart_type} chart for {ticker}. Try running an analysis first."

        # Inject chart into the pending chart list on the agent
        if not hasattr(agent, "_pending_charts"):
            agent._pending_charts = []
        agent._pending_charts.append({
            "ticker": ticker,
            "chart_type": chart_type,
            "title": f"{ticker} — {chart_type.replace('_', ' ').title()}",
            "figure": fig,
        })
        return f"I've generated the {chart_type} chart for **{ticker}**. It appears below."

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
