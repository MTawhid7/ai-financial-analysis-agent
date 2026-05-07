"""ConversationalAgent — top-level multi-turn chat agent.

Sits above the Researcher → Quant → Editor pipeline and handles:
  - Intent classification (financial_analysis / financial_question / memory_query /
    off_topic / clarification_needed)
  - Routing user messages to the appropriate handler
  - Session-scoped budget tracking, LLM instances, and memory
  - Propagation of step_callback to the inner pipeline for live UI updates

The inner pipeline (run_pipeline) is treated as a black box; this agent
never touches AgentState directly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..core.budget_tracker import RequestBudgetTracker
from ..core.conversation_state import (
    ConversationState,
    append_messages,
    new_session,
)
from ..core.llm import content_to_str, get_primary_llm_with_fallback, get_subllm
from ..memory.long_term import LongTermMemory
from ..memory.memory_manager import MemoryManager
from ..memory.short_term import ShortTermMemory
from .intent_classifier import IntentType, classify
from .orchestrator import run_pipeline

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_BASE = (
    "You are an expert AI Financial Analyst. You help users understand stocks, "
    "investment concepts, financial metrics, and market dynamics. "
    "Be concise, accurate, and professional. "
    "Always clarify when a question is outside your financial expertise. "
    "You have access to real-time web search and can run quantitative analysis pipelines."
)

_REJECTION_RESPONSE = (
    "I'm specialised as an AI Financial Analyst. I can help with:\n\n"
    "- **Stock analysis** — e.g. *\"Analyse AAPL\"* or *\"Is TSLA worth buying?\"*\n"
    "- **Financial concepts** — e.g. *\"What is a P/E ratio?\"*\n"
    "- **Calculations** — e.g. *\"Calculate 5-year CAGR for 12% annual growth\"*\n\n"
    "That topic falls outside finance and investing. Is there a financial question I can help with?"
)

_CLARIFICATION_RESPONSE = (
    "I'd love to help! Could you give me a bit more context?\n\n"
    "For example:\n"
    "- **Stock analysis**: *\"Analyse Apple\"* or *\"Tell me about NVDA\"*\n"
    "- **Financial question**: *\"How does compound interest work?\"*"
)


class ConversationalAgent:
    """Session-scoped conversational agent wrapping the financial pipeline.

    Create one instance per chat session and keep it in session state.
    The budget_tracker and memory accumulate across the entire session.
    """

    def __init__(self) -> None:
        self.budget = RequestBudgetTracker()
        self._primary_llm = get_primary_llm_with_fallback(budget_tracker=self.budget)
        self._subllm = get_subllm(budget_tracker=self.budget)
        self._memory = MemoryManager(LongTermMemory(), subllm=self._subllm)

    async def process_message(
        self,
        message: str,
        state: ConversationState,
        step_callback: Callable[[dict], None] | None = None,
    ) -> tuple[str, ConversationState]:
        """Process one user turn and return (assistant_response, updated_state).

        Args:
            message: Raw user message text.
            state: Current conversation state.
            step_callback: Optional; called after each pipeline tool invocation
                           with a step event dict for live UI updates.

        Returns:
            (response_text, new_conversation_state)
        """
        memory_ctx = await self._memory.build_memory_context(
            messages=state.get("messages", []),
            query=message,
        )

        try:
            await self._memory.maybe_extract_preferences(message)
        except Exception as exc:
            logger.debug("Preference extraction skipped: %s", exc)

        intent, tickers = await classify(message, self._subllm)

        if intent == "financial_analysis":
            response = await self._handle_financial_analysis(
                message, tickers, state, step_callback
            )
        elif intent == "memory_query":
            response = await self._handle_memory_query(message, tickers)
        elif intent == "financial_question":
            response = await self._handle_financial_question(message, state, memory_ctx)
        elif intent == "off_topic":
            response = _REJECTION_RESPONSE
        else:
            response = _CLARIFICATION_RESPONSE

        new_state = append_messages(
            state,
            user_content=message,
            assistant_content=response,
            intent=intent,
            tickers=tickers,
        )
        return response, new_state

    async def _handle_memory_query(self, message: str, tickers: list[str]) -> str:
        """Answer a question about past analyses from stored summaries.

        Searches by ticker first (most precise), then falls back to keyword
        search across summary text. If nothing is found, offers to run fresh analysis.
        """
        summaries: list[dict] = []

        try:
            if tickers:
                seen_summaries: set[str] = set()
                for ticker in tickers:
                    results = await self._memory._lt.search_summaries(ticker, limit=2)
                    for r in results:
                        if r["summary"] not in seen_summaries:
                            summaries.append(r)
                            seen_summaries.add(r["summary"])

            if not summaries:
                summaries = await self._memory._lt.search_summaries(message, limit=3)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)

        if not summaries:
            offer = ""
            if tickers:
                offer = (
                    f"\n\nWould you like me to run a fresh analysis? "
                    f"Try: *\"Analyse {', '.join(tickers)}\"*"
                )
            return (
                "I don't have any stored analyses that match your question."
                + offer
            )

        ticker_label = summaries[0]["tickers"]
        lines = [f"Here's what I found from a previous analysis of **{ticker_label}**:\n"]
        for s in summaries[:3]:
            if s["tickers"] != ticker_label:
                lines.append(f"**{s['tickers']}**: {s['summary']}")
            else:
                lines.append(s["summary"])

        lines.append(
            "\n---\n*This is from a stored past analysis. "
            "Would you like me to run a fresh analysis to get the latest data?*"
        )
        return "\n\n".join(lines)

    async def _handle_financial_analysis(
        self,
        message: str,
        tickers: list[str],
        state: ConversationState,
        step_callback: Callable[[dict], None] | None,
    ) -> str:
        """Run the full Researcher → Quant → Editor pipeline and return the report."""
        if not tickers:
            return (
                "I'd be happy to run a full analysis! Could you tell me which "
                "stock(s) you'd like me to analyse? For example: *\"Analyse AAPL\"* "
                "or *\"Tell me about Tesla and Microsoft\"*."
            )

        ticker_str = ", ".join(tickers)
        logger.info("Running financial analysis for tickers: %s", ticker_str)

        try:
            final_state, _trace_path, _artifacts_path = await run_pipeline(
                query=message,
                tickers=tickers,
                dry_run=False,
                step_callback=step_callback,
            )
        except Exception as exc:
            logger.exception("Pipeline failed for tickers %s", ticker_str)
            return (
                f"I ran into an error while analysing **{ticker_str}**: `{exc}`\n\n"
                "Please check your API keys and try again. "
                "If the issue persists, try fewer tickers or wait a minute for rate limits to reset."
            )

        report = final_state.get("report_markdown", "")
        errors = final_state.get("errors", [])
        status = final_state.get("status", "COMPLETE")

        if not report:
            return (
                f"The pipeline completed with status **{status}** but did not produce a report. "
                f"Errors: {errors}"
            )

        try:
            await self._memory.maybe_save_analysis_summary(
                session_id=state.get("session_id", ""),
                tickers=tickers,
                report_markdown=report,
                run_id=final_state.get("run_id", ""),
            )
        except Exception as exc:
            logger.debug("Analysis summary save skipped: %s", exc)

        header = f"Here is my analysis for **{ticker_str}**:\n\n"
        if status != "COMPLETE" and errors:
            header += (
                f"> ⚠️ Pipeline completed with status **{status}** — "
                f"{len(errors)} error(s) encountered. The report may be partial.\n\n"
            )

        return header + report

    async def _handle_financial_question(
        self,
        message: str,
        state: ConversationState,
        memory_ctx: str = "",
    ) -> str:
        """Answer a general financial question using the primary LLM."""
        system_content = _SYSTEM_PROMPT_BASE
        if memory_ctx:
            system_content = f"{_SYSTEM_PROMPT_BASE}\n\n---\n{memory_ctx}"

        recent = ShortTermMemory.get_windowed_messages(
            state.get("messages", []), max_tokens=3000
        )

        lc_messages: list[Any] = [SystemMessage(content=system_content)]
        for msg in recent:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        lc_messages.append(HumanMessage(content=message))

        try:
            response = await self._primary_llm.ainvoke(lc_messages)
            return content_to_str(response.content if hasattr(response, "content") else response)
        except Exception as exc:
            logger.error("LLM call failed for financial question: %s", exc)
            return (
                f"I encountered an error while generating a response: `{exc}`\n\n"
                "Please try again in a moment."
            )
