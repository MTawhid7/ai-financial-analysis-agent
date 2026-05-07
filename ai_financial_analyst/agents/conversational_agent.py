"""ConversationalAgent — top-level multi-turn chat agent.

Sits above the Researcher → Quant → Editor pipeline and handles:
  - Intent classification (financial_analysis / financial_question / off_topic / clarification_needed)
  - Routing user messages to the appropriate handler
  - Session-scoped budget tracking and LLM instances
  - Propagation of step_callback to the inner pipeline for live UI updates

The inner pipeline (run_pipeline) is treated as a black box; this agent
never touches AgentState directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.budget_tracker import RequestBudgetTracker
from ..core.conversation_state import (
    ConversationState,
    append_messages,
    get_recent_context,
    new_session,
)
from ..core.llm import content_to_str, get_primary_llm_with_fallback, get_subllm
from .intent_classifier import IntentType, classify
from .orchestrator import run_pipeline

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
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
    The budget_tracker accumulates across the entire session.
    """

    def __init__(self) -> None:
        self.budget = RequestBudgetTracker()
        self._primary_llm = get_primary_llm_with_fallback(budget_tracker=self.budget)
        self._subllm = get_subllm(budget_tracker=self.budget)

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
        intent, tickers = await classify(message, self._subllm)

        if intent == "financial_analysis":
            response = await self._handle_financial_analysis(
                message, tickers, state, step_callback
            )
        elif intent == "financial_question":
            response = await self._handle_financial_question(message, state)
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
    ) -> str:
        """Answer a general financial question using the primary LLM."""
        recent = get_recent_context(state, max_messages=6)

        # Build message list for the LLM: system prompt + recent history + current question.
        lc_messages: list[Any] = [SystemMessage(content=_SYSTEM_PROMPT)]
        for msg in recent:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # LangChain expects AIMessage for assistant turns.
                from langchain_core.messages import AIMessage
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
