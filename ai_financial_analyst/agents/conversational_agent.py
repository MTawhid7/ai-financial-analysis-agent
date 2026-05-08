"""ConversationalAgent — top-level multi-turn chat agent.

Handles seven intents:
  financial_analysis  → full pipeline
  comparison          → multi-ticker pipeline + comparison table
  refinement          → LLM-guided modification of the last stored report
  financial_question  → direct LLM answer with conversation history
  memory_query        → retrieves stored analysis summaries
  off_topic           → polite rejection
  clarification_needed → asks for more context
"""

from __future__ import annotations

import logging
import os
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

_DB_PATH = os.getenv("MEMORY_DB_PATH", ".memory/memory.db")

_SYSTEM_PROMPT_BASE = (
    "You are an expert AI Financial Analyst. You help users understand stocks, "
    "investment concepts, financial metrics, and market dynamics. "
    "Be concise, accurate, and professional. "
    "Always clarify when a question is outside your financial expertise."
)

_REJECTION_RESPONSE = (
    "I'm specialised as an AI Financial Analyst. I can help with:\n\n"
    "- **Stock analysis** — e.g. *\"Analyse AAPL\"*\n"
    "- **Comparisons** — e.g. *\"Compare AAPL vs MSFT\"*\n"
    "- **Refinements** — e.g. *\"Make the bear case more pessimistic\"*\n"
    "- **Financial concepts** — e.g. *\"What is a P/E ratio?\"*\n\n"
    "That topic falls outside finance and investing. Is there a financial question I can help with?"
)

_CLARIFICATION_RESPONSE = (
    "I'd love to help! Could you give me a bit more context?\n\n"
    "For example:\n"
    "- **Stock analysis**: *\"Analyse Apple\"* or *\"Tell me about NVDA\"*\n"
    "- **Comparison**: *\"Compare Tesla vs Ford\"*\n"
    "- **Financial question**: *\"How does compound interest work?\"*"
)


class ConversationalAgent:
    """Session-scoped conversational agent — one instance per user session."""

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        self.budget = RequestBudgetTracker()
        self._primary_llm = get_primary_llm_with_fallback(budget_tracker=self.budget)
        self._subllm = get_subllm(budget_tracker=self.budget)
        self._memory = MemoryManager(LongTermMemory(user_id=user_id), subllm=self._subllm)
        self.last_analysis_state: dict | None = None

    async def process_message(
        self,
        message: str,
        state: ConversationState,
        step_callback: Callable[[dict], None] | None = None,
        conversation_id: str | None = None,
    ) -> tuple[str, ConversationState]:
        """Process one user turn and return (response, updated_state)."""
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
        elif intent == "comparison":
            response = await self._handle_comparison(message, tickers, state, step_callback)
        elif intent == "refinement":
            response = await self._handle_refinement(message, conversation_id)
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

    # ------------------------------------------------------------------
    # Financial analysis
    # ------------------------------------------------------------------

    async def _handle_financial_analysis(
        self,
        message: str,
        tickers: list[str],
        state: ConversationState,
        step_callback: Callable[[dict], None] | None,
    ) -> str:
        if not tickers:
            return (
                "I'd be happy to run a full analysis! Which stock(s) would you like me to analyse? "
                "For example: *\"Analyse AAPL\"* or *\"Tell me about Tesla and Microsoft\"*."
            )

        ticker_str = ", ".join(tickers)
        logger.info("Running financial analysis for: %s", ticker_str)

        try:
            final_state, _trace, _artifacts = await run_pipeline(
                query=message, tickers=tickers, step_callback=step_callback
            )
        except Exception as exc:
            logger.exception("Pipeline failed for %s", ticker_str)
            return (
                f"Error analysing **{ticker_str}**: `{exc}`\n\n"
                "Please check your API keys and try again."
            )

        report = final_state.get("report_markdown", "")
        errors = final_state.get("errors", [])
        status = final_state.get("status", "COMPLETE")

        if not report:
            return f"The pipeline completed with status **{status}** but produced no report. Errors: {errors}"

        self.last_analysis_state = final_state

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
            header += f"> ⚠️ Status **{status}** — {len(errors)} error(s). Report may be partial.\n\n"

        return header + report

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    async def _handle_comparison(
        self,
        message: str,
        tickers: list[str],
        state: ConversationState,
        step_callback: Callable[[dict], None] | None,
    ) -> str:
        from .comparison_agent import run_comparison

        response, final_state = await run_comparison(
            message=message,
            tickers=tickers,
            primary_llm=self._primary_llm,
            step_callback=step_callback,
        )

        if final_state:
            self.last_analysis_state = final_state
            try:
                await self._memory.maybe_save_analysis_summary(
                    session_id=state.get("session_id", ""),
                    tickers=tickers,
                    report_markdown=response,
                    run_id=final_state.get("run_id", ""),
                )
            except Exception:
                pass

        return response

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    async def _handle_refinement(
        self,
        message: str,
        conversation_id: str | None,
    ) -> str:
        if not conversation_id:
            return (
                "I need to know which conversation's analysis to refine. "
                "Please run an analysis first in this conversation."
            )

        from .refinement_handler import refine_analysis

        return await refine_analysis(
            message=message,
            conversation_id=conversation_id,
            user_id=self.user_id,
            primary_llm=self._primary_llm,
            db_path=_DB_PATH,
        )

    # ------------------------------------------------------------------
    # Memory query
    # ------------------------------------------------------------------

    async def _handle_memory_query(self, message: str, tickers: list[str]) -> str:
        summaries: list[dict] = []
        try:
            if tickers:
                seen: set[str] = set()
                for ticker in tickers:
                    for r in await self._memory._lt.search_summaries(ticker, limit=2):
                        if r["summary"] not in seen:
                            summaries.append(r)
                            seen.add(r["summary"])
            if not summaries:
                summaries = await self._memory._lt.search_summaries(message, limit=3)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)

        if not summaries:
            offer = (
                f"\n\nWould you like me to run a fresh analysis? "
                f"Try: *\"Analyse {', '.join(tickers)}\"*"
                if tickers else ""
            )
            return "I don't have any stored analyses that match your question." + offer

        label = summaries[0]["tickers"]
        lines = [f"Here's what I found from a previous analysis of **{label}**:\n"]
        for s in summaries[:3]:
            if s["tickers"] != label:
                lines.append(f"**{s['tickers']}**: {s['summary']}")
            else:
                lines.append(s["summary"])

        lines.append(
            "\n---\n*This is from a stored past analysis. "
            "Would you like me to run a fresh analysis for the latest data?*"
        )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Financial question
    # ------------------------------------------------------------------

    async def _handle_financial_question(
        self,
        message: str,
        state: ConversationState,
        memory_ctx: str = "",
    ) -> str:
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
            logger.error("LLM call failed: %s", exc)
            return f"Error generating response: `{exc}`\n\nPlease try again in a moment."
