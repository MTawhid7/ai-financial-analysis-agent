"""MemoryManager — facade coordinating short-term context and long-term SQLite memory.

Provides:
  build_memory_context()          → ≤500-token string injected into system prompt
  maybe_extract_preferences()     → detect and persist stated user preferences
  maybe_save_analysis_summary()   → generate and persist a one-paragraph analysis summary
  get_preferences() / count_analyses() / clear_all()  → UI accessors
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from ..core.conversation_state import ChatMessage
from ..core.llm import content_to_str
from .long_term import LongTermMemory
from .short_term import ShortTermMemory

logger = logging.getLogger(__name__)

# ~500 tokens ≈ 2000 chars (using len/4 heuristic consistent with codebase).
_MAX_CONTEXT_CHARS = 2000

# Signals that the user is stating a preference — LLM extraction only runs when matched.
_PREFERENCE_SIGNALS = re.compile(
    r"\b(i prefer|always show|always include|i like|i'?m a|i am a|"
    r"please always|please never|keep it|make it|focus on|"
    r"i want .{0,20}(analysis|report|summary))\b",
    re.IGNORECASE,
)

_SUMMARY_PROMPT = """\
Summarise this financial analysis in ONE paragraph (max 80 words).
Include: which tickers were analysed, the most important metric (e.g. CAGR or P/E), and the key conclusion.
Be specific — use the actual numbers.

Report excerpt:
{report_excerpt}

One-paragraph summary:"""

_PREFERENCE_PROMPT = """\
Extract any explicit user preferences from the message below.
Return a JSON object mapping preference keys to string values.
Only include explicit statements — do not infer or guess.
Return {{}} if no preferences are found.

Examples:
  "I prefer conservative analysis"    → {{"investment_style": "conservative"}}
  "always show brief summaries"       → {{"summary_length": "brief"}}
  "I'm a long-term investor"          → {{"investor_type": "long-term"}}
  "focus on dividend stocks"          → {{"focus": "dividend stocks"}}

Message: {message}

JSON:"""


class MemoryManager:
    """Coordinates short-term context and long-term SQLite memory for the agent."""

    def __init__(self, long_term: LongTermMemory, subllm: Any = None) -> None:
        self._lt = long_term
        self._subllm = subllm

    # ------------------------------------------------------------------
    # Context building — called on every turn
    # ------------------------------------------------------------------

    async def build_memory_context(
        self, messages: list[ChatMessage], query: str
    ) -> str:
        """Return a ≤500-token string for injection into the system prompt.

        Includes long-term preferences and summaries of relevant past analyses.
        Returns an empty string when nothing is stored yet.
        """
        parts: list[str] = []

        try:
            prefs = await self._lt.get_all_preferences()
            if prefs:
                lines = "\n".join(f"  - {k}: {v}" for k, v in prefs.items())
                parts.append(f"User preferences (from past sessions):\n{lines}")
        except Exception as exc:
            logger.debug("Could not load preferences: %s", exc)

        try:
            if query.strip():
                summaries = await self._lt.search_summaries(query.strip(), limit=2)
                if summaries:
                    lines = "\n".join(
                        f"  - [{s['tickers']}]: {s['summary'][:200]}"
                        for s in summaries
                    )
                    parts.append(f"Relevant past analyses:\n{lines}")
        except Exception as exc:
            logger.debug("Could not search summaries: %s", exc)

        if not parts:
            return ""

        context = "\n\n".join(parts)
        if len(context) > _MAX_CONTEXT_CHARS:
            context = context[:_MAX_CONTEXT_CHARS] + "…"
        return context

    # ------------------------------------------------------------------
    # Preference extraction — called after each user message
    # ------------------------------------------------------------------

    async def maybe_extract_preferences(self, message: str) -> None:
        """Detect and persist explicit user preferences from a message.

        A regex pre-filter avoids an LLM call on every message — the LLM
        is only invoked when preference signal keywords are present.
        """
        if self._subllm is None or not _PREFERENCE_SIGNALS.search(message):
            return

        try:
            prompt = _PREFERENCE_PROMPT.format(message=message)
            response = await self._subllm.ainvoke([HumanMessage(content=prompt)])
            raw = content_to_str(
                response.content if hasattr(response, "content") else response
            ).strip()

            # Strip markdown code fences if present.
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rstrip("`").strip()

            prefs = json.loads(raw)
            if not isinstance(prefs, dict):
                return

            for key, value in prefs.items():
                if isinstance(key, str) and isinstance(value, str) and key and value:
                    await self._lt.save_preference(key, value)
                    logger.info("Saved preference: %s = %s", key, value)

        except Exception as exc:
            logger.debug("Preference extraction skipped: %s", exc)

    # ------------------------------------------------------------------
    # Analysis summary — called after each pipeline run
    # ------------------------------------------------------------------

    async def maybe_save_analysis_summary(
        self,
        session_id: str,
        tickers: list[str],
        report_markdown: str,
        run_id: str = "",
    ) -> None:
        """Generate a one-paragraph summary of the completed analysis and persist it."""
        if not report_markdown or self._subllm is None:
            return

        try:
            prompt = _SUMMARY_PROMPT.format(report_excerpt=report_markdown[:3000])
            response = await self._subllm.ainvoke([HumanMessage(content=prompt)])
            summary = content_to_str(
                response.content if hasattr(response, "content") else response
            ).strip()

            if summary:
                await self._lt.save_analysis_summary(
                    session_id=session_id,
                    tickers=tickers,
                    summary_text=summary,
                    run_id=run_id,
                )
                logger.info("Saved analysis summary for tickers: %s", tickers)

        except Exception as exc:
            logger.warning("Could not save analysis summary: %s", exc)

    # ------------------------------------------------------------------
    # UI accessors — memory stats
    # ------------------------------------------------------------------

    async def get_preferences(self) -> dict[str, str]:
        return await self._lt.get_all_preferences()

    async def count_analyses(self) -> int:
        return await self._lt.count_summaries()

    async def clear_all(self) -> None:
        await self._lt.delete_all()

    # ------------------------------------------------------------------
    # Conversation persistence (proxy methods for chat_app.py)
    # ------------------------------------------------------------------

    async def create_conversation(self, conversation_id: str, title: str) -> None:
        await self._lt.save_conversation(conversation_id, title)

    async def update_conversation_title(self, conversation_id: str, title: str) -> None:
        await self._lt.update_conversation_title(conversation_id, title)

    async def update_conversation_updated_at(self, conversation_id: str) -> None:
        await self._lt.update_conversation_updated_at(conversation_id)

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        return await self._lt.list_conversations(limit)

    async def load_conversation(self, conversation_id: str) -> list[dict]:
        """Return all messages for a conversation in chronological order."""
        return await self._lt.get_conversation_messages(conversation_id)

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: str = "",
        tickers: str = "",
    ) -> None:
        await self._lt.save_message(conversation_id, role, content, intent, tickers)

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._lt.delete_conversation(conversation_id)
