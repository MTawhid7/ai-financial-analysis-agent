"""Intent classifier for the conversational agent.

Classifies each user message into one of five intents and extracts
any stock tickers mentioned. Uses Flash-Lite (sub-LLM) to preserve
the primary-model budget.

Intent taxonomy:
  financial_analysis  — user wants a NEW analysis of one or more specific stocks
  financial_question  — general finance/investing question (no specific stock)
  memory_query        — user wants to RECALL something from a previous interaction
  off_topic           — unrelated to finance or investing
  clarification_needed — too vague to act on; need more information
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

IntentType = Literal[
    "financial_analysis",
    "financial_question",
    "memory_query",
    "off_topic",
    "clarification_needed",
]

_VALID_INTENTS: frozenset[str] = frozenset(
    [
        "financial_analysis",
        "financial_question",
        "memory_query",
        "off_topic",
        "clarification_needed",
    ]
)

# Regex for potential stock tickers: 1-5 uppercase letters standing alone.
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

_EXCLUDED_CAPS = frozenset(
    [
        "I", "A", "AN", "THE", "AND", "OR", "BUT", "IN", "ON", "AT", "TO",
        "FOR", "OF", "BY", "FROM", "WITH", "AS", "IS", "IT", "BE", "DO",
        "IF", "NO", "SO", "UP", "US", "MY", "ME", "WE", "HE", "SHE", "YOU",
        "HOW", "WHY", "CAN", "WILL", "MAY", "HAD", "DID", "HAS", "ARE",
        "WAS", "WERE", "NOT", "YES", "ANY", "ALL", "NEW", "NOW", "OUT",
        "ITS", "OUR", "YOUR", "THEIR", "THIS", "THAT", "THAN", "THEN",
        "WHEN", "WHERE", "WHAT", "WHO", "WHICH", "EPS", "PE", "TTM",
        "YOY", "QOQ", "IPO", "ETF", "GDP", "CPI", "CEO", "CFO", "COO",
        "AI", "ML", "API", "UI", "UX", "OK", "VS", "PER",
    ]
)

_CLASSIFICATION_PROMPT = """\
You are classifying a user message for an AI Financial Analyst assistant.

Classify the message into exactly one intent and extract any stock tickers.

INTENT DEFINITIONS:
- "financial_analysis": user wants a NEW analysis or information about one or more specific stocks.
  (e.g. "Analyse AAPL", "Is Tesla worth buying?", "Tell me about Microsoft", "How is NVDA doing?")
- "financial_question": general finance/investing question with no specific stock to analyse.
  (e.g. "What is a P/E ratio?", "How does dollar-cost averaging work?")
- "memory_query": user wants to RECALL something from a PREVIOUS conversation or past analysis
  — they are asking about something already discussed, not requesting new analysis.
  (e.g. "What did we find about AAPL earlier?", "Remind me what you said about Tesla",
   "What was the P/E ratio we found last time?", "What did we discuss previously?",
   "Do you remember our analysis of Microsoft?", "What did you find out about NVDA?")
- "off_topic": unrelated to finance, investing, or economics.
  (e.g. "What's the weather?", "Write me a poem")
- "clarification_needed": too vague to determine what financial topic is meant.
  (e.g. "Tell me about that", "What do you think?")

CRITICAL RULES:
1. If the message contains retrospective phrases — "earlier", "last time", "previously",
   "what did we", "what did you", "what you found", "remind me", "do you remember",
   "recall", "from before", "last session" — classify as "memory_query" EVEN IF a ticker is present.
2. If the user clearly wants fresh analysis ("analyse X", "tell me about X", "is X a good buy"),
   use "financial_analysis" regardless of whether X was discussed before.
3. tickers: extract stock symbols even for memory_query (they help retrieve the right memory).
4. Output ONLY the JSON object below. No explanation, no markdown.

OUTPUT FORMAT:
{{"intent": "<one of the five intents>", "tickers": ["TICK1", "TICK2"]}}

USER MESSAGE:
{message}
"""


async def classify(message: str, subllm) -> tuple[IntentType, list[str]]:
    """Classify a user message and extract any mentioned tickers.

    Returns:
        (intent, tickers) tuple. Never raises — falls back to
        "financial_question" on any LLM or parsing failure.
    """
    prompt = _CLASSIFICATION_PROMPT.format(message=message.strip())
    try:
        from langchain_core.messages import HumanMessage
        response = await subllm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)

        from ..core.llm import content_to_str
        raw = content_to_str(raw).strip()

        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rstrip("`").strip()

        parsed = json.loads(raw)
        intent = parsed.get("intent", "")
        tickers = [t.upper().strip() for t in parsed.get("tickers", []) if t.strip()]

        if intent not in _VALID_INTENTS:
            logger.warning("Unexpected intent %r; defaulting to financial_question", intent)
            intent = "financial_question"
            tickers = []

        # Regex fallback for ticker extraction when the LLM missed them.
        if intent in ("financial_analysis", "memory_query") and not tickers:
            tickers = _extract_tickers_regex(message)

        logger.debug("Classified %r → intent=%s tickers=%s", message[:60], intent, tickers)
        return intent, tickers  # type: ignore[return-value]

    except Exception as exc:
        logger.warning("Intent classification failed (%s); defaulting to financial_question", exc)
        return "financial_question", []


def _extract_tickers_regex(message: str) -> list[str]:
    """Best-effort ticker extraction from ALL_CAPS tokens in the message."""
    return [t for t in _TICKER_RE.findall(message) if t not in _EXCLUDED_CAPS]
