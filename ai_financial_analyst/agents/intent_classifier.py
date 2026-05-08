"""Intent classifier for the conversational agent.

Classifies each user message into one of seven intents and extracts
any stock tickers. Uses Flash-Lite (sub-LLM) to preserve primary budget.

Intent taxonomy:
  financial_analysis   — user wants a NEW analysis of one or more specific stocks
  comparison           — user wants a SIDE-BY-SIDE comparison of two or more stocks
  refinement           — user wants to MODIFY or UPDATE a previous analysis result
  financial_question   — general finance/investing question (no specific stock)
  memory_query         — user wants to RECALL something from a previous interaction
  off_topic            — unrelated to finance or investing
  clarification_needed — too vague to act on
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

IntentType = Literal[
    "financial_analysis",
    "comparison",
    "refinement",
    "financial_question",
    "memory_query",
    "off_topic",
    "clarification_needed",
]

_VALID_INTENTS: frozenset[str] = frozenset(
    [
        "financial_analysis",
        "comparison",
        "refinement",
        "financial_question",
        "memory_query",
        "off_topic",
        "clarification_needed",
    ]
)

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
- "financial_analysis": user wants a NEW analysis of one or more specific stocks.
  (e.g. "Analyse AAPL", "Is Tesla worth buying?", "Tell me about Microsoft")
- "comparison": user wants a SIDE-BY-SIDE comparison of two or more stocks.
  (e.g. "Compare AAPL and MSFT", "NVDA vs AMD", "which is better: Tesla or Ford?",
   "compare Apple against Google")
- "refinement": user wants to MODIFY, UPDATE, or IMPROVE a previous analysis result.
  (e.g. "Make the bear case more pessimistic", "Redo with 15% growth assumption",
   "Update the conclusion", "Add a risks section", "Be more conservative",
   "Change the outlook to bearish")
- "financial_question": general finance/investing question with no specific stock to analyse.
  (e.g. "What is a P/E ratio?", "How does compound interest work?")
- "memory_query": user wants to RECALL something from a PREVIOUS conversation or past analysis.
  (e.g. "What did we find about AAPL earlier?", "Remind me what you said about Tesla",
   "What was the P/E we found last time?")
- "off_topic": unrelated to finance, investing, or economics.
- "clarification_needed": too vague to determine the financial topic.

CRITICAL RULES:
1. If the message contains "vs", "versus", "compare", "against", "or" between two companies
   and seems to be asking which is better, use "comparison".
2. If the message contains "make it", "redo", "update", "change to", "assume X%",
   "more pessimistic/optimistic", "add a section" — use "refinement".
3. If the message contains retrospective phrases ("earlier", "last time", "what did we find",
   "remind me") — use "memory_query" even if a ticker is present.
4. tickers: extract even for comparison, refinement, and memory_query intents.
5. Output ONLY the JSON below. No explanation, no markdown.

OUTPUT FORMAT:
{{"intent": "<one of the seven intents>", "tickers": ["TICK1", "TICK2"]}}

USER MESSAGE:
{message}
"""


async def classify(message: str, subllm) -> tuple[IntentType, list[str]]:
    """Classify a user message and extract any mentioned tickers."""
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

        if intent in ("financial_analysis", "comparison", "memory_query", "refinement") and not tickers:
            tickers = _extract_tickers_regex(message)

        logger.debug("Classified %r → intent=%s tickers=%s", message[:60], intent, tickers)
        return intent, tickers  # type: ignore[return-value]

    except Exception as exc:
        logger.warning("Intent classification failed (%s); defaulting to financial_question", exc)
        return "financial_question", []


def _extract_tickers_regex(message: str) -> list[str]:
    return [t for t in _TICKER_RE.findall(message) if t not in _EXCLUDED_CAPS]
