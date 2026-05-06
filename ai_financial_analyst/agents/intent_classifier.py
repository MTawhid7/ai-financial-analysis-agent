"""Intent classifier for the conversational agent.

Classifies each user message into one of four intents and extracts
any stock tickers mentioned. Uses Flash-Lite (sub-LLM) to preserve
the primary-model budget.

Intent taxonomy:
  financial_analysis  — user wants analysis of one or more specific stocks
  financial_question  — general finance/investing question (no specific stock)
  off_topic           — unrelated to finance or investing
  clarification_needed — too vague to act on; need more information
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

# Intent type alias used throughout the module.
IntentType = Literal[
    "financial_analysis",
    "financial_question",
    "off_topic",
    "clarification_needed",
]

_VALID_INTENTS: frozenset[str] = frozenset(
    ["financial_analysis", "financial_question", "off_topic", "clarification_needed"]
)

# Regex for potential stock tickers: 1-5 uppercase letters standing alone.
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# Common English ALL-CAPS words to exclude from ticker candidates.
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
- "financial_analysis": user wants analysis of one or more specific stocks/companies
  (even if the company name is spelled out, e.g. "Tell me about Apple" or "Is Tesla worth buying?")
- "financial_question": general finance/investing question with no specific stock
  (e.g. "What is a P/E ratio?", "How does dollar-cost averaging work?")
- "off_topic": unrelated to finance, investing, or economics
  (e.g. "What's the weather?", "Write me a poem", "How do I cook pasta?")
- "clarification_needed": too vague to determine what financial topic is meant
  (e.g. "Tell me about that", "What do you think?", "Can you help me?")

RULES:
1. If the user mentions a company name or stock symbol and seems to want information or analysis about it, use "financial_analysis".
2. If asking for general financial education, use "financial_question".
3. tickers: extract only the actual stock symbols or map company names to their ticker.
   Examples: Apple→AAPL, Tesla→TSLA, Microsoft→MSFT, Amazon→AMZN, Google/Alphabet→GOOGL,
   Meta→META, Netflix→NFLX, NVIDIA→NVDA, AMD→AMD, Intel→INTC.
   If no specific stock is mentioned, return an empty list.
4. Output ONLY the JSON object below. No explanation, no markdown.

OUTPUT FORMAT:
{{"intent": "<one of the four intents>", "tickers": ["TICK1", "TICK2"]}}

USER MESSAGE:
{message}
"""


async def classify(message: str, subllm) -> tuple[IntentType, list[str]]:
    """Classify a user message and extract any mentioned tickers.

    Args:
        message: The raw user message.
        subllm: Flash-Lite LLM instance (from get_subllm()).

    Returns:
        (intent, tickers) tuple. Never raises — falls back to
        "financial_question" on any LLM or parsing failure.
    """
    prompt = _CLASSIFICATION_PROMPT.format(message=message.strip())
    try:
        from langchain_core.messages import HumanMessage
        response = await subllm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)

        # Normalize content (handles google-genai list-of-blocks format)
        from ..core.llm import content_to_str
        raw = content_to_str(raw).strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rstrip("`").strip()

        parsed = json.loads(raw)
        intent = parsed.get("intent", "")
        tickers = [t.upper().strip() for t in parsed.get("tickers", []) if t.strip()]

        if intent not in _VALID_INTENTS:
            logger.warning("Unexpected intent from classifier: %r; defaulting to financial_question", intent)
            intent = "financial_question"
            tickers = []

        # If intent is financial_analysis but no tickers extracted, try regex fallback.
        if intent == "financial_analysis" and not tickers:
            tickers = _extract_tickers_regex(message)

        logger.debug("Classified '%s...' → intent=%s tickers=%s", message[:60], intent, tickers)
        return intent, tickers  # type: ignore[return-value]

    except Exception as exc:
        logger.warning("Intent classification failed (%s); falling back to financial_question", exc)
        return "financial_question", []


def _extract_tickers_regex(message: str) -> list[str]:
    """Best-effort ticker extraction from ALL_CAPS tokens in the message."""
    candidates = _TICKER_RE.findall(message)
    return [t for t in candidates if t not in _EXCLUDED_CAPS]
