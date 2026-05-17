"""WebSearchTool — Tavily search with regex sanitization.

Pipeline: TavilySearch → regex sanitizer → structured output
Tavily returns clean, AI-optimised summaries so no Flash-Lite
compression step is needed (saving sub-LLM quota).
The regex sanitizer still runs to catch any adversarial content.
"""

from __future__ import annotations

import json
import logging

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from pydantic import Field

from ..config import settings
from ..core.cache import ResultCache, TTL_WEB_SEARCH
from ..core.sanitizer import ContentSanitizer, build_sanitizer
from .base import StrictToolInput, safe_tool_call

logger = logging.getLogger(__name__)

_MAX_FACTS_PER_RESULT = 5
_MIN_CONTENT_CHARS    = settings.search_min_content_chars
_cache                = ResultCache()

# Sanitizer (regex-only — no Flash-Lite needed since Tavily pre-summarises).
_sanitizer: ContentSanitizer = build_sanitizer(subllm=None)


def configure(subllm=None) -> None:
    """Optional: inject Flash-Lite for deeper extraction if desired."""
    global _sanitizer
    _sanitizer = build_sanitizer(subllm=subllm)


class WebSearchInput(StrictToolInput):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=3, ge=1, le=5, description="Number of results to retrieve")


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, max_results: int = 3) -> str:
    """Search the web via Tavily and return sanitised summaries.

    Tavily is purpose-built for AI agents and returns structured,
    pre-summarised results. The regex sanitizer still runs to neutralise
    any adversarial content before results reach the agent.
    """
    args = {"query": query, "max_results": max_results}

    def _fetch():
        return _search_and_sanitize(query, max_results)

    result, hit = _cache.get_or_fetch("web_search", args, _fetch, ttl=TTL_WEB_SEARCH)
    if hit:
        logger.debug("Web search cache HIT query=%s", query[:60])
    return result


def _search_and_sanitize(query: str, max_results: int) -> str:
    def _run():
        # search_depth="basic" reliably returns results; "advanced" (the default)
        # often returns empty arrays for financial queries.
        tavily = TavilySearch(
            max_results=max_results,
            api_key=settings.tavily_api_key,
            search_depth="basic",
        )
        raw_results = tavily.invoke({"query": query})

        # Normalise the response: TavilySearch may return a bare list OR a dict
        # {"results": [...], "answer": ...} depending on the SDK version.
        if isinstance(raw_results, str):
            raw_results = json.loads(raw_results)
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("results", raw_results.get("data", []))
        if not isinstance(raw_results, list):
            raw_results = []

        if not raw_results:
            logger.warning("Tavily returned 0 results for query: %s", query[:80])

        summaries = []
        data_truncated = False

        for item in raw_results[:max_results]:
            content = item.get("content", "") or ""
            title = item.get("title", "") or ""

            # Run regex sanitizer on content to strip injection patterns.
            cleaned = _sanitizer._regex_filter(content)
            if cleaned is None:
                logger.warning("Tavily result sanitized away: title='%s'", title[:60])
                continue

            # Discard results that are navigation menus or near-empty pages —
            # strip markdown links and check for enough substantive text.
            import re as _re
            text_only = _re.sub(r'\[.*?\]\(.*?\)', '', cleaned).strip()
            if len(text_only) < _MIN_CONTENT_CHARS:
                logger.warning(
                    "Tavily result skipped (low content, %d chars): title='%s'",
                    len(text_only), title[:60],
                )
                continue

            # Truncate if content is excessively long.
            if len(cleaned) > _MAX_FACTS_PER_RESULT * 200:
                cleaned = cleaned[: _MAX_FACTS_PER_RESULT * 200]
                data_truncated = True

            summaries.append({
                "headline": _sanitizer._regex_filter(title) or title,
                "url": item.get("url", ""),
                "content": cleaned,
                "score": item.get("score", 0.0),
            })

        return json.dumps({
            "query": query,
            "result_count": len(summaries),
            "data_truncated": data_truncated,
            "summaries": summaries,
        })

    return safe_tool_call("web_search", _run, {"query": query, "max_results": max_results})
