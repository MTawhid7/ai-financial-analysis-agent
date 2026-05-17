"""WebSearchTool — thin LangChain @tool wrapper over data/search/TavilySearchClient.

All search logic (credibility scoring, date filtering, retry) lives in
data/search/tavily.py. This wrapper handles input validation, caching, and
JSON serialisation. No global state: TavilySearchClient is instantiated per call.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool
from pydantic import Field

from ..config import settings
from ..core.cache import ResultCache, TTL_WEB_SEARCH
from ..data.search.tavily import TavilySearchClient
from .base import StrictToolInput, safe_tool_call

logger = logging.getLogger(__name__)

_cache = ResultCache()


class WebSearchInput(StrictToolInput):
    query:       str = Field(description="Search query string")
    max_results: int = Field(default=3, ge=1, le=5, description="Number of results to retrieve")


# Legacy configure() shim — kept for backward compat with orchestrator call sites.
# The subllm parameter is no longer used (Tavily pre-summarises results).
def configure(subllm=None) -> None:
    """No-op shim: TavilySearchClient is now stateless, no global injection needed."""


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, max_results: int = 3) -> str:
    """Search the web via Tavily with source credibility scoring and date filtering.

    Results are sorted by source credibility (Reuters > CNBC > unknown blog).
    If no results are found, the query is reformulated and retried once.
    Cached for settings.ttl_web_search_s (default 1 hour).
    """
    args = {"query": query, "max_results": max_results}

    def _fetch() -> str:
        client  = TavilySearchClient(max_results=max_results)
        results = client.search(query)
        payload = {
            "query":        query,
            "result_count": len(results),
            "data_truncated": any(len(r.content) >= 1000 for r in results),
            "summaries":    [
                {
                    "headline":    r.headline,
                    "url":         r.url,
                    "content":     r.content,
                    "score":       r.score,
                    "source_tier": r.source_tier,
                }
                for r in results if r.content
            ],
        }
        return json.dumps(payload)

    result, hit = _cache.get_or_fetch("web_search", args, _fetch, ttl=TTL_WEB_SEARCH)
    if hit:
        logger.debug("Web search cache HIT query=%s", query[:60])
    return result
