"""Injectable Tavily web search client with credibility scoring and date filtering.

Key improvements over the old web_search.py:
- No global state: TavilySearchClient is instantiated with injected config
- Source credibility: results sorted by (source_tier ASC, tavily_score DESC)
- Date filtering: settings.search_days_window limits result recency (default 90 days)
- Zero-result retry: reformulates query and retries once before giving up
- No configure() global mutation: create a new instance with desired settings
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ...config import settings
from ...core.sanitizer import ContentSanitizer, build_sanitizer
from .credibility import score_source

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single web search result, enriched with credibility metadata."""
    headline:    str
    url:         str
    content:     str
    score:       float
    source_tier: int = 4
    published_date: str | None = None


class TavilySearchClient:
    """Tavily search wrapper that applies credibility scoring and date filtering.

    Stateless after construction — safe to create per request.
    """

    def __init__(
        self,
        api_key:        str | None = None,
        days_window:    int | None = None,
        max_results:    int | None = None,
        min_content_chars: int | None = None,
        sanitizer:      ContentSanitizer | None = None,
    ) -> None:
        self._api_key        = api_key or settings.tavily_api_key
        self._days_window    = days_window if days_window is not None else settings.search_days_window
        self._max_results    = max_results or settings.search_max_results
        self._min_chars      = min_content_chars or settings.search_min_content_chars
        self._sanitizer      = sanitizer or ContentSanitizer()

    def search(self, query: str) -> list[SearchResult]:
        """Search Tavily; retry with reformulated query on zero results.

        Results are sorted by (source_tier ASC, tavily_score DESC) so Tier 1
        sources appear first regardless of Tavily's own ranking.
        """
        results = self._invoke(query)
        if not results:
            reformulated = _reformulate_query(query)
            logger.warning("Zero Tavily results for %r — retrying as %r", query[:60], reformulated[:60])
            results = self._invoke(reformulated)
        scored = [self._enrich(r) for r in results]
        return sorted(scored, key=lambda r: (r.source_tier, -r.score))

    def _invoke(self, query: str) -> list[dict]:
        """Call Tavily API; return raw result list (normalised from various response shapes)."""
        try:
            from langchain_tavily import TavilySearch
            kwargs: dict[str, Any] = {
                "max_results":  self._max_results,
                "api_key":      self._api_key,
                "search_depth": "basic",
            }
            if self._days_window > 0:
                kwargs["days"] = self._days_window
            tavily = TavilySearch(**kwargs)
            raw = tavily.invoke({"query": query})

            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict):
                raw = raw.get("results", raw.get("data", []))
            if not isinstance(raw, list):
                return []
            return raw
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query[:60], exc)
            return []

    def _enrich(self, item: dict) -> SearchResult:
        """Convert a raw Tavily result dict to a SearchResult with credibility data."""
        url     = item.get("url", "")
        title   = item.get("title", "") or ""
        content = item.get("content", "") or ""
        score   = float(item.get("score", 0.0))

        # Sanitize content
        cleaned = self._sanitizer._regex_filter(content)
        if cleaned is None:
            cleaned = ""

        # Skip near-empty results
        text_only = re.sub(r'\[.*?\]\(.*?\)', '', cleaned).strip()
        if len(text_only) < self._min_chars:
            cleaned = ""

        # Truncate if excessively long
        if len(cleaned) > 1000:
            cleaned = cleaned[:1000]

        return SearchResult(
            headline    = self._sanitizer._regex_filter(title) or title,
            url         = url,
            content     = cleaned,
            score       = score,
            source_tier = score_source(url),
            published_date = item.get("published_date"),
        )


def _reformulate_query(query: str) -> str:
    """Broaden a query that returned zero Tavily results.

    Removes the year (too narrow) and 'analyst outlook' (overly specific).
    """
    q = re.sub(r'\b20\d{2}\b', '', query).strip()
    q = re.sub(r'\banalyst\s+outlook\b', '', q, flags=re.I).strip()
    return re.sub(r'\s+', ' ', q).strip() or query
