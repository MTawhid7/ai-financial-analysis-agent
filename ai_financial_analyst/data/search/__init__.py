"""Web search data access — Tavily client with credibility scoring and date filtering."""

from .tavily import TavilySearchClient, SearchResult
from .credibility import SOURCE_TIERS, score_source

__all__ = ["TavilySearchClient", "SearchResult", "SOURCE_TIERS", "score_source"]
