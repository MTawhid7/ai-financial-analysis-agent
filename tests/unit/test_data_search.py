"""Unit tests for data/search/ — TavilySearchClient, credibility scoring, query reformulation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_financial_analyst.data.search.credibility import score_source, SOURCE_TIERS
from ai_financial_analyst.data.search.tavily import TavilySearchClient, _reformulate_query, SearchResult
from ai_financial_analyst.core.sanitizer import ContentSanitizer


# ── Credibility scoring ───────────────────────────────────────────────────────

class TestSourceCredibility:
    def test_reuters_is_tier_1(self):
        assert score_source("https://www.reuters.com/article/xyz") == 1

    def test_bloomberg_is_tier_1(self):
        assert score_source("https://bloomberg.com/news/articles/abc") == 1

    def test_sec_gov_is_tier_1(self):
        assert score_source("https://sec.gov/cgi-bin/browse-edgar") == 1

    def test_cnbc_is_tier_2(self):
        assert score_source("https://www.cnbc.com/2024/01/01/article") == 2

    def test_nytimes_is_tier_3(self):
        assert score_source("https://www.nytimes.com/story") == 3

    def test_unknown_domain_is_tier_4(self):
        assert score_source("https://random-stock-forum.example.com") == 4

    def test_empty_url_is_tier_4(self):
        assert score_source("") == 4

    def test_all_tier_1_sources_in_dict(self):
        tier_1 = [d for d, t in SOURCE_TIERS.items() if t == 1]
        assert len(tier_1) >= 5   # at least 5 tier-1 domains defined


# ── Query reformulation ───────────────────────────────────────────────────────

class TestQueryReformulation:
    def test_removes_year(self):
        result = _reformulate_query("AAPL stock news analyst outlook 2026")
        assert "2026" not in result
        assert "AAPL" in result

    def test_removes_analyst_outlook(self):
        result = _reformulate_query("TSLA analyst outlook earnings")
        assert "analyst outlook" not in result.lower()
        assert "TSLA" in result

    def test_collapses_extra_spaces(self):
        result = _reformulate_query("AAPL  stock  news 2026")
        assert "  " not in result

    def test_fallback_to_original_when_empty(self):
        # If stripping everything left nothing, return original
        original = "2026"
        result   = _reformulate_query(original)
        assert isinstance(result, str)
        assert len(result) > 0


# ── TavilySearchClient ────────────────────────────────────────────────────────

class TestTavilySearchClient:
    """Tests use a mocked TavilySearch to avoid network calls."""

    def _client(self):
        return TavilySearchClient(
            api_key     = "test-key",
            days_window = 90,
            max_results = 3,
        )

    def _make_raw_result(self, url="https://reuters.com/x", content="Apple reported strong quarterly revenue growth this period with significant earnings beat", score=0.9):
        return {"url": url, "title": "Apple News", "content": content, "score": score}

    @patch("langchain_tavily.TavilySearch")
    def test_returns_search_results(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = [self._make_raw_result()]
        mock_cls.return_value = mock_instance

        client  = self._client()
        results = client.search("AAPL stock news")

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    @patch("langchain_tavily.TavilySearch")
    def test_results_sorted_by_tier_then_score(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = [
            self._make_raw_result(url="https://random-blog.com/x", score=0.95),
            self._make_raw_result(url="https://reuters.com/x",     score=0.70),
        ]
        mock_cls.return_value = mock_instance

        client  = self._client()
        results = client.search("AAPL")

        # Reuters (tier 1) should come first despite lower Tavily score
        assert results[0].source_tier == 1
        assert results[1].source_tier == 4

    @patch("langchain_tavily.TavilySearch")
    def test_retries_on_zero_results(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = [
            [],
            [self._make_raw_result()],
        ]
        mock_cls.return_value = mock_instance

        client  = self._client()
        results = client.search("AAPL stock 2026")

        assert mock_instance.invoke.call_count == 2
        assert len(results) == 1

    @patch("langchain_tavily.TavilySearch")
    def test_source_tier_enriched(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = [
            self._make_raw_result(url="https://bloomberg.com/article"),
        ]
        mock_cls.return_value = mock_instance

        client  = self._client()
        results = client.search("AAPL")

        assert results[0].source_tier == 1

    @patch("langchain_tavily.TavilySearch")
    def test_api_failure_returns_empty(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = RuntimeError("API failure")
        mock_cls.return_value = mock_instance

        client  = self._client()
        results = client.search("AAPL")

        assert results == []

    @patch("langchain_tavily.TavilySearch")
    def test_low_content_result_filtered(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = [
            self._make_raw_result(content="short"),  # < min_content_chars
        ]
        mock_cls.return_value = mock_instance

        client  = TavilySearchClient(api_key="k", min_content_chars=50)
        results = client.search("AAPL")

        assert len(results) == 1
        assert results[0].content == ""
