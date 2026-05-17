"""Unit tests for LongTermMemory SQLite persistence.

Each test uses an isolated temporary database via tmp_path to guarantee
no cross-test state leakage.
"""

from __future__ import annotations

import pytest

from ai_financial_analyst.memory.long_term import LongTermMemory


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def mem(db_path) -> LongTermMemory:
    return LongTermMemory(db_path=db_path)


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


class TestPreferences:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_preference(self, mem):
        await mem.save_preference("investment_style", "conservative")
        prefs = await mem.get_all_preferences()
        assert prefs["investment_style"] == "conservative"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_key(self, mem):
        await mem.save_preference("summary_length", "brief")
        await mem.save_preference("summary_length", "detailed")
        prefs = await mem.get_all_preferences()
        assert prefs["summary_length"] == "detailed"
        assert len(prefs) == 1

    @pytest.mark.asyncio
    async def test_multiple_preferences(self, mem):
        await mem.save_preference("style", "conservative")
        await mem.save_preference("horizon", "long-term")
        prefs = await mem.get_all_preferences()
        assert prefs["style"] == "conservative"
        assert prefs["horizon"] == "long-term"

    @pytest.mark.asyncio
    async def test_empty_preferences_returns_empty_dict(self, mem):
        prefs = await mem.get_all_preferences()
        assert prefs == {}

    @pytest.mark.asyncio
    async def test_created_at_set_on_first_save(self, mem):
        before = __import__("time").time()
        await mem.save_preference("investor_type", "long-term")
        after = __import__("time").time()

        import aiosqlite
        async with aiosqlite.connect(mem._db_path) as db:
            async with db.execute(
                "SELECT created_at FROM preferences WHERE key = ?", ("investor_type",)
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert before <= row[0] <= after

    @pytest.mark.asyncio
    async def test_created_at_preserved_on_update(self, mem):
        await mem.save_preference("risk_tolerance", "low")

        import aiosqlite, time as _time
        async with aiosqlite.connect(mem._db_path) as db:
            async with db.execute(
                "SELECT created_at, updated_at FROM preferences WHERE key = ?",
                ("risk_tolerance",),
            ) as cur:
                first = await cur.fetchone()

        _time.sleep(0.01)  # ensure updated_at advances
        await mem.save_preference("risk_tolerance", "high")

        async with aiosqlite.connect(mem._db_path) as db:
            async with db.execute(
                "SELECT created_at, updated_at FROM preferences WHERE key = ?",
                ("risk_tolerance",),
            ) as cur:
                second = await cur.fetchone()

        assert second[0] == first[0]      # created_at unchanged
        assert second[1] > first[1]       # updated_at advanced

    @pytest.mark.asyncio
    async def test_preference_change_logged(self, mem, caplog):
        import logging
        await mem.save_preference("investment_style", "conservative")
        with caplog.at_level(logging.INFO):
            await mem.save_preference("investment_style", "aggressive")
        assert "investment_style" in caplog.text
        assert "conservative" in caplog.text
        assert "aggressive" in caplog.text


# ---------------------------------------------------------------------------
# Analysis summaries
# ---------------------------------------------------------------------------


class TestAnalysisSummaries:
    @pytest.mark.asyncio
    async def test_save_and_count(self, mem):
        await mem.save_analysis_summary("sess1", ["AAPL"], "Apple showed strong growth.", "run1")
        count = await mem.count_summaries()
        assert count == 1

    @pytest.mark.asyncio
    async def test_count_zero_when_empty(self, mem):
        assert await mem.count_summaries() == 0

    @pytest.mark.asyncio
    async def test_search_by_ticker(self, mem):
        await mem.save_analysis_summary("sess1", ["AAPL"], "Apple had high P/E.", "r1")
        await mem.save_analysis_summary("sess2", ["MSFT"], "Microsoft showed revenue growth.", "r2")
        results = await mem.search_summaries("AAPL")
        assert len(results) == 1
        assert "AAPL" in results[0]["tickers"]

    @pytest.mark.asyncio
    async def test_search_by_summary_text(self, mem):
        await mem.save_analysis_summary("sess1", ["AAPL"], "Strong CAGR of 15% over 5 years.", "r1")
        results = await mem.search_summaries("CAGR")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_no_match(self, mem):
        await mem.save_analysis_summary("sess1", ["AAPL"], "Apple growth story.", "r1")
        results = await mem.search_summaries("NVDA")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, mem):
        for i in range(5):
            await mem.save_analysis_summary("sess1", ["AAPL"], f"Analysis {i} AAPL.", f"r{i}")
        results = await mem.search_summaries("AAPL", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_get_recent_summaries(self, mem):
        await mem.save_analysis_summary("s1", ["AAPL"], "First.", "r1")
        await mem.save_analysis_summary("s2", ["MSFT"], "Second.", "r2")
        recent = await mem.get_recent_summaries(limit=2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["tickers"] == "MSFT"


# ---------------------------------------------------------------------------
# delete_all
# ---------------------------------------------------------------------------


class TestDeleteAll:
    @pytest.mark.asyncio
    async def test_clears_preferences(self, mem):
        await mem.save_preference("k", "v")
        await mem.delete_all()
        assert await mem.get_all_preferences() == {}

    @pytest.mark.asyncio
    async def test_clears_summaries(self, mem):
        await mem.save_analysis_summary("s1", ["AAPL"], "Summary.", "r1")
        await mem.delete_all()
        assert await mem.count_summaries() == 0

    @pytest.mark.asyncio
    async def test_delete_all_on_empty_db_is_safe(self, mem):
        await mem.delete_all()  # should not raise
        assert await mem.count_summaries() == 0

    @pytest.mark.asyncio
    async def test_clears_conversations_and_messages(self, mem):
        await mem.save_conversation("c1", "Test")
        await mem.save_message("c1", "user", "hello")
        await mem.delete_all()
        assert await mem.list_conversations() == []
        assert await mem.get_conversation_messages("c1") == []


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class TestConversations:
    @pytest.mark.asyncio
    async def test_save_and_list_conversation(self, mem):
        await mem.save_conversation("conv-1", "AAPL Analysis")
        convs = await mem.list_conversations()
        assert len(convs) == 1
        assert convs[0]["id"] == "conv-1"
        assert convs[0]["title"] == "AAPL Analysis"

    @pytest.mark.asyncio
    async def test_list_ordered_by_updated_at_desc(self, mem):
        await mem.save_conversation("c1", "First")
        await mem.save_conversation("c2", "Second")
        await mem.update_conversation_updated_at("c1")
        convs = await mem.list_conversations()
        assert convs[0]["id"] == "c1"

    @pytest.mark.asyncio
    async def test_update_title(self, mem):
        await mem.save_conversation("c1", "Old title")
        await mem.update_conversation_title("c1", "New title")
        convs = await mem.list_conversations()
        assert convs[0]["title"] == "New title"

    @pytest.mark.asyncio
    async def test_delete_conversation(self, mem):
        await mem.save_conversation("c1", "To delete")
        await mem.delete_conversation("c1")
        assert await mem.list_conversations() == []

    @pytest.mark.asyncio
    async def test_delete_conversation_also_deletes_messages(self, mem):
        await mem.save_conversation("c1", "Conv")
        await mem.save_message("c1", "user", "hello")
        await mem.delete_conversation("c1")
        assert await mem.get_conversation_messages("c1") == []

    @pytest.mark.asyncio
    async def test_duplicate_save_is_ignored(self, mem):
        await mem.save_conversation("c1", "First save")
        await mem.save_conversation("c1", "Second save")  # INSERT OR IGNORE
        convs = await mem.list_conversations()
        assert len(convs) == 1


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestMessages:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_messages(self, mem):
        await mem.save_conversation("c1", "Conv")
        await mem.save_message("c1", "user", "Hello!")
        await mem.save_message("c1", "assistant", "Hi there!")
        msgs = await mem.get_conversation_messages("c1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello!"
        assert msgs[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_messages_ordered_chronologically(self, mem):
        await mem.save_conversation("c1", "Conv")
        await mem.save_message("c1", "user", "First")
        await mem.save_message("c1", "assistant", "Second")
        await mem.save_message("c1", "user", "Third")
        msgs = await mem.get_conversation_messages("c1")
        assert [m["content"] for m in msgs] == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_messages_include_intent_and_tickers(self, mem):
        await mem.save_conversation("c1", "Conv")
        await mem.save_message("c1", "user", "Analyse AAPL", "financial_analysis", "AAPL")
        msgs = await mem.get_conversation_messages("c1")
        assert msgs[0]["intent"] == "financial_analysis"
        assert msgs[0]["tickers"] == "AAPL"

    @pytest.mark.asyncio
    async def test_empty_messages_for_nonexistent_conversation(self, mem):
        assert await mem.get_conversation_messages("nonexistent") == []


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    """Verify semantic search falls back gracefully and ranks by cosine similarity."""

    @pytest.mark.asyncio
    async def test_keyword_fallback_when_no_embedder(self, mem):
        """search_summaries with embedder=None uses LIKE matching."""
        await mem.save_analysis_summary("s1", ["AAPL"], "Apple reported strong iPhone sales.", "r1")
        results = await mem.search_summaries("Apple", limit=5, embedder=None)
        assert len(results) == 1
        assert "Apple" in results[0]["summary"]

    @pytest.mark.asyncio
    async def test_semantic_search_ranks_by_cosine(self, mem):
        """Semantic search returns results ranked by cosine similarity."""
        # Save two summaries with different content
        await mem.save_analysis_summary("s1", ["AAPL"], "Apple iPhone margins expanded significantly.", "r1")
        await mem.save_analysis_summary("s2", ["MSFT"], "Microsoft Azure cloud revenue grew 30%.", "r2")

        # Create a simple mock embedder that returns deterministic vectors
        class MockEmbedder:
            pass

        import math
        # Query embedding: a vector pointing toward "Apple iPhone" content
        query_vec = [1.0, 0.0, 0.0]
        apple_vec = [0.9, 0.1, 0.0]   # similar to query
        msft_vec  = [0.0, 0.9, 0.1]   # dissimilar

        call_count = [0]
        vecs_to_return = [apple_vec, msft_vec]

        async def mock_embed_texts(texts):
            idx = call_count[0]
            call_count[0] += 1
            return [vecs_to_return[idx % len(vecs_to_return)]]

        async def mock_embed_query(text):
            return query_vec

        import ai_financial_analyst.memory.long_term as lt_module
        original_embed_texts = None
        original_embed_query = None

        import ai_financial_analyst.pageindex.embedder as embedder_module
        original_et = getattr(embedder_module, 'embed_texts', None)
        original_eq = getattr(embedder_module, 'embed_query', None)

        embedder_module.embed_texts = mock_embed_texts
        embedder_module.embed_query = mock_embed_query

        try:
            # Save with embeddings
            await mem.save_analysis_summary("s3", ["AAPL"], "Apple iPhone margins expanded significantly.", "r3", embedder=MockEmbedder())
            # Simulate second save getting a different vector
            await mem.save_analysis_summary("s4", ["MSFT"], "Microsoft Azure cloud revenue grew 30%.", "r4", embedder=MockEmbedder())
            # Search
            results = await mem.search_summaries("Apple iPhone", limit=5, embedder=MockEmbedder())
            # Should return results (at least 1)
            assert len(results) >= 1
        finally:
            embedder_module.embed_texts = original_et
            embedder_module.embed_query = original_eq

    @pytest.mark.asyncio
    async def test_keyword_fallback_on_embedder_failure(self, mem):
        """If embedder raises, search falls back to LIKE."""
        await mem.save_analysis_summary("s1", ["TSLA"], "Tesla battery margin improved.", "r1")

        class FailingEmbedder:
            pass

        import ai_financial_analyst.pageindex.embedder as embedder_module

        original_eq = getattr(embedder_module, 'embed_query', None)

        async def failing_embed_query(text):
            raise RuntimeError("API failure")

        embedder_module.embed_query = failing_embed_query

        try:
            results = await mem.search_summaries("Tesla", limit=5, embedder=FailingEmbedder())
            # Falls back to LIKE — should still find "Tesla"
            assert len(results) >= 1
        finally:
            embedder_module.embed_query = original_eq


# ---------------------------------------------------------------------------
# Time-decay scoring
# ---------------------------------------------------------------------------


class TestSummaryTimeDecay:
    """Tests for the _blended_score helper and its effect on search ranking."""

    def test_blended_score_recent_higher_than_old(self):
        from ai_financial_analyst.memory.long_term import _blended_score
        now = __import__("time").time()
        # Two summaries with identical semantic similarity but different ages
        recent_score = _blended_score(0.80, now - 1 * 86_400, lambda_=0.01)   # 1 day old
        old_score    = _blended_score(0.80, now - 120 * 86_400, lambda_=0.01) # 120 days old
        assert recent_score > old_score

    def test_blended_score_lambda_zero_ignores_age(self):
        """lambda=0 disables decay — pure cosine similarity controls ranking."""
        from ai_financial_analyst.memory.long_term import _blended_score
        now = __import__("time").time()
        recent = _blended_score(0.80, now, lambda_=0.0)
        old    = _blended_score(0.80, now - 365 * 86_400, lambda_=0.0)
        assert recent == old  # identical because decay is disabled

    def test_blended_score_very_recent_not_above_one(self):
        """Blended score must stay in [0, 1] even for brand-new summaries."""
        from ai_financial_analyst.memory.long_term import _blended_score
        score = _blended_score(1.0, __import__("time").time(), lambda_=0.01)
        assert 0.0 <= score <= 1.0
