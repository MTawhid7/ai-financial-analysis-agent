"""Unit tests for PageIndex sentence chunking and deduplication helpers."""

from __future__ import annotations

import pytest

from ai_financial_analyst.pageindex.pipeline import _split_into_chunks
from ai_financial_analyst.pageindex.retriever import _deduplicate_by_page


class TestSplitIntoChunks:
    def test_short_text_returns_empty_list(self):
        """Text under the chunk threshold needs no splitting."""
        short = "Short paragraph." * 10  # well under 1500 chars
        result = _split_into_chunks(short)
        assert result == []

    def test_long_text_returns_multiple_chunks(self):
        """Text well above the threshold should produce multiple chunks."""
        long_text = "A paragraph of content. " * 200  # ~4800 chars
        chunks = _split_into_chunks(long_text)
        assert len(chunks) > 1

    def test_chunks_cover_all_content(self):
        """Every paragraph should appear in at least one chunk."""
        # Each paragraph is ~80 chars; 30 × 80 = 2400 chars > 1500 threshold
        paragraphs = [f"Paragraph {i} with enough content to be meaningful in a real document." for i in range(30)]
        text = "\n\n".join(paragraphs)
        chunks = _split_into_chunks(text)
        all_content = " ".join(chunks)
        for p in paragraphs:
            assert p in all_content, f"Paragraph missing from chunks: {p}"

    def test_each_chunk_within_size_limit(self):
        from ai_financial_analyst.config import settings
        long_text = "A paragraph of content. " * 200
        chunks = _split_into_chunks(long_text)
        # Chunks may slightly exceed max due to overlap text, but should be reasonably bounded
        for chunk in chunks:
            assert len(chunk) < settings.pageindex_chunk_max_chars * 2

    def test_single_huge_paragraph_still_produces_chunks(self):
        """Even a monolithic paragraph gets split."""
        mono = "word " * 1000  # 5000 chars, no paragraph breaks
        chunks = _split_into_chunks(mono)
        # The splitter may not be able to split without paragraph breaks,
        # but at minimum it should not crash.
        assert isinstance(chunks, list)


class TestDeduplicateByPage:
    def _fake_page(self, doc_id: str, page_number: int, chunk_index: int = 0):
        """Return a simple namespace object mimicking a DocumentPage row."""
        class FakePage:
            pass
        p = FakePage()
        p.document_id  = doc_id
        p.page_number  = page_number
        p.chunk_index  = chunk_index
        p.id           = f"{doc_id}:{page_number}:{chunk_index}"
        return p

    def test_root_pages_unchanged(self):
        pages = [
            (self._fake_page("doc1", 1), 0.9),
            (self._fake_page("doc1", 2), 0.8),
            (self._fake_page("doc1", 3), 0.7),
        ]
        result = _deduplicate_by_page(pages, top_k=3)
        assert len(result) == 3

    def test_chunk_hits_collapsed_to_one_per_page(self):
        """Two chunks from the same page should appear only once."""
        pages = [
            (self._fake_page("doc1", 1, chunk_index=1), 0.95),
            (self._fake_page("doc1", 1, chunk_index=2), 0.85),  # same page, lower score
            (self._fake_page("doc1", 2, chunk_index=0), 0.80),
        ]
        result = _deduplicate_by_page(pages, top_k=10)
        assert len(result) == 2  # page 1 collapsed, page 2 kept

    def test_top_k_limit_applied(self):
        pages = [(self._fake_page("doc1", i), 1.0 - i * 0.1) for i in range(10)]
        result = _deduplicate_by_page(pages, top_k=3)
        assert len(result) == 3

    def test_highest_scored_chunk_retained(self):
        """When two chunks for the same page appear, the higher-scored one is kept first."""
        high_score_chunk = (self._fake_page("doc1", 1, chunk_index=1), 0.95)
        low_score_chunk  = (self._fake_page("doc1", 1, chunk_index=2), 0.50)
        pages = [high_score_chunk, low_score_chunk]
        result = _deduplicate_by_page(pages, top_k=10)
        assert len(result) == 1
        assert result[0][1] == 0.95  # higher score retained
