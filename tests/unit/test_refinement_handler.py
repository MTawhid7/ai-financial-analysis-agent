"""Unit tests for refinement_handler.py.

Tests the fuzzy match logic (_flexible_str_replace), the INSERT-based
versioning approach in _save_updated_report, and the optimistic locking
that prevents silent data loss on concurrent edits.
"""

from __future__ import annotations

import time
import uuid

import aiosqlite
import pytest

from ai_financial_analyst.agents.refinement_handler import (
    EditConflictError,
    _extract_section_context,
    _flexible_str_replace,
    _infer_target_section,
    _load_latest_report,
    _save_updated_report,
    _strip_lines,
)


class TestFlexibleStrReplace:
    def test_exact_match(self):
        report = "## Bull Case\n- Strong revenue growth\n\n## Bear Case\n- Competition"
        old = "## Bull Case\n- Strong revenue growth"
        new = "## Bull Case\n- Very strong revenue growth\n- Expanding margins"
        result = _flexible_str_replace(report, old, new)
        assert result is not None
        assert "Very strong revenue growth" in result
        assert "Expanding margins" in result
        assert "## Bear Case" in result  # unchanged section preserved

    def test_line_strip_fallback(self):
        """Trailing whitespace difference should be tolerated."""
        report = "## Bull Case   \n- Revenue growth\n\n## Bear Case"
        # LLM returned old_string without trailing spaces
        old = "## Bull Case\n- Revenue growth"
        new = "## Bull Case\n- Massive revenue growth"
        result = _flexible_str_replace(report, old, new)
        assert result is not None
        assert "Massive revenue growth" in result
        assert "## Bear Case" in result

    def test_returns_none_when_not_found(self):
        report = "## Executive Summary\nApple is strong."
        old = "This text does not exist in the report."
        new = "replacement"
        result = _flexible_str_replace(report, old, new)
        assert result is None

    def test_replaces_only_first_occurrence(self):
        report = "## Section\ncontent\n\n## Section\ncontent"
        old = "## Section\ncontent"
        new = "## Section\nupdated content"
        result = _flexible_str_replace(report, old, new)
        assert result is not None
        assert result.count("updated content") == 1
        # Second occurrence is unchanged
        assert "content" in result


class TestStripLines:
    def test_removes_trailing_spaces(self):
        assert _strip_lines("hello   \nworld  ") == "hello\nworld"

    def test_preserves_leading_spaces(self):
        assert _strip_lines("  indented  \n  line  ") == "  indented\n  line"

    def test_empty_string(self):
        assert _strip_lines("") == ""


# ── Optimistic locking helpers ────────────────────────────────────────────────


async def _seed_report(db_path: str, conversation_id: str, user_id: str, markdown: str) -> float:
    """Insert a single report row and return its created_at timestamp."""
    created_at = time.time()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS reports"
            " (id TEXT, conversation_id TEXT, user_id TEXT, tickers TEXT,"
            "  report_markdown TEXT, raw_data_json TEXT, analysis_json TEXT, created_at REAL)"
        )
        await db.execute(
            "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, user_id, "AAPL",
             markdown, "{}", "{}", created_at),
        )
        await db.commit()
    return created_at


class TestSectionAwareEditing:
    """Tests for section inference and extraction helpers."""

    def test_infer_bull_case(self):
        assert _infer_target_section("make the bull case more optimistic") == "Bull Case"

    def test_infer_bear_case(self):
        assert _infer_target_section("strengthen the bear case argument") == "Bear Case"

    def test_infer_conclusion(self):
        assert _infer_target_section("update the conclusion") == "Conclusion"

    def test_infer_executive_summary(self):
        assert _infer_target_section("rewrite the executive summary") == "Executive Summary"

    def test_infer_returns_none_for_unrecognised_message(self):
        assert _infer_target_section("make it more detailed") is None

    def test_case_insensitive_inference(self):
        assert _infer_target_section("BULL CASE needs work") == "Bull Case"

    def test_extract_section_returns_none_when_not_found(self):
        report = "## Executive Summary\nSome content."
        result = _extract_section_context(report, "Bear Case")
        assert result is None

    def test_extract_section_splits_correctly(self):
        report = (
            "## Executive Summary\nSummary content.\n\n"
            "## Bull Case\n- Growth driver\n- Market share\n\n"
            "## Bear Case\n- Competition risk"
        )
        result = _extract_section_context(report, "Bull Case")
        assert result is not None
        before, section, after = result
        assert "## Bull Case" in section
        assert "Growth driver" in section
        # Bear case should not be in the extracted section
        assert "Bear Case" not in section
        assert "Bear Case" in after

    def test_extract_last_section_has_empty_after(self):
        report = "## Executive Summary\nContent.\n\n## Conclusion\nFinal thoughts."
        result = _extract_section_context(report, "Conclusion")
        assert result is not None
        before, section, after = result
        assert "Final thoughts" in section
        assert after == ""


class TestOptimisticLocking:
    """Concurrent edit protection via base_created_at lock token."""

    @pytest.mark.asyncio
    async def test_load_latest_report_includes_created_at(self, tmp_path):
        db_path = str(tmp_path / "reports.db")
        await _seed_report(db_path, "conv1", "user1", "# Report")

        result = await _load_latest_report("conv1", "user1", db_path)

        assert result is not None
        assert "created_at" in result
        assert isinstance(result["created_at"], float)
        assert result["created_at"] > 0

    @pytest.mark.asyncio
    async def test_save_succeeds_when_version_matches(self, tmp_path):
        db_path = str(tmp_path / "reports.db")
        created_at = await _seed_report(db_path, "conv1", "user1", "# Original")

        # Save with the correct lock token — should succeed
        await _save_updated_report(
            "# Updated", "conv1", "user1", db_path, base_created_at=created_at
        )

        result = await _load_latest_report("conv1", "user1", db_path)
        assert result["report_markdown"] == "# Updated"

    @pytest.mark.asyncio
    async def test_save_raises_conflict_when_version_stale(self, tmp_path):
        db_path = str(tmp_path / "reports.db")
        created_at = await _seed_report(db_path, "conv1", "user1", "# Original")

        # Simulate a concurrent write that happened between our load and save
        await _seed_report(db_path, "conv1", "user1", "# Concurrent edit")

        # Our save should be rejected because a newer version now exists
        with pytest.raises(EditConflictError):
            await _save_updated_report(
                "# Our edit", "conv1", "user1", db_path, base_created_at=created_at
            )

    @pytest.mark.asyncio
    async def test_save_without_lock_token_always_succeeds(self, tmp_path):
        """base_created_at=None disables the check (backward-compatible)."""
        db_path = str(tmp_path / "reports.db")
        await _seed_report(db_path, "conv1", "user1", "# Original")

        # No lock token — should always succeed regardless of concurrent writes
        await _save_updated_report(
            "# No lock", "conv1", "user1", db_path, base_created_at=None
        )

        result = await _load_latest_report("conv1", "user1", db_path)
        assert result["report_markdown"] == "# No lock"
