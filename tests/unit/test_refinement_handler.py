"""Unit tests for refinement_handler.py.

Tests the fuzzy match logic (_flexible_str_replace) and the INSERT-based
versioning approach in _save_updated_report.
"""

from __future__ import annotations

import pytest

from ai_financial_analyst.agents.refinement_handler import (
    _flexible_str_replace,
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
