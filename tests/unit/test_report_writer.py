"""Unit tests for report_writer.py — quality scoring and section validation."""

from __future__ import annotations

import pytest

from ai_financial_analyst.tools.report_writer import (
    _DISCLAIMER,
    _REQUIRED_SECTIONS,
    _check_quality,
    _enforce_disclaimer,
    _validate_sections,
)


class TestValidateSections:
    def test_all_sections_present_no_stubs_added(self):
        report = "\n".join(f"## {s}\nContent here." for s in _REQUIRED_SECTIONS)
        result = _validate_sections(report)
        assert "*(Section could not be generated)*" not in result

    def test_missing_section_stub_appended(self):
        report = "## Executive Summary\nContent."
        result = _validate_sections(report)
        # All other required sections should have stubs
        for section in _REQUIRED_SECTIONS:
            if section != "Executive Summary":
                assert section in result

    def test_h1_heading_also_accepted(self):
        report = "# Executive Summary\nContent.\n\n" + "\n".join(
            f"## {s}\nContent." for s in _REQUIRED_SECTIONS if s != "Executive Summary"
        )
        result = _validate_sections(report)
        assert "*(Section could not be generated)*" not in result


class TestEnforceDisclaimer:
    def test_disclaimer_appended_when_missing(self):
        report = "## Conclusion\nFinal thoughts."
        result = _enforce_disclaimer(report)
        assert "This is not financial advice" in result

    def test_disclaimer_not_duplicated_when_present(self):
        report = "## Conclusion\nFinal thoughts." + _DISCLAIMER
        result = _enforce_disclaimer(report)
        count = result.count("This is not financial advice")
        assert count == 1


class TestCheckQuality:
    def _full_report(self, sections: dict[str, str]) -> str:
        """Build a fake report with given section → content mapping."""
        parts = []
        for section, content in sections.items():
            parts.append(f"## {section}\n{content}")
        return "\n\n".join(parts)

    def test_no_warnings_for_adequate_sections(self):
        # Give each section 200 words — well above all thresholds
        content = " ".join(["word"] * 200)
        sections = {s: content for s in _REQUIRED_SECTIONS}
        report = self._full_report(sections)
        result = _check_quality(report)
        assert result == []

    def test_stub_section_flagged(self):
        sections = {s: " ".join(["word"] * 200) for s in _REQUIRED_SECTIONS}
        # Make Quantitative Analysis a stub (< 80 words threshold)
        sections["Quantitative Analysis"] = "*(Section could not be generated)*"
        report = self._full_report(sections)
        result = _check_quality(report)
        assert "Quantitative Analysis" in result

    def test_empty_report_returns_empty_list(self):
        result = _check_quality("")
        assert result == []

    def test_multiple_short_sections_all_flagged(self):
        sections = {s: " ".join(["word"] * 200) for s in _REQUIRED_SECTIONS}
        sections["Bull Case"] = "one word"    # below 40-word threshold
        sections["Bear Case"] = "two words"   # below 40-word threshold
        report = self._full_report(sections)
        flagged = _check_quality(report)
        assert "Bull Case" in flagged
        assert "Bear Case" in flagged

    def test_returns_list_not_raises(self):
        result = _check_quality("## Random Heading\nContent here.")
        assert isinstance(result, list)


class TestDisclaimerConstant:
    def test_disclaimer_contains_required_text(self):
        assert "This is not financial advice" in _DISCLAIMER
        assert "AI system" in _DISCLAIMER
