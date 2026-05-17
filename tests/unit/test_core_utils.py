"""Unit tests for core/utils.py shared utilities."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ai_financial_analyst.core.utils import (
    assess_data_quality,
    estimate_tokens,
    extract_domain,
    get_first_row,
    null_result,
    safe_float,
)


class TestSafeFloat:
    def test_basic_float(self):
        assert safe_float(3.14) == pytest.approx(3.14)

    def test_int_converted(self):
        assert safe_float(42) == pytest.approx(42.0)

    def test_string_number(self):
        assert safe_float("1.5") == pytest.approx(1.5)

    def test_none_returns_none(self):
        assert safe_float(None) is None

    def test_nan_returns_none(self):
        assert safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert safe_float(float("inf")) is None
        assert safe_float(float("-inf")) is None

    def test_unparseable_string_returns_none(self):
        assert safe_float("not_a_number") is None

    def test_rounds_to_6_decimal_places(self):
        result = safe_float(1.1234567890)
        assert result is not None
        assert len(str(result).split(".")[-1]) <= 6


class TestAssessDataQuality:
    def test_all_present_returns_full(self):
        grade, note = assess_data_quality(
            required={"price": 200.0, "sector": "IT"},
            optional={"pe_ratio": 28.5},
        )
        assert grade == "FULL"
        assert note is None

    def test_missing_required_returns_partial(self):
        grade, note = assess_data_quality(
            required={"price": 200.0, "sector": None},
            optional={"pe_ratio": 28.5},
        )
        assert grade == "PARTIAL"
        assert "sector" in note

    def test_no_optional_with_all_required_returns_full(self):
        # When optional dict is empty/None, treat as "no optional required" → FULL
        grade, note = assess_data_quality(
            required={"price": 200.0},
        )
        assert grade == "FULL"
        assert note is None


class TestNullResult:
    def test_structure(self):
        result = null_result("AAPL", "price_history", "No data")
        assert result["ticker"]       == "AAPL"
        assert result["data_type"]    == "price_history"
        assert result["data_quality"] == "UNAVAILABLE"
        assert result["result"]       is None
        assert result["reason"]       == "No data"
        assert "data_timestamp" in result


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_prose_uses_4_chars_per_token(self):
        # 400 plain chars ÷ 4 = 100 tokens
        result = estimate_tokens("a" * 400)
        assert result == 100

    def test_json_uses_2_chars_per_token(self):
        # JSON with > 10% structural density → 2 chars/token
        json_str = '{"key": "value"}' * 20  # heavy on {, ", :
        result = estimate_tokens(json_str)
        # Should be approximately len / 2
        assert result == pytest.approx(len(json_str) / 2, rel=0.1)

    def test_minimum_one_token(self):
        assert estimate_tokens("x") == 1


class TestGetFirstRow:
    def _df(self):
        return pd.DataFrame(
            {"col": {"Operating Cash Flow": 100.0, "Net Income": 50.0}}
        )

    def test_first_matching_name_returned(self):
        df  = self._df()
        row = get_first_row(df, "Operating Cash Flow", "Total Cash")
        assert row is not None
        assert row["col"] == 100.0

    def test_second_name_used_when_first_missing(self):
        df  = self._df()
        row = get_first_row(df, "NonExistent", "Net Income")
        assert row is not None
        assert row["col"] == 50.0

    def test_none_returned_when_no_match(self):
        df  = self._df()
        row = get_first_row(df, "Foo", "Bar")
        assert row is None

    def test_none_df_returns_none(self):
        assert get_first_row(None, "anything") is None

    def test_empty_df_returns_none(self):
        assert get_first_row(pd.DataFrame(), "anything") is None


class TestExtractDomain:
    def test_standard_url(self):
        assert extract_domain("https://www.reuters.com/article/xyz") == "reuters.com"

    def test_strips_www(self):
        assert extract_domain("https://www.bloomberg.com/news") == "bloomberg.com"

    def test_no_www(self):
        assert extract_domain("https://sec.gov/cgi-bin/browse") == "sec.gov"

    def test_empty_string(self):
        assert extract_domain("") == ""

    def test_invalid_url(self):
        result = extract_domain("not_a_url")
        assert isinstance(result, str)  # no exception; empty or partial string
