"""Unit tests for the centralised Settings configuration class.

Verifies that:
- Default values are correct
- Env var overrides work
- Validators enforce ordering and range constraints
- get_ttl() dispatches correctly per data type
- String parsing helpers (allowed_origins, admin_user_ids) handle both forms
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_financial_analyst.config.settings import Settings


class TestDefaults:
    def test_primary_model_default(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert "flash" in s.llm_primary_model.lower()

    def test_fallback_model_default(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.llm_fallback_model != s.llm_primary_model

    def test_ttl_price_is_15_minutes(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.ttl_price_s == 15 * 60

    def test_ttl_damodaran_is_30_days(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.ttl_damodaran_s == 30 * 24 * 3600

    def test_budget_thresholds_ordered(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.llm_budget_soft_warn_pct < s.llm_budget_warn_pct < s.llm_budget_defer_pct

    def test_primary_rpm_limit(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.llm_primary_rpm_limit == 15

    def test_fallback_rpm_limit(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.llm_fallback_rpm_limit == 30


class TestEnvOverrides:
    def test_primary_model_overridable(self, monkeypatch):
        monkeypatch.setenv("LLM_PRIMARY_MODEL", "my-custom-model")
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.llm_primary_model == "my-custom-model"

    def test_daily_budget_overridable(self, monkeypatch):
        monkeypatch.setenv("LLM_DAILY_BUDGET", "500")
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.llm_daily_budget == 500

    def test_ttl_price_overridable(self, monkeypatch):
        monkeypatch.setenv("TTL_PRICE_S", "60")
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.ttl_price_s == 60


class TestValidators:
    def test_budget_pct_must_be_positive(self):
        with pytest.raises(ValidationError, match="Budget percentage"):
            Settings(
                google_api_key="k",
                tavily_api_key="k",
                llm_budget_soft_warn_pct=0.0,  # invalid: must be > 0
            )

    def test_budget_pct_must_be_le_one(self):
        with pytest.raises(ValidationError, match="Budget percentage"):
            Settings(
                google_api_key="k",
                tavily_api_key="k",
                llm_budget_soft_warn_pct=1.5,
            )

    def test_budget_thresholds_must_be_ordered(self):
        with pytest.raises(ValidationError, match="ordered"):
            Settings(
                google_api_key="k",
                tavily_api_key="k",
                llm_budget_soft_warn_pct=0.90,  # greater than warn
                llm_budget_warn_pct=0.80,
                llm_budget_defer_pct=0.95,
            )


class TestGetTTL:
    def test_price_history_ttl(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.get_ttl("price_history") == s.ttl_price_s

    def test_fundamentals_ttl(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.get_ttl("fundamentals") == s.ttl_fundamentals_s

    def test_web_search_ttl(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.get_ttl("web_search") == s.ttl_web_search_s

    def test_damodaran_ttl(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.get_ttl("damodaran_sector") == s.ttl_damodaran_s

    def test_unknown_type_returns_default(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.get_ttl("nonexistent_type") == s.ttl_default_s


class TestStringParsing:
    def test_allowed_origins_default_has_localhost(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert any("localhost" in o for o in s.allowed_origins)

    def test_allowed_origins_from_comma_string(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://a.com,http://b.com")
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert "http://a.com" in s.allowed_origins
        assert len(s.allowed_origins) == 2

    def test_allowed_origins_property_strips_spaces(self):
        s = Settings(
            google_api_key="k",
            tavily_api_key="k",
            allowed_origins_raw="  http://a.com , http://b.com  ",
        )
        assert "http://a.com" in s.allowed_origins
        assert "http://b.com" in s.allowed_origins

    def test_admin_ids_from_comma_string(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USER_IDS", "user1,user2, user3")
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert "user1" in s.admin_user_ids
        assert "user3" in s.admin_user_ids
        assert len(s.admin_user_ids) == 3

    def test_admin_ids_empty_by_default(self):
        s = Settings(google_api_key="k", tavily_api_key="k")
        assert s.admin_user_ids == []
