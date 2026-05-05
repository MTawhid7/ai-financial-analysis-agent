"""Unit tests for RequestBudgetTracker."""

import logging
import pytest

from ai_financial_analyst.core.budget_tracker import RequestBudgetTracker


class TestRequestBudgetTracker:
    def test_initial_state(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        stats = tracker.get_stats()
        assert stats["total_calls"] == 0
        assert stats["cache_hits"] == 0
        assert stats["budget_used_pct"] == 0.0

    def test_primary_call_counted(self):
        tracker = RequestBudgetTracker(daily_budget=100)
        tracker.record_primary_call()
        assert tracker.get_stats()["primary_calls"] == 1
        assert tracker.total_calls == 1

    def test_sub_call_counted(self):
        tracker = RequestBudgetTracker(daily_budget=100)
        tracker.record_sub_call()
        assert tracker.get_stats()["sub_calls"] == 1

    def test_cache_hit_counted(self):
        tracker = RequestBudgetTracker(daily_budget=100)
        tracker.record_cache_hit()
        assert tracker.get_stats()["cache_hits"] == 1

    def test_cache_hit_does_not_count_toward_api_calls(self):
        tracker = RequestBudgetTracker(daily_budget=100)
        tracker.record_cache_hit()
        assert tracker.total_calls == 0

    def test_budget_warning_at_80_percent(self, caplog):
        tracker = RequestBudgetTracker(daily_budget=10)
        with caplog.at_level(logging.WARNING):
            for _ in range(8):
                tracker.record_primary_call()
        assert "Budget alert" in caplog.text

    def test_budget_warning_fires_only_once(self, caplog):
        tracker = RequestBudgetTracker(daily_budget=10)
        with caplog.at_level(logging.WARNING):
            for _ in range(15):
                tracker.record_primary_call()
        warning_count = caplog.text.count("Budget alert")
        assert warning_count == 1
