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


class TestPerToolTracking:
    def test_record_tool_call_increments_per_tool_counter(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        tracker.record_tool_call("yahoo_finance", is_primary=False)
        stats = tracker.get_stats()
        assert stats["per_tool_calls"]["yahoo_finance"] == 1

    def test_record_tool_call_primary_increments_primary_calls(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        tracker.record_tool_call("report_writer", is_primary=True)
        assert tracker.get_stats()["primary_calls"] == 1

    def test_record_tool_call_sub_increments_sub_calls(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        tracker.record_tool_call("web_search", is_primary=False)
        assert tracker.get_stats()["sub_calls"] == 1

    def test_multiple_tools_tracked_independently(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        tracker.record_tool_call("yahoo_finance", is_primary=False)
        tracker.record_tool_call("yahoo_finance", is_primary=False)
        tracker.record_tool_call("benchmark_lookup", is_primary=True)
        stats = tracker.get_stats()
        assert stats["per_tool_calls"]["yahoo_finance"] == 2
        assert stats["per_tool_calls"]["benchmark_lookup"] == 1

    def test_cache_hit_with_tool_name_tracks_cache_credits(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        tracker.record_cache_hit("yahoo_finance")
        stats = tracker.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_credits"] == 1
        assert stats["per_tool_calls"].get("cache:yahoo_finance") == 1

    def test_cache_credits_reduce_effective_calls(self):
        tracker = RequestBudgetTracker(daily_budget=100)
        tracker.record_primary_call()
        tracker.record_primary_call()
        tracker.record_cache_hit()  # credit one back
        assert tracker.effective_calls == 1

    def test_effective_calls_never_negative(self):
        tracker = RequestBudgetTracker(daily_budget=100)
        tracker.record_cache_hit()
        tracker.record_cache_hit()
        assert tracker.effective_calls == 0  # max(0, -2) = 0


class TestRpmTracking:
    def test_get_stats_includes_rpm_fields(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        stats = tracker.get_stats()
        assert "primary_rpm_current" in stats
        assert "sub_rpm_current" in stats
        assert stats["primary_rpm_current"] == 0
        assert stats["sub_rpm_current"] == 0

    def test_rpm_increments_with_calls(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        tracker.record_primary_call()
        tracker.record_primary_call()
        assert tracker.get_stats()["primary_rpm_current"] == 2

    def test_sub_rpm_tracked_separately(self):
        tracker = RequestBudgetTracker(daily_budget=1500)
        tracker.record_sub_call()
        stats = tracker.get_stats()
        assert stats["sub_rpm_current"] == 1
        assert stats["primary_rpm_current"] == 0

    def test_rpm_warning_fires_when_over_limit(self, caplog):
        tracker = RequestBudgetTracker(daily_budget=10000)
        with caplog.at_level(logging.WARNING):
            # Exceed the 15 RPM primary limit
            for _ in range(16):
                tracker.record_primary_call()
        assert "RPM alert" in caplog.text
