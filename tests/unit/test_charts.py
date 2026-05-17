"""Unit tests for the charts module.

Tests the three Group D improvements:
  1. Intraday data — finer period→interval mapping + _interval_for_range()
  2. Volume profile — generate_volume_profile_chart() returns valid Plotly structure
  3. P/E gradient coloring — _pe_color() produces correct continuous gradient

No yfinance network calls are made — chart generator tests mock _fetch_hist.
"""

from __future__ import annotations

import re

import pytest


# ── Period → interval mapping ─────────────────────────────────────────────────


class TestPeriodIntervals:
    """_interval_for() now returns finer intraday intervals for short periods."""

    def _interval_for(self, period: str) -> str:
        from ai_financial_analyst.charts._data import _interval_for
        return _interval_for(period)

    def test_1d_uses_5m(self):
        assert self._interval_for("1d") == "5m"

    def test_5d_uses_15m(self):
        assert self._interval_for("5d") == "15m"

    def test_1mo_uses_1h(self):
        assert self._interval_for("1mo") == "1h"

    def test_3mo_uses_1d(self):
        assert self._interval_for("3mo") == "1d"

    def test_1y_uses_1d(self):
        assert self._interval_for("1y") == "1d"

    def test_2y_uses_1wk(self):
        assert self._interval_for("2y") == "1wk"

    def test_10y_uses_1mo(self):
        assert self._interval_for("10y") == "1mo"


class TestIntervalForRange:
    """_interval_for_range() uses intraday intervals for short date spans."""

    def _interval_for_range(self, start: str, end: str | None = None) -> str:
        from ai_financial_analyst.charts._data import _interval_for_range
        return _interval_for_range(start, end, period="1y")

    def test_3_day_range_uses_5m(self):
        from datetime import date, timedelta
        today = date.today()
        start = str(today - timedelta(days=3))
        assert self._interval_for_range(start) == "5m"

    def test_20_day_range_uses_1h(self):
        from datetime import date, timedelta
        today = date.today()
        start = str(today - timedelta(days=20))
        assert self._interval_for_range(start) == "1h"

    def test_90_day_range_uses_1wk(self):
        # 90 days > 60-day threshold → weekly bars
        from datetime import date, timedelta
        today = date.today()
        start = str(today - timedelta(days=90))
        assert self._interval_for_range(start) == "1wk"

    def test_no_start_falls_back_to_period(self):
        from ai_financial_analyst.charts._data import _interval_for_range
        # Without start, falls back to period-based mapping
        assert _interval_for_range(None, None, period="5d") == "15m"
        assert _interval_for_range(None, None, period="1y") == "1d"


# ── P/E gradient coloring ─────────────────────────────────────────────────────


class TestPEColorGradient:
    """_pe_color() produces a continuously interpolated green→blue→red color."""

    def _color(self, premium: float) -> str:
        from ai_financial_analyst.charts.pipeline import _pe_color
        return _pe_color(premium)

    def test_returns_valid_hex_string(self):
        assert re.match(r"^#[0-9a-f]{6}$", self._color(0))

    def test_large_discount_is_green(self):
        color = self._color(-50)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        # Green channel should dominate
        assert g > r and g > b

    def test_parity_is_blue(self):
        color = self._color(0)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        # Blue channel should dominate
        assert b > r and b > g

    def test_large_premium_is_red(self):
        color = self._color(50)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        # Red channel should dominate
        assert r > g and r > b

    def test_clamped_below_minus50(self):
        # premium=-100 and premium=-50 should produce the same color
        assert self._color(-100) == self._color(-50)

    def test_clamped_above_plus50(self):
        # premium=100 and premium=50 should produce the same color
        assert self._color(100) == self._color(50)

    def test_gradient_is_monotone_red_channel(self):
        """As premium increases from -50 to +50, red channel increases."""
        colors = [self._color(p) for p in range(-50, 51, 10)]
        reds   = [int(c[1:3], 16) for c in colors]
        assert reds == sorted(reds)

    def test_gradient_green_channel_lower_at_premium_than_discount(self):
        """Green channel at large premium should be significantly lower than at large discount.

        The gradient transitions green→blue→red, so the green channel has a slight
        bump near parity (both green and blue have high green values), but the
        overall direction from discount to premium is downward.
        """
        discount_green = int(self._color(-50)[3:5], 16)
        premium_green  = int(self._color(+50)[3:5], 16)
        assert premium_green < discount_green - 50  # meaningful drop across the range


# ── Volume profile chart ──────────────────────────────────────────────────────


class TestVolumeProfileChart:
    """generate_volume_profile_chart() returns valid Plotly structure."""

    def _make_hist(self, n: int = 50):
        """Return a minimal DataFrame mimicking yfinance OHLCV output."""
        import pandas as pd
        import numpy as np

        rng = pd.date_range("2024-01-01", periods=n, freq="D")
        prices = 100 + np.cumsum(np.random.randn(n))
        return pd.DataFrame({
            "Open":   prices * 0.99,
            "High":   prices * 1.01,
            "Low":    prices * 0.98,
            "Close":  prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        }, index=rng)

    def test_returns_dict_with_data_and_layout(self):
        from unittest.mock import patch
        from ai_financial_analyst.charts.volume import generate_volume_profile_chart

        hist = self._make_hist()
        with patch("ai_financial_analyst.charts.volume._fetch_hist", return_value=(hist, "AAPL")):
            result = generate_volume_profile_chart("AAPL", period="1y")

        assert result is not None
        assert "data" in result
        assert "layout" in result
        assert len(result["data"]) == 1

    def test_bar_orientation_is_horizontal(self):
        from unittest.mock import patch
        from ai_financial_analyst.charts.volume import generate_volume_profile_chart

        hist = self._make_hist()
        with patch("ai_financial_analyst.charts.volume._fetch_hist", return_value=(hist, "AAPL")):
            result = generate_volume_profile_chart("AAPL")

        assert result["data"][0]["orientation"] == "h"

    def test_bin_count_matches_bins_param(self):
        from unittest.mock import patch
        from ai_financial_analyst.charts.volume import generate_volume_profile_chart

        hist = self._make_hist(100)
        with patch("ai_financial_analyst.charts.volume._fetch_hist", return_value=(hist, "AAPL")):
            result = generate_volume_profile_chart("AAPL", bins=20)

        assert len(result["data"][0]["y"]) == 20

    def test_highest_volume_bin_is_amber(self):
        from unittest.mock import patch
        from ai_financial_analyst.charts.volume import generate_volume_profile_chart
        from ai_financial_analyst.charts._theme import _AMBER

        hist = self._make_hist()
        with patch("ai_financial_analyst.charts.volume._fetch_hist", return_value=(hist, "AAPL")):
            result = generate_volume_profile_chart("AAPL", bins=10)

        colors = result["data"][0]["marker"]["color"]
        volumes = result["data"][0]["x"]
        max_idx = volumes.index(max(volumes))
        assert colors[max_idx] == _AMBER

    def test_handles_empty_dataframe(self):
        import pandas as pd
        from unittest.mock import patch
        from ai_financial_analyst.charts.volume import generate_volume_profile_chart

        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        with patch("ai_financial_analyst.charts.volume._fetch_hist", return_value=(empty, "AAPL")):
            result = generate_volume_profile_chart("AAPL")

        assert result is None
