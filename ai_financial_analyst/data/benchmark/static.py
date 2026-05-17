"""Static JSON benchmark fallback — lazily loaded on first access.

The critical fix vs the old benchmark_lookup.py: the JSON file is NOT loaded
at module import time. It is loaded on the first call to load() and then
cached for the process lifetime. Import errors from a missing file no longer
propagate at module import time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_PATH  = Path(__file__).parent.parent.parent / "data" / "benchmarks.json"
_benchmarks: dict | None = None


def load() -> dict:
    """Return the static benchmarks dict, loading from disk on first call."""
    global _benchmarks
    if _benchmarks is None:
        try:
            with _DATA_PATH.open() as f:
                _benchmarks = json.load(f)
        except Exception as exc:
            logger.error("Failed to load static benchmarks from %s: %s", _DATA_PATH, exc)
            _benchmarks = {"sectors": {}}
    return _benchmarks


def get_static(gics_sector: str) -> dict:
    """Return static benchmark data for a GICS sector."""
    data = load().get("sectors", {}).get(gics_sector, {})
    return {
        "sector":                    gics_sector,
        "pe_ratio_sector_avg":       data.get("pe_ratio"),
        "forward_pe_sector_avg":     None,
        "ev_ebitda_sector_avg":      data.get("ev_ebitda"),
        "price_to_book_sector_avg":  data.get("price_to_book"),
        "price_to_sales_sector_avg": data.get("price_to_sales"),
        "operating_margin_pct":      None,
        "beta_sector_avg":           None,
        "peer_examples":             data.get("peer_examples", []),
        "source":                    "Bundled static data (approximate 2024 averages)",
        "data_year":                 2024,
    }


def sector_names() -> list[str]:
    """Return the list of known GICS sector names from the static data."""
    return list(load().get("sectors", {}).keys())
