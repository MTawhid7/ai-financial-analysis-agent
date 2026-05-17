"""Sector benchmark lookup — live Damodaran with static fallback.

Extracts the lookup logic from tools/benchmark_lookup.py into a pure data
module that has no LangChain @tool dependency. The tool wrapper delegates here.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings
from ...core.cache import ResultCache
from .normalizer import normalise_sector
from .static import get_static

logger = logging.getLogger(__name__)

_cache = ResultCache()


def get_sector_benchmarks(raw_sector: str, country: str | None = None) -> dict[str, Any]:
    """Return sector valuation benchmarks for the given sector string.

    Resolves raw_sector via normalise_sector() (exact → alias → fuzzy),
    fetches from Damodaran with 30-day cache, falls back to static JSON.
    Adds geographic context when country is provided and non-US.

    Returns the same dict structure as the old benchmark_lookup_tool.
    """
    sector, matched_from = normalise_sector(raw_sector)
    if sector is None:
        return {
            "error_type": "TOOL_ERROR",
            "tool":       "benchmark_lookup",
            "message":    f"Sector '{raw_sector}' not recognised.",
        }

    data = _get_benchmarks(sector)

    if matched_from and matched_from != sector:
        data["sector_matched_from"] = matched_from
        from .normalizer import SECTOR_ALIASES
        data["sector_match_method"] = (
            "alias" if matched_from.lower() in SECTOR_ALIASES else "fuzzy"
        )

    if country:
        geo = _geographic_context(country)
        if geo is not None:
            data["geographic_context"] = geo

    return data


def _get_benchmarks(gics_sector: str) -> dict:
    def _fetch() -> dict:
        try:
            from ...tools.benchmark_lookup import _fetch_damodaran_all
            live = _fetch_damodaran_all()
            if live and gics_sector in live:
                return live[gics_sector]
        except Exception as exc:
            logger.warning("Damodaran live fetch failed: %s", exc)
        return get_static(gics_sector)

    result, _ = _cache.get_or_fetch(
        "damodaran_sector", {"sector": gics_sector}, _fetch, ttl=settings.ttl_damodaran_s
    )
    return result or get_static(gics_sector)


# ── Geographic context (re-used from tools/benchmark_lookup.py) ──────────────

_EM_COUNTRIES: frozenset[str] = frozenset({
    "china", "india", "brazil", "russia", "taiwan", "south korea", "indonesia",
    "thailand", "malaysia", "philippines", "vietnam", "turkey", "mexico",
    "argentina", "colombia", "chile", "peru", "egypt", "saudi arabia",
    "united arab emirates", "qatar", "south africa", "nigeria", "kenya",
    "hong kong", "pakistan", "bangladesh", "sri lanka", "myanmar",
    "czechia", "czech republic", "poland", "hungary", "romania", "ukraine",
})

_DEVELOPED_EX_US: frozenset[str] = frozenset({
    "united kingdom", "germany", "france", "japan", "australia", "canada",
    "netherlands", "switzerland", "sweden", "norway", "denmark", "finland",
    "belgium", "austria", "ireland", "spain", "italy", "portugal",
    "singapore", "new zealand", "israel", "luxembourg",
})


def _geographic_context(country: str) -> dict | None:
    c = country.lower().strip()
    if c in ("united states", "usa", "us", "u.s.", "u.s.a."):
        return None
    if c in _EM_COUNTRIES:
        return {
            "geographic_scope": "emerging_market",
            "country": country,
            "benchmark_note": (
                f"{country} is classified as an Emerging Market. "
                "EM companies typically trade at a 25–35% P/E discount to US sector averages."
            ),
            "typical_pe_discount_vs_us": -30,
        }
    if c in _DEVELOPED_EX_US:
        return {
            "geographic_scope": "developed_ex_us",
            "country": country,
            "benchmark_note": (
                f"{country} is a developed market outside the US. "
                "Sector multiples may differ by 10–20%."
            ),
            "typical_pe_discount_vs_us": -12,
        }
    return {
        "geographic_scope": "non_us_unclassified",
        "country": country,
        "benchmark_note": (
            f"{country} market data is not in our classification. "
            "US benchmarks are provided as directional reference only."
        ),
        "typical_pe_discount_vs_us": None,
    }
