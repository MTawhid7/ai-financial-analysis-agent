"""Sector name normalisation: exact GICS → alias → difflib fuzzy match."""

from __future__ import annotations

import difflib

from .static import sector_names

# Common yfinance sector strings that differ from the canonical GICS names.
SECTOR_ALIASES: dict[str, str] = {
    "technology":                "Information Technology",
    "healthcare":                "Health Care",
    "financial services":        "Financials",
    "consumer cyclical":         "Consumer Discretionary",
    "consumer defensive":        "Consumer Staples",
    "basic materials":           "Materials",
    "communication services":    "Communication Services",
    "industrials":               "Industrials",
    "energy":                    "Energy",
    "utilities":                 "Utilities",
    "real estate":               "Real Estate",
    "reits":                     "Real Estate",
}

_FUZZY_CUTOFF = 0.60


def normalise_sector(raw: str) -> tuple[str | None, str | None]:
    """Map any sector string to a canonical GICS sector name.

    Returns (canonical_sector, original_input) on success, (None, None) on failure.

    Resolution order:
      1. Exact case-insensitive GICS match
      2. Alias dict (common yfinance → GICS mappings)
      3. difflib fuzzy match at cutoff=0.60
    """
    sectors = sector_names()
    key     = raw.strip()

    # 1. Exact GICS match (case-insensitive)
    exact = next((s for s in sectors if s.lower() == key.lower()), None)
    if exact:
        return exact, key

    # 2. Alias lookup
    alias = SECTOR_ALIASES.get(key.lower())
    if alias and alias in sectors:
        return alias, key

    # 3. Fuzzy match
    close = difflib.get_close_matches(key, sectors, n=1, cutoff=_FUZZY_CUTOFF)
    if close:
        return close[0], key

    return None, None
