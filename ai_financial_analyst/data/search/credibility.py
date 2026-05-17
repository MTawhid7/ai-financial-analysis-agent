"""Source credibility scoring for web search results.

Classifies source domains into tiers (1 = highest confidence → 4 = unknown).
Used by TavilySearchClient to sort results by credibility before returning them.
"""

from __future__ import annotations

from ...core.utils import extract_domain

# Tier 1: Major financial wire services + regulators
# Tier 2: Established financial media
# Tier 3: General reputable press with financial coverage
# Tier 4: Default — unranked / unknown source (not in this dict)
SOURCE_TIERS: dict[str, int] = {
    # Tier 1 — highest confidence
    "reuters.com":         1,
    "apnews.com":          1,
    "bloomberg.com":       1,
    "ft.com":              1,
    "wsj.com":             1,
    "sec.gov":             1,
    "finra.org":           1,
    "federalreserve.gov":  1,
    "bls.gov":             1,
    "treasury.gov":        1,
    # Tier 2 — established financial media
    "cnbc.com":            2,
    "marketwatch.com":     2,
    "barrons.com":         2,
    "forbes.com":          2,
    "businessinsider.com": 2,
    "seekingalpha.com":    2,
    "investopedia.com":    2,
    "thestreet.com":       2,
    "morningstar.com":     2,
    # Tier 3 — general reputable press
    "nytimes.com":         3,
    "bbc.com":             3,
    "economist.com":       3,
    "washingtonpost.com":  3,
    "theguardian.com":     3,
    "ap.org":              3,
}

_DEFAULT_TIER = 4


def score_source(url: str) -> int:
    """Return the credibility tier (1–4) for a given URL.

    1 = major wire service / regulator (highest confidence)
    4 = unknown source (lowest confidence)
    """
    domain = extract_domain(url)
    return SOURCE_TIERS.get(domain, _DEFAULT_TIER)
