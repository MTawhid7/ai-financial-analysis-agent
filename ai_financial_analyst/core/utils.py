"""Shared utility functions used across multiple modules.

Extracted from yahoo_finance.py (_sf, _null, _get_row) and short_term.py
(_estimate_tokens) to eliminate duplication and enable independent testing.

All functions are stateless — no I/O, no side effects.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ── Numeric helpers ───────────────────────────────────────────────────────────

def safe_float(val: Any) -> float | None:
    """Convert val to float, returning None for non-numeric or NaN/Inf values.

    Handles None, strings that aren't numeric, and IEEE edge cases gracefully.
    """
    try:
        f = float(val)
        return round(f, 6) if not (math.isnan(f) or math.isinf(f)) else None
    except (TypeError, ValueError):
        return None


# ── Data quality assessment ───────────────────────────────────────────────────

def assess_data_quality(
    required: dict[str, Any],
    optional: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Return (quality_grade, degradation_note) for a fetched data payload.

    quality_grade:
      "FULL"    — all required fields present; at least one optional field present
      "PARTIAL" — some required or all optional fields missing
      "UNAVAILABLE" — should not reach here; caller returns _null() directly

    Args:
        required: mapping of field_name → value for mandatory fields
        optional: mapping of field_name → value for bonus fields (may be None/empty)

    Example:
        grade, note = assess_data_quality(
            required={"price": 200.0, "sector": "IT", "market_cap": 3e12},
            optional={"pe_ratio": 28.5, "ev_to_ebitda": None},
        )
    """
    optional = optional or {}

    missing_required = [k for k, v in required.items() if v is None]
    has_any_optional = any(v is not None for v in optional.values()) if optional else True

    if not missing_required and has_any_optional:
        return "FULL", None

    parts: list[str] = []
    if missing_required:
        parts.append(f"Missing required fields: {', '.join(missing_required)}")
    if optional and not has_any_optional:
        parts.append("No optional fields available")

    return "PARTIAL", "; ".join(parts) if parts else "Incomplete data"


# ── Null / unavailable sentinel ───────────────────────────────────────────────

def null_result(ticker: str, data_type: str, reason: str) -> dict[str, Any]:
    """Return a standardised null payload for unavailable data.

    Used by all Yahoo Finance data-type fetch functions when the API returns
    nothing usable. The caller typically JSON-serialises this dict.
    """
    from datetime import datetime, timezone
    return {
        "ticker":         ticker,
        "data_type":      data_type,
        "data_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_quality":   "UNAVAILABLE",
        "result":         None,
        "reason":         reason,
    }


# ── Token estimation ──────────────────────────────────────────────────────────

def estimate_tokens(content: str) -> int:
    """Estimate LLM token count from a string with content-type awareness.

    JSON-heavy content (high density of `{`, `[`, `"`) is approximately
    2 chars/token because JSON syntax characters are dense tokens.
    Prose and markdown are approximately 4 chars/token.

    This is a heuristic, not a precise tokenizer.  Use for budget tracking
    and window management; do not rely on it for hard token-count limits.
    """
    if not content:
        return 0
    structural = content.count("{") + content.count("[") + content.count('"')
    density    = structural / len(content)
    chars_per_token = 2.0 if density > 0.10 else 4.0
    return max(1, int(len(content) / chars_per_token))


# ── DataFrame helper ──────────────────────────────────────────────────────────

def get_first_row(df: "pd.DataFrame | None", *names: str) -> "pd.Series | None":
    """Return the first matching row from a financial DataFrame by row label.

    Tries each name in order; returns None if the DataFrame is empty or no
    name matches. Used by Yahoo Finance data-type functions to extract rows
    from statement DataFrames that use different column labels across API versions.

    Example:
        ocf = get_first_row(cashflow_df, "Operating Cash Flow", "Total Cash From Operating Activities")
    """
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


# ── URL helpers ───────────────────────────────────────────────────────────────

def extract_domain(url: str) -> str:
    """Extract the registered domain from a URL string.

    Returns an empty string if the URL cannot be parsed.

    Example:
        extract_domain("https://www.reuters.com/article/xyz")  → "reuters.com"
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host   = parsed.hostname or ""
        # Strip www. prefix
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""
