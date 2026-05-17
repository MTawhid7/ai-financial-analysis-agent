"""Shared data structures and utilities for the data access layer.

DataResult is the canonical return type for every data-fetch function.
The tool wrappers in tools/ call DataResult.to_json() to produce the
same JSON wire format that callers (researcher, quant_analyst) already expect.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


# ── DataResult ────────────────────────────────────────────────────────────────

@dataclass
class DataResult:
    """Canonical return type for all financial data-fetch functions.

    Fields:
        ticker          Stock ticker symbol (uppercase)
        data_type       One of the 7 Yahoo data types or a custom source name
        data_quality    "FULL" | "PARTIAL" | "UNAVAILABLE"
        data_timestamp  ISO 8601 UTC timestamp of when the data was fetched
        degradation_note  Human-readable explanation when quality < FULL; None otherwise
        payload         The data-specific fields (varies per data_type)
    """

    ticker:           str
    data_type:        str
    data_quality:     str
    data_timestamp:   str
    degradation_note: str | None
    payload:          dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the full flat dict that tools/yahoo_finance.py JSON-serialises.

        Merges the envelope fields (ticker, data_type, …) with the payload so
        callers can do result["current_price"] as before.
        """
        return {
            "ticker":           self.ticker,
            "data_type":        self.data_type,
            "data_timestamp":   self.data_timestamp,
            "data_quality":     self.data_quality,
            "degradation_note": self.degradation_note,
            **self.payload,
        }

    def to_json(self) -> str:
        """Serialize to the same JSON string the old tool functions produced."""
        return json.dumps(self.to_dict())


def null_result(ticker: str, data_type: str, reason: str) -> DataResult:
    """Return an UNAVAILABLE DataResult sentinel for when a fetch yields nothing."""
    return DataResult(
        ticker           = ticker,
        data_type        = data_type,
        data_quality     = "UNAVAILABLE",
        data_timestamp   = utc_now(),
        degradation_note = reason,
        payload          = {"result": None, "reason": reason},
    )


# ── Shared fetch utilities ────────────────────────────────────────────────────

def utc_now() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def safe_float(val: Any) -> float | None:
    """Convert val to a rounded float; return None for non-numeric / NaN / Inf."""
    try:
        f = float(val)
        return round(f, 6) if not (math.isnan(f) or math.isinf(f)) else None
    except (TypeError, ValueError):
        return None


def get_first_row(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
    """Return the first matching index row from a financial DataFrame."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def assess_quality(
    required: dict[str, Any],
    optional: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Return (quality_grade, degradation_note) for a fetched payload.

    "FULL"    — all required fields non-None; at least one optional non-None (or no optionals)
    "PARTIAL" — at least one required field is None, or all optionals are None
    """
    optional = optional or {}
    missing  = [k for k, v in required.items() if v is None]
    has_opt  = any(v is not None for v in optional.values()) if optional else True

    if not missing and has_opt:
        return "FULL", None

    parts: list[str] = []
    if missing:
        parts.append(f"Missing: {', '.join(missing)}")
    if optional and not has_opt:
        parts.append("No optional fields available")
    return "PARTIAL", "; ".join(parts) or "Incomplete data"
