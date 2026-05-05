"""BenchmarkLookupTool — static GICS sector benchmark data (zero API cost)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.tools import tool
from pydantic import Field

from .base import StrictToolInput, ToolError, ErrorType, safe_tool_call

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parent.parent / "data" / "benchmarks.json"

# Load once at import time — no network call required.
with _DATA_PATH.open() as _f:
    _BENCHMARKS: dict = json.load(_f)


class BenchmarkLookupInput(StrictToolInput):
    gics_sector: str = Field(
        description=(
            "GICS sector name. One of: Information Technology, Health Care, "
            "Financials, Consumer Discretionary, Consumer Staples, Industrials, "
            "Communication Services, Energy, Utilities, Real Estate, Materials."
        )
    )


@tool("benchmark_lookup", args_schema=BenchmarkLookupInput)
def benchmark_lookup_tool(gics_sector: str) -> str:
    """Return sector-average P/E, EV/EBITDA, and Price/Book for a GICS sector.

    Data is bundled statically — no network call, zero API cost.
    """
    def _run():
        sectors = _BENCHMARKS.get("sectors", {})
        # Case-insensitive match
        match = next(
            (k for k in sectors if k.lower() == gics_sector.strip().lower()),
            None,
        )
        if match is None:
            available = list(sectors.keys())
            return ToolError(
                error_type=ErrorType.TOOL_ERROR,
                tool="benchmark_lookup",
                message=f"Sector '{gics_sector}' not found. Available: {available}",
                input={"gics_sector": gics_sector},
            ).to_json()

        data = sectors[match]
        return json.dumps({
            "sector": match,
            "pe_ratio_sector_avg": data["pe_ratio"],
            "ev_ebitda_sector_avg": data["ev_ebitda"],
            "price_to_book_sector_avg": data["price_to_book"],
            "peer_examples": data["peer_examples"],
            "source": "Bundled static data (approximate 2024 averages)",
        })

    return safe_tool_call("benchmark_lookup", _run, {"gics_sector": gics_sector})
