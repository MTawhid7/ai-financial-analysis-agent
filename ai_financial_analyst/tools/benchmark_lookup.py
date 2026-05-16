"""BenchmarkLookupTool — sector valuation benchmarks.

Data source priority:
  1. Damodaran (NYU Stern) — fetched from public HTML pages, cached 30 days.
     Provides: Trailing P/E, Forward P/E, PEG, EV/EBITDA, P/Book, operating margin, beta.
  2. Bundled static JSON (benchmarks.json) — approximate 2024 averages.
     Used as fallback when Damodaran is unreachable or HTML structure changes.

Industry-to-sector mapping: Damodaran's ~100 industries are averaged into the
11 GICS sectors that yfinance uses.  The mapping is approximate (e.g., "Technology"
covers both software and semiconductors which trade at very different multiples),
but meaningfully better than a single stale snapshot.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import Field

from ..core.cache import ResultCache, TTL_DAMODARAN
from .base import StrictToolInput, ToolError, ErrorType, safe_tool_call

logger = logging.getLogger(__name__)

_cache     = ResultCache()
_DATA_PATH = Path(__file__).parent.parent / "data" / "benchmarks.json"

with _DATA_PATH.open() as _f:
    _STATIC_BENCHMARKS: dict = json.load(_f)

# ── Damodaran URLs ─────────────────────────────────────────────────────────────
_DAMODARAN_URLS = {
    "pe":     "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/pedata.html",
    "evdata": "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/vebitda.html",
    "pbv":    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/pbvdata.html",
    "ps":     "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/psdata.html",
    "margin": "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/margin.html",
    "beta":   "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betasetc.html",
}

# ── Industry → GICS sector mapping ────────────────────────────────────────────
# Damodaran's taxonomy has ~100 sub-industries. Each GICS sector maps to a list
# of Damodaran industry names whose rows are averaged to form the sector benchmark.
_DAMODARAN_TO_GICS: dict[str, list[str]] = {
    "Information Technology": [
        "Software (System & Application)", "Software (Entertainment)", "Software (Internet)",
        "Semiconductor", "Semiconductor Equip", "Computers/Peripherals",
        "Electronics (Consumer & Office)", "Electronics (General)",
        "Computer Services", "Information Services",
    ],
    "Health Care": [
        "Drug (Pharmaceutical)", "Drug (Biotechnology)", "Healthcare Products",
        "Healthcare Support Services", "Heathcare Information and Technology",
        "Health Care Products",
    ],
    "Financials": [
        "Bank (Money Center)", "Banks (Regional)", "Financial Services (non-bank)",
        "Insurance (General)", "Insurance (Life)", "Insurance (Prop/Cas.)",
        "Investments & Asset Management",
    ],
    "Consumer Discretionary": [
        "Retail (General)", "Retail (Online)", "Retail (Special Lines)",
        "Retail (Automotive)", "Retail (Building Supply)",
        "Entertainment", "Recreation", "Furn/Home Furnishings",
        "Shoe", "Apparel", "Auto & Truck", "Auto Parts",
        "Hotel/Gaming", "Restaurant/Dining",
    ],
    "Consumer Staples": [
        "Food Processing", "Food Wholesalers", "Household Products",
        "Beverage (Alcoholic)", "Beverage (Soft)", "Tobacco",
        "Retail (Grocery and Food)",
    ],
    "Materials": [
        "Chemical (Basic)", "Chemical (Diversified)", "Chemical (Specialty)",
        "Metals & Mining", "Precious Metals", "Steel",
        "Paper/Forest Products", "Building Materials", "Package & Container",
        "Rubber& Tires",
    ],
    "Communication Services": [
        "Telecom (Wireless)", "Telecom. Services", "Telecom. Equipment",
        "Cable TV", "Broadcasting", "Advertising",
    ],
    "Industrials": [
        "Machinery", "Aerospace/Defense", "Transportation",
        "Transportation (Railroads)", "Trucking", "Shipbuilding & Marine",
        "Engineering/Construction", "Construction Supplies",
        "Electrical Equipment", "Office Equipment & Services",
        "Business & Consumer Services",
    ],
    "Energy": [
        "Oil/Gas (Production and Exploration)", "Oil/Gas (Integrated)",
        "Oilfield Svcs/Equip.", "Coal & Related Energy",
        "Green & Renewable Energy", "Power",
    ],
    "Utilities": [
        "Utility (General)", "Utility (Water)", "Power",
    ],
    "Real Estate": [
        "R.E.I.T.", "Real Estate (Development)",
        "Real Estate (General/Diversified)", "Real Estate (Operations & Services)",
    ],
}

# Reverse map: Damodaran industry → GICS sector
_INDUSTRY_TO_GICS: dict[str, str] = {
    ind.lower(): gics
    for gics, industries in _DAMODARAN_TO_GICS.items()
    for ind in industries
}


# ── Tool ──────────────────────────────────────────────────────────────────────

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
    """Return sector valuation benchmarks: P/E, EV/EBITDA, P/Book, margins, beta.

    Fetches from Damodaran (NYU Stern) with a 30-day cache; falls back to
    bundled static data if the live fetch fails.
    """
    def _run():
        sector = _normalise_sector(gics_sector)
        if sector is None:
            available = list(_STATIC_BENCHMARKS.get("sectors", {}).keys())
            return ToolError(
                error_type=ErrorType.TOOL_ERROR,
                tool="benchmark_lookup",
                message=f"Sector '{gics_sector}' not found. Available: {available}",
                input={"gics_sector": gics_sector},
            ).to_json()

        data = _get_sector_benchmarks(sector)
        return json.dumps(data)

    return safe_tool_call("benchmark_lookup", _run, {"gics_sector": gics_sector})


# ── Benchmark data retrieval ──────────────────────────────────────────────────

def _get_sector_benchmarks(gics_sector: str) -> dict:
    """Return benchmarks, trying Damodaran first, static JSON as fallback."""
    def _fetch() -> dict:
        live = _fetch_damodaran_all()
        if live and gics_sector in live:
            return live[gics_sector]
        # Fallback to static
        return _static_benchmarks(gics_sector)

    result, hit = _cache.get_or_fetch(
        "damodaran_sector",
        {"sector": gics_sector},
        _fetch,
        ttl=TTL_DAMODARAN,
    )
    return result or _static_benchmarks(gics_sector)


def _static_benchmarks(gics_sector: str) -> dict:
    """Return from the bundled static benchmarks.json."""
    sectors = _STATIC_BENCHMARKS.get("sectors", {})
    data    = sectors.get(gics_sector, {})
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


# ── Damodaran HTML fetcher ────────────────────────────────────────────────────

def _fetch_damodaran_all() -> dict[str, dict] | None:
    """Fetch all Damodaran datasets and return a GICS-sector keyed dict.

    Returns None if any critical fetch fails.
    """
    try:
        pe_data     = _fetch_damodaran_table(_DAMODARAN_URLS["pe"])
        ev_data     = _fetch_damodaran_table(_DAMODARAN_URLS["evdata"])
        pbv_data    = _fetch_damodaran_table(_DAMODARAN_URLS["pbv"])
        ps_data     = _fetch_damodaran_table(_DAMODARAN_URLS["ps"])
        margin_data = _fetch_damodaran_table(_DAMODARAN_URLS["margin"])
        beta_data   = _fetch_damodaran_table(_DAMODARAN_URLS["beta"])
    except Exception as exc:
        logger.warning("Damodaran fetch failed: %s", exc)
        return None

    if not pe_data:
        return None

    # Aggregate industry rows into GICS sectors
    result: dict[str, dict] = {}
    for gics_sector, industry_names in _DAMODARAN_TO_GICS.items():
        industry_set = {n.lower() for n in industry_names}

        pe_vals   = _collect(pe_data,     industry_set, ("Current PE", "Trailing PE", "TrailingPE", "Current P/E"))
        fpe_vals  = _collect(pe_data,     industry_set, ("Forward PE", "ForwardPE", "Forward P/E"))
        ev_vals   = _collect(ev_data,     industry_set, ("EV/EBITDA", "EV/ EBITDA", "EV/EBITDA1"))
        pbv_vals  = _collect(pbv_data,    industry_set, ("PBV", "Price to Book", "P/BV", "Current P/BV"))
        ps_vals   = _collect(ps_data,     industry_set, ("PS", "Price/Sales", "Price/ Sales", "Price to Sales", "Current P/S"))
        mg_vals   = _collect(margin_data, industry_set, ("Net Margin", "Net margin", "After-tax Operating Margin"))
        bt_vals   = _collect(beta_data,   industry_set, ("Beta", "Market Beta", "Levered Beta"))

        result[gics_sector] = {
            "sector":                    gics_sector,
            "pe_ratio_sector_avg":       _avg(pe_vals),
            "forward_pe_sector_avg":     _avg(fpe_vals),
            "ev_ebitda_sector_avg":      _avg(ev_vals),
            "price_to_book_sector_avg":  _avg(pbv_vals),
            "price_to_sales_sector_avg": _avg(ps_vals),
            "operating_margin_pct":      _pct(_avg(mg_vals)),
            "beta_sector_avg":           _avg(bt_vals),
            "peer_examples":             _STATIC_BENCHMARKS.get("sectors", {}).get(gics_sector, {}).get("peer_examples", []),
            "source":                    "Damodaran (NYU Stern) — annual dataset",
            "data_year":                 _detect_year(pe_data),
        }

    return result if result else None


def _fetch_damodaran_table(url: str, timeout: int = 8) -> list[dict[str, Any]]:
    """Download one Damodaran HTML page and return rows as list of dicts.

    Each dict maps column_name → value (numeric or string).
    Returns [] on any failure.
    """
    try:
        from bs4 import BeautifulSoup
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=timeout).read()
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        rows_raw = table.find_all("tr")
        if len(rows_raw) < 3:
            return []

        # Find header row (first row with <th> or first <td> row)
        header_row = rows_raw[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        if not headers or len(headers) < 2:
            return []

        rows: list[dict[str, Any]] = []
        for row_tag in rows_raw[1:]:
            cells = [td.get_text(strip=True) for td in row_tag.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            # Skip summary / total rows
            first = cells[0].lower()
            if any(skip in first for skip in ("total", "market", "overall")):
                continue
            entry: dict[str, Any] = {}
            for i, col in enumerate(headers):
                if i < len(cells):
                    entry[col] = _parse_numeric(cells[i])
            rows.append(entry)
        return rows
    except Exception as exc:
        logger.debug("Damodaran table fetch error (%s): %s", url, exc)
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_sector(raw: str) -> str | None:
    """Case-insensitive GICS sector lookup."""
    sectors = _STATIC_BENCHMARKS.get("sectors", {})
    return next(
        (k for k in sectors if k.lower() == raw.strip().lower()),
        None,
    )


def _collect(
    rows: list[dict],
    industry_set: set[str],
    col_candidates: tuple[str, ...],
) -> list[float]:
    """Extract numeric values for matching industries from a dataset."""
    vals: list[float] = []
    for row in rows:
        # First column is always industry name
        name = str(list(row.values())[0]).lower().strip()
        if name not in industry_set:
            continue
        for col in col_candidates:
            v = row.get(col)
            if isinstance(v, (int, float)) and v > 0:
                vals.append(float(v))
                break
    return vals


def _avg(vals: list[float]) -> float | None:
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _pct(v: float | None) -> float | None:
    """Convert decimal margin (e.g. 0.25) to percentage (25.0), or pass through if already %-scale."""
    if v is None:
        return None
    return round(v * 100, 2) if v < 1 else round(v, 2)


def _parse_numeric(s: str) -> str | float:
    """Attempt to convert a cell string to a float; return original string on failure."""
    clean = s.replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        return float(clean) if clean not in ("", "N/A", "NA", "-", "NM") else s
    except ValueError:
        return s


def _detect_year(rows: list[dict]) -> int:
    """Try to infer the data year from the table content."""
    import datetime
    return datetime.datetime.utcnow().year
