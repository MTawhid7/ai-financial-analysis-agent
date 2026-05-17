"""Data access layer — fetches financial data from all sources.

Public API:

    from ai_financial_analyst.data.yahoo import fetch_ticker_data
    from ai_financial_analyst.data.base import DataResult, null_result
    from ai_financial_analyst.data.search.tavily import TavilySearchClient
    from ai_financial_analyst.data.benchmark import get_sector_benchmarks

Design principles:
- Each data source in its own sub-package (yahoo/, market/, benchmark/, search/)
- Every fetch function returns a plain dict — JSON serialisation is the tool layer's job
- Cache injection: all fetch functions accept an optional cache argument so tests
  can pass a no-op cache without touching module-level state
- Concurrent fetching: data/yahoo/__init__.py orchestrates asyncio.gather across
  the 7 data types with a configurable semaphore
"""
