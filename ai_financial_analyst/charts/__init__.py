"""AI Financial Analyst — charts package.

Public API:
  generate_on_demand_chart(ticker, chart_type, ...)  — on-demand via Manager tool
  generate_all_charts(final_state)                   — bulk post-analysis charts

Individual generators are importable from their submodules if needed.
"""
from .dispatcher import generate_on_demand_chart, generate_all_charts

__all__ = ["generate_on_demand_chart", "generate_all_charts"]
