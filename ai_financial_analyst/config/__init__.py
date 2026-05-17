"""Centralised configuration for the AI Financial Analyst Agent.

All tunable values are sourced from environment variables with sensible defaults.
Zero hardcoded constants should remain in business-logic modules.

Usage:
    from ai_financial_analyst.config import settings
    model_name = settings.llm_primary_model
"""

from .settings import Settings, settings

__all__ = ["Settings", "settings"]
