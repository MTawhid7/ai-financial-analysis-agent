"""Centralised, env-backed configuration using pydantic-settings.

Every tunable value lives here. Business-logic modules import `settings`
and read from it — never from os.environ directly, never from hardcoded literals.

Environment variables are read from the process environment and from the `.env`
file in the project root (loaded by python-dotenv at application startup).

All fields have production-ready defaults so the system works out of the box
for local development without any env file. Required fields (API keys) must be
supplied via the environment or .env file.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all application configuration.

    Field names use snake_case; the corresponding env var is the
    field's `alias` (ALL_CAPS). pydantic-settings reads both.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # don't raise on unexpected env vars
        populate_by_name=True,   # allow field name OR alias
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    google_api_key: str = Field(
        default="",
        alias="GOOGLE_API_KEY",
        description="Gemini + Google Embeddings API key (required for AI features)",
    )
    tavily_api_key: str = Field(
        default="",
        alias="TAVILY_API_KEY",
        description="Tavily web-search API key (required for news search)",
    )
    langsmith_api_key: str = Field(
        default="",
        alias="LANGSMITH_API_KEY",
        description="LangSmith tracing key (optional)",
    )
    langsmith_tracing: bool = Field(
        default=False,
        alias="LANGSMITH_TRACING",
        description="Enable LangSmith trace export",
    )

    # ── LLM: model names (overridable without code changes) ──────────────────
    llm_primary_model: str = Field(
        default="gemini-2.5-flash-preview-05-20",
        alias="LLM_PRIMARY_MODEL",
        description="Primary Gemini model identifier",
    )
    llm_fallback_model: str = Field(
        default="gemini-2.5-flash-8b",
        alias="LLM_FALLBACK_MODEL",
        description="Fallback model when primary is rate-limited",
    )
    llm_embedding_model: str = Field(
        default="models/text-embedding-004",
        alias="LLM_EMBEDDING_MODEL",
        description="Gemini embedding model for semantic search",
    )

    # ── LLM: rate limiting ────────────────────────────────────────────────────
    llm_primary_rpm_limit: int = Field(
        default=15,
        alias="LLM_PRIMARY_RPM_LIMIT",
        description="Primary model requests-per-minute free-tier limit",
    )
    llm_fallback_rpm_limit: int = Field(
        default=30,
        alias="LLM_FALLBACK_RPM_LIMIT",
        description="Fallback model requests-per-minute free-tier limit",
    )
    llm_call_timeout_s: float = Field(
        default=120.0,
        alias="LLM_CALL_TIMEOUT_S",
        description="Per-call timeout (seconds) for primary async LLM invocations",
    )

    # ── LLM: circuit breaker ──────────────────────────────────────────────────
    llm_cb_max_failures: int = Field(
        default=5,
        alias="LLM_CB_MAX_FAILURES",
        description="Number of rate-limit errors that trip the circuit breaker",
    )
    llm_cb_window_s: float = Field(
        default=60.0,
        alias="LLM_CB_WINDOW_S",
        description="Rolling window (seconds) for counting rate-limit failures",
    )
    llm_cb_half_open_delay_s: float = Field(
        default=60.0,
        alias="LLM_CB_HALF_OPEN_DELAY_S",
        description="Seconds to wait before sending a recovery probe after tripping",
    )

    # ── LLM: budget / quota ───────────────────────────────────────────────────
    llm_daily_budget: int = Field(
        default=1500,
        alias="LLM_DAILY_BUDGET",
        description="Estimated daily request budget (Gemini free tier ≈ 1500 RPD)",
    )
    llm_budget_soft_warn_pct: float = Field(
        default=0.60,
        alias="LLM_BUDGET_SOFT_WARN_PCT",
        description="Soft warning threshold as a fraction of daily budget (0–1)",
    )
    llm_budget_warn_pct: float = Field(
        default=0.80,
        alias="LLM_BUDGET_WARN_PCT",
        description="Hard warning threshold as a fraction of daily budget (0–1)",
    )
    llm_budget_defer_pct: float = Field(
        default=0.95,
        alias="LLM_BUDGET_DEFER_PCT",
        description="Deferral threshold — activate caching-only mode above this fraction",
    )

    # ── Cache TTLs (seconds) ──────────────────────────────────────────────────
    ttl_price_s: int = Field(
        default=15 * 60,
        alias="TTL_PRICE_S",
        description="Price history cache TTL (15 minutes)",
    )
    ttl_fundamentals_s: int = Field(
        default=6 * 3600,
        alias="TTL_FUNDAMENTALS_S",
        description="Fundamentals cache TTL (6 hours)",
    )
    ttl_financials_s: int = Field(
        default=24 * 3600,
        alias="TTL_FINANCIALS_S",
        description="Financial statements cache TTL (24 hours)",
    )
    ttl_market_benchmark_s: int = Field(
        default=24 * 3600,
        alias="TTL_MARKET_BENCHMARK_S",
        description="Market/benchmark cache TTL (24 hours)",
    )
    ttl_risk_free_s: int = Field(
        default=3600,
        alias="TTL_RISK_FREE_S",
        description="Risk-free rate cache TTL (1 hour)",
    )
    ttl_damodaran_s: int = Field(
        default=30 * 24 * 3600,
        alias="TTL_DAMODARAN_S",
        description="Damodaran sector benchmark cache TTL (30 days)",
    )
    ttl_web_search_s: int = Field(
        default=3600,
        alias="TTL_WEB_SEARCH_S",
        description="Web search result cache TTL (1 hour)",
    )
    ttl_default_s: int = Field(
        default=4 * 3600,
        alias="TTL_DEFAULT_S",
        description="Default cache TTL for uncategorised entries (4 hours)",
    )

    # ── File paths ────────────────────────────────────────────────────────────
    cache_dir: str = Field(
        default=".cache",
        alias="CACHE_DIR",
        description="Directory for diskcache storage",
    )
    memory_db_path: str = Field(
        default=".memory/memory.db",
        alias="MEMORY_DB_PATH",
        description="SQLite path for long-term memory (preferences, summaries)",
    )
    checkpoint_db_path: str = Field(
        default=".checkpoints/state.db",
        alias="CHECKPOINT_DB_PATH",
        description="SQLite path for LangGraph pipeline checkpoints",
    )
    upload_dir: str = Field(
        default=".uploads",
        alias="UPLOAD_DIR",
        description="Directory for user-uploaded document files",
    )
    artifacts_dir: str = Field(
        default="debug_artifacts",
        alias="ARTIFACTS_DIR",
        description="Directory for per-run trace and artifact JSON files",
    )

    # ── Web search ────────────────────────────────────────────────────────────
    search_days_window: int = Field(
        default=90,
        alias="SEARCH_DAYS_WINDOW",
        description="Only return Tavily results from the last N days (0 = no filter)",
    )
    search_max_results: int = Field(
        default=3,
        alias="SEARCH_MAX_RESULTS",
        description="Maximum number of web search results to retrieve",
    )
    search_min_content_chars: int = Field(
        default=150,
        alias="SEARCH_MIN_CONTENT_CHARS",
        description="Minimum substantive content length to keep a search result",
    )

    # ── Data fetching ─────────────────────────────────────────────────────────
    yahoo_fetch_concurrency: int = Field(
        default=3,
        alias="YAHOO_FETCH_CONCURRENCY",
        description="Max concurrent yfinance HTTP calls per ticker (asyncio.Semaphore)",
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline_max_tool_rounds: int = Field(
        default=5,
        alias="PIPELINE_MAX_TOOL_ROUNDS",
        description="Maximum tool-use rounds before the Manager LLM stops",
    )
    pipeline_session_ttl_s: int = Field(
        default=1800,
        alias="PIPELINE_SESSION_TTL_S",
        description="ConversationalAgent inactivity TTL in the session LRU cache (30 min)",
    )
    pipeline_node_max_retries: int = Field(
        default=1,
        alias="PIPELINE_NODE_MAX_RETRIES",
        description=(
            "Max retry attempts for transient errors in pipeline nodes. "
            "PartialStateError, CircuitBreakerError, and SanitizationAlert are never retried. "
            "Set to 0 to disable retries."
        ),
    )
    pipeline_node_retry_delay_s: float = Field(
        default=0.5,
        alias="PIPELINE_NODE_RETRY_DELAY_S",
        description="Base delay (seconds) between node retry attempts (linear: delay × attempt_number).",
    )
    memory_context_summaries_limit: int = Field(
        default=3,
        alias="MEMORY_CONTEXT_SUMMARIES_LIMIT",
        description="Max past analysis summaries injected into the system prompt context",
    )
    memory_decay_lambda: float = Field(
        default=0.01,
        alias="MEMORY_DECAY_LAMBDA",
        description=(
            "Exponential decay rate for summary age in semantic search scoring. "
            "lambda=0.01 → 30-day-old summary retains ~74%% recency weight; "
            "180-day-old retains ~16%%. Set to 0 to disable decay."
        ),
    )

    # ── Researcher ────────────────────────────────────────────────────────────
    researcher_ticker_concurrency: int = Field(
        default=3,
        alias="RESEARCHER_TICKER_CONCURRENCY",
        description=(
            "Max concurrent ticker fetches in the researcher node. "
            "Each fetch uses yfinance + Tavily (no Gemini RPM consumed), "
            "so concurrency is safe. Semaphore-gated."
        ),
    )

    # ── PageIndex ─────────────────────────────────────────────────────────────
    pageindex_rrf_k: int = Field(
        default=60,
        alias="PAGEINDEX_RRF_K",
        description="RRF constant k in score = 1/(k + rank). Higher k = softer rank differences.",
    )
    pageindex_embed_batch_size: int = Field(
        default=100,
        alias="PAGEINDEX_EMBED_BATCH_SIZE",
        description="Max texts per Gemini embedding API request.",
    )
    pageindex_chunk_max_chars: int = Field(
        default=1500,
        alias="PAGEINDEX_CHUNK_MAX_CHARS",
        description=(
            "Pages longer than this (chars) are split into overlapping sub-page chunks "
            "for more precise vector search. Each chunk gets its own embedding."
        ),
    )
    pageindex_chunk_overlap_chars: int = Field(
        default=150,
        alias="PAGEINDEX_CHUNK_OVERLAP_CHARS",
        description="Character overlap between consecutive sub-page chunks to preserve context.",
    )

    # ── Database (optional Postgres) ──────────────────────────────────────────
    database_url: str | None = Field(
        default=None,
        alias="DATABASE_URL",
        description="Postgres connection URL. When absent, SQLite is used for memory.",
    )

    # ── Backend / Auth ────────────────────────────────────────────────────────
    fastapi_jwt_secret: str = Field(
        default="change-me-in-production",
        alias="FASTAPI_JWT_SECRET",
        description="JWT signing secret (must be overridden in production)",
    )
    google_client_id: str = Field(
        default="",
        alias="GOOGLE_CLIENT_ID",
        description="Google OAuth 2.0 Client ID for user authentication",
    )
    google_client_secret: str = Field(
        default="",
        alias="GOOGLE_CLIENT_SECRET",
        description="Google OAuth 2.0 Client Secret",
    )
    # Stored as comma-separated strings; use .allowed_origins / .admin_user_ids properties.
    # Using str avoids pydantic-settings attempting JSON-parsing on the raw env value.
    allowed_origins_raw: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        alias="ALLOWED_ORIGINS",
        description="CORS-allowed origins (comma-separated)",
    )
    admin_user_ids_raw: str = Field(
        default="",
        alias="ADMIN_USER_IDS",
        description="Comma-separated user IDs allowed to manage system documents",
    )
    workspace_dir: str = Field(
        default="workspace",
        alias="WORKSPACE_DIR",
        description="Workspace directory for file operations",
    )

    # ── List properties (parsed from comma-separated raw strings) ────────────

    @property
    def allowed_origins(self) -> list[str]:
        """CORS-allowed origins as a list."""
        return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]

    @property
    def admin_user_ids(self) -> list[str]:
        """User IDs permitted to call /admin/* endpoints."""
        return [uid.strip() for uid in self.admin_user_ids_raw.split(",") if uid.strip()]

    # ── Derived helpers (not env-configurable) ────────────────────────────────

    def get_ttl(self, data_type: str) -> int:
        """Return the appropriate cache TTL (seconds) for a given data type."""
        return {
            "price_history":    self.ttl_price_s,
            "fundamentals":     self.ttl_fundamentals_s,
            "balance_sheet":    self.ttl_financials_s,
            "cash_flow":        self.ttl_financials_s,
            "earnings":         self.ttl_fundamentals_s,
            "price_metrics":    self.ttl_fundamentals_s,
            "financials_trend": self.ttl_financials_s,
            "risk_free_rate":   self.ttl_risk_free_s,
            "sp500_data":       self.ttl_market_benchmark_s,
            "damodaran_sector": self.ttl_damodaran_s,
            "web_search":       self.ttl_web_search_s,
        }.get(data_type, self.ttl_default_s)

    @field_validator("llm_budget_soft_warn_pct", "llm_budget_warn_pct", "llm_budget_defer_pct")
    @classmethod
    def _pct_in_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"Budget percentage must be between 0 and 1, got {v}")
        return v

    @model_validator(mode="after")
    def _budget_thresholds_ordered(self) -> "Settings":
        if not (self.llm_budget_soft_warn_pct
                < self.llm_budget_warn_pct
                < self.llm_budget_defer_pct):
            raise ValueError(
                "Budget thresholds must be ordered: "
                "soft_warn < warn < defer "
                f"(got {self.llm_budget_soft_warn_pct} < "
                f"{self.llm_budget_warn_pct} < {self.llm_budget_defer_pct})"
            )
        return self



# Module-level singleton — imported by all other modules.
# In tests, override individual fields by creating a fresh Settings(field=value).
settings = Settings()
