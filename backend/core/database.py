"""Database initialisation and session management.

Runs on FastAPI startup via the lifespan context manager.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

# Fallback to local SQLite if DATABASE_URL is not set (for testing/development)
_DATABASE_URL = os.getenv("DATABASE_URL")
if not _DATABASE_URL:
    # Use SQLite for backwards compatibility if .env is missing
    _db_path = os.getenv("MEMORY_DB_PATH", ".memory/memory.db")
    os.makedirs(os.path.dirname(_db_path) or ".", exist_ok=True)
    _DATABASE_URL = f"sqlite+aiosqlite:///{_db_path}"

if _DATABASE_URL.startswith("postgresql+asyncpg"):
    engine = create_async_engine(
        _DATABASE_URL,
        echo=False,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        }
    )
else:
    engine = create_async_engine(_DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db_session() -> AsyncSession: # type: ignore
    """Dependency to get an async SQLAlchemy session."""
    async with async_session_factory() as session:
        yield session


async def run_migrations() -> None:
    """Create all tables idempotently; enable pgvector extension on PostgreSQL."""
    logger.info("Running database migrations via SQLAlchemy...")
    is_pg = _DATABASE_URL.startswith("postgresql")
    async with engine.begin() as conn:
        if is_pg:
            # Enable pgvector extension (idempotent; no-op if already enabled)
            try:
                await conn.execute(
                    __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
                )
                await conn.execute(
                    __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                )
            except Exception as ext_err:
                logger.warning("Could not create pgvector extension (may need superuser): %s", ext_err)

        await conn.run_sync(Base.metadata.create_all)

        if is_pg:
            # Add GIN full-text-search index on document_pages if not present
            await conn.execute(__import__("sqlalchemy").text("""
                CREATE INDEX IF NOT EXISTS idx_pages_content_fts
                ON document_pages USING GIN (to_tsvector('english', content))
            """))
            # Add IVFFlat ANN index for vector similarity if pgvector is available
            from .models import HAS_PGVECTOR
            if HAS_PGVECTOR:
                await conn.execute(__import__("sqlalchemy").text("""
                    CREATE INDEX IF NOT EXISTS idx_pages_embedding_ivfflat
                    ON document_pages USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))

    logger.info("Database migrations complete.")
