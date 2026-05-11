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
    """Create all tables idempotently."""
    logger.info("Running database migrations via SQLAlchemy...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database migrations complete.")
