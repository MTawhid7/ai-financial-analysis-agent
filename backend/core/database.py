"""Database initialisation and schema migration.

Runs on FastAPI startup via the lifespan context manager.
All migrations use IF NOT EXISTS / ADD COLUMN IF NOT EXISTS so they are
safe to re-run on every startup and backward-compatible with the existing
Phase 2.5 .memory/memory.db.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH = os.getenv("MEMORY_DB_PATH", ".memory/memory.db")


def get_db_path() -> str:
    return _DB_PATH


async def run_migrations() -> None:
    """Apply all schema migrations idempotently."""
    db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")

        # Phase 4A: users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                email        TEXT NOT NULL UNIQUE,
                display_name TEXT DEFAULT '',
                picture_url  TEXT DEFAULT '',
                created_at   REAL NOT NULL,
                last_seen_at REAL NOT NULL
            )
        """)

        # Phase 2 tables (ensure they exist if DB is fresh)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analysis_summaries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL,
                tickers      TEXT    NOT NULL,
                summary_text TEXT    NOT NULL,
                run_id       TEXT    DEFAULT '',
                created_at   REAL    NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New conversation',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                intent          TEXT DEFAULT '',
                tickers         TEXT DEFAULT '',
                created_at      REAL NOT NULL
            )
        """)

        # Phase 4A: add user_id to multi-tenant tables
        for table in ("preferences", "analysis_summaries", "conversations", "messages"):
            try:
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT 'default'"
                )
                logger.info("Migrated %s: added user_id column", table)
            except Exception:
                pass  # Column already exists — expected on subsequent startups

        # Indexes for user-scoped queries
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user"
            " ON conversations(user_id, updated_at DESC)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv"
            " ON messages(conversation_id, created_at)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_user"
            " ON analysis_summaries(user_id, created_at DESC)"
        )

        await db.commit()

    logger.info("Database migrations complete: %s", db_path)
