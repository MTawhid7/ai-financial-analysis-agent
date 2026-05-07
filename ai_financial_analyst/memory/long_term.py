"""Long-term cross-session memory backed by SQLite.

Four tables:
  preferences          — user-stated preferences (key-value, upserted)
  analysis_summaries   — one-paragraph summaries of completed pipeline runs
  conversations        — one row per chat thread (title, timestamps)
  messages             — all user and assistant turns, linked to conversations

Uses lazy initialisation: the database and schema are created on the first
async call. `CREATE TABLE IF NOT EXISTS` makes all additions backward-compatible
with existing databases created in earlier phases.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = ".memory/memory.db"


class LongTermMemory:
    """Persistent cross-session memory store."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        await self._init_db()
        self._initialized = True

    async def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            # Phase 2 tables
            await db.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
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
            # Phase 2.5 tables
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
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conv"
                " ON messages(conversation_id, created_at)"
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    async def save_preference(self, key: str, value: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                (key.strip(), value.strip(), time.time()),
            )
            await db.commit()

    async def get_all_preferences(self) -> dict[str, str]:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT key, value FROM preferences ORDER BY key"
            ) as cursor:
                rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    # ------------------------------------------------------------------
    # Analysis summaries
    # ------------------------------------------------------------------

    async def save_analysis_summary(
        self,
        session_id: str,
        tickers: list[str],
        summary_text: str,
        run_id: str = "",
    ) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO analysis_summaries"
                " (session_id, tickers, summary_text, run_id, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, ", ".join(tickers), summary_text.strip(), run_id, time.time()),
            )
            await db.commit()

    async def search_summaries(self, query: str, limit: int = 3) -> list[dict]:
        """Return summaries whose tickers or summary_text contain the query."""
        await self._ensure_init()
        like = f"%{query.strip()}%"
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT tickers, summary_text, created_at FROM analysis_summaries"
                " WHERE tickers LIKE ? OR summary_text LIKE ?"
                " ORDER BY created_at DESC LIMIT ?",
                (like, like, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [{"tickers": r[0], "summary": r[1], "created_at": r[2]} for r in rows]

    async def get_recent_summaries(self, limit: int = 5) -> list[dict]:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT tickers, summary_text, created_at FROM analysis_summaries"
                " ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [{"tickers": r[0], "summary": r[1], "created_at": r[2]} for r in rows]

    async def count_summaries(self) -> int:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM analysis_summaries") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def save_conversation(self, conversation_id: str, title: str) -> None:
        """Create a new conversation record."""
        await self._ensure_init()
        now = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (conversation_id, title[:80], now, now),
            )
            await db.commit()

    async def update_conversation_title(self, conversation_id: str, title: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title[:80], conversation_id),
            )
            await db.commit()

    async def update_conversation_updated_at(self, conversation_id: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (time.time(), conversation_id),
            )
            await db.commit()

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        """Return conversations ordered by most recently updated."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations"
                " ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await db.commit()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: str = "",
        tickers: str = "",
    ) -> None:
        """Persist one message turn to the database."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, intent, tickers, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conversation_id, role, content, intent, tickers, time.time()),
            )
            await db.commit()

    async def get_conversation_messages(self, conversation_id: str) -> list[dict]:
        """Return all messages for a conversation in chronological order."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT role, content, intent, tickers, created_at FROM messages"
                " WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {
                "role": r[0],
                "content": r[1],
                "intent": r[2],
                "tickers": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def delete_all(self) -> None:
        """Permanently delete all stored preferences, summaries, and conversations."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM preferences")
            await db.execute("DELETE FROM analysis_summaries")
            await db.execute("DELETE FROM messages")
            await db.execute("DELETE FROM conversations")
            await db.commit()
        logger.info("Long-term memory cleared.")
