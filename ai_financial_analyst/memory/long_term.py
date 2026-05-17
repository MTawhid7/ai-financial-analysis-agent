"""Long-term cross-session memory backed by SQLite.

Four tables:
  preferences          — user-stated preferences (key-value, upserted)
  analysis_summaries   — one-paragraph summaries of completed pipeline runs
  conversations        — one row per chat thread (title, timestamps)
  messages             — all user and assistant turns, linked to conversations

All preference/summary/conversation queries are scoped by user_id.
Defaults to "default" for backward compatibility with Streamlit and tests.

Semantic search (optional):
  When an embedder is provided to save_analysis_summary(), a 768-dim vector
  is stored in embedding_json. search_summaries() uses cosine similarity when
  an embedder is provided; falls back to LIKE matching otherwise.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from ..config import settings

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = settings.memory_db_path


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class LongTermMemory:
    """Persistent cross-session memory store.

    The user_id parameter scopes all queries to a single user.
    Defaults to "default" for backward compatibility with Streamlit and tests.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH, user_id: str = "default") -> None:
        self._db_path = db_path
        self._user_id = user_id
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
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conv"
                " ON messages(conversation_id, created_at)"
            )
            # user_id columns added by backend migration at startup;
            # silently ignore if already present.
            for table in ("preferences", "analysis_summaries", "conversations", "messages"):
                try:
                    await db.execute(
                        f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT 'default'"
                    )
                except Exception:
                    pass
            # Semantic search column for analysis summaries — nullable, added lazily.
            try:
                await db.execute(
                    "ALTER TABLE analysis_summaries ADD COLUMN embedding_json TEXT"
                )
            except Exception:
                pass
            await db.commit()

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    async def save_preference(self, key: str, value: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO preferences (key, value, updated_at, user_id) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated_at=excluded.updated_at, user_id=excluded.user_id",
                (key.strip(), value.strip(), time.time(), self._user_id),
            )
            await db.commit()

    async def get_all_preferences(self) -> dict[str, str]:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT key, value FROM preferences WHERE user_id = ? ORDER BY key",
                (self._user_id,),
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
        embedder: Any = None,
    ) -> None:
        await self._ensure_init()
        embedding_json: str | None = None
        if embedder is not None:
            try:
                from ..pageindex.embedder import embed_texts
                vecs = await embed_texts([summary_text])
                if vecs and any(v != 0 for v in vecs[0]):
                    embedding_json = json.dumps(vecs[0])
            except Exception as exc:
                logger.debug("Could not embed summary for semantic indexing: %s", exc)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO analysis_summaries"
                " (session_id, tickers, summary_text, run_id, created_at, user_id, embedding_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, ", ".join(tickers), summary_text.strip(), run_id,
                 time.time(), self._user_id, embedding_json),
            )
            await db.commit()

    async def search_summaries(
        self, query: str, limit: int = 3, embedder: Any = None
    ) -> list[dict]:
        """Return summaries relevant to query.

        If an embedder is provided, performs semantic cosine-similarity search
        against stored embeddings (falling back to LIKE for un-embedded rows).
        Otherwise uses keyword LIKE matching.
        """
        await self._ensure_init()

        if embedder is not None:
            return await self._semantic_search_summaries(query, limit, embedder)

        like = f"%{query.strip()}%"
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT tickers, summary_text, created_at FROM analysis_summaries"
                " WHERE user_id = ? AND (tickers LIKE ? OR summary_text LIKE ?)"
                " ORDER BY created_at DESC LIMIT ?",
                (self._user_id, like, like, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [{"tickers": r[0], "summary": r[1], "created_at": r[2]} for r in rows]

    async def _semantic_search_summaries(
        self, query: str, limit: int, embedder: Any
    ) -> list[dict]:
        """Semantic cosine-similarity search over stored summary embeddings."""
        try:
            from ..pageindex.embedder import embed_query
            q_vec = await embed_query(query)
        except Exception as exc:
            logger.warning("Semantic search failed — falling back to LIKE: %s", exc)
            return await self.search_summaries(query, limit, embedder=None)

        # Fetch all summaries for this user
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT tickers, summary_text, created_at, embedding_json"
                " FROM analysis_summaries WHERE user_id = ? ORDER BY created_at DESC",
                (self._user_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        scored: list[tuple[float, dict]] = []
        keyword_fallback: list[dict] = []

        q_lower = query.lower()
        for tickers, summary_text, created_at, emb_json in rows:
            row_dict = {"tickers": tickers, "summary": summary_text, "created_at": created_at}
            if emb_json:
                try:
                    d_vec = json.loads(emb_json)
                    score = _cosine(q_vec, d_vec)
                    scored.append((score, row_dict))
                except Exception:
                    keyword_fallback.append(row_dict)
            else:
                # Un-embedded row: include if keyword matches
                if q_lower in tickers.lower() or q_lower in summary_text.lower():
                    keyword_fallback.append(row_dict)

        # Rank by cosine similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [row for _, row in scored[:limit]]

        # Append keyword-only results (deduped by summary text) up to limit
        seen = {r["summary"] for r in results}
        for row in keyword_fallback:
            if len(results) >= limit:
                break
            if row["summary"] not in seen:
                results.append(row)
                seen.add(row["summary"])

        return results[:limit]

    async def get_recent_summaries(self, limit: int = 5) -> list[dict]:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT tickers, summary_text, created_at FROM analysis_summaries"
                " WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (self._user_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [{"tickers": r[0], "summary": r[1], "created_at": r[2]} for r in rows]

    async def count_summaries(self) -> int:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM analysis_summaries WHERE user_id = ?",
                (self._user_id,),
            ) as cursor:
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
                "INSERT OR IGNORE INTO conversations (id, title, user_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (conversation_id, title[:80], self._user_id, now, now),
            )
            await db.commit()

    async def update_conversation_title(self, conversation_id: str, title: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
                (title[:80], conversation_id, self._user_id),
            )
            await db.commit()

    async def update_conversation_updated_at(self, conversation_id: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ? AND user_id = ?",
                (time.time(), conversation_id, self._user_id),
            )
            await db.commit()

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        """Return conversations ordered by most recently updated."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations"
                " WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (self._user_id, limit),
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
            await db.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, self._user_id),
            )
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
                "INSERT INTO messages"
                " (id, conversation_id, role, content, intent, tickers, created_at, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conversation_id, role, content,
                 intent, tickers, time.time(), self._user_id),
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
            {"role": r[0], "content": r[1], "intent": r[2], "tickers": r[3], "created_at": r[4]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def delete_all(self) -> None:
        """Permanently delete all data for this user."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM preferences WHERE user_id = ?", (self._user_id,))
            await db.execute(
                "DELETE FROM analysis_summaries WHERE user_id = ?", (self._user_id,)
            )
            # Delete messages for all conversations owned by this user
            await db.execute(
                "DELETE FROM messages WHERE conversation_id IN"
                " (SELECT id FROM conversations WHERE user_id = ?)",
                (self._user_id,),
            )
            await db.execute("DELETE FROM conversations WHERE user_id = ?", (self._user_id,))
            await db.commit()
        logger.info("Long-term memory cleared for user %s.", self._user_id[:8])
