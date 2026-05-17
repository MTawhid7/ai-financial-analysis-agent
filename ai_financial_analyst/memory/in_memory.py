"""InMemoryBackend — a pure-in-memory MemoryBackend for testing.

Zero file I/O, zero async SQLite setup. Tests that need memory can use this
backend without tmp_path fixtures or database cleanup.

All data is lost when the object is garbage-collected — that's by design.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryBackend:
    """Pure in-memory implementation of the MemoryBackend Protocol.

    Satisfies the Protocol without importing aiosqlite, making it
    suitable for unit tests that don't need real persistence.
    """

    def __init__(self, user_id: str = "test") -> None:
        self._user_id      = user_id
        # Internal structure: {key: {value, created_at, updated_at}}
        # get_all_preferences() returns {key: value} — Protocol unchanged.
        self._preferences: dict[str, dict]  = {}
        self._summaries:   list[dict]       = []
        self._conversations: dict[str, dict] = {}
        self._messages:    list[dict]        = []

    # ── Preferences ───────────────────────────────────────────────────────────

    async def save_preference(self, key: str, value: str) -> None:
        k, v = key.strip(), value.strip()
        now  = time.time()
        existing = self._preferences.get(k)
        if existing and existing["value"] != v:
            logger.info(
                "Preference '%s' updated: '%s' → '%s'", k, existing["value"], v
            )
        self._preferences[k] = {
            "value":      v,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }

    async def get_all_preferences(self) -> dict[str, str]:
        return {k: v["value"] for k, v in self._preferences.items()}

    # ── Analysis summaries ────────────────────────────────────────────────────

    async def save_analysis_summary(
        self,
        session_id:   str,
        tickers:      list[str],
        summary_text: str,
        run_id:       str = "",
        embedder:     Any = None,
    ) -> None:
        self._summaries.append({
            "session_id":   session_id,
            "tickers":      ", ".join(tickers),
            "summary":      summary_text.strip(),
            "run_id":       run_id,
            "created_at":   time.time(),
            "embedding_json": None,
        })

    async def search_summaries(
        self,
        query:    str,
        limit:    int = 3,
        embedder: Any = None,
    ) -> list[dict]:
        q = query.lower()
        results = [
            {"tickers": s["tickers"], "summary": s["summary"], "created_at": s["created_at"]}
            for s in reversed(self._summaries)
            if q in s["tickers"].lower() or q in s["summary"].lower()
        ]
        return results[:limit]

    async def get_recent_summaries(self, limit: int = 5) -> list[dict]:
        return [
            {"tickers": s["tickers"], "summary": s["summary"], "created_at": s["created_at"]}
            for s in reversed(self._summaries)
        ][:limit]

    async def count_summaries(self) -> int:
        return len(self._summaries)

    # ── Conversations ─────────────────────────────────────────────────────────

    async def save_conversation(self, conversation_id: str, title: str) -> None:
        now = time.time()
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                "id":         conversation_id,
                "title":      title[:80],
                "created_at": now,
                "updated_at": now,
            }

    async def update_conversation_title(self, conversation_id: str, title: str) -> None:
        if conversation_id in self._conversations:
            self._conversations[conversation_id]["title"]      = title[:80]
            self._conversations[conversation_id]["updated_at"] = time.time()

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        convos = sorted(
            self._conversations.values(),
            key=lambda c: c["updated_at"],
            reverse=True,
        )
        return convos[:limit]

    async def delete_conversation(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
        self._messages = [m for m in self._messages if m["conversation_id"] != conversation_id]

    # ── Messages ──────────────────────────────────────────────────────────────

    async def save_message(
        self,
        conversation_id: str,
        role:            str,
        content:         str,
        intent:          str = "",
        tickers:         str = "",
    ) -> None:
        self._messages.append({
            "id":              str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role":            role,
            "content":         content,
            "intent":          intent,
            "tickers":         tickers,
            "created_at":      time.time(),
        })
        # Touch updated_at on conversation
        if conversation_id in self._conversations:
            self._conversations[conversation_id]["updated_at"] = time.time()

    async def get_conversation_messages(self, conversation_id: str) -> list[dict]:
        msgs = [m for m in self._messages if m["conversation_id"] == conversation_id]
        return sorted(msgs, key=lambda m: m["created_at"])

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def delete_all(self) -> None:
        self._preferences.clear()
        self._summaries.clear()
        self._conversations.clear()
        self._messages.clear()
