"""MemoryBackend Protocol — the interface contract for all memory backends.

Any class implementing this protocol can be used wherever memory is expected.
This enables substituting SQLiteMemoryBackend with InMemoryBackend in tests
without file I/O, and PostgresMemoryBackend for horizontal scaling.

Usage:
    from ai_financial_analyst.memory.protocol import MemoryBackend
    from ai_financial_analyst.memory.in_memory import InMemoryBackend

    backend: MemoryBackend = InMemoryBackend()
    await backend.save_preference("style", "conservative")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryBackend(Protocol):
    """Async key-value + semantic-search memory store.

    All methods are async to support both SQLite (aiosqlite) and Postgres
    (asyncpg) backends without blocking the event loop.
    """

    # ── Preferences ───────────────────────────────────────────────────────────

    async def save_preference(self, key: str, value: str) -> None:
        """Persist a user preference (upsert by key)."""
        ...

    async def get_all_preferences(self) -> dict[str, str]:
        """Return all stored preferences for the current user."""
        ...

    # ── Analysis summaries ────────────────────────────────────────────────────

    async def save_analysis_summary(
        self,
        session_id:   str,
        tickers:      list[str],
        summary_text: str,
        run_id:       str = "",
        embedder:     Any = None,
    ) -> None:
        """Persist a one-paragraph analysis summary, optionally with embedding."""
        ...

    async def search_summaries(
        self,
        query:    str,
        limit:    int = 3,
        embedder: Any = None,
    ) -> list[dict]:
        """Return summaries relevant to query (semantic when embedder provided)."""
        ...

    async def get_recent_summaries(self, limit: int = 5) -> list[dict]:
        """Return the most recent summaries regardless of query."""
        ...

    async def count_summaries(self) -> int:
        """Return total number of stored summaries."""
        ...

    # ── Conversations ─────────────────────────────────────────────────────────

    async def save_conversation(self, conversation_id: str, title: str) -> None:
        """Create or update a conversation record."""
        ...

    async def update_conversation_title(self, conversation_id: str, title: str) -> None:
        """Update the display title of a conversation."""
        ...

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        """Return recent conversations ordered by last-updated descending."""
        ...

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages."""
        ...

    # ── Messages ──────────────────────────────────────────────────────────────

    async def save_message(
        self,
        conversation_id: str,
        role:            str,
        content:         str,
        intent:          str = "",
        tickers:         str = "",
    ) -> None:
        """Append a message to a conversation."""
        ...

    async def get_conversation_messages(
        self, conversation_id: str
    ) -> list[dict]:
        """Return all messages for a conversation in chronological order."""
        ...

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def delete_all(self) -> None:
        """Delete all data for the current user (used in tests and account deletion)."""
        ...
