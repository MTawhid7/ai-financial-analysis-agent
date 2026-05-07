"""In-memory registry mapping event_id → asyncio.Queue for SSE streaming.

Each POST /chat/{conv_id} creates a queue entry; the corresponding
GET /stream/{event_id} consumes it.  The queue is removed after streaming
completes or times out.
"""

from __future__ import annotations

import asyncio

_store: dict[str, asyncio.Queue] = {}


def create_queue(event_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _store[event_id] = q
    return q


def get_queue(event_id: str) -> asyncio.Queue | None:
    return _store.get(event_id)


def remove_queue(event_id: str) -> None:
    _store.pop(event_id, None)
