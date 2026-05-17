"""LRU cache of ConversationalAgent instances keyed by user_id.

Creating a ConversationalAgent is expensive (initialises LLMs and the memory
system).  This module caches one instance per user and evicts it after 30
minutes of inactivity to free memory.
"""

from __future__ import annotations

import logging
import time

from ai_financial_analyst.config import settings

logger = logging.getLogger(__name__)

_TTL_SECONDS = settings.pipeline_session_ttl_s
_cache: dict[str, tuple[object, float]] = {}  # user_id → (agent, last_access_ts)


def get_or_create(user_id: str) -> object:
    """Return the cached ConversationalAgent for user_id, creating one if needed."""
    # Lazy import avoids loading AI stack at module import time.
    from ai_financial_analyst.agents.conversational_agent import ConversationalAgent

    now = time.monotonic()

    if user_id in _cache:
        agent, ts = _cache[user_id]
        if now - ts < _TTL_SECONDS:
            _cache[user_id] = (agent, now)
            return agent
        logger.debug("Session expired for user %s; creating new agent", user_id[:8])

    agent = ConversationalAgent.create(user_id=user_id)
    _cache[user_id] = (agent, now)
    logger.info("Created new ConversationalAgent for user %s", user_id[:8])
    return agent


def evict(user_id: str) -> None:
    """Remove a cached agent (e.g. after logout)."""
    _cache.pop(user_id, None)
