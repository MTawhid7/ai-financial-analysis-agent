"""Feedback endpoints — 👍/👎 ratings on assistant messages.

POST /feedback   Store a rating for a message.
GET  /feedback/stats/{conversation_id}   Return rating counts for a conversation.
"""

from __future__ import annotations

import time
import uuid

import aiosqlite
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from ..core.database import get_db_path
from ..core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_index: int = Field(ge=0, description="0-based index of the assistant message")
    rating: int = Field(description="1 for thumbs-up, -1 for thumbs-down")

    @property
    def is_valid_rating(self) -> bool:
        return self.rating in (1, -1)


@router.post("", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(
    body: FeedbackRequest,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    if not body.is_valid_rating:
        return  # silently ignore invalid ratings

    async with aiosqlite.connect(get_db_path()) as db:
        # Upsert: one rating per (user, conversation, message_index)
        await db.execute(
            "INSERT INTO feedback (id, conversation_id, user_id, message_index, rating, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT DO NOTHING",
            (str(uuid.uuid4()), body.conversation_id, user.id,
             body.message_index, body.rating, time.time()),
        )
        await db.commit()


@router.get("/stats/{conversation_id}")
async def feedback_stats(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Return per-message rating for a conversation."""
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT message_index, rating FROM feedback"
            " WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user.id),
        ) as cursor:
            rows = await cursor.fetchall()
    return {r[0]: r[1] for r in rows}
