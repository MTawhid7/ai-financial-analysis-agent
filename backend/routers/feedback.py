"""Feedback endpoints — 👍/👎 ratings on assistant messages.

POST /feedback   Store a rating for a message.
GET  /feedback/stats/{conversation_id}   Return rating counts for a conversation.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

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
    from ..core.database import async_session_factory
    from ..core.models import Feedback
    
    if not body.is_valid_rating:
        return  # silently ignore invalid ratings

    async with async_session_factory() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(Feedback).where(
                Feedback.conversation_id == body.conversation_id,
                Feedback.user_id == user.id,
                Feedback.message_index == body.message_index
            )
        )
        fb = result.scalar_one_or_none()
        if not fb:
            session.add(Feedback(
                id=str(uuid.uuid4()),
                conversation_id=body.conversation_id,
                user_id=user.id,
                message_index=body.message_index,
                rating=body.rating,
                created_at=time.time()
            ))
            await session.commit()


@router.get("/stats/{conversation_id}")
async def feedback_stats(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Return per-message rating for a conversation."""
    from sqlalchemy import select
    from ..core.database import async_session_factory
    from ..core.models import Feedback
    
    async with async_session_factory() as session:
        result = await session.execute(
            select(Feedback).where(
                Feedback.conversation_id == conversation_id,
                Feedback.user_id == user.id
            )
        )
        rows = result.scalars().all()
        
    return {r.message_index: r.rating for r in rows}
