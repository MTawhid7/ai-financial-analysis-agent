"""Conversation CRUD endpoints.

GET    /conversations          List the current user's conversations.
POST   /conversations          Create a new conversation.
GET    /conversations/{id}     Load full message history for a conversation.
DELETE /conversations/{id}     Delete a conversation and all its messages.
PATCH  /conversations/{id}     Update the conversation title.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, delete, update

from ..core.database import async_session_factory
from ..core.deps import CurrentUser, get_current_user
from ..core.models import Conversation, Message

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float


class MessageOut(BaseModel):
    role: str
    content: str
    intent: str
    tickers: str
    charts: list[dict] | None = None
    report_id: str | None = None
    created_at: float


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[MessageOut]


class CreateConversationRequest(BaseModel):
    title: str = "New conversation"


class UpdateTitleRequest(BaseModel):
    title: str


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(
    user: CurrentUser = Depends(get_current_user),
) -> list[ConversationSummary]:
    async with async_session_factory() as session:
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
            .limit(50)
        )
        rows = result.scalars().all()
    
    return [ConversationSummary(id=r.id, title=r.title, created_at=r.created_at, updated_at=r.updated_at) for r in rows]


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest,
    user: CurrentUser = Depends(get_current_user),
) -> ConversationSummary:
    conv_id = str(uuid.uuid4())
    now = time.time()
    
    conv = Conversation(
        id=conv_id,
        title=body.title[:80],
        user_id=user.id,
        created_at=now,
        updated_at=now
    )
    
    async with async_session_factory() as session:
        session.add(conv)
        await session.commit()
        
    return ConversationSummary(id=conv_id, title=body.title[:80], created_at=now, updated_at=now)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> ConversationDetail:
    async with async_session_factory() as session:
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
        )
        conv = result.scalar_one_or_none()

        if not conv:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        msg_result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        msg_rows = msg_result.scalars().all()

    messages = [
        MessageOut(
            role=r.role, 
            content=r.content, 
            intent=r.intent or "", 
            tickers=r.tickers or "", 
            charts=r.charts or [],
            report_id=r.report_id,
            created_at=r.created_at
        )
        for r in msg_rows
    ]
    return ConversationDetail(id=conv.id, title=conv.title, messages=messages)


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def update_title(
    conversation_id: str,
    body: UpdateTitleRequest,
    user: CurrentUser = Depends(get_current_user),
) -> ConversationSummary:
    now = time.time()
    async with async_session_factory() as session:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user.id)
            .values(title=body.title[:80], updated_at=now)
        )
        await session.commit()
        
        result = await session.execute(select(Conversation).where(Conversation.id == conversation_id))
        conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return ConversationSummary(id=conv.id, title=conv.title, created_at=conv.created_at, updated_at=conv.updated_at)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with async_session_factory() as session:
        await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
        await session.execute(delete(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id))
        await session.commit()

