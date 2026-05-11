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

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..core.database import get_db_path
from ..core.deps import CurrentUser, get_current_user

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
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations"
            " WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50",
            (user.id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [ConversationSummary(id=r[0], title=r[1], created_at=r[2], updated_at=r[3]) for r in rows]


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest,
    user: CurrentUser = Depends(get_current_user),
) -> ConversationSummary:
    conv_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT INTO conversations (id, title, user_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (conv_id, body.title[:80], user.id, now, now),
        )
        await db.commit()
    return ConversationSummary(id=conv_id, title=body.title[:80], created_at=now, updated_at=now)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> ConversationDetail:
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user.id),
        ) as cursor:
            conv_row = await cursor.fetchone()

        if not conv_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        async with db.execute(
            "SELECT role, content, intent, tickers, created_at FROM messages"
            " WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        ) as cursor:
            msg_rows = await cursor.fetchall()

    messages = [
        MessageOut(role=r[0], content=r[1], intent=r[2] or "", tickers=r[3] or "", created_at=r[4])
        for r in msg_rows
    ]
    return ConversationDetail(id=conv_row[0], title=conv_row[1], messages=messages)


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def update_title(
    conversation_id: str,
    body: UpdateTitleRequest,
    user: CurrentUser = Depends(get_current_user),
) -> ConversationSummary:
    now = time.time()
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "UPDATE conversations SET title = ?, updated_at = ?"
            " WHERE id = ? AND user_id = ?",
            (body.title[:80], now, conversation_id, user.id),
        )
        await db.commit()
        async with db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return ConversationSummary(id=row[0], title=row[1], created_at=row[2], updated_at=row[3])


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        await db.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user.id),
        )
        await db.commit()
