"""Chat endpoints with SSE streaming.

POST /chat/{conversation_id}    Accept a user message, start the pipeline in the
                                background, return an event_id for the SSE stream.
GET  /stream/{event_id}         SSE stream — tool step events followed by the
                                final assistant response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core import event_store, session_manager
from ..core.database import get_db_path
from ..core.deps import CurrentUser, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# POST /chat/{conversation_id}
# ---------------------------------------------------------------------------


@router.post("/chat/{conversation_id}")
async def send_message(
    conversation_id: str,
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Accept a message, launch the pipeline in the background, return event_id."""
    # Verify the conversation belongs to this user (create it if missing).
    await _ensure_conversation(conversation_id, user.id, body.message)

    state = await _build_conversation_state(conversation_id, user.id)
    agent = session_manager.get_or_create(user.id)

    event_id = str(uuid.uuid4())
    queue = event_store.create_queue(event_id)

    asyncio.create_task(
        _run_pipeline_and_enqueue(
            agent=agent,
            message=body.message,
            state=state,
            queue=queue,
            conversation_id=conversation_id,
            user_id=user.id,
        )
    )

    return {"event_id": event_id}


# ---------------------------------------------------------------------------
# GET /stream/{event_id}
# ---------------------------------------------------------------------------


@router.get("/stream/{event_id}")
async def stream_events(
    event_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    """SSE stream that emits step events and the final response."""
    queue = event_store.get_queue(event_id)
    if not queue:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stream not found")

    async def generate():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=90.0)
                except asyncio.TimeoutError:
                    yield 'data: {"type":"ping"}\n\n'
                    continue

                if event is None:  # sentinel — pipeline finished
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            event_store.remove_queue(event_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _ensure_conversation(conversation_id: str, user_id: str, first_message: str) -> None:
    """Create the conversation row if it doesn't exist yet."""
    now = time.time()
    title = first_message.strip()[:55] + ("…" if len(first_message.strip()) > 55 else "")
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT OR IGNORE INTO conversations (id, title, user_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (conversation_id, title, user_id, now, now),
        )
        await db.commit()


async def _build_conversation_state(conversation_id: str, user_id: str) -> object:
    """Reconstruct a ConversationState from persisted messages."""
    from ai_financial_analyst.core.conversation_state import ChatMessage, new_session

    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT role, content, intent, tickers, created_at FROM messages"
            " WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        ) as cursor:
            rows = await cursor.fetchall()

    conv_messages = [
        ChatMessage(
            role=r[0],
            content=r[1],
            metadata={"intent": r[2], "tickers": r[3]},
            timestamp=r[4],
        )
        for r in rows
    ]
    state = new_session()
    state["session_id"] = conversation_id
    state["messages"] = conv_messages
    return state


async def _run_pipeline_and_enqueue(
    agent,
    message: str,
    state,
    queue: asyncio.Queue,
    conversation_id: str,
    user_id: str,
) -> None:
    """Run the conversational agent and push events + final response to the queue."""
    def step_callback(event: dict) -> None:
        queue.put_nowait({"type": "step", **event})

    try:
        response_text, new_state = await agent.process_message(
            message, state, step_callback=step_callback
        )
        await _persist_turn(conversation_id, user_id, message, response_text, new_state)
        queue.put_nowait({"type": "complete", "response": response_text})
    except Exception as exc:
        logger.exception("Pipeline error for conversation %s", conversation_id)
        queue.put_nowait({"type": "error", "detail": str(exc)})
    finally:
        queue.put_nowait(None)  # sentinel — closes the SSE stream


async def _persist_turn(
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    new_state,
) -> None:
    """Save both turns to the messages table and update conversation timestamp."""
    intent = new_state.get("current_intent", "") if new_state else ""
    tickers_str = ", ".join(new_state.get("pending_tickers", [])) if new_state else ""
    now = time.time()
    msg_id_user = str(uuid.uuid4())
    msg_id_assistant = str(uuid.uuid4())

    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, intent, tickers, created_at)"
            " VALUES (?, ?, 'user', ?, '', '', ?)",
            (msg_id_user, conversation_id, user_message, now - 0.001),
        )
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, intent, tickers, created_at)"
            " VALUES (?, ?, 'assistant', ?, ?, ?, ?)",
            (msg_id_assistant, conversation_id, assistant_response, intent, tickers_str, now),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await db.commit()
