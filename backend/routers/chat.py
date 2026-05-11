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

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from ..core import event_store, session_manager
from ..core.database import async_session_factory
from ..core.deps import CurrentUser, get_current_user
from ..core.models import Conversation, Message, Report

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
    title = first_message.strip()[:55] + ("…" if len(first_message.strip()) > 55 else "")
    async with async_session_factory() as session:
        result = await session.execute(select(Conversation).where(Conversation.id == conversation_id))
        conv = result.scalar_one_or_none()
        if not conv:
            conv = Conversation(id=conversation_id, title=title, user_id=user_id)
            session.add(conv)
            await session.commit()


async def _build_conversation_state(conversation_id: str, user_id: str) -> object:
    """Reconstruct a ConversationState from persisted messages."""
    from ai_financial_analyst.core.conversation_state import ChatMessage, new_session

    async with async_session_factory() as session:
        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        rows = result.scalars().all()

    conv_messages = [
        ChatMessage(
            role=r.role,
            content=r.content,
            metadata={"intent": r.intent, "tickers": r.tickers},
            timestamp=r.created_at,
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
            message, state,
            step_callback=step_callback,
            conversation_id=conversation_id,
        )

        # Collect on-demand charts generated by the generate_chart tool
        charts: list[dict] = list(getattr(agent, "_pending_charts", []))
        if hasattr(agent, "_pending_charts"):
            agent._pending_charts = []  # reset for next turn

        report_id: str | None = None
        final_state = getattr(agent, "last_analysis_state", None)
        
        async with async_session_factory() as session:
            # If a financial analysis ran, generate pipeline charts and save report for export.
            if final_state:
                agent.last_analysis_state = None  # consume immediately

                # Generate Plotly charts (non-blocking, best-effort)
                try:
                    from ai_financial_analyst.tools.chart_generator import generate_all_charts
                    charts = generate_all_charts(final_state)
                except Exception as exc:
                    logger.warning("Chart generation failed: %s", exc)

                # Persist report to the reports table for later export
                try:
                    import json as _json
                    report_id = str(uuid.uuid4())
                    tickers_str = ", ".join(final_state.get("tickers", []))
                    
                    report = Report(
                        id=report_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        tickers=tickers_str,
                        report_markdown=final_state.get("report_markdown", ""),
                        raw_data_json=_json.dumps(final_state.get("raw_data", {}), default=str),
                        analysis_json=_json.dumps(final_state.get("analysis", {}), default=str),
                    )
                    session.add(report)
                except Exception as exc:
                    logger.warning("Could not save report for export: %s", exc)

            await _persist_turn(session, conversation_id, user_id, message, response_text, new_state, charts, report_id)
            await session.commit()

        queue.put_nowait({
            "type": "complete",
            "response": response_text,
            "charts": charts,
            "report_id": report_id,
        })
    except Exception as exc:
        logger.exception("Pipeline error for conversation %s", conversation_id)
        queue.put_nowait({"type": "error", "detail": str(exc)})
    finally:
        queue.put_nowait(None)  # sentinel — closes the SSE stream


async def _persist_turn(
    session,
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    new_state,
    charts: list,
    report_id: str | None,
) -> None:
    """Save both turns to the messages table and update conversation timestamp."""
    intent = new_state.get("current_intent", "") if new_state else ""
    tickers_str = ", ".join(new_state.get("pending_tickers", [])) if new_state else ""
    now = time.time()
    
    msg_user = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        content=user_message,
        intent="",
        tickers="",
        created_at=now - 0.001
    )
    
    msg_assistant = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        content=assistant_response,
        intent=intent,
        tickers=tickers_str,
        charts=charts,
        report_id=report_id,
        created_at=now
    )
    
    session.add(msg_user)
    session.add(msg_assistant)
    
    result = await session.execute(select(Conversation).where(Conversation.id == conversation_id))
    conv = result.scalar_one_or_none()
    if conv:
        conv.updated_at = now

