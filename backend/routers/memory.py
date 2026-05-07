"""Memory management endpoints (preferences and analysis summaries).

GET    /memory/preferences           Return all stored preferences.
PATCH  /memory/preferences           Upsert a preference key-value pair.
DELETE /memory/preferences/{key}     Delete one preference.
GET    /memory/summaries             List analysis summaries (paginated).
DELETE /memory/summaries/{id}        Delete one summary.
POST   /memory/clear                 Delete ALL memory for the current user.
"""

from __future__ import annotations

import time

import aiosqlite
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from ..core.database import get_db_path
from ..core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


class PreferencePatch(BaseModel):
    key: str
    value: str


@router.get("/preferences")
async def get_preferences(
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, str]:
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT key, value FROM preferences WHERE user_id = ? ORDER BY key",
            (user.id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return {r[0]: r[1] for r in rows}


@router.patch("/preferences", status_code=status.HTTP_204_NO_CONTENT)
async def upsert_preference(
    body: PreferencePatch,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT INTO preferences (key, value, updated_at, user_id) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at"
            " WHERE user_id = ?",
            (body.key, body.value, time.time(), user.id, user.id),
        )
        await db.commit()


@router.delete("/preferences/{key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preference(
    key: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "DELETE FROM preferences WHERE key = ? AND user_id = ?", (key, user.id)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Analysis summaries
# ---------------------------------------------------------------------------


class SummaryOut(BaseModel):
    id: int
    tickers: str
    summary_text: str
    created_at: float


@router.get("/summaries", response_model=list[SummaryOut])
async def list_summaries(
    limit: int = 20,
    offset: int = 0,
    user: CurrentUser = Depends(get_current_user),
) -> list[SummaryOut]:
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT id, tickers, summary_text, created_at FROM analysis_summaries"
            " WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user.id, limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
    return [SummaryOut(id=r[0], tickers=r[1], summary_text=r[2], created_at=r[3]) for r in rows]


@router.delete("/summaries/{summary_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_summary(
    summary_id: int,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "DELETE FROM analysis_summaries WHERE id = ? AND user_id = ?",
            (summary_id, user.id),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Clear all
# ---------------------------------------------------------------------------


@router.post("/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_all_memory(
    user: CurrentUser = Depends(get_current_user),
) -> None:
    """Delete all preferences and analysis summaries for the current user."""
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("DELETE FROM preferences WHERE user_id = ?", (user.id,))
        await db.execute("DELETE FROM analysis_summaries WHERE user_id = ?", (user.id,))
        await db.commit()
