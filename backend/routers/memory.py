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

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy import select, delete

from ..core.database import async_session_factory
from ..core.deps import CurrentUser, get_current_user
from ..core.models import Preference, AnalysisSummary

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
    async with async_session_factory() as session:
        result = await session.execute(
            select(Preference).where(Preference.user_id == user.id).order_by(Preference.key)
        )
        rows = result.scalars().all()
    return {r.key: r.value for r in rows}


@router.patch("/preferences", status_code=status.HTTP_204_NO_CONTENT)
async def upsert_preference(
    body: PreferencePatch,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with async_session_factory() as session:
        result = await session.execute(
            select(Preference).where(Preference.key == body.key, Preference.user_id == user.id)
        )
        pref = result.scalar_one_or_none()
        if pref:
            pref.value = body.value
            pref.updated_at = time.time()
        else:
            session.add(Preference(key=body.key, value=body.value, user_id=user.id))
        await session.commit()


@router.delete("/preferences/{key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preference(
    key: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with async_session_factory() as session:
        await session.execute(
            delete(Preference).where(Preference.key == key, Preference.user_id == user.id)
        )
        await session.commit()


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
    async with async_session_factory() as session:
        result = await session.execute(
            select(AnalysisSummary)
            .where(AnalysisSummary.user_id == user.id)
            .order_by(AnalysisSummary.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = result.scalars().all()
    return [SummaryOut(id=r.id, tickers=r.tickers, summary_text=r.summary_text, created_at=r.created_at) for r in rows]


@router.delete("/summaries/{summary_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_summary(
    summary_id: int,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    async with async_session_factory() as session:
        await session.execute(
            delete(AnalysisSummary).where(AnalysisSummary.id == summary_id, AnalysisSummary.user_id == user.id)
        )
        await session.commit()


# ---------------------------------------------------------------------------
# Clear all
# ---------------------------------------------------------------------------


@router.post("/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_all_memory(
    user: CurrentUser = Depends(get_current_user),
) -> None:
    """Delete all preferences and analysis summaries for the current user."""
    async with async_session_factory() as session:
        await session.execute(delete(Preference).where(Preference.user_id == user.id))
        await session.execute(delete(AnalysisSummary).where(AnalysisSummary.user_id == user.id))
        await session.commit()
