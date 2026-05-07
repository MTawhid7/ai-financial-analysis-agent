"""Authentication endpoints.

POST /auth/google   Validate a Google ID token, upsert user, set session cookie.
GET  /auth/me       Return the current user's profile.
POST /auth/logout   Clear the session cookie.
"""

from __future__ import annotations

import logging
import time

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from ..core.auth import AuthError, COOKIE_NAME, create_jwt, verify_google_id_token
from ..core.database import get_db_path
from ..core.deps import CurrentUser, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class GoogleTokenRequest(BaseModel):
    id_token: str


class UserProfile(BaseModel):
    id: str
    email: str
    display_name: str
    picture_url: str


@router.post("/google")
async def google_sign_in(body: GoogleTokenRequest, response: Response) -> UserProfile:
    """Validate a Google ID token, upsert the user, and set a session cookie."""
    try:
        payload = verify_google_id_token(body.id_token)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    user_id: str = payload["sub"]
    email: str = payload.get("email", "")
    display_name: str = payload.get("name", "")
    picture_url: str = payload.get("picture", "")
    now = time.time()

    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT INTO users (id, email, display_name, picture_url, created_at, last_seen_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            "   display_name=excluded.display_name,"
            "   picture_url=excluded.picture_url,"
            "   last_seen_at=excluded.last_seen_at",
            (user_id, email, display_name, picture_url, now, now),
        )
        await db.commit()

    jwt_token = create_jwt(user_id, email)
    response.set_cookie(
        key=COOKIE_NAME,
        value=jwt_token,
        httponly=True,
        samesite="lax",
        max_age=30 * 24 * 3600,
        secure=False,  # Set to True in production (HTTPS)
    )

    logger.info("User signed in: %s", email)
    return UserProfile(
        id=user_id,
        email=email,
        display_name=display_name,
        picture_url=picture_url,
    )


@router.get("/me")
async def get_me(user: CurrentUser = Depends(get_current_user)) -> UserProfile:
    """Return the authenticated user's profile from the database."""
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT id, email, display_name, picture_url FROM users WHERE id = ?",
            (user.id,),
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserProfile(id=row[0], email=row[1], display_name=row[2], picture_url=row[3])


@router.post("/logout")
async def logout(response: Response) -> dict:
    """Clear the session cookie."""
    response.delete_cookie(key=COOKIE_NAME)
    return {"status": "logged out"}
