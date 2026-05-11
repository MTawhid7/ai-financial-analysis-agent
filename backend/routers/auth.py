"""Authentication endpoints.

POST /auth/google   Validate a Google ID token, upsert user, set session cookie.
GET  /auth/me       Return the current user's profile.
POST /auth/logout   Clear the session cookie.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from ..core.auth import AuthError, COOKIE_NAME, create_jwt, verify_google_id_token
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
    from sqlalchemy import select
    from ..core.database import async_session_factory
    from ..core.models import User
    
    try:
        payload = verify_google_id_token(body.id_token)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    user_id: str = payload["sub"]
    email: str = payload.get("email", "")
    display_name: str = payload.get("name", "")
    picture_url: str = payload.get("picture", "")
    now = time.time()

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user_row = result.scalar_one_or_none()
        
        if user_row:
            user_row.display_name = display_name
            user_row.picture_url = picture_url
            user_row.last_seen_at = now
        else:
            session.add(User(
                id=user_id, email=email, display_name=display_name,
                picture_url=picture_url, created_at=now, last_seen_at=now
            ))
        await session.commit()

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
    from sqlalchemy import select
    from ..core.database import async_session_factory
    from ..core.models import User
    
    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user.id))
        user_row = result.scalar_one_or_none()

    if not user_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserProfile(id=user_row.id, email=user_row.email, display_name=user_row.display_name, picture_url=user_row.picture_url)


@router.post("/logout")
async def logout(response: Response) -> dict:
    """Clear the session cookie.

    delete_cookie must match the original set_cookie attributes (httponly, samesite)
    so every browser recognises them as the same cookie to delete.
    """
    response.delete_cookie(
        key=COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return {"status": "logged out"}
