"""FastAPI dependency injection helpers."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Cookie, HTTPException, status

from .auth import AuthError, COOKIE_NAME, decode_jwt


@dataclass
class CurrentUser:
    id: str        # Google sub (stable user identifier)
    email: str


async def get_current_user(
    fin_session: str | None = Cookie(default=None, alias=COOKIE_NAME),
) -> CurrentUser:
    """Extract and validate the session JWT from the httpOnly cookie."""
    if not fin_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = decode_jwt(fin_session)
        return CurrentUser(id=payload["sub"], email=payload["email"])
    except (AuthError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
