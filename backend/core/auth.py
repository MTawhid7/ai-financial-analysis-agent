"""JWT signing/verification and Google ID token validation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

_JWT_SECRET = os.environ.get("FASTAPI_JWT_SECRET", "change-me-in-production")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_SECONDS = 30 * 24 * 3600  # 30 days
_GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
_COOKIE_NAME = "fin_session"


class AuthError(Exception):
    """Raised when token validation fails."""


def create_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + _JWT_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_jwt(token: str) -> dict[str, Any]:
    """Decode and validate a JWT. Raises AuthError on failure."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except JWTError as exc:
        raise AuthError(f"Invalid token: {exc}") from exc


def verify_google_id_token(id_token_str: str) -> dict[str, Any]:
    """Validate a Google ID token and return the payload.

    Returns a dict with at least: sub, email, name, picture.
    Raises AuthError if the token is invalid or expired.
    """
    if not _GOOGLE_CLIENT_ID:
        raise AuthError("GOOGLE_CLIENT_ID environment variable not set")

    try:
        payload = google_id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            _GOOGLE_CLIENT_ID,
        )
        return payload
    except Exception as exc:
        raise AuthError(f"Google token verification failed: {exc}") from exc


COOKIE_NAME = _COOKIE_NAME
