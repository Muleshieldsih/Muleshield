# -*- coding: utf-8 -*-
"""
MuleShield AI -- request-level authentication.
SIH26184 | MHA / I4C
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException

from backend import db


def _token_from_header(authorization: str) -> str:
    """Pull the token out of an 'Authorization: Bearer <token>' header."""
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def optional_user(authorization: str = Header(default="")) -> Optional[dict]:
    """Resolve the caller if they presented a live token; never raise."""
    return db.resolve_session(_token_from_header(authorization))


def current_user(authorization: str = Header(default="")) -> dict:
    """Resolve the caller, or raise 401."""
    user = db.resolve_session(_token_from_header(authorization))
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    """Resolve the caller and require the administrator flag."""
    if not user.get("is_admin"):
        raise HTTPException(status_code=403,
                            detail="Administrator access required.")
    return user


def actor_for(user: Optional[dict], claimed: str, fallback: str = "ANALYST") -> str:
    """Decide whose name goes on an audit entry."""
    if user:
        return str(user.get("display_name") or user.get("username") or fallback)
    return claimed or fallback

