# -*- coding: utf-8 -*-
"""
MuleShield AI -- request-level authentication.
SIH26184 | MHA / I4C

Two dependencies, and the difference between them is the whole design:

    current_user   401s when there is no valid token. For endpoints that must
                   know who is acting.
    optional_user  returns None instead of raising. For the existing
                   case-workflow endpoints.

`optional_user` was introduced when this system had no authentication at all, to
let a token upgrade the audit trail from "asserted" to "verified" without
breaking the callers that predated it. That was the right call while the only
thing at stake was attribution.

It stopped being the right call once someone noticed that POST
/api/v1/bank/micro-freeze -- the one irreversible action in the product, taken
against a real person's bank account -- accepted a request from anybody with
curl, as did every endpoint returning a victim's name and account number. A
login form in the React app is not access control; it is a suggestion to
browsers.

ENDPOINT POLICY, and the line it draws
--------------------------------------
    Tier A  current_user   401 without a token. Everything that mutates state,
                           returns PII, or reveals forward operational
                           intelligence -- freeze, complaints, graph,
                           embeddings, predictions, the audit trail, hotspots
                           and alerts.
    Tier B  open           GET /api/v1/intel/atms ONLY. Aggregate counts over
                           machines: no account numbers, no victims, no case
                           ids. See that router's own docstring.
    Tier C  no dependency  /health, /api, /docs, the SPA catch-all.

The line is: **retrospective aggregates stay open, forward operational forecasts
close.** Knowing which machines have historically seen cash-outs is a statistic.
Knowing where a patrol is about to be sent in the next ninety minutes is
operational intelligence, and it is exactly what an offender would like to read.

`optional_user` is kept because that upgrade rule is still correct and still
used: `actor_for()` below applies it. A verified identity beats the body's
claimed `officer_id` for the AUDIT ACTOR. What the body can no longer do is
serve as authorisation.
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


# --- SYNC ON PURPOSE ---------------------------------------------------------
# Both dependencies below are plain `def`, not `async def`, which looks wrong
# next to every other handler in this codebase. It is deliberate.
#
# Starlette runs a SYNC dependency in a threadpool and an ASYNC one directly on
# the event loop. These call into sqlite3, which blocks. The app ships with
# `uvicorn --workers 1` (Dockerfile), so anything that blocks the loop blocks
# every other request and stalls the /ws/feed sockets along with them.
#
# The session lookup is only ~100us, but the login handler in routers/auth.py
# runs PBKDF2 at 600k iterations -- measured at 354ms on this machine. As
# `async def` that is a third of a second with the whole server frozen, on every
# sign-in. As `def` it is 354ms in a worker thread and nobody else notices.
# -----------------------------------------------------------------------------


def optional_user(authorization: str = Header(default="")) -> Optional[dict]:
    """Resolve the caller if they presented a live token; never raise."""
    return db.resolve_session(_token_from_header(authorization))


def current_user(authorization: str = Header(default="")) -> dict:
    """Resolve the caller, or 401."""
    user = db.resolve_session(_token_from_header(authorization))
    if user is None:
        # WWW-Authenticate is what tells a client this is an auth failure rather
        # than a permissions one.
        raise HTTPException(
            status_code=401,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    """Resolve the caller and require the administrator flag.

    403, not 401: the caller HAS proven who they are, they are simply not allowed
    to do this. Returning 401 would tell a signed-in officer to log in again,
    which they would do, and it would not help.
    """
    if not user.get("is_admin"):
        raise HTTPException(status_code=403,
                            detail="Administrator access required.")
    return user


def actor_for(user: Optional[dict], claimed: str, fallback: str = "ANALYST") -> str:
    """Decide whose name goes on an audit entry.

    A verified identity always wins. This is the single place that rule lives, so
    it cannot drift between the freeze path and the case-workflow path.
    """
    if user:
        return str(user.get("display_name") or user.get("username") or fallback)
    return claimed or fallback
