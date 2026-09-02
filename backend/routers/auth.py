# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 4: Auth Router
SIH26184 | MHA / I4C

Officer-facing:
  POST /api/v1/auth/login            -- exchange credentials for a bearer token
  POST /api/v1/auth/logout           -- revoke the presented token
  GET  /api/v1/auth/me               -- who am I (restores a session on reload)
  POST /api/v1/auth/forgot-password  -- RAISE a reset request for an admin to action
  POST /api/v1/auth/reset-password   -- spend an APPROVED token

Administrator-only:
  GET  /api/v1/auth/users            -- the officer roster
  POST /api/v1/auth/users            -- issue credentials to a new officer
  POST /api/v1/auth/users/{id}/unlock
  GET  /api/v1/auth/reset-requests
  POST /api/v1/auth/reset-requests/{id}/approve
  POST /api/v1/auth/reset-requests/{id}/deny

WHY RESETS NEED AN APPROVER
---------------------------
An earlier version of this router returned the reset token directly to whoever
asked for it, on the grounds that there is no mail server to send it to. That was
not a limitation, it was an authentication bypass: knowing an officer's username
was enough to mint a token and take over their account.

A reset is now a request that a named administrator approves. The token goes to
the administrator, who passes it on by a channel they already trust. When a mail
server exists, the approval step sends the mail and nothing else changes.

Every handler is a plain `def`, not `async def` -- see the note in backend/auth.py.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from backend import db
from backend.auth import _token_from_header, admin_user, current_user
from backend.models.schemas import (
    CreateUserRequest,
    ForgotPasswordRequest,
    LoginRequest,
    LoginResponse,
    ResetPasswordRequest,
    ResetRequestOut,
    ResetApprovalResponse,
    UserOut,
)

logger = logging.getLogger("muleshield.auth")

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# officer-facing
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse,
             summary="Exchange credentials for a token")
def login(request: LoginRequest) -> LoginResponse:
    try:
        user = db.verify_login(request.username, request.password)
    except db.LockedOut as e:
        # 423 Locked. Distinguishable from a wrong password on purpose: an officer
        # who cannot get in has to be told why, or they will just keep trying.
        logger.warning("[AUTH] locked-out login attempt for '%s'", request.username)
        raise HTTPException(
            status_code=423,
            detail=f"Account locked after {db.MAX_FAILED_LOGINS} failed attempts. "
                   f"Try again in {e.minutes} minute(s), or ask an administrator "
                   f"to unlock it.",
        ) from e

    if user is None:
        # One message for "no such user" and for "wrong password". Distinguishing
        # them turns this endpoint into a way to enumerate the roster.
        logger.info("[AUTH] failed login for '%s'", request.username)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    session = db.create_session(user["id"])
    logger.info("[AUTH] %s signed in", user["username"])
    return LoginResponse(
        access_token=session["token"],
        expires_at=session["expires_at"],
        user=UserOut(**user),
    )


@router.post("/logout", summary="Revoke the presented token")
def logout(
    authorization: str = Header(default=""),
    user: dict = Depends(current_user),
) -> dict:
    """Revoke exactly the token that was presented.

    Only this one row dies -- the same officer signed in at a second terminal
    stays signed in, which is what you want when a duty desk is shared.
    """
    db.delete_session(_token_from_header(authorization))
    logger.info("[AUTH] %s signed out", user["username"])
    return {"detail": "Signed out."}


@router.get("/me", response_model=UserOut, summary="The signed-in officer")
def me(user: dict = Depends(current_user)) -> UserOut:
    return UserOut(**user)


@router.post("/forgot-password", summary="Raise a password reset request")
def forgot_password(request: ForgotPasswordRequest) -> dict:
    """Queue a reset request for an administrator to action.

    The response is IDENTICAL whether or not the account exists, and it never
    contains a token. Nothing an unauthenticated caller can do here changes any
    password.
    """
    db.request_password_reset(request.username)
    logger.info("[AUTH] reset requested for '%s'", request.username)
    return {"detail": "If that account exists, a reset request has been raised. "
                      "An administrator will action it."}


@router.post("/reset-password", summary="Spend an approved reset token")
def reset_password(request: ResetPasswordRequest) -> dict:
    if not db.consume_reset_token(request.token, request.new_password):
        raise HTTPException(
            status_code=400,
            detail="That token is invalid, not yet approved, expired, or already used.",
        )
    logger.info("[AUTH] password reset completed")
    # set_password already dropped every session for that officer, so anyone
    # holding a token minted before the reset is now signed out.
    return {"detail": "Password updated. Sign in with your new password."}


# ---------------------------------------------------------------------------
# administrator-only
# ---------------------------------------------------------------------------

@router.get("/users", response_model=list[UserOut], summary="The officer roster")
def list_users(admin: dict = Depends(admin_user)) -> list[UserOut]:
    return [UserOut(**u) for u in db.list_users()]


@router.post("/users", response_model=UserOut, status_code=201,
             summary="Issue credentials to a new officer")
def create_officer(request: CreateUserRequest,
                   admin: dict = Depends(admin_user)) -> UserOut:
    """Create an account with an initial password the administrator hands over.

    There is no self-registration: an officer gets an account because somebody
    with authority created one, which is the same way they get a warrant card.
    """
    if db.get_user_by_name(request.username) is not None:
        raise HTTPException(status_code=409,
                            detail=f"'{request.username}' already exists.")
    user = db.create_user(request.username, request.display_name,
                          request.password, is_admin=request.is_admin)
    logger.info("[AUTH] %s created account '%s'",
                admin["username"], request.username)
    return UserOut(**user)


@router.post("/users/{user_id}/unlock", summary="Clear a lockout immediately")
def unlock(user_id: int, admin: dict = Depends(admin_user)) -> dict:
    if not db.unlock_user(user_id):
        raise HTTPException(status_code=404, detail="No such officer.")
    logger.info("[AUTH] %s unlocked user %d", admin["username"], user_id)
    return {"detail": "Account unlocked."}


@router.get("/reset-requests", response_model=list[ResetRequestOut],
            summary="Password reset requests awaiting a decision")
def reset_requests(status: str = "pending",
                   admin: dict = Depends(admin_user)) -> list[ResetRequestOut]:
    return [ResetRequestOut(**r) for r in db.list_reset_requests(status)]


@router.post("/reset-requests/{request_id}/approve",
             response_model=ResetApprovalResponse,
             summary="Approve a reset and mint its single-use token")
def approve_reset(request_id: int,
                  admin: dict = Depends(admin_user)) -> ResetApprovalResponse:
    """Approve, and return the token ONCE to the approving administrator.

    It is shown here and nowhere else: not stored in the clear, not retrievable
    later, not sent anywhere. The administrator passes it to the officer by a
    channel they already trust.
    """
    token = db.approve_reset(request_id, admin["display_name"])
    if token is None:
        raise HTTPException(status_code=404,
                            detail="No such pending request.")
    logger.info("[AUTH] %s approved reset request %d",
                admin["username"], request_id)
    return ResetApprovalResponse(
        detail="Approved. Give this token to the officer -- it is shown once.",
        reset_token=token,
        expires_in_minutes=db.RESET_TOKEN_MINUTES,
    )


@router.post("/reset-requests/{request_id}/deny", summary="Refuse a reset request")
def deny_reset(request_id: int, admin: dict = Depends(admin_user)) -> dict:
    if not db.deny_reset(request_id, admin["display_name"]):
        raise HTTPException(status_code=404, detail="No such pending request.")
    logger.info("[AUTH] %s denied reset request %d", admin["username"], request_id)
    return {"detail": "Request denied."}
