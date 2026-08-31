# -*- coding: utf-8 -*-
"""
MuleShield AI -- persistence for officer accounts and sessions.
SIH26184 | MHA / I4C

This is the project's FIRST persistence layer. Everything else in backend/state.py
is loaded from CSV at boot and mutated in memory, which is fine for case data that
is regenerated anyway -- and completely unacceptable for credentials. An account
that vanishes on restart is not an account.

Deliberately stdlib-only: sqlite3, hashlib, secrets, hmac. Adding passlib/bcrypt
or an ORM would mean editing the Dockerfile's pip layer, which has never been
built or tested. PBKDF2-HMAC-SHA256 at OWASP's recommended iteration count is a
defensible answer to "how are passwords stored" and costs no new dependency.

    from backend import db
    db.init()                                    # schema + first-run seed
    user = db.verify_login("officer", "hunter2")  # None when wrong
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("muleshield.db")

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "muleshield.db"

# OWASP's current floor for PBKDF2-HMAC-SHA256. Costs roughly 0.2s per login on a
# laptop, which is the point: it is the attacker's cost that matters.
PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
SESSION_HOURS = 12          # a shift, not a fortnight
RESET_TOKEN_MINUTES = 30

# Lockout after repeated failures, to stop an unlimited guessing run against a
# known username. The lock EXPIRES on its own rather than needing a human: an
# officer who fat-fingers their password three times at 2am must not be stuck
# until an administrator wakes up, and an attacker throttled to three guesses
# per quarter-hour is stopped just as effectively as one locked out forever.
# An administrator can also clear it immediately -- see unlock_user().
MAX_FAILED_LOGINS = 3
LOCKOUT_MINUTES = 15

# sqlite3 is synchronous and the endpoints calling it are async. Every query here
# is a microsecond-scale local read against a table with a handful of rows, so a
# single connection behind a lock is simpler -- and has fewer failure modes -- than
# introducing aiosqlite for a workload that never blocks.
_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    UNIQUE NOT NULL,
    display_name  TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until  TEXT,
    created_at    TEXT    NOT NULL,
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    issued_at  TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);

-- A reset is a REQUEST an administrator approves, not a token anyone can
-- mint for anyone. The first version of this table handed the token straight
-- back to whoever asked, which meant knowing a username was enough to take over
-- the account. `token_hash` stays NULL until an admin approves.
CREATE TABLE IF NOT EXISTS password_resets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status      TEXT    NOT NULL DEFAULT 'pending',   -- pending | approved | used | denied
    requested_at TEXT   NOT NULL,
    token_hash  TEXT,
    expires_at  TEXT,
    decided_by  TEXT,
    decided_at  TEXT,
    used_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_resets_status ON password_resets(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_resets_token ON password_resets(token_hash)
    WHERE token_hash IS NOT NULL;
"""


def _token_digest(token: str) -> str:
    """Store the digest of a bearer token, never the token itself.

    Same reasoning as the password column: if this file is ever copied off the
    box, the attacker must not walk away with a set of live sessions. The raw
    token is handed to the client once and never written down here.

    Plain SHA-256 rather than PBKDF2 is correct for this one: the token is
    already 256 bits of `secrets` output, so there is no dictionary to slow an
    attacker down -- only a per-request cost to pay on every authenticated call.
    Stretching is for low-entropy secrets chosen by humans.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# password hashing
# ---------------------------------------------------------------------------

def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Return 'pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>'.

    The salt is per-user and random, so two officers who pick the same password
    do not share a hash, and a precomputed table is worthless against the store.
    """
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification against a stored hash string."""
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
    except Exception:
        # A malformed record must fail closed, never raise into the login path.
        return False
    # compare_digest, not ==, so the comparison does not leak via timing.
    return hmac.compare_digest(dk, expected)


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def init() -> None:
    """Create the schema and, on an empty database only, seed one account."""
    global _conn
    with _lock:
        if _conn is None:
            _conn = _connect()
        _conn.executescript(SCHEMA)
        _conn.commit()
        n = _conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if n == 0:
        _seed_first_user()
    logger.info("[DB] ready at %s (%d user(s))", DB_PATH.name, max(n, 1))


def _seed_first_user() -> None:
    """Create the initial officer account.

    The password comes from MULESHIELD_ADMIN_PASSWORD when set. When it is not,
    a random one is generated and logged ONCE at WARNING -- a known default baked
    into a public repository would make every deployment of this system trivially
    accessible, which is a worse outcome than an operator having to read the boot
    log once.
    """
    username = os.environ.get("MULESHIELD_ADMIN_USER", "officer")
    password = os.environ.get("MULESHIELD_ADMIN_PASSWORD", "")
    generated = False
    if not password:
        password = secrets.token_urlsafe(12)
        generated = True

    # The first account is the administrator: somebody has to be able to
    # create the others and action reset requests, and on an empty database
    # there is nobody else to grant it.
    create_user(username, "Duty Officer", password, is_admin=True)

    if generated:
        logger.warning("=" * 66)
        logger.warning("  FIRST RUN -- seeded account '%s'", username)
        logger.warning("  password: %s", password)
        logger.warning("  Shown once. Set MULESHIELD_ADMIN_PASSWORD to pin it.")
        logger.warning("=" * 66)
    else:
        logger.info("[DB] seeded account '%s' from MULESHIELD_ADMIN_PASSWORD", username)


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def _require() -> sqlite3.Connection:
    if _conn is None:
        init()
    assert _conn is not None
    return _conn
# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------

_USER_COLS = ("id, username, display_name, is_admin, failed_logins,"
              " locked_until, created_at, last_login")


def _as_user(row) -> dict:
    d = dict(row)
    d["is_admin"] = bool(d.get("is_admin", 0))
    d["locked"] = _is_locked(d.get("locked_until"))
    return d


def _is_locked(locked_until: Optional[str]) -> bool:
    if not locked_until:
        return False
    try:
        return datetime.fromisoformat(locked_until) > datetime.now(timezone.utc)
    except ValueError:
        return False


def create_user(username: str, display_name: str, password: str,
                is_admin: bool = False) -> dict:
    conn = _require()
    with _lock:
        cur = conn.execute(
            "INSERT INTO users (username, display_name, password_hash, is_admin,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (username, display_name, hash_password(password),
             1 if is_admin else 0, _now()),
        )
        conn.commit()
        uid = cur.lastrowid
    return get_user(uid)


def get_user(user_id: int) -> Optional[dict]:
    row = _require().execute(
        f"SELECT {_USER_COLS} FROM users WHERE id = ?", (user_id,)).fetchone()
    return _as_user(row) if row else None


def get_user_by_name(username: str) -> Optional[dict]:
    row = _require().execute(
        "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    rows = _require().execute(
        f"SELECT {_USER_COLS} FROM users ORDER BY id").fetchall()
    return [_as_user(r) for r in rows]


class LockedOut(Exception):
    """Raised instead of returning None, so the caller can say WHY."""

    def __init__(self, minutes: int):
        super().__init__("Account temporarily locked.")
        self.minutes = minutes


def verify_login(username: str, password: str) -> Optional[dict]:
    """Return the user on success, None on a bad credential, raise on lockout.

    "No such user" and "wrong password" are deliberately indistinguishable, so
    this cannot be used to enumerate the roster. A lockout IS distinguishable,
    which is a considered trade: an officer who cannot get in has to be told why
    or they will simply keep trying, and by that point three failed attempts have
    already revealed as much as the message does.
    """
    rec = get_user_by_name(username)
    if rec is None:
        # Spend the work anyway, so a missing user is not measurably faster.
        hash_password(password)
        return None

    if _is_locked(rec.get("locked_until")):
        remaining = datetime.fromisoformat(rec["locked_until"]) - datetime.now(timezone.utc)
        raise LockedOut(max(1, int(remaining.total_seconds() // 60) + 1))

    conn = _require()

    if not verify_password(password, rec["password_hash"]):
        failed = int(rec.get("failed_logins", 0)) + 1
        lock_until = None
        if failed >= MAX_FAILED_LOGINS:
            lock_until = (datetime.now(timezone.utc)
                          + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
            logger.warning("[DB] '%s' locked after %d failed attempts",
                           username, failed)
        with _lock:
            conn.execute(
                "UPDATE users SET failed_logins = ?, locked_until = ? WHERE id = ?",
                (0 if lock_until else failed, lock_until, rec["id"]))
            conn.commit()
        if lock_until:
            raise LockedOut(LOCKOUT_MINUTES)
        return None

    # A good password clears the counter -- the limit is on consecutive failures,
    # not on a tally that follows an officer around forever.
    with _lock:
        conn.execute(
            "UPDATE users SET last_login = ?, failed_logins = 0, locked_until = NULL"
            " WHERE id = ?", (_now(), rec["id"]))
        conn.commit()
    return get_user(rec["id"])


def unlock_user(user_id: int) -> bool:
    """Clear a lockout immediately. For an administrator who has verified the
    officer another way and does not want them waiting out the timer."""
    conn = _require()
    with _lock:
        cur = conn.execute(
            "UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = ?",
            (user_id,))
        conn.commit()
    return cur.rowcount > 0


def set_password(user_id: int, password: str) -> None:
    conn = _require()
    with _lock:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (hash_password(password), user_id))
        # Changing a password invalidates every existing session for that user.
        # If it is being changed because it was compromised, leaving the
        # intruder's session alive would defeat the point of changing it.
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def create_session(user_id: int, hours: int = SESSION_HOURS) -> dict:
    """Mint a session. The raw token is returned once and never stored."""
    token = secrets.token_urlsafe(32)
    issued = datetime.now(timezone.utc)
    expires = issued + timedelta(hours=hours)
    conn = _require()
    with _lock:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, issued_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (_token_digest(token), user_id, issued.isoformat(), expires.isoformat()))
        conn.commit()
    return {"token": token, "expires_at": expires.isoformat()}


def resolve_session(token: str) -> Optional[dict]:
    """Return the user behind a live token, or None. Expired rows are reaped."""
    if not token:
        return None
    row = _require().execute(
        "SELECT user_id, expires_at FROM sessions WHERE token_hash = ?",
        (_token_digest(token),)).fetchone()
    if row is None:
        return None
    if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
        delete_session(token)
        return None
    return get_user(row["user_id"])


def delete_session(token: str) -> None:
    conn = _require()
    with _lock:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?",
                     (_token_digest(token),))
        conn.commit()


def purge_expired() -> int:
    conn = _require()
    with _lock:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (_now(),))
        conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# password reset -- an approval queue, not a self-service dispenser
# ---------------------------------------------------------------------------
#
# The first version of this handed a working reset token straight back to whoever
# asked for it. With no mail server to send it to instead, that made knowing a
# username sufficient to take over the account: an authentication bypass wearing
# the clothes of a convenience feature.
#
# An officer now RAISES A REQUEST. An administrator -- a real person with their own
# credentials -- approves it and passes the token on by whatever channel they
# already trust for this: a phone call, a desk visit, the station roster. When a
# mail server exists, the approval step sends the mail and nothing else changes.


def request_password_reset(username: str) -> bool:
    """Queue a reset request.

    Returns False when no such user exists, and the caller must NOT reveal which
    happened -- a distinguishable answer here is a way to enumerate the roster.
    """
    rec = get_user_by_name(username)
    if rec is None:
        return False
    conn = _require()
    with _lock:
        # One open request per officer. Asking twice must not stack up work for
        # whoever reviews the queue.
        existing = conn.execute(
            "SELECT id FROM password_resets WHERE user_id = ? AND status = 'pending'",
            (rec["id"],)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO password_resets (user_id, status, requested_at)"
                " VALUES (?, 'pending', ?)", (rec["id"], _now()))
            conn.commit()
    return True


def list_reset_requests(status: str = "pending") -> list[dict]:
    """Requests for an administrator to action. Pass '' for every status."""
    rows = _require().execute(
        "SELECT r.id, r.status, r.requested_at, r.decided_by, r.decided_at,"
        "       r.expires_at, u.username, u.display_name"
        "  FROM password_resets r JOIN users u ON u.id = r.user_id"
        " WHERE (? = '' OR r.status = ?)"
        " ORDER BY r.requested_at DESC", (status, status)).fetchall()
    return [dict(r) for r in rows]


def approve_reset(request_id: int, decided_by: str) -> Optional[str]:
    """Approve a pending request and mint its single-use token.

    The raw token is returned ONCE, to the administrator, who hands it to the
    officer. It is never stored in the clear and never returned again.
    """
    conn = _require()
    row = conn.execute(
        "SELECT status FROM password_resets WHERE id = ?", (request_id,)).fetchone()
    if row is None or row["status"] != "pending":
        return None

    token = secrets.token_urlsafe(24)
    expires = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_MINUTES)
    with _lock:
        conn.execute(
            "UPDATE password_resets SET status = 'approved', token_hash = ?,"
            " expires_at = ?, decided_by = ?, decided_at = ? WHERE id = ?",
            (_token_digest(token), expires.isoformat(), decided_by,
             _now(), request_id))
        conn.commit()
    return token


def deny_reset(request_id: int, decided_by: str) -> bool:
    conn = _require()
    row = conn.execute(
        "SELECT status FROM password_resets WHERE id = ?", (request_id,)).fetchone()
    if row is None or row["status"] != "pending":
        return False
    with _lock:
        conn.execute(
            "UPDATE password_resets SET status = 'denied', decided_by = ?,"
            " decided_at = ? WHERE id = ?", (decided_by, _now(), request_id))
        conn.commit()
    return True


def consume_reset_token(token: str, new_password: str) -> bool:
    """Spend an APPROVED token. False if unknown, unapproved, expired or used."""
    if not token:
        return False
    row = _require().execute(
        "SELECT id, user_id, status, expires_at FROM password_resets"
        " WHERE token_hash = ?", (_token_digest(token),)).fetchone()
    if row is None or row["status"] != "approved":
        return False
    if not row["expires_at"]:
        return False
    if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
        return False

    set_password(row["user_id"], new_password)
    conn = _require()
    with _lock:
        # Clearing token_hash means a spent token cannot even be looked up again.
        conn.execute(
            "UPDATE password_resets SET status = 'used', used_at = ?,"
            " token_hash = NULL WHERE id = ?", (_now(), row["id"]))
        conn.commit()
    return True
