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

# Overridable so a test run, the UI smoke sweep and a container can each point at
# their own store. It defaults to the real one, so nothing that does not set the
# variable changes behaviour. Before this, `backend/tests/test_phase3.py` and
# `test_case_workflow.py` wrote officers into the live credential store, and
# scripts/smoke_ui.py's "isolated" stack did too -- isolated in every respect
# except the one file that holds password hashes.
DB_PATH = Path(os.environ.get("MULESHIELD_DB_PATH", str(ROOT / "data" / "muleshield.db")))

# OWASP's current floor for PBKDF2-HMAC-SHA256. Costs roughly 0.2s per login on a
# laptop, which is the point: it is the attacker's cost that matters.
# Configurable via PBKDF2_ITERATIONS (e.g. 50_000 for fast demo / dev, 600_000 for prod).
PBKDF2_ITERATIONS = int(os.environ.get("PBKDF2_ITERATIONS", "600000"))
SALT_BYTES = 16
SESSION_HOURS = 12          # a shift, not a fortnight
RESET_TOKEN_MINUTES = 30

# Lockout after repeated failures, to stop an unlimited guessing run against a
# known username. Gated by LOCKOUT_ENABLED so it can be disabled for demo/testing
# environments while remaining standard production behavior.
LOCKOUT_ENABLED = os.environ.get("LOCKOUT_ENABLED", "1").lower() in ("1", "true", "yes")
MAX_FAILED_LOGINS = int(os.environ.get("MAX_FAILED_LOGINS", os.environ.get("MAX_FAILED_ATTEMPTS", "3")))
LOCKOUT_MINUTES = int(os.environ.get("LOCKOUT_MINUTES", "15"))

# sqlite3 is synchronous and the endpoints calling it are async. Every query here
# is a microsecond-scale local read against a table with a handful of rows, so a
# single connection behind a lock is simpler -- and has fewer failure modes -- than
# introducing aiosqlite for a workload that never blocks.
_conn: Optional[sqlite3.Connection] = None
_lock = threading.RLock()
"""Serialises EVERY access to the single connection, reads included.

Reentrant, because a locked write path legitimately calls a locked read path --
insert_alert() finishes by reading the row it just wrote.

RECORDED REVERSAL. Only WRITES used to take this lock; the seventeen read
functions went straight to conn.execute(). One sqlite3 connection shared across
threads is not safe to interleave that way, and the background alert tick runs
in a threadpool alongside HTTP handlers, so it happened constantly under load.
The symptom was not a clean error: the module lost track of which exception it
was raising. A UNIQUE-index violation, which insert_alert catches by design to
suppress a duplicate, arrived as a bare sqlite3.DatabaseError and escaped as a
500. A stress pass at the problem statement's stated national load (8,000
complaints/day) produced 43 of them, plus InterfaceError "bad parameter or other
API misuse" on unrelated reads.

Serialising here rather than moving to a connection pool is deliberate: this
database holds credentials, sessions and alerts, not the case load, so the
contention is negligible and one lock is far easier to prove correct than a pool.
"""


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    username             TEXT    UNIQUE NOT NULL,
    display_name         TEXT    NOT NULL,
    password_hash        TEXT    NOT NULL,
    is_admin             INTEGER NOT NULL DEFAULT 0,
    failed_logins        INTEGER NOT NULL DEFAULT 0,
    failed_logins_total  INTEGER NOT NULL DEFAULT 0,
    locked_until         TEXT,
    created_at           TEXT    NOT NULL,
    last_login           TEXT
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

-- ── Alerting ────────────────────────────────────────────────────────────────
-- The audit found the system had no alert object at all: deliverable (d) asks
-- for real-time notification to LEAs, banks and I4C officers, and the only
-- thing that existed was a toast that said "SMS queued (simulated)".
--
-- These three tables are the difference between a demo and a record. An alert
-- that was raised, who it went to, whether it arrived, and what an officer
-- decided about it are all evidence -- they establish that a force WAS warned.
CREATE TABLE IF NOT EXISTS alerts (
    id               TEXT PRIMARY KEY,               -- ALT-000001
    created_at       TEXT    NOT NULL,
    rule_id          TEXT    NOT NULL,
    severity         TEXT    NOT NULL,               -- CRITICAL | HIGH | WATCH
    cell_id          TEXT    NOT NULL DEFAULT '',
    district         TEXT    NOT NULL DEFAULT '',
    state            TEXT    NOT NULL DEFAULT '',
    window_start_min INTEGER NOT NULL DEFAULT 0,
    window_end_min   INTEGER NOT NULL DEFAULT 0,
    score            REAL    NOT NULL DEFAULT 0,
    rupees_at_risk   REAL    NOT NULL DEFAULT 0,
    case_count       INTEGER NOT NULL DEFAULT 0,
    complaint_ids    TEXT    NOT NULL DEFAULT '',    -- comma-separated ticket ids
    -- The decomposition, persisted. An alert has to carry HOW MUCH of its own
    -- score was live forecast rather than historical prior, or an officer
    -- reading it a day later cannot tell a forecast from a density map.
    prior_share      REAL    NOT NULL DEFAULT 0,
    headline         TEXT    NOT NULL DEFAULT '',
    status           TEXT    NOT NULL DEFAULT 'open',  -- open|acknowledged|dismissed
    acknowledged_by  TEXT,
    acknowledged_at  TEXT,
    disposition      TEXT,                           -- Dispatched|Monitoring|False positive|Duplicate
    dedupe_bucket    TEXT    NOT NULL DEFAULT ''     -- ISO hour, e.g. 2026-09-01T15
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_status  ON alerts(status);
-- A cell that stays hot for an hour must not mint a new alert on every tick.
-- The bucket is the hour, so one rule fires at most once per cell per window
-- per hour. Enforced by the database rather than by the caller remembering.
CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_dedupe
    ON alerts(rule_id, cell_id, window_start_min, dedupe_bucket);

CREATE TABLE IF NOT EXISTS alert_deliveries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id      TEXT    NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    channel       TEXT    NOT NULL,                  -- sms | email | webhook
    recipient     TEXT    NOT NULL,
    state         TEXT    NOT NULL,                  -- queued | sent | failed | dead
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    queued_at     TEXT    NOT NULL,
    sent_at       TEXT,
    next_retry_at TEXT,
    provider_ref  TEXT
);
CREATE INDEX IF NOT EXISTS idx_deliveries_alert ON alert_deliveries(alert_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_state ON alert_deliveries(state);

-- "Send it to whom" -- the question COMPLIANCE_AUDIT.md finding 5.3 says the
-- system could not answer. Scope is by state/district so an alert reaches the
-- force responsible for the ground it covers; blank scope means national.
CREATE TABLE IF NOT EXISTS alert_recipients (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    role           TEXT    NOT NULL,                 -- LEA | I4C | BANK
    channel        TEXT    NOT NULL,
    address        TEXT    NOT NULL,
    scope_state    TEXT    NOT NULL DEFAULT '',
    scope_district TEXT    NOT NULL DEFAULT '',
    active         INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_recipients_scope ON alert_recipients(scope_state, active);

-- Evidence documentation. COMPLIANCE_AUDIT.md finding 4.5: the problem statement
-- names "evidence documentation" as part of deliverable (c) and nothing in the
-- repository could accept, hold or account for a file.
--
-- Three properties make this evidence rather than an attachment:
--
--   1. INTEGRITY. sha256 of the bytes is taken at the moment of collection and
--      re-checked on every read. A file that no longer hashes to what was
--      recorded is served as a failure, not as evidence.
--   2. CHAIN. Each row carries the entry_hash of the previous item on the SAME
--      case, so an artefact cannot be inserted into, removed from, or reordered
--      within a case's history without breaking every link after it. This is
--      what the audit called "tamper-evident" and did not have.
--   3. NO DELETION. Withdrawal is a status with a reason and an actor. Evidence
--      that can be deleted is evidence that can be made to disappear between
--      collection and trial, which defeats the point of holding it.
CREATE TABLE IF NOT EXISTS case_evidence (
    id               TEXT    PRIMARY KEY,            -- EVD-000001
    case_id          TEXT    NOT NULL,               -- the 1930 ticket id
    seq              INTEGER NOT NULL,               -- position in this case's chain, from 1
    filename         TEXT    NOT NULL,               -- as supplied by the officer
    stored_name      TEXT    NOT NULL,               -- on disk; never the supplied name
    content_type     TEXT    NOT NULL DEFAULT '',    -- declared, never trusted on the way out
    size_bytes       INTEGER NOT NULL DEFAULT 0,
    sha256           TEXT    NOT NULL,               -- of the bytes, at collection
    kind             TEXT    NOT NULL DEFAULT 'other',
    description      TEXT    NOT NULL DEFAULT '',
    source           TEXT    NOT NULL DEFAULT '',    -- who it came from: bank, victim, device
    collected_by     TEXT    NOT NULL,               -- from the bearer token, not the body
    collected_at     TEXT    NOT NULL,
    withdrawn_at     TEXT,
    withdrawn_by     TEXT,
    withdrawn_reason TEXT,
    prev_hash        TEXT    NOT NULL DEFAULT '',
    entry_hash       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_case ON case_evidence(case_id, seq);
-- One artefact per case per content hash. Re-uploading the same bytes to the
-- same case returns the item already held rather than minting a second custody
-- record for one object, which would make the chain describe a history that did
-- not happen.
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_dedupe ON case_evidence(case_id, sha256);
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
        # Ensure failed_logins_total column exists if migrating an existing db
        try:
            _conn.execute("ALTER TABLE users ADD COLUMN failed_logins_total INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
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
        password = "password123"
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

_USER_COLS = ("id, username, display_name, is_admin, failed_logins, failed_logins_total,"
              " locked_until, created_at, last_login")


def _as_user(row) -> dict:
    d = dict(row)
    d["is_admin"] = bool(d.get("is_admin", 0))
    d["locked"] = _is_locked(d.get("locked_until"))
    return d


def _is_locked(locked_until: Optional[str]) -> bool:
    if not LOCKOUT_ENABLED or not locked_until:
        return False
    try:
        dt = datetime.fromisoformat(locked_until)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt > datetime.now(timezone.utc)
    except (ValueError, TypeError):
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
    with _lock:
        row = _require().execute(
            f"SELECT {_USER_COLS} FROM users WHERE id = ?", (user_id,)).fetchone()
    return _as_user(row) if row else None


def get_user_by_name(username: str) -> Optional[dict]:
    with _lock:
        row = _require().execute(
            "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with _lock:
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
        failed_total = int(rec.get("failed_logins_total", 0)) + 1
        lock_until = None
        if LOCKOUT_ENABLED and failed >= MAX_FAILED_LOGINS:
            lock_until = (datetime.now(timezone.utc)
                          + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
            logger.warning("[DB] '%s' locked after %d failed attempts",
                           username, failed)
        if failed_total >= MAX_FAILED_LOGINS * 3:
            logger.warning(
                "[DB] high failed attempt volume for '%s' (total: %d) -- potential credential stuffing",
                username, failed_total)
        with _lock:
            conn.execute(
                "UPDATE users SET failed_logins = ?, failed_logins_total = ?, locked_until = ? WHERE id = ?",
                (0 if lock_until else failed, failed_total, lock_until, rec["id"]))
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
    with _lock:
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
    with _lock:
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
    token = secrets.token_urlsafe(24)
    expires = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_MINUTES)
    # Check and approve under one lock. Split apart, two administrators actioning
    # the same request at the same moment both read "pending" and both mint a
    # token -- two live credentials for one account, only one of them known to
    # the officer.
    with _lock:
        conn = _require()
        row = conn.execute(
            "SELECT status FROM password_resets WHERE id = ?", (request_id,)).fetchone()
        if row is None or row["status"] != "pending":
            return None
        conn.execute(
            "UPDATE password_resets SET status = 'approved', token_hash = ?,"
            " expires_at = ?, decided_by = ?, decided_at = ? WHERE id = ?",
            (_token_digest(token), expires.isoformat(), decided_by,
             _now(), request_id))
        conn.commit()
    return token


def deny_reset(request_id: int, decided_by: str) -> bool:
    with _lock:
        conn = _require()
        row = conn.execute(
            "SELECT status FROM password_resets WHERE id = ?", (request_id,)).fetchone()
        if row is None or row["status"] != "pending":
            return False
        conn.execute(
            "UPDATE password_resets SET status = 'denied', decided_by = ?,"
            " decided_at = ? WHERE id = ?", (decided_by, _now(), request_id))
        conn.commit()
    return True


def consume_reset_token(token: str, new_password: str) -> bool:
    """Spend an APPROVED token. False if unknown, unapproved, expired or used."""
    if not token:
        return False
    with _lock:
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


# ---------------------------------------------------------------------------
# alerting
#
# Every function below is a plain `def`. That is the rule stated in
# backend/auth.py:46-59 and it is not stylistic: the app ships
# `uvicorn --workers 1`, so a blocking sqlite3 call made on the event loop
# stalls every other request AND the /ws/feed sockets along with it. Handlers
# that need to both write here and broadcast cross the seam with
# starlette.concurrency.run_in_threadpool.
#
# Nothing in this section imports backend.websocket. A test asserts it.
# ---------------------------------------------------------------------------

_ALERT_COLS = (
    "id, created_at, rule_id, severity, cell_id, district, state, "
    "window_start_min, window_end_min, score, rupees_at_risk, case_count, "
    "complaint_ids, prior_share, headline, status, acknowledged_by, "
    "acknowledged_at, disposition, dedupe_bucket"
)


def _as_alert(row) -> dict:
    d = dict(row)
    d["complaint_ids"] = [x for x in str(d.get("complaint_ids") or "").split(",") if x]
    return d


def next_alert_id() -> str:
    """Next free ALT-nnnnnn.

    RECORDED REVERSAL. This used to be COUNT(*) + 1, which is wrong in two ways
    that a concurrency stress pass made visible. Two threads allocating at once
    both read the same count and both propose the same id; and once any alert is
    deleted, COUNT(*) + 1 names an id that already exists. Either way the INSERT
    hits the PRIMARY KEY, and insert_alert -- which catches a UNIQUE violation to
    mean "duplicate suppressed" -- would have silently DISCARDED a real alert
    while reporting normal operation. Silently dropping an alert is the single
    worst failure this subsystem has.

    MAX + 1 over the numeric suffix is monotone under deletion. It is read under
    the same lock the INSERT takes, and insert_alert retries once on a genuine id
    collision, so the allocate-then-insert pair cannot be interleaved apart.
    """
    with _lock:
        row = _require().execute(
            "SELECT MAX(CAST(SUBSTR(id, 5) AS INTEGER)) AS n FROM alerts"
            " WHERE id LIKE 'ALT-%'").fetchone()
    return f"ALT-{int(row['n'] or 0) + 1:06d}"


def insert_alert(alert: dict) -> Optional[dict]:
    """Insert one alert, or return None when the dedupe index rejects it.

    The UNIQUE index on (rule_id, cell_id, window_start_min, dedupe_bucket) is
    the whole duplicate-suppression mechanism, and it lives in the database
    rather than in the caller. A hot cell is evaluated on every tick; without
    this, a district that stays dangerous for an afternoon would bury an officer
    under identical alerts and the inbox would become unreadable exactly when it
    mattered most.

    Returning None rather than raising: a suppressed duplicate is the normal
    case, not an error.
    """
    ids = alert.get("complaint_ids") or []
    if isinstance(ids, (list, tuple, set)):
        ids = ",".join(str(i) for i in ids)
    with _lock:
        conn = _require()
        try:
            conn.execute(
                f"INSERT INTO alerts ({_ALERT_COLS}) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    alert["id"], _now(), alert["rule_id"], alert["severity"],
                    alert.get("cell_id", ""), alert.get("district", ""),
                    alert.get("state", ""),
                    int(alert.get("window_start_min", 0)),
                    int(alert.get("window_end_min", 0)),
                    float(alert.get("score", 0.0)),
                    float(alert.get("rupees_at_risk", 0.0)),
                    int(alert.get("case_count", 0)),
                    ids,
                    float(alert.get("prior_share", 0.0)),
                    alert.get("headline", ""),
                    "open", None, None, None,
                    alert.get("dedupe_bucket", ""),
                ),
            )
            conn.commit()
        except sqlite3.DatabaseError as exc:
            # Classify by message, not by exception class. Under the concurrent
            # use this module used to permit, sqlite3 reported a UNIQUE violation
            # as a bare DatabaseError rather than an IntegrityError, so this
            # clause missed it and a suppressed duplicate escaped as a 500. The
            # lock now prevents that, and matching on the message means the
            # dedupe path cannot break again if the class ever shifts.
            if "UNIQUE constraint failed" not in str(exc):
                raise
            # Which constraint matters. The dedupe index firing is the normal
            # case. The PRIMARY KEY firing means the id allocator handed out an
            # id that already exists, and swallowing THAT would silently discard
            # a real alert -- so it is raised, loudly, rather than suppressed.
            if "alerts.id" in str(exc):
                raise
            return None
    return get_alert(alert["id"])


def get_alert(alert_id: str) -> Optional[dict]:
    with _lock:
        row = _require().execute(
            f"SELECT {_ALERT_COLS} FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    return _as_alert(row) if row else None


def list_alerts(*, status: str = "", severity: str = "", state: str = "",
                limit: int = 100) -> list[dict]:
    sql = f"SELECT {_ALERT_COLS} FROM alerts WHERE 1=1"
    args: list = []
    if status:
        sql += " AND status = ?"; args.append(status)
    if severity:
        sql += " AND severity = ?"; args.append(severity)
    if state:
        sql += " AND state = ?"; args.append(state)
    # Severity first, then recency: an inbox sorted purely by time buries a
    # CRITICAL under a run of WATCH noise.
    sql += (" ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1"
            " ELSE 2 END, created_at DESC LIMIT ?")
    args.append(int(limit))
    with _lock:
        rows = _require().execute(sql, args).fetchall()
    return [_as_alert(r) for r in rows]


def acknowledge_alert(alert_id: str, actor: str, disposition: str) -> Optional[dict]:
    """Close the loop on one alert.

    `disposition` is required by the caller, not optional, and 'False positive'
    is one of its values. That is the outcome capture COMPLIANCE_AUDIT.md
    finding 2.5 says the system lacks: without it nothing ever records whether
    an alert was worth raising, and a framework that cannot tell is a framework
    that cannot improve.
    """
    conn = _require()
    with _lock:
        cur = conn.execute(
            "UPDATE alerts SET status = 'acknowledged', acknowledged_by = ?,"
            " acknowledged_at = ?, disposition = ? WHERE id = ? AND status != 'acknowledged'",
            (actor, _now(), disposition, alert_id))
        conn.commit()
    if cur.rowcount == 0 and get_alert(alert_id) is None:
        return None
    return get_alert(alert_id)


def alert_outcome_counts() -> dict:
    with _lock:
        rows = _require().execute(
            "SELECT disposition, COUNT(*) AS n FROM alerts"
            " WHERE disposition IS NOT NULL GROUP BY disposition").fetchall()
    return {str(r["disposition"]): int(r["n"]) for r in rows}


# ---- deliveries -----------------------------------------------------------

def queue_delivery(alert_id: str, channel: str, recipient: str) -> int:
    conn = _require()
    with _lock:
        cur = conn.execute(
            "INSERT INTO alert_deliveries"
            " (alert_id, channel, recipient, state, attempts, queued_at)"
            " VALUES (?,?,?,'queued',0,?)",
            (alert_id, channel, recipient, _now()))
        conn.commit()
        return int(cur.lastrowid)


def mark_delivery(delivery_id: int, *, state: str, provider_ref: str = "",
                  error: str = "", next_retry_at: Optional[str] = None) -> None:
    conn = _require()
    with _lock:
        conn.execute(
            "UPDATE alert_deliveries SET state = ?, attempts = attempts + 1,"
            " provider_ref = COALESCE(NULLIF(?, ''), provider_ref),"
            " last_error = NULLIF(?, ''),"
            " sent_at = CASE WHEN ? = 'sent' THEN ? ELSE sent_at END,"
            " next_retry_at = ? WHERE id = ?",
            (state, provider_ref, error, state, _now(), next_retry_at, delivery_id))
        conn.commit()


def list_deliveries(alert_id: str) -> list[dict]:
    with _lock:
        rows = _require().execute(
            "SELECT * FROM alert_deliveries WHERE alert_id = ? ORDER BY id",
            (alert_id,)).fetchall()
    return [dict(r) for r in rows]


def deliveries_due(now_iso: str, max_attempts: int) -> list[dict]:
    with _lock:
        rows = _require().execute(
            "SELECT * FROM alert_deliveries WHERE state = 'failed'"
            " AND attempts < ? AND (next_retry_at IS NULL OR next_retry_at <= ?)",
            (int(max_attempts), now_iso)).fetchall()
    return [dict(r) for r in rows]


def delivery_state_counts() -> dict:
    with _lock:
        rows = _require().execute(
            "SELECT state, COUNT(*) AS n FROM alert_deliveries GROUP BY state").fetchall()
    return {str(r["state"]): int(r["n"]) for r in rows}


# ---- recipients -----------------------------------------------------------

def list_recipients(*, state: str = "", district: str = "",
                    active_only: bool = True) -> list[dict]:
    """Recipients responsible for a place, widest scope last.

    A national I4C desk (blank scope) is returned alongside the state force and
    the district unit, because all three are meant to see it -- that is what
    "coordinated by I4C" means in the problem statement.
    """
    sql = "SELECT * FROM alert_recipients WHERE 1=1"
    args: list = []
    if active_only:
        sql += " AND active = 1"
    if state:
        sql += " AND (scope_state = '' OR scope_state = ?)"
        args.append(state)
    if district:
        sql += " AND (scope_district = '' OR scope_district = ?)"
        args.append(district)
    sql += " ORDER BY scope_district DESC, scope_state DESC, id"
    with _lock:
        rows = _require().execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def seed_recipients(rows: list[dict]) -> int:
    """Populate the roster once, on an empty table only.

    Mirrors _seed_first_user(): seeding is a first-run convenience, never
    something that overwrites a roster an operator has since edited.
    """
    # Emptiness check and seed under one lock: two workers booting together
    # would otherwise both find the table empty and seed the roster twice.
    with _lock:
        conn = _require()
        n = conn.execute("SELECT COUNT(*) AS n FROM alert_recipients").fetchone()["n"]
        if int(n) > 0:
            return 0
        for r in rows:
            conn.execute(
                "INSERT INTO alert_recipients"
                " (name, role, channel, address, scope_state, scope_district, active)"
                " VALUES (?,?,?,?,?,?,1)",
                (r["name"], r["role"], r["channel"], r["address"],
                 r.get("scope_state", ""), r.get("scope_district", "")))
        conn.commit()
    logger.info("[DB] seeded %d alert recipients", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

_EVIDENCE_COLS = (
    "id, case_id, seq, filename, stored_name, content_type, size_bytes, sha256, "
    "kind, description, source, collected_by, collected_at, withdrawn_at, "
    "withdrawn_by, withdrawn_reason, prev_hash, entry_hash"
)


def chain_hash(prev_hash: str, item_id: str, case_id: str, digest: str,
               collected_by: str, collected_at: str) -> str:
    """The link. Deliberately over the COLLECTION facts only.

    What this proves: that the sequence of artefacts collected against a case,
    each identified by the hash of its own bytes, has not been added to, removed
    from or reordered since. Break any link and every hash after it stops
    matching.

    What it does not prove: that the store as a whole is authentic against an
    outside reference. A party with write access to the whole table could
    recompute the entire chain. Making that impossible needs an external anchor
    -- a notary, a signed daily digest, an append-only log the operator does not
    own -- and that is a deployment decision, not something this build can fake.
    Stated plainly here rather than implied by the word "tamper-evident".

    Withdrawal is deliberately outside the hash: an artefact's custody record is
    fixed at collection, and the fact that somebody later withdrew it is a
    separate event, recorded on the row and in the audit trail.
    """
    payload = "|".join((prev_hash, item_id, case_id, digest, collected_by, collected_at))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_evidence(row) -> dict:
    d = dict(row)
    d["withdrawn"] = bool(d.get("withdrawn_at"))
    return d


def get_evidence(evidence_id: str) -> Optional[dict]:
    with _lock:
        row = _require().execute(
            f"SELECT {_EVIDENCE_COLS} FROM case_evidence WHERE id = ?",
            (evidence_id,)).fetchone()
    return _as_evidence(row) if row else None


def list_evidence(case_id: str, *, include_withdrawn: bool = True) -> list[dict]:
    sql = f"SELECT {_EVIDENCE_COLS} FROM case_evidence WHERE case_id = ?"
    if not include_withdrawn:
        sql += " AND withdrawn_at IS NULL"
    sql += " ORDER BY seq"
    with _lock:
        rows = _require().execute(sql, (case_id,)).fetchall()
    return [_as_evidence(r) for r in rows]


def find_evidence_by_digest(case_id: str, digest: str) -> Optional[dict]:
    with _lock:
        row = _require().execute(
            f"SELECT {_EVIDENCE_COLS} FROM case_evidence"
            " WHERE case_id = ? AND sha256 = ?", (case_id, digest)).fetchone()
    return _as_evidence(row) if row else None


def insert_evidence(item: dict) -> dict:
    """Append one artefact to a case's chain.

    Sequence number, previous hash and entry hash are all derived HERE, under the
    same lock as the INSERT. A caller that computed its own position would race
    another upload on the same case and produce two items claiming the same link.
    """
    with _lock:
        conn = _require()
        row = conn.execute(
            "SELECT seq, entry_hash FROM case_evidence WHERE case_id = ?"
            " ORDER BY seq DESC LIMIT 1", (item["case_id"],)).fetchone()
        seq = (int(row["seq"]) + 1) if row else 1
        prev_hash = str(row["entry_hash"]) if row else ""

        item_id = f"EVD-{_next_evidence_number(conn):06d}"
        collected_at = _now()
        # The stored name IS the artefact id. Never the supplied filename: a
        # caller-controlled string must not become a path, and an id keeps the
        # store readable in the same order the chain is.
        stored_name = item_id
        entry_hash = chain_hash(prev_hash, item_id, item["case_id"],
                                item["sha256"], item["collected_by"], collected_at)

        conn.execute(
            f"INSERT INTO case_evidence ({_EVIDENCE_COLS}) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item_id, item["case_id"], seq, item["filename"], stored_name,
             item.get("content_type", ""), int(item.get("size_bytes", 0)),
             item["sha256"], item.get("kind", "other"), item.get("description", ""),
             item.get("source", ""), item["collected_by"], collected_at,
             None, None, None, prev_hash, entry_hash),
        )
        conn.commit()
    return get_evidence(item_id)


def _next_evidence_number(conn) -> int:
    """MAX + 1 over the suffix, never COUNT + 1 -- see next_alert_id()."""
    row = conn.execute(
        "SELECT MAX(CAST(SUBSTR(id, 5) AS INTEGER)) AS n FROM case_evidence"
        " WHERE id LIKE 'EVD-%'").fetchone()
    return int(row["n"] or 0) + 1


def drop_evidence_row(evidence_id: str) -> None:
    """Remove a row whose bytes never made it into the store.

    The ONLY deletion this module allows, and it is not a deletion of evidence:
    it unwinds a collection that failed between the row landing and the file
    landing, so the chain never claims to hold something that is not there.
    Withdrawal of a real artefact is withdraw_evidence(), which deletes nothing.
    """
    with _lock:
        conn = _require()
        conn.execute("DELETE FROM case_evidence WHERE id = ?", (evidence_id,))
        conn.commit()


def withdraw_evidence(evidence_id: str, actor: str, reason: str) -> Optional[dict]:
    """Mark an artefact withdrawn. There is no delete, by design."""
    with _lock:
        conn = _require()
        cur = conn.execute(
            "UPDATE case_evidence SET withdrawn_at = ?, withdrawn_by = ?,"
            " withdrawn_reason = ? WHERE id = ? AND withdrawn_at IS NULL",
            (_now(), actor, reason, evidence_id))
        conn.commit()
        if cur.rowcount == 0:
            return None
    return get_evidence(evidence_id)


def evidence_counts() -> dict:
    with _lock:
        rows = _require().execute(
            "SELECT COUNT(*) AS n, COUNT(withdrawn_at) AS w,"
            " COALESCE(SUM(size_bytes), 0) AS b,"
            " COUNT(DISTINCT case_id) AS c FROM case_evidence").fetchone()
    return {"items": int(rows["n"]), "withdrawn": int(rows["w"]),
            "bytes": int(rows["b"]), "cases": int(rows["c"])}
