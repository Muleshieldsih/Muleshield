# -*- coding: utf-8 -*-
"""
Authentication, the officer roster, lockout, and cross-case intelligence.

    python -m pytest backend/tests/test_auth.py -v

Two tests here check security properties rather than mechanisms, and they are the
ones to read first:

  TestAttribution.test_authenticated_actor_overrides_the_body
      a caller can no longer decide whose name goes on an irreversible action

  TestPasswordReset.test_forgot_password_never_returns_a_token
      the first version of this flow handed a working reset token to anyone who
      asked for it, which made knowing a username enough to seize the account
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from backend import db                       # noqa: E402
from backend.main import app                 # noqa: E402

from backend.tests.conftest import ADMIN_PASSWORD as PASSWORD  # noqa: E402
from backend.tests.conftest import ADMIN_USER                 # noqa: E402


# `client` (anonymous) now comes from backend/tests/conftest.py, which owns the
# throwaway credential store and the pinned admin for the whole suite. This module
# keeps the ANONYMOUS client under that name, because most of what it asserts is
# what happens to a caller with no token.


@pytest.fixture
def admin(client):
    """Headers for the seeded administrator, freshly signed in.

    Fresh per test on purpose: several tests below change passwords, and changing
    a password revokes that officer's sessions. A module-scoped token would be
    legitimately dead by the time a later test used it, and the failure would read
    as "attribution is broken" when nothing is.
    """
    r = client.post("/api/v1/auth/login",
                    json={"username": ADMIN_USER, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def make_officer(client, admin, *, password="issued-password-1", is_admin=False):
    """Create a throwaway officer and return (username, headers, id).

    Tests that deliberately fail logins use one of these rather than the shared
    administrator, so a lockout cannot strand every test that follows.
    """
    username = f"IO-{uuid.uuid4().hex[:6]}"
    r = client.post("/api/v1/auth/users", headers=admin, json={
        "username": username, "display_name": f"Officer {username}",
        "password": password, "is_admin": is_admin,
    })
    assert r.status_code == 201, r.text
    uid = r.json()["id"]
    tok = client.post("/api/v1/auth/login",
                      json={"username": username, "password": password}
                      ).json()["access_token"]
    return username, {"Authorization": f"Bearer {tok}"}, uid


class TestPasswordStorage:
    def test_hash_round_trips(self):
        h = db.hash_password("correct horse")
        assert db.verify_password("correct horse", h)
        assert not db.verify_password("Correct horse", h)

    def test_same_password_hashes_differently(self):
        """A per-user salt, so a stolen store cannot be attacked once for everyone."""
        assert db.hash_password("same") != db.hash_password("same")

    def test_hash_is_not_the_password(self):
        h = db.hash_password("plaintext-secret")
        assert "plaintext-secret" not in h
        assert h.startswith("pbkdf2_sha256$")

    def test_malformed_hash_fails_closed(self):
        """A corrupt record must reject the login, never raise into it."""
        assert db.verify_password("anything", "not-a-real-hash") is False
        assert db.verify_password("anything", "") is False

    def test_verify_reads_iterations_from_stored_record(self, monkeypatch):
        """Verification must extract iteration count from the hash string itself,
        so existing credentials remain valid when PBKDF2_ITERATIONS is changed."""
        h50k = db.hash_password("my-password", iterations=50_000)
        assert "$50000$" in h50k
        monkeypatch.setattr(db, "PBKDF2_ITERATIONS", 600_000)
        assert db.verify_password("my-password", h50k) is True


class TestLogin:
    def test_login_returns_a_token(self, client):
        r = client.post("/api/v1/auth/login",
                        json={"username": "testofficer", "password": PASSWORD})
        assert r.status_code == 200
        body = r.json()
        assert body["token_type"] == "bearer"
        assert len(body["access_token"]) > 20
        assert body["user"]["username"] == "testofficer"
        assert body["user"]["is_admin"] is True

    def test_wrong_password_is_rejected(self, client, admin):
        _, _, _ = make_officer(client, admin)          # isolate from other tests
        r = client.post("/api/v1/auth/login",
                        json={"username": "nobody-at-all", "password": "wrong"})
        assert r.status_code == 401

    def test_failures_are_indistinguishable(self, client, admin):
        """A different message for a missing user would enumerate the roster."""
        username, _, _ = make_officer(client, admin)
        a = client.post("/api/v1/auth/login",
                        json={"username": username, "password": "wrong"}).json()
        b = client.post("/api/v1/auth/login",
                        json={"username": "no-such-officer", "password": "wrong"}).json()
        assert a["detail"] == b["detail"]


class TestLockout:
    def test_locks_after_three_failures(self, client, admin):
        username, _, _ = make_officer(client, admin, password="right-password-1")
        for _ in range(db.MAX_FAILED_LOGINS - 1):
            assert client.post("/api/v1/auth/login",
                               json={"username": username,
                                     "password": "wrong"}).status_code == 401
        # The attempt that trips the limit reports the lock, not a bad password.
        assert client.post("/api/v1/auth/login",
                           json={"username": username,
                                 "password": "wrong"}).status_code == 423

    def test_correct_password_is_refused_while_locked(self, client, admin):
        """The whole point of a lockout. If the right password still worked, an
        attacker who guessed it on attempt four would simply walk in."""
        username, _, _ = make_officer(client, admin, password="right-password-2")
        for _ in range(db.MAX_FAILED_LOGINS):
            client.post("/api/v1/auth/login",
                        json={"username": username, "password": "wrong"})
        r = client.post("/api/v1/auth/login",
                        json={"username": username, "password": "right-password-2"})
        assert r.status_code == 423

    def test_admin_can_unlock(self, client, admin):
        username, _, uid = make_officer(client, admin, password="right-password-3")
        for _ in range(db.MAX_FAILED_LOGINS):
            client.post("/api/v1/auth/login",
                        json={"username": username, "password": "wrong"})
        assert client.post(f"/api/v1/auth/users/{uid}/unlock",
                           headers=admin).status_code == 200
        assert client.post("/api/v1/auth/login",
                           json={"username": username,
                                 "password": "right-password-3"}).status_code == 200

    def test_a_good_login_clears_the_counter(self, client, admin):
        """The limit is on CONSECUTIVE failures, not a tally that follows an
        officer around forever."""
        username, _, _ = make_officer(client, admin, password="right-password-4")
        for _ in range(db.MAX_FAILED_LOGINS - 1):
            client.post("/api/v1/auth/login",
                        json={"username": username, "password": "wrong"})
        assert client.post("/api/v1/auth/login",
                           json={"username": username,
                                 "password": "right-password-4"}).status_code == 200
        # Back to a full allowance.
        for _ in range(db.MAX_FAILED_LOGINS - 1):
            assert client.post("/api/v1/auth/login",
                               json={"username": username,
                                     "password": "wrong"}).status_code == 401

    def test_lockout_disabled_flag_allows_retries_while_recording_failures(self, client, admin, monkeypatch, caplog):
        """When LOCKOUT_ENABLED is False (e.g. demo mode), unlimited retries are allowed,
        while the failed_logins signal is preserved for security audit/telemetry.
        
        NOTE: db.LOCKOUT_ENABLED is read at module import time, so tests must patch
        db.LOCKOUT_ENABLED directly via monkeypatch, not os.environ."""
        import logging
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(db, "LOCKOUT_ENABLED", False)
        username, _, uid = make_officer(client, admin, password="right-password-5")
        for _ in range(10):
            assert client.post("/api/v1/auth/login",
                               json={"username": username,
                                     "password": "wrong"}).status_code == 401
        u = db.get_user(uid)
        assert u["failed_logins"] == 10
        assert u["failed_logins_total"] == 10
        assert u["locked"] is False
        assert "potential credential stuffing" in caplog.text
        # Correct password still works after multiple retries
        assert client.post("/api/v1/auth/login",
                           json={"username": username,
                                 "password": "right-password-5"}).status_code == 200
        # Windowed counter resets to 0, cumulative counter preserves total for credential stuffing detection
        u_after = db.get_user(uid)
        assert u_after["failed_logins"] == 0
        assert u_after["failed_logins_total"] == 10

    def test_is_locked_handles_naive_and_corrupt_timestamps(self):
        """_is_locked must never raise TypeError on offset-naive timestamps or ValueError on corrupt strings."""
        assert db._is_locked("2020-01-01T00:00:00") is False  # past naive timestamp
        assert db._is_locked("not-a-timestamp") is False


class TestSession:
    def test_me_requires_a_token(self, client):
        """401, not 200. A 200 would mean the SPA catch-all had swallowed the route."""
        assert client.get("/api/v1/auth/me").status_code == 401

    def test_me_returns_the_officer(self, client, admin):
        r = client.get("/api/v1/auth/me", headers=admin)
        assert r.status_code == 200
        assert r.json()["username"] == "testofficer"

    def test_garbage_token_is_rejected(self, client):
        assert client.get("/api/v1/auth/me",
                          headers={"Authorization": "Bearer nope"}).status_code == 401

    def test_malformed_header_is_rejected(self, client):
        assert client.get("/api/v1/auth/me",
                          headers={"Authorization": "just-a-token"}).status_code == 401

    def test_logout_revokes_only_that_session(self, client, admin):
        """Signing out at one terminal must not sign the officer out everywhere."""
        second = client.post("/api/v1/auth/login",
                             json={"username": "testofficer", "password": PASSWORD}
                             ).json()["access_token"]
        h2 = {"Authorization": f"Bearer {second}"}

        assert client.post("/api/v1/auth/logout", headers=admin).status_code == 200
        assert client.get("/api/v1/auth/me", headers=admin).status_code == 401
        assert client.get("/api/v1/auth/me", headers=h2).status_code == 200

    def test_token_is_not_stored_in_the_clear(self, client):
        """A copied database must not hand over live sessions."""
        import sqlite3
        raw = client.post("/api/v1/auth/login",
                          json={"username": "testofficer", "password": PASSWORD}
                          ).json()["access_token"]
        con = sqlite3.connect(str(db.DB_PATH))
        stored = [r[0] for r in con.execute("SELECT token_hash FROM sessions")]
        con.close()
        assert raw not in stored
        assert all(len(s) == 64 for s in stored)      # sha256 hex


class TestRoster:
    def test_admin_can_issue_credentials(self, client, admin):
        username, _, _ = make_officer(client, admin)
        names = [u["username"] for u in client.get("/api/v1/auth/users",
                                                   headers=admin).json()]
        assert username in names

    def test_issued_officer_is_not_an_admin_by_default(self, client, admin):
        _, officer, _ = make_officer(client, admin)
        assert client.get("/api/v1/auth/me", headers=officer).json()["is_admin"] is False

    def test_duplicate_username_is_refused(self, client, admin):
        username, _, _ = make_officer(client, admin)
        r = client.post("/api/v1/auth/users", headers=admin, json={
            "username": username, "display_name": "Impostor",
            "password": "another-password"})
        assert r.status_code == 409

    def test_officer_cannot_reach_admin_routes(self, client, admin):
        """403, not 401: they have proven who they are, they are just not allowed."""
        _, officer, _ = make_officer(client, admin)
        assert client.get("/api/v1/auth/users", headers=officer).status_code == 403
        assert client.get("/api/v1/auth/reset-requests",
                          headers=officer).status_code == 403

    def test_anonymous_cannot_reach_admin_routes(self, client):
        assert client.get("/api/v1/auth/users").status_code == 401
        assert client.post("/api/v1/auth/users", json={
            "username": "x", "display_name": "x",
            "password": "12345678"}).status_code == 401

    def test_short_initial_password_is_refused(self, client, admin):
        r = client.post("/api/v1/auth/users", headers=admin, json={
            "username": "IO-short", "display_name": "x", "password": "abc"})
        assert r.status_code == 422


class TestPasswordReset:
    def test_forgot_password_never_returns_a_token(self, client, admin):
        """The bypass this flow was rebuilt to close.

        The first version returned a working token to whoever asked, so knowing a
        username was enough to take the account. Nothing an unauthenticated caller
        can do here may change any password.
        """
        username, _, _ = make_officer(client, admin)
        r = client.post("/api/v1/auth/forgot-password", json={"username": username})
        assert r.status_code == 200
        assert "reset_token" not in r.json()
        assert "token" not in r.text.lower()

    def test_unknown_user_gets_the_same_answer(self, client, admin):
        username, _, _ = make_officer(client, admin)
        a = client.post("/api/v1/auth/forgot-password", json={"username": username})
        b = client.post("/api/v1/auth/forgot-password", json={"username": "ghost"})
        assert a.status_code == b.status_code == 200
        assert a.json() == b.json()

    def test_admin_approval_flow(self, client, admin):
        username, _, _ = make_officer(client, admin, password="original-password")
        client.post("/api/v1/auth/forgot-password", json={"username": username})

        pending = [r for r in client.get("/api/v1/auth/reset-requests",
                                         headers=admin).json()
                   if r["username"] == username]
        assert pending, "the request never reached the queue"

        approved = client.post(
            f"/api/v1/auth/reset-requests/{pending[0]['id']}/approve",
            headers=admin)
        assert approved.status_code == 200
        token = approved.json()["reset_token"]

        assert client.post("/api/v1/auth/reset-password", json={
            "token": token, "new_password": "the-new-password"}).status_code == 200
        assert client.post("/api/v1/auth/login", json={
            "username": username,
            "password": "the-new-password"}).status_code == 200

    def test_unapproved_request_yields_no_working_token(self, client, admin):
        """A pending request must not be spendable. Only approval mints a token."""
        username, _, _ = make_officer(client, admin)
        client.post("/api/v1/auth/forgot-password", json={"username": username})
        # There is no token to guess; a fabricated one must be refused.
        assert client.post("/api/v1/auth/reset-password", json={
            "token": "made-up-token", "new_password": "whatever-1"}).status_code == 400

    def test_token_is_single_use(self, client, admin):
        username, _, _ = make_officer(client, admin)
        client.post("/api/v1/auth/forgot-password", json={"username": username})
        rid = [r for r in client.get("/api/v1/auth/reset-requests", headers=admin).json()
               if r["username"] == username][0]["id"]
        token = client.post(f"/api/v1/auth/reset-requests/{rid}/approve",
                            headers=admin).json()["reset_token"]

        assert client.post("/api/v1/auth/reset-password", json={
            "token": token, "new_password": "first-new-password"}).status_code == 200
        assert client.post("/api/v1/auth/reset-password", json={
            "token": token, "new_password": "second-attempt"}).status_code == 400

    def test_reset_revokes_existing_sessions(self, client, admin):
        """If a password is reset because it leaked, the intruder's live session
        must die with it."""
        username, officer, _ = make_officer(client, admin)
        assert client.get("/api/v1/auth/me", headers=officer).status_code == 200

        client.post("/api/v1/auth/forgot-password", json={"username": username})
        rid = [r for r in client.get("/api/v1/auth/reset-requests", headers=admin).json()
               if r["username"] == username][0]["id"]
        token = client.post(f"/api/v1/auth/reset-requests/{rid}/approve",
                            headers=admin).json()["reset_token"]
        client.post("/api/v1/auth/reset-password",
                    json={"token": token, "new_password": "rotated-password"})

        assert client.get("/api/v1/auth/me", headers=officer).status_code == 401

    def test_admin_can_deny(self, client, admin):
        username, _, _ = make_officer(client, admin)
        client.post("/api/v1/auth/forgot-password", json={"username": username})
        rid = [r for r in client.get("/api/v1/auth/reset-requests", headers=admin).json()
               if r["username"] == username][0]["id"]
        assert client.post(f"/api/v1/auth/reset-requests/{rid}/deny",
                           headers=admin).status_code == 200
        still = [r for r in client.get("/api/v1/auth/reset-requests", headers=admin).json()
                 if r["username"] == username]
        assert not still

    def test_short_new_password_is_refused(self, client, admin):
        username, _, _ = make_officer(client, admin)
        client.post("/api/v1/auth/forgot-password", json={"username": username})
        rid = [r for r in client.get("/api/v1/auth/reset-requests", headers=admin).json()
               if r["username"] == username][0]["id"]
        token = client.post(f"/api/v1/auth/reset-requests/{rid}/approve",
                            headers=admin).json()["reset_token"]
        assert client.post("/api/v1/auth/reset-password", json={
            "token": token, "new_password": "short"}).status_code == 422


class TestAttribution:
    """The point of the whole exercise."""

    def _a_complaint(self, client, admin):
        rows = client.get("/api/v1/complaint/list?limit=1", headers=admin).json()
        if not rows:
            pytest.skip("no complaints loaded")
        return rows[0]["ticket_id"]

    def test_authenticated_actor_overrides_the_body(self, client, admin):
        """A signed-in caller cannot pin a freeze on somebody else.

        This is the property the audit trail previously could not offer: it
        recorded faithfully whatever it was told, and it was told by the client.
        """
        cid = self._a_complaint(client, admin)
        acct = f"TEST-{uuid.uuid4().hex[:8]}"
        r = client.post("/api/v1/bank/micro-freeze", headers=admin, json={
            "account_id": acct, "complaint_id": cid,
            "officer_id": "SOMEONE-ELSE",       # a lie
        })
        assert r.status_code == 200
        assert r.json()["officer_id"] != "SOMEONE-ELSE"

        mine = [e for e in client.get(f"/api/v1/audit/{cid}", headers=admin).json()
                if acct in e.get("object", "")]
        assert mine, "the freeze left no audit entry"
        assert mine[0]["actor"] == "Duty Officer"

    def test_unauthenticated_freeze_is_refused(self, client):
        """The reversal of a deliberate earlier decision, recorded not hidden.

        This test previously asserted that an anonymous freeze WORKED. The
        reasoning was that every caller predating authentication should keep
        working, and that the old audit trail was not worthless -- it recorded
        exactly what it was given, it simply could not verify it. Both arguments
        were sound while the only thing at stake was attribution.

        Neither survives the observation that POST /api/v1/bank/micro-freeze is
        the one irreversible action in the product, taken against a real
        person's bank account, and that it was accepting it from anybody with
        curl and no credentials at all.

        What did NOT change: the body's `officer_id` still names the audit actor
        when no verified identity overrides it. See
        test_authenticated_actor_overrides_the_body. It just no longer serves as
        authorisation.
        """
        r = client.post("/api/v1/bank/micro-freeze", json={
            "account_id": "TEST-0001",
            "complaint_id": "any",
            "officer_id": "IO-LEGACY",
        })
        assert r.status_code == 401, r.text
        assert "bearer" in r.headers.get("www-authenticate", "").lower()

    def test_authenticated_note_is_attributed(self, client, admin):
        cid = self._a_complaint(client, admin)
        r = client.post(f"/api/v1/complaint/{cid}/note", headers=admin,
                        json={"text": "Checked the terminal account.",
                              "author": "NOT-ME"})
        assert r.status_code == 200
        assert r.json()["author"] == "Duty Officer"


class TestIntelligence:
    def test_recurring_atms(self, client):
        r = client.get("/api/v1/intel/atms?limit=5")
        assert r.status_code == 200
        rows = r.json()
        if not rows:
            pytest.skip("no cash-out data loaded")
        assert len(rows) <= 5
        assert rows[0]["cashouts"] >= 1
        assert rows[0]["atm_id"]

    def test_ranked_by_distinct_cases_not_raw_count(self, client):
        """A chain that pools four ways into one machine is one case, not four."""
        rows = client.get("/api/v1/intel/atms?limit=50").json()
        if len(rows) < 2:
            pytest.skip("not enough data")
        for a, b in zip(rows, rows[1:]):
            assert a["distinct_complaints"] >= b["distinct_complaints"]

    def test_limit_is_bounded(self, client):
        assert client.get("/api/v1/intel/atms?limit=0").status_code == 422
        assert client.get("/api/v1/intel/atms?limit=9999").status_code == 422
