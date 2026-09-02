# -*- coding: utf-8 -*-
"""
MuleShield AI -- shared test fixtures.
SIH26184 | MHA / I4C

    Run: python -m pytest backend/tests -v

WHY THIS FILE EXISTS
--------------------
Until now every backend test module declared its own `client` fixture and its own
sys.path bootstrap. Three consequences, all of which this file fixes:

1. `test_phase3.py` and `test_case_workflow.py` ran against the REAL credential
   store at data/muleshield.db, creating officers in it. Only test_auth.py
   repointed db.DB_PATH, and it did so by assigning the module global directly.
2. Each module triggered its own lifespan, so state.load_all() parsed a 622k-row
   CSV and loaded torch three times per suite.
3. There was no way to make a request AS somebody. That was survivable while no
   endpoint required a token. It stopped being survivable the moment they did.

THE TWO CLIENTS, AND THE DIFFERENCE BETWEEN THEM
------------------------------------------------
    client       anonymous. Name and semantics deliberately UNCHANGED, so every
                 existing test that asserts a 401/403/422 against a stranger
                 keeps meaning exactly what it meant before.
    auth_client  the same app, signed in as the seeded administrator.

A module whose endpoints now require a token overrides `client` with a four-line
alias returning `auth_client` (see test_case_workflow.py). That is the whole
migration: no test body changes, and the anonymous client stays available under
the name `anon` for tests that need to assert the refusal.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT), str(ROOT / "engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

ADMIN_USER = "testofficer"
ADMIN_PASSWORD = "test-password-not-a-secret"


@pytest.fixture(scope="session", autouse=True)
def _isolated_env():
    """Point the suite at a throwaway credential store, BEFORE anything imports.

    Autouse and session-scoped on purpose: db.DB_PATH is read at import time, and
    the app's lifespan calls db.init() the moment the first TestClient is
    constructed. Setting these after that point would be too late, and the suite
    would seed officers into the real store -- which is exactly the bug this
    replaces.

    MULESHIELD_SCHEDULER=off keeps the background rule-evaluation tick from
    running under test. Tests that need a rule pass call notify.evaluate()
    directly, which is deterministic; a timer firing mid-assertion is not.
    """
    tmp = Path(tempfile.mkdtemp(prefix="muleshield-test-"))
    os.environ["MULESHIELD_DB_PATH"] = str(tmp / "test.db")
    os.environ["MULESHIELD_ADMIN_USER"] = ADMIN_USER
    os.environ["MULESHIELD_ADMIN_PASSWORD"] = ADMIN_PASSWORD
    os.environ["MULESHIELD_SCHEDULER"] = "off"
    # Evidence bytes go to disk. Without this the suite would collect artefacts
    # into data/evidence/ alongside real ones -- the file-store equivalent of the
    # credential-store bug this fixture exists to prevent.
    os.environ["MULESHIELD_EVIDENCE_DIR"] = str(tmp / "evidence")

    from backend import db
    db.close()
    db.DB_PATH = Path(os.environ["MULESHIELD_DB_PATH"])

    # Same reason db.DB_PATH is reassigned: the module read the environment at
    # import time, and a test that imported it earlier would otherwise keep the
    # real store.
    from backend import evidence as _evidence
    _evidence.EVIDENCE_DIR = Path(os.environ["MULESHIELD_EVIDENCE_DIR"])
    yield
    db.close()


@pytest.fixture(scope="session")
def app_client(_isolated_env):
    """One TestClient for the whole suite, so the lifespan runs exactly once.

    state.load_all() parses data/transactions.csv and loads two model
    checkpoints. Doing that once instead of once per module is the difference
    between a suite that takes seconds and one that takes minutes.
    """
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def client(app_client):
    """Anonymous. Unchanged from what every module used to declare locally."""
    return app_client


@pytest.fixture(scope="module")
def anon(app_client):
    """A genuinely credential-free client.

    NOT an alias for `app_client`. `auth_client` signs in by setting a default
    Authorization header on the shared client, so anything holding that same
    object is authenticated too -- which silently turned the first version of
    this fixture into a second authenticated client, and made a test that meant
    to assert 401 assert 200 instead.

    A second TestClient over the same app is safe here precisely because it is
    NOT used as a context manager: the lifespan has already been run by
    `app_client`, so this one only borrows the loaded state to send requests.
    """
    from backend.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def admin_token(app_client):
    r = app_client.post("/api/v1/auth/login",
                        json={"username": ADMIN_USER, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, f"seeded admin could not sign in: {r.text}"
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def admin(admin_token):
    """Bearer headers for the seeded administrator."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="module")
def auth_client(app_client, admin_token):
    """The same app, signed in.

    Implemented by setting a default header on the shared client rather than by
    constructing a second one -- a second TestClient would run the lifespan
    again. httpx merges per-request `headers=` over the client defaults, so the
    existing tests that pass `headers=admin` explicitly still work, and so does
    a test that overrides Authorization to assert a bad token is rejected.
    """
    app_client.headers.update({"Authorization": f"Bearer {admin_token}"})
    yield app_client
    app_client.headers.pop("Authorization", None)


@pytest.fixture(scope="module")
def case_id(auth_client):
    """A real ticket id from the loaded corpus.

    Skips rather than fails when the queue is empty: data/transactions.csv is
    gitignored and a fresh clone has not run scripts/generate_data.py yet. That
    is a missing artefact, not a broken test.
    """
    r = auth_client.get("/api/v1/complaint/list", params={"limit": 1})
    assert r.status_code == 200, r.text
    rows = r.json()
    if not rows:
        pytest.skip("no complaints loaded; run: python scripts/generate_data.py")
    return rows[0]["ticket_id"]
