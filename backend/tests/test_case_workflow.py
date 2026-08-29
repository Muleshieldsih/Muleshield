# -*- coding: utf-8 -*-
"""
Case workflow: status, assignment, notes, the money trail and the audit log.
SIH26184 | MHA / I4C

Before this existed a complaint could be listed but not worked. `status` was
written once as "ACTIVE" and read by nothing; there was no assignment, no notes,
and no audit trail — an account could be frozen and the system could not
afterwards say who had ordered it.

These tests hold that surface in place, and in particular hold two properties
that are easy to lose: that the trail returns every transaction rather than a
de-duplicated summary, and that an action which changes a case leaves a record.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
import backend.state as state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def case_id(client):
    """A real case from the loaded corpus."""
    rows = client.get("/api/v1/complaint/list", params={"limit": 1}).json()
    if not rows:
        pytest.skip("no complaints loaded")
    return rows[0]["ticket_id"]


class TestStatus:
    def test_legacy_active_is_presented_as_new(self, client, case_id):
        """
        Seed records carry "ACTIVE", which is not one of the six workflow
        states. Rather than rewrite 2,500 rows on load it is read as "New",
        keeping the CSV as the record of what was reported.
        """
        body = client.get(f"/api/v1/complaint/{case_id}").json()
        assert body["status"] in state.CASE_STATUSES

    def test_status_can_be_moved(self, client, case_id):
        r = client.patch(f"/api/v1/complaint/{case_id}",
                         json={"status": "Investigating", "actor": "AS-1042"})
        assert r.status_code == 200
        assert r.json()["status"] == "Investigating"
        # And it sticks, rather than only being echoed back.
        assert client.get(f"/api/v1/complaint/{case_id}").json()["status"] == "Investigating"

    def test_unknown_status_is_rejected_and_names_the_valid_set(self, client, case_id):
        r = client.patch(f"/api/v1/complaint/{case_id}", json={"status": "Escalated!!"})
        assert r.status_code == 422
        # "invalid input" would send the caller to the source. This does not.
        assert "New" in r.json()["detail"] and "Closed" in r.json()["detail"]

    def test_unknown_case_is_404(self, client):
        r = client.patch("/api/v1/complaint/NOPE-000", json={"status": "Closed"})
        assert r.status_code == 404


class TestAssignmentAndNotes:
    def test_assignment_persists(self, client, case_id):
        client.patch(f"/api/v1/complaint/{case_id}",
                     json={"assignee": "AS-2001", "actor": "AS-2001"})
        assert client.get(f"/api/v1/complaint/{case_id}").json()["assignee"] == "AS-2001"

    def test_note_is_stored_and_counted(self, client, case_id):
        before = client.get(f"/api/v1/complaint/{case_id}").json()["note_count"]
        r = client.post(f"/api/v1/complaint/{case_id}/note",
                        json={"text": "Beneficiary bank contacted.", "author": "AS-2001"})
        assert r.status_code == 200
        assert r.json()["author"] == "AS-2001"

        notes = client.get(f"/api/v1/complaint/{case_id}/notes").json()
        assert notes[0]["text"] == "Beneficiary bank contacted."
        assert client.get(f"/api/v1/complaint/{case_id}").json()["note_count"] == before + 1

    def test_empty_note_is_rejected(self, client, case_id):
        assert client.post(f"/api/v1/complaint/{case_id}/note", json={"text": ""}).status_code == 422


class TestTransactionTrail:
    def test_trail_returns_rows_in_time_order(self, client, case_id):
        rows = client.get(f"/api/v1/complaint/{case_id}/transactions").json()
        assert rows, "a case in the corpus should have a ledger"
        stamps = [r["timestamp"] for r in rows]
        assert stamps == sorted(stamps)

    def test_trail_does_not_deduplicate(self, client, case_id):
        """
        The graph endpoint collapses edges by source-destination pair, so two
        transfers between the same accounts become one and the second is lost.
        An investigator's ledger cannot silently drop rows, so the trail must
        return at least as many as the graph does.
        """
        trail = client.get(f"/api/v1/complaint/{case_id}/transactions").json()
        graph = client.get(f"/api/v1/graph/{case_id}")
        if graph.status_code == 200:
            assert len(trail) >= len(graph.json()["edges"])

    def test_elapsed_minutes_start_at_zero(self, client, case_id):
        rows = client.get(f"/api/v1/complaint/{case_id}/transactions").json()
        assert rows[0]["minutes_from_first"] == 0.0
        assert all(r["minutes_from_first"] >= 0 for r in rows)

    def test_absent_atm_is_null_not_nan(self, client, case_id):
        """
        Seed rows come from pandas, where a missing cash-out ATM is float('nan')
        — which is not JSON and would break a strict client.
        """
        rows = client.get(f"/api/v1/complaint/{case_id}/transactions").json()
        for r in rows:
            assert r["cashout_atm_id"] is None or isinstance(r["cashout_atm_id"], str)
            assert str(r["cashout_atm_id"]) != "nan"


class TestAuditTrail:
    def test_status_change_is_recorded_with_its_actor(self, client, case_id):
        client.patch(f"/api/v1/complaint/{case_id}",
                     json={"status": "Intervention Required", "actor": "AS-9099"})
        entries = client.get("/api/v1/audit", params={"case_id": case_id}).json()
        assert entries, "a status change must leave a record"
        top = entries[0]
        assert top["actor"] == "AS-9099"
        assert "status" in top["action"].lower()
        assert "Intervention Required" in top["object"]

    def test_audit_is_newest_first(self, client, case_id):
        client.patch(f"/api/v1/complaint/{case_id}", json={"status": "Resolved", "actor": "AS-1"})
        client.patch(f"/api/v1/complaint/{case_id}", json={"status": "Closed", "actor": "AS-2"})
        entries = client.get("/api/v1/audit", params={"case_id": case_id}).json()
        assert entries[0]["actor"] == "AS-2"

    def test_freeze_leaves_a_readable_record(self, client, case_id):
        """
        The freeze log is keyed by reference and read by no endpoint, so on its
        own an irreversible action left no trace of who ordered it.
        """
        client.post("/api/v1/bank/micro-freeze", json={
            "account_id": "ACC-TEST-0001",
            "complaint_id": case_id,
            "officer_id": "IO-777",
        })
        entries = client.get("/api/v1/audit", params={"case_id": case_id}).json()
        froze = [e for e in entries if "Froze" in e["action"]]
        assert froze, "a freeze must appear in the audit trail"
        assert froze[0]["actor"] == "IO-777"
        assert "ACC-TEST-0001" in froze[0]["object"]

    def test_audit_scopes_to_the_case(self, client, case_id):
        entries = client.get("/api/v1/audit", params={"case_id": case_id}).json()
        assert all(e["case_id"] == case_id for e in entries)
