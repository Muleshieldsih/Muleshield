# -*- coding: utf-8 -*-
"""
Evidence documentation -- deliverable (c)'s last open clause.

    python -m pytest backend/tests/test_evidence.py -v

COMPLIANCE_AUDIT.md finding 4.5 recorded that the problem statement names
"evidence documentation" and the repository had no upload path, no store and no
accounting of any kind. These tests exist to make sure what replaced that is
evidence rather than an attachment: a hash taken at collection and re-checked on
every read, a chain that breaks visibly if the set is altered, no delete, and a
BSA 2023 s.63 certificate generated from the store rather than typed.

The adversarial tests are the point. Anyone can assert that an upload returns
201; TestTampering below edits the bytes on disk and rewrites a row in sqlite
behind the module's back, because a custody system that only works when nobody
touches it is not a custody system.
"""

import hashlib
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT), str(ROOT / "engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend import db, evidence  # noqa: E402


PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n%%EOF\n"


def _upload(client, case_id, content=PDF, filename="statement.pdf",
            kind="bank_statement", description="", source="", headers=None):
    return client.post(
        f"/api/v1/evidence/{case_id}",
        files={"file": (filename, io.BytesIO(content), "application/pdf")},
        data={"kind": kind, "description": description, "source": source},
        **({"headers": headers} if headers else {}),
    )


@pytest.fixture(scope="module")
def client(auth_client):
    """Every endpoint here is Tier A, so the module acts as a signed-in officer.
    `anon` remains available for the tests that assert the refusal."""
    return auth_client


@pytest.fixture(autouse=True)
def _clean_evidence(app_client):
    """Each test starts from an empty store.

    Depends on app_client so the schema exists and DB_PATH already points at the
    suite's throwaway store -- never the production one.
    """
    conn = db._require()
    with db._lock:
        conn.execute("DELETE FROM case_evidence")
        conn.commit()
    yield


class TestAuth:
    """TIER A. Evidence is victim material, bank material, and what a
    prosecution rests on."""

    def test_collect_requires_authentication(self, anon, case_id):
        r = _upload(anon, case_id)
        assert r.status_code == 401

    def test_listing_requires_authentication(self, anon, case_id):
        assert anon.get(f"/api/v1/evidence/case/{case_id}").status_code == 401

    def test_certificate_requires_authentication(self, anon, case_id):
        assert anon.get(
            f"/api/v1/evidence/case/{case_id}/certificate").status_code == 401

    def test_download_requires_authentication(self, anon, client, case_id):
        item = _upload(client, case_id).json()
        assert anon.get(
            f"/api/v1/evidence/{item['id']}/download").status_code == 401


class TestCollection:
    def test_an_artefact_is_taken_into_custody(self, client, case_id):
        r = _upload(client, case_id, description="SBI statement, Aug 2026",
                    source="State Bank of India")
        assert r.status_code == 201, r.text
        item = r.json()
        assert item["id"].startswith("EVD-")
        assert item["case_id"] == case_id
        assert item["seq"] == 1
        assert item["prev_hash"] == ""
        assert item["entry_hash"]
        assert item["size_bytes"] == len(PDF)
        assert item["kind"] == "bank_statement"
        assert item["description"] == "SBI statement, Aug 2026"

    def test_the_hash_is_of_the_bytes_that_arrived(self, client, case_id):
        item = _upload(client, case_id).json()
        assert item["sha256"] == hashlib.sha256(PDF).hexdigest()

    def test_the_collecting_officer_comes_from_the_token(self, client, case_id):
        """Not from the body. Who collected a piece of evidence is the one field
        a custody record cannot let the caller choose."""
        item = _upload(client, case_id).json()
        assert item["collected_by"]
        assert "testofficer" in item["collected_by"].lower()

    def test_an_unknown_case_is_refused(self, client):
        r = _upload(client, "not-a-real-ticket")
        assert r.status_code == 404

    def test_an_empty_artefact_is_refused(self, client, case_id):
        r = _upload(client, case_id, content=b"")
        assert r.status_code == 422

    def test_an_unknown_kind_is_refused(self, client, case_id):
        r = _upload(client, case_id, kind="nonsense")
        assert r.status_code == 422
        assert "kind must be one of" in r.text

    def test_every_offered_kind_is_accepted(self, client, case_id):
        for i, kind in enumerate(evidence.KINDS):
            r = _upload(client, case_id, content=b"artefact-%d" % i, kind=kind)
            assert r.status_code == 201, f"{kind}: {r.text}"

    def test_an_oversize_artefact_is_refused(self, client, case_id, monkeypatch):
        monkeypatch.setattr(evidence, "MAX_BYTES", 64)
        r = _upload(client, case_id, content=b"x" * 4096)
        assert r.status_code == 413

    def test_a_supplied_filename_never_becomes_a_path(self, client, case_id):
        """Directory traversal comes off the name, and the name is not the path
        anyway -- the stored file is named by the artefact id."""
        item = _upload(client, case_id,
                       filename="../../../../etc/passwd").json()
        assert "/" not in item["filename"] and "\\" not in item["filename"]
        assert ".." not in item["filename"]
        stored = evidence.case_dir(case_id) / item["id"]
        assert stored.exists(), "bytes are stored under the artefact id"

    def test_the_same_bytes_twice_is_one_custody_record(self, client, case_id):
        """One object, one record. A second row for the same artefact would make
        the chain describe a history that did not happen."""
        first = _upload(client, case_id).json()
        second = _upload(client, case_id).json()
        assert first["id"] == second["id"]
        assert len(db.list_evidence(case_id)) == 1

    def test_different_bytes_are_different_artefacts(self, client, case_id):
        a = _upload(client, case_id, content=b"one").json()
        b = _upload(client, case_id, content=b"two").json()
        assert a["id"] != b["id"]
        assert b["seq"] == a["seq"] + 1


class TestChain:
    def test_the_chain_links_each_artefact_to_the_last(self, client, case_id):
        items = [_upload(client, case_id, content=b"artefact-%d" % i).json()
                 for i in range(3)]
        assert [i["seq"] for i in items] == [1, 2, 3]
        assert items[0]["prev_hash"] == ""
        assert items[1]["prev_hash"] == items[0]["entry_hash"]
        assert items[2]["prev_hash"] == items[1]["entry_hash"]

    def test_a_clean_case_verifies_intact(self, client, case_id):
        for i in range(3):
            _upload(client, case_id, content=b"artefact-%d" % i)
        v = client.get(f"/api/v1/evidence/case/{case_id}/verify").json()
        assert v["items"] == 3
        assert v["content_ok"] is True
        assert v["chain_ok"] is True
        assert v["intact"] is True

    def test_an_empty_case_verifies_trivially(self, client, case_id):
        v = client.get(f"/api/v1/evidence/case/{case_id}/verify").json()
        assert v["items"] == 0 and v["intact"] is True


class TestTampering:
    """The tests that decide whether any of this is worth anything.

    Each one reaches past the module -- editing bytes on disk, rewriting a row
    in sqlite -- because that is what tampering looks like. A custody system
    that only holds when nobody touches it is not a custody system.
    """

    def test_edited_bytes_fail_verification(self, client, case_id):
        item = _upload(client, case_id).json()
        path = evidence.case_dir(case_id) / item["id"]
        path.write_bytes(PDF + b"\n% appended by somebody\n")

        v = client.get(f"/api/v1/evidence/case/{case_id}/verify").json()
        assert v["content_ok"] is False
        assert v["intact"] is False
        assert v["content"][0]["reason"] == "content hash mismatch"

    def test_edited_bytes_are_not_served(self, client, case_id):
        """409, and the artefact stays in the record. Handing back altered bytes
        would let them be produced in court as though nothing had happened."""
        item = _upload(client, case_id).json()
        (evidence.case_dir(case_id) / item["id"]).write_bytes(b"different")

        r = client.get(f"/api/v1/evidence/{item['id']}/download")
        assert r.status_code == 409
        assert "integrity" in r.text.lower()

    def test_missing_bytes_fail_verification(self, client, case_id):
        item = _upload(client, case_id).json()
        (evidence.case_dir(case_id) / item["id"]).unlink()
        v = client.get(f"/api/v1/evidence/case/{case_id}/verify").json()
        assert v["content_ok"] is False
        assert "missing" in v["content"][0]["reason"]

    def test_a_rewritten_row_breaks_the_chain(self, client, case_id):
        """Swap one artefact's recorded hash in sqlite, behind the module's
        back. Content verification catches the bytes; the CHAIN catches the row,
        and every link after it."""
        items = [_upload(client, case_id, content=b"artefact-%d" % i).json()
                 for i in range(3)]
        conn = db._require()
        with db._lock:
            conn.execute("UPDATE case_evidence SET sha256 = ? WHERE id = ?",
                         ("0" * 64, items[1]["id"]))
            conn.commit()

        v = client.get(f"/api/v1/evidence/case/{case_id}/verify").json()
        assert v["chain_ok"] is False
        broken = [c for c in v["chain"] if not c["ok"]]
        assert items[1]["id"] in [c["id"] for c in broken]
        assert items[2]["id"] in [c["id"] for c in broken], (
            "a rewritten row must break every link after it, not only its own"
        )

    def test_a_removed_row_breaks_the_chain(self, client, case_id):
        items = [_upload(client, case_id, content=b"artefact-%d" % i).json()
                 for i in range(3)]
        conn = db._require()
        with db._lock:
            conn.execute("DELETE FROM case_evidence WHERE id = ?", (items[1]["id"],))
            conn.commit()

        v = client.get(f"/api/v1/evidence/case/{case_id}/verify").json()
        assert v["chain_ok"] is False, "removing an artefact must be visible"


class TestWithdrawal:
    def test_withdrawal_needs_a_reason(self, client, case_id):
        item = _upload(client, case_id).json()
        r = client.post(f"/api/v1/evidence/{item['id']}/withdraw",
                        json={"reason": "no"})
        assert r.status_code == 422

    def test_withdrawal_records_who_and_why(self, client, case_id):
        item = _upload(client, case_id).json()
        r = client.post(f"/api/v1/evidence/{item['id']}/withdraw",
                        json={"reason": "duplicate of EVD-000001"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["withdrawn"] is True
        assert out["withdrawn_by"]
        assert out["withdrawn_reason"] == "duplicate of EVD-000001"

    def test_a_withdrawn_artefact_is_not_deleted(self, client, case_id):
        """THE POINT OF THE WHOLE MODULE. Evidence that can be deleted is
        evidence that can be made to disappear between collection and trial."""
        item = _upload(client, case_id).json()
        client.post(f"/api/v1/evidence/{item['id']}/withdraw",
                    json={"reason": "collected in error"})

        still = client.get(f"/api/v1/evidence/{item['id']}").json()
        assert still["id"] == item["id"]
        assert (evidence.case_dir(case_id) / item["id"]).exists(), (
            "the bytes must survive withdrawal"
        )
        listed = client.get(f"/api/v1/evidence/case/{case_id}").json()
        assert [i["id"] for i in listed] == [item["id"]]

    def test_withdrawn_items_can_be_filtered_out_of_the_list(self, client, case_id):
        item = _upload(client, case_id).json()
        client.post(f"/api/v1/evidence/{item['id']}/withdraw",
                    json={"reason": "collected in error"})
        live = client.get(f"/api/v1/evidence/case/{case_id}",
                          params={"include_withdrawn": False}).json()
        assert live == []

    def test_withdrawal_does_not_break_the_chain(self, client, case_id):
        """Withdrawal is deliberately outside the hash: a custody record is
        fixed at collection, and a later withdrawal is a separate event."""
        items = [_upload(client, case_id, content=b"artefact-%d" % i).json()
                 for i in range(3)]
        client.post(f"/api/v1/evidence/{items[1]['id']}/withdraw",
                    json={"reason": "not relevant to this case"})
        v = client.get(f"/api/v1/evidence/case/{case_id}/verify").json()
        assert v["chain_ok"] is True and v["content_ok"] is True

    def test_withdrawing_twice_is_refused(self, client, case_id):
        item = _upload(client, case_id).json()
        body = {"reason": "collected in error"}
        assert client.post(f"/api/v1/evidence/{item['id']}/withdraw",
                           json=body).status_code == 200
        assert client.post(f"/api/v1/evidence/{item['id']}/withdraw",
                           json=body).status_code == 409

    def test_there_is_no_delete_endpoint(self, client, case_id):
        item = _upload(client, case_id).json()
        r = client.delete(f"/api/v1/evidence/{item['id']}")
        assert r.status_code in (404, 405), (
            "an evidence store must not offer a delete"
        )


class TestDownload:
    def test_the_bytes_come_back_unchanged(self, client, case_id):
        item = _upload(client, case_id).json()
        r = client.get(f"/api/v1/evidence/{item['id']}/download")
        assert r.status_code == 200
        assert r.content == PDF

    def test_the_response_carries_the_recorded_hash(self, client, case_id):
        item = _upload(client, case_id).json()
        r = client.get(f"/api/v1/evidence/{item['id']}/download")
        assert r.headers["x-evidence-sha256"] == hashlib.sha256(PDF).hexdigest()

    def test_content_type_is_never_the_one_the_uploader_declared(self, client, case_id):
        """An uploaded .html or .svg served back under its declared type, on the
        same origin as the console, is stored XSS against the next officer who
        opens the case."""
        r = client.post(
            f"/api/v1/evidence/{case_id}",
            files={"file": ("payload.html", io.BytesIO(b"<script>alert(1)</script>"),
                            "text/html")},
            data={"kind": "other"})
        assert r.status_code == 201
        item = r.json()
        assert item["content_type"] == "text/html", "the declaration is recorded"

        d = client.get(f"/api/v1/evidence/{item['id']}/download")
        assert d.headers["content-type"] == "application/octet-stream"
        assert d.headers["x-content-type-options"] == "nosniff"
        assert "attachment" in d.headers.get("content-disposition", "")

    def test_an_unknown_artefact_is_a_404(self, client):
        assert client.get("/api/v1/evidence/EVD-999999").status_code == 404
        assert client.get("/api/v1/evidence/EVD-999999/download").status_code == 404


class TestCertificate:
    def test_it_cites_the_statute_in_force(self, client, case_id):
        _upload(client, case_id)
        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        assert "Bharatiya Sakshya Adhiniyam" in c["statute"]
        assert "63" in c["statute"]
        assert "65B" in c["statute_note"], (
            "a reader who knows the old section must be able to find their way"
        )

    def test_it_lists_every_artefact_with_its_hash(self, client, case_id):
        items = [_upload(client, case_id, content=b"artefact-%d" % i).json()
                 for i in range(3)]
        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        assert c["artefact_count"] == 3
        assert [a["id"] for a in c["artefacts"]] == [i["id"] for i in items]
        for a, i in zip(c["artefacts"], items):
            assert a["sha256"] == i["sha256"]
            assert a["entry_hash"] == i["entry_hash"]

    def test_it_carries_a_live_verification(self, client, case_id):
        _upload(client, case_id)
        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        assert c["verification"]["intact"] is True
        assert all(a["reverified_ok"] for a in c["artefacts"])

    def test_it_reports_a_failure_rather_than_printing_anyway(self, client, case_id):
        """A certificate that could only ever say 'intact' would be decoration."""
        item = _upload(client, case_id).json()
        (evidence.case_dir(case_id) / item["id"]).write_bytes(b"tampered")

        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        assert c["verification"]["intact"] is False
        assert c["artefacts"][0]["reverified_ok"] is False

    def test_the_custodian_is_the_officer_producing_it(self, client, case_id):
        _upload(client, case_id)
        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        assert "testofficer" in c["custodian"].lower()

    def test_it_states_its_own_limits(self, client, case_id):
        """Including that the chain is not externally anchored. A document that
        overclaims is worse than none."""
        _upload(client, case_id)
        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        blob = " ".join(c["limitations"]).lower()
        assert "anchor" in blob
        assert "simulated" in blob
        assert "not a legal opinion" in blob

    def test_withdrawn_artefacts_are_counted_separately(self, client, case_id):
        a = _upload(client, case_id, content=b"one").json()
        _upload(client, case_id, content=b"two")
        client.post(f"/api/v1/evidence/{a['id']}/withdraw",
                    json={"reason": "collected in error"})
        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        assert c["artefact_count"] == 1
        assert c["withdrawn_count"] == 1
        assert len(c["artefacts"]) == 2, "withdrawn items stay on the record"

    def test_it_carries_the_case_it_describes(self, client, case_id):
        _upload(client, case_id)
        c = client.get(f"/api/v1/evidence/case/{case_id}/certificate").json()
        assert c["case_id"] == case_id
        assert c["complaint"] is not None
        assert c["complaint"]["victim_bank"]

    def test_an_unknown_case_has_no_certificate(self, client):
        assert client.get(
            "/api/v1/evidence/case/nope/certificate").status_code == 404


class TestAuditTrail:
    def test_collection_download_and_withdrawal_are_all_recorded(self, client, case_id):
        item = _upload(client, case_id).json()
        client.get(f"/api/v1/evidence/{item['id']}/download")
        client.post(f"/api/v1/evidence/{item['id']}/withdraw",
                    json={"reason": "collected in error"})

        trail = client.get("/api/v1/audit", params={"limit": 200}).json()
        actions = " | ".join(e["action"] for e in trail)
        for expected in ("Collected evidence", "Downloaded evidence",
                         "Withdrew evidence"):
            assert expected in actions, f"{expected!r} left no audit entry"

    def test_the_entries_name_the_case(self, client, case_id):
        _upload(client, case_id)
        trail = client.get("/api/v1/audit", params={"case_id": case_id,
                                                    "limit": 50}).json()
        assert any(e["action"] == "Collected evidence" for e in trail)


class TestConcurrencySeam:
    def test_no_sqlite_function_is_a_coroutine(self):
        """The same rule backend/notify.py is held to: anything that touches
        sqlite is a plain def, so it can only ever run in a threadpool."""
        import inspect
        from backend import evidence as mod
        offenders = [name for name, fn in vars(mod).items()
                     if inspect.iscoroutinefunction(fn)]
        assert offenders == [], (
            f"{offenders} are coroutines in a module that touches sqlite and the "
            f"filesystem; they would block the event loop"
        )

    def test_the_store_is_isolated_from_the_real_one(self):
        """The suite must never write an artefact into data/evidence/."""
        assert "muleshield-test-" in str(evidence.EVIDENCE_DIR), (
            f"tests are writing to {evidence.EVIDENCE_DIR}"
        )
