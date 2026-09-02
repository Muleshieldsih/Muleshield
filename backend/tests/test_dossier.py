# -*- coding: utf-8 -*-
"""
Tests for the police intelligence dossier -- deliverable (c)'s middle clause.

Structured like test_evidence.py, because the dossier is the same kind of object:
a document generated from the store rather than typed, which must not be able to
describe a state the system is not in.

The tests that matter most here are the two agreement tests. A dossier is printed
and handed to somebody; if it disagreed with the console an officer was looking
at five seconds earlier, the divergence would be discovered in front of the
person we were trying to convince. `TestAgreesWithTheConsole` is the guard that
makes `backend/forecast.py` worth having.
"""

import io

import pytest


@pytest.fixture(scope="module")
def client(auth_client):
    """Tier A throughout, so the module acts as a signed-in officer.
    `anon` stays available for the refusal tests."""
    return auth_client


@pytest.fixture(scope="module")
def doc(client, case_id):
    r = client.get(f"/api/v1/dossier/{case_id}")
    assert r.status_code == 200, r.text
    return r.json()


# ── Access ───────────────────────────────────────────────────────────────────

class TestAuth:
    def test_anonymous_is_refused(self, anon, case_id):
        assert anon.get(f"/api/v1/dossier/{case_id}").status_code == 401

    def test_unknown_case_is_404_not_an_empty_document(self, client):
        """A typo must not produce a blank dossier with an official header on it."""
        r = client.get("/api/v1/dossier/TKT-DOES-NOT-EXIST")
        assert r.status_code == 404

    def test_signed_in_officer_gets_the_document(self, client, case_id):
        assert client.get(f"/api/v1/dossier/{case_id}").status_code == 200


# ── Content ──────────────────────────────────────────────────────────────────

class TestComplaintBlock:
    def test_it_names_the_case_it_claims_to_be_about(self, doc, case_id):
        assert doc["case_id"] == case_id
        assert doc["complaint"]["ticket_id"] == case_id

    def test_it_names_the_officer_who_produced_it(self, doc):
        """A document that forecasts a deployment has to say who asked for it."""
        assert doc["produced_by"]
        assert doc["produced_by"] != "UNKNOWN"

    def test_it_carries_the_operator_and_problem_statement(self, doc):
        assert doc["system"]["problem_statement"] == "SIH26184"
        assert "I4C" in doc["system"]["operator"]


class TestMoneyTrail:
    def test_every_hop_carries_an_ifsc(self, doc):
        """The IFSC is what a bank nodal officer acts on. A hop without one is a
        hop that cannot be actioned."""
        assert doc["money_trail"], "no money trail on a corpus case"
        assert all(r["ifsc_code"] for r in doc["money_trail"])

    def test_the_trail_is_in_time_order(self, doc):
        stamps = [r["timestamp"] for r in doc["money_trail"]]
        assert stamps == sorted(stamps)

    def test_the_count_matches_the_rows(self, doc):
        assert doc["money_trail_count"] == len(doc["money_trail"])

    def test_banks_involved_is_derived_from_the_trail(self, doc):
        assert set(doc["banks_involved"]) == {
            r["bank_name"] for r in doc["money_trail"] if r["bank_name"]
        }


class TestDetections:
    def test_the_rules_are_stated_in_words(self, doc):
        """A printed page has to say what the threshold was, not just that
        something tripped. These two strings are already written for a human."""
        det = doc["detections"]
        assert "outgoing txns" in det["velocity_rule"]
        assert "of mean" in det["fund_split_rule"]

    def test_counts_match_their_lists(self, doc):
        det = doc["detections"]
        assert det["velocity_count"] == len(det["velocity_flagged"])
        assert det["fund_split_count"] == len(det["fund_split_flagged"])
        assert det["terminal_count"] == len(det["terminal_leaves"])


class TestForecast:
    def test_the_search_zone_is_present(self, doc):
        """The zone is the deliverable; the ranked list is the drill-down."""
        fc = doc["forecast"]
        assert fc is not None
        assert fc["search_zone"] is not None
        assert fc["search_zone"]["radius_km"] > 0

    def test_it_returns_the_operating_k(self, doc):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
        from xgb_model import OPERATING_K
        assert len(doc["forecast"]["ranked_candidates"]) == OPERATING_K

    def test_candidates_are_ranked_and_descending(self, doc):
        cands = doc["forecast"]["ranked_candidates"]
        assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
        confs = [c["confidence"] for c in cands]
        assert confs == sorted(confs, reverse=True)

    def test_candidates_carry_an_address_a_bank_and_coordinates(self, doc):
        """A patrol is dispatched to a place, not to an id."""
        for c in doc["forecast"]["ranked_candidates"]:
            assert c["address"]
            assert c["bank"]
            assert isinstance(c["lat"], float) and isinstance(c["lon"], float)

    def test_the_countdown_is_a_band_not_a_point(self, doc):
        fc = doc["forecast"]
        assert fc["time_to_cashout_low"] is not None
        assert fc["time_to_cashout_high"] is not None
        assert fc["time_to_cashout_low"] < fc["time_to_cashout_high"]


# ── The agreement tests: the reason backend/forecast.py exists ───────────────

class TestAgreesWithTheConsole:
    def test_the_forecast_matches_the_prediction_endpoint_exactly(self, client, case_id, doc):
        """A printed dossier and the Interception screen must never name a
        different top ATM. Both call backend.forecast.compute; this is the guard
        that keeps it that way."""
        p = client.get(f"/api/v1/predict/cashout/{case_id}").json()
        assert [c["atm_id"] for c in doc["forecast"]["ranked_candidates"]] == \
               [c["atm_id"] for c in p["ranked_candidates"]]
        assert doc["forecast"]["search_zone"] == p["search_zone"]
        assert doc["forecast"]["terminal_account"] == p["terminal_account"]

    def test_the_money_trail_matches_the_transactions_endpoint(self, client, case_id, doc):
        """The ledger is the non-lossy source; graph edges are de-duplicated by
        src->dst pair and would silently drop a repeated transfer."""
        txns = client.get(f"/api/v1/complaint/{case_id}/transactions").json()
        assert doc["money_trail_count"] == len(txns)


# ── Caveats ──────────────────────────────────────────────────────────────────

class TestCaveats:
    def test_the_wire_advisory_appears_verbatim(self, doc):
        """The same sentence that goes to CFCFRMS goes on the printed page. If
        it is ever softened it must be softened in both places."""
        from backend.adapters.webhook import samanvaya_payload
        wire = samanvaya_payload({
            "created_at": "", "id": "", "severity": "", "headline": "",
            "state": "", "district": "", "cell_id": "",
            "window_start_min": 0, "window_end_min": 0,
            "complaint_ids": [], "rupees_at_risk": 0.0, "prior_share": 0.0,
        })["advisory"]
        assert wire in doc["caveats"]

    def test_it_denies_being_a_predicted_atm(self, doc):
        joined = " ".join(doc["caveats"]).lower()
        assert "not a predicted atm" in joined

    def test_it_states_the_countdown_is_not_a_deadline(self, doc):
        joined = " ".join(doc["caveats"]).lower()
        assert "not a deadline" in joined

    def test_it_refuses_to_be_grounds_to_act_against_a_person(self, doc):
        joined = " ".join(doc["caveats"]).lower()
        assert "never grounds alone" in joined


# ── The embedded evidence certificate ────────────────────────────────────────

class TestEvidenceEmbedded:
    def test_it_carries_the_statute(self, doc):
        assert "Bharatiya Sakshya Adhiniyam" in doc["evidence"]["statute"]
        assert "63" in doc["evidence"]["statute"]

    def test_it_carries_a_live_verification(self, doc):
        v = doc["evidence"]["verification"]
        assert set(v) >= {"content_ok", "chain_ok", "intact", "verified_at"}

    def test_a_case_with_no_artefacts_still_produces_a_dossier(self, client, case_id):
        """An empty custody chain is an honest statement, not an error. This is
        the common case: most cases have no artefacts collected yet."""
        d = client.get(f"/api/v1/dossier/{case_id}").json()
        assert d["evidence"]["artefact_count"] >= 0
        assert isinstance(d["evidence"]["artefacts"], list)

    def test_a_collected_artefact_appears_on_the_dossier(self, client, case_id):
        r = client.post(
            f"/api/v1/evidence/{case_id}",
            files={"file": ("statement.pdf", io.BytesIO(b"victim statement"),
                            "application/pdf")},
            data={"kind": "fir_copy", "description": "Victim statement",
                  "source": "Complainant"},
        )
        assert r.status_code == 201, r.text
        item = r.json()
        d = client.get(f"/api/v1/dossier/{case_id}").json()
        ids = [a["id"] for a in d["evidence"]["artefacts"]]
        assert item["id"] in ids
        assert d["evidence"]["artefact_count"] >= 1


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_two_identical_calls_agree(self, client, case_id):
        """Precedent: tests/test_ranked_candidates.py -- 'a briefing an officer
        acts on must not change between two identical calls.' Only the
        production timestamp, the measured latency, and the audit trail (which
        this very call appends to) are allowed to move."""
        a = client.get(f"/api/v1/dossier/{case_id}").json()
        b = client.get(f"/api/v1/dossier/{case_id}").json()
        for d in (a, b):
            d.pop("produced_at", None)
            d.pop("actions", None)
            d["evidence"].pop("produced_at", None)
            d["evidence"]["verification"].pop("verified_at", None)
            if d.get("forecast"):
                d["forecast"].pop("inference_time_ms", None)
        assert a == b


# ── Audit ────────────────────────────────────────────────────────────────────

class TestAuditTrail:
    def test_production_is_recorded_against_the_case(self, client, case_id):
        client.get(f"/api/v1/dossier/{case_id}")
        rows = client.get(f"/api/v1/audit/{case_id}").json()
        assert any(r["action"] == "Produced case dossier" for r in rows)

    def test_the_audit_row_names_the_officer_not_a_claim(self, client, case_id):
        client.get(f"/api/v1/dossier/{case_id}")
        rows = client.get(f"/api/v1/audit/{case_id}").json()
        row = next(r for r in rows if r["action"] == "Produced case dossier")
        assert row["actor"] and row["actor"] != "UNKNOWN"


# ── The seam ─────────────────────────────────────────────────────────────────

class TestConcurrencySeam:
    def test_the_handler_is_not_a_coroutine(self):
        """It reaches sqlite through the nested certificate and runs the full
        forecast, so it must run in the threadpool rather than on the event
        loop. Same rule as backend/routers/evidence.py; see backend/auth.py."""
        import inspect
        from backend.routers import dossier as router_mod
        assert not inspect.iscoroutinefunction(router_mod.case_dossier)

    def test_the_assembler_is_not_a_coroutine(self):
        import inspect
        from backend import dossier as mod
        assert not inspect.iscoroutinefunction(mod.dossier)
