# -*- coding: utf-8 -*-
"""
MuleShield AI -- alerting: rules, delivery, acknowledgement.
SIH26184 | MHA / I4C

    Run: python -m pytest backend/tests/test_alerts.py -v

The properties checked here are the ones that decide whether this is an
early-warning system or a dashboard with extra tables: that a rule fires without
a human present, that a hot cell cannot spam the inbox, that a delivery failure
is recorded rather than swallowed, and that a HIGH alert cannot be raised on
historical density alone.
"""

import inspect
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(ROOT), str(ROOT / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend import db, notify  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────

def _cell(cell_id="CELL-001", *, rupees=1_000.0, cases=1, prior_share=0.1,
          w=(0, 30), district="Nuh", state="Haryana", registry=0):
    return {
        "cell_id": cell_id, "district": district, "state": state,
        "lat": 28.1, "lon": 77.0, "atm_count": 6,
        "score": rupees, "conditional_rupees": rupees * (1 - prior_share),
        "prior_rupees": rupees * prior_share, "prior_share": prior_share,
        "case_count": cases, "complaint_ids": [f"C-{i}" for i in range(cases)],
        "suspect_registry_hits": registry,
        "windows": [{
            "window_start_min": w[0], "window_end_min": w[1], "score": rupees,
            "conditional_rupees": rupees * (1 - prior_share),
            "prior_rupees": rupees * prior_share,
            "prior_share": prior_share, "case_count": cases,
        }],
    }


def _background(n=20, cases=1):
    """Quiet cells to sit the hot one against.

    R-HIGH-CONVERGE is relative to the surface it reads: a cell has to carry
    CONVERGE_EXCESS times the mean case count of the live cell-windows. A test
    that hands the rule a single cell is therefore comparing that cell to itself,
    which is not a scenario the rule can ever see in production -- the real
    surface carries 222 cells.
    """
    return [_cell(f"BG-{i:03d}", rupees=1_000.0, cases=cases, district="Quiet")
            for i in range(n)]


def _surface(cells, degraded=False):
    return {"as_of": "2026-09-01T12:00:00", "degraded": degraded,
            "prior_weight": 0.15, "n_cells": len(cells),
            "n_open_complaints": sum(c["case_count"] for c in cells),
            "cells": cells}


@pytest.fixture(autouse=True)
def _clean_alert_tables(app_client):
    """Each test starts from an empty inbox.

    Depends on app_client so the schema exists and DB_PATH already points at the
    suite's throwaway store -- never the production one.
    """
    conn = db._require()
    with db._lock:
        conn.execute("DELETE FROM alert_deliveries")
        conn.execute("DELETE FROM alerts")
        conn.commit()
    db.seed_recipients([
        {"name": "I4C Desk", "role": "I4C", "channel": "email",
         "address": "i4c@example.gov.in", "scope_state": "", "scope_district": ""},
        {"name": "Haryana Cyber", "role": "LEA", "channel": "sms",
         "address": "+919000000009", "scope_state": "Haryana", "scope_district": ""},
    ])
    yield


# ── the seam ─────────────────────────────────────────────────────────────────

class TestConcurrencySeam:
    def test_no_sqlite_function_is_a_coroutine(self):
        """Every sqlite-touching function is a plain def.

        The app ships uvicorn --workers 1. An async function that blocks on
        sqlite freezes the event loop, and the /ws/feed sockets with it -- so the
        alert broadcast would be blocked by the alert write. See auth.py:46-59.
        """
        fns = [f for f in vars(notify).values()
               if callable(f) and getattr(f, "__module__", "") == "backend.notify"]
        assert fns, "no functions found in backend.notify"
        offenders = [f.__name__ for f in fns if inspect.iscoroutinefunction(f)]
        assert not offenders, f"coroutines touching sqlite: {offenders}"

    def test_notify_does_not_import_the_websocket_manager(self):
        """Broadcasting is the caller's job, on the loop side of the seam.

        Checked by walking the AST rather than searching the source text: the
        first version of this test grepped for the string and failed on the
        module docstring that EXPLAINS the rule, which is a good reminder that
        a source-text assertion tests the prose as well as the code.
        """
        import ast
        tree = ast.parse((ROOT / "backend" / "notify.py").read_text(encoding="utf-8"))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add(n.module or "")
        offenders = {m for m in imported if "websocket" in m}
        assert not offenders, f"notify imports {offenders} across the seam"

    def test_db_alerting_functions_are_all_sync(self):
        for name in ("insert_alert", "get_alert", "list_alerts",
                     "acknowledge_alert", "queue_delivery", "mark_delivery",
                     "list_deliveries", "deliveries_due", "list_recipients"):
            assert not inspect.iscoroutinefunction(getattr(db, name)), name


# ── rules ────────────────────────────────────────────────────────────────────

class TestRules:
    def test_a_degraded_surface_raises_nothing(self):
        """No live cases means the surface is pure history. Alerting on it would
        be alerting on a density map -- the exact thing this must not become."""
        out = notify.evaluate(_surface([_cell(rupees=9e9, cases=99)], degraded=True))
        assert out == []

    def test_critical_fires_on_a_large_sum_inside_the_golden_hour(self):
        out = notify.evaluate(_surface([
            _cell(rupees=notify.CRIT_RUPEES + 1, cases=1, w=(0, 30))]))
        assert any(a["severity"] == "CRITICAL" for a in out)

    def test_critical_does_not_fire_outside_the_golden_hour(self):
        """A large sum expected in 60-120 minutes is a watch item, not a
        scramble: the window is still open, so the response is different."""
        out = notify.evaluate(_surface([
            _cell(rupees=notify.CRIT_RUPEES + 1, cases=1, w=(60, 120))]))
        assert not any(a["rule_id"] == "R-CRIT-RUPEES" for a in out)

    def test_converge_needs_cases_money_and_live_evidence(self):
        """All three, against a realistic background."""
        out = notify.evaluate(_surface(
            [_cell(rupees=notify.HIGH_MIN_RUPEES + 1, cases=30, prior_share=0.05)]
            + _background()), dry_run=True)
        assert any(a["rule_id"] == "R-HIGH-CONVERGE" for a in out)

    def test_convergence_is_relative_to_the_current_load(self):
        """THE FIX FOR THE 685-ALERT PASS.

        Four open cases pointing at one cluster is remarkable on a quiet
        surface and unremarkable when every cluster is carrying twelve. The
        rule has to know which it is looking at, or it fires on the average --
        which is what it did at the national load the problem statement names:
        664 of 666 cell-windows cleared a fixed floor of three, and one pass
        raised 685 alerts.

        Identical hot cell in both surfaces. Only the background moves.
        """
        hot = _cell("CELL-HOT", rupees=notify.HIGH_MIN_RUPEES + 1,
                    cases=4, prior_share=0.05)

        quiet = notify.evaluate(_surface([hot] + _background(cases=1)), dry_run=True)
        assert any(a["rule_id"] == "R-HIGH-CONVERGE" and a["cell_id"] == "CELL-HOT"
                   for a in quiet), "four converging cases on a quiet surface is news"

        busy = notify.evaluate(_surface([hot] + _background(cases=12)), dry_run=True)
        assert not any(a["rule_id"] == "R-HIGH-CONVERGE" and a["cell_id"] == "CELL-HOT"
                       for a in busy), (
            "the same four cases is below the average of a busy surface and must "
            "not raise a HIGH alert there")

    def test_a_high_alert_must_concern_real_money(self):
        """R-HIGH-CONVERGE had no rupee floor at all; the stress pass raised a
        HIGH over a Rs 2,452 forecast."""
        out = notify.evaluate(_surface(
            [_cell(rupees=notify.HIGH_MIN_RUPEES - 1, cases=30, prior_share=0.05)]
            + _background()), dry_run=True)
        assert not any(a["rule_id"] == "R-HIGH-CONVERGE" for a in out)

    def test_a_prior_driven_cell_cannot_raise_a_high_alert(self):
        """THE ANTI-PRATIBIMB FLOOR, as policy rather than display.

        Same cell, same case count, same money -- enough of both to clear every
        other condition, so the conditional-share floor is the only thing that
        can reject it. Its score simply comes mostly from historical density.
        I4C already has a map of where fraud has happened; an alert that fires on
        that tells them nothing new and teaches officers to ignore us.
        """
        out = notify.evaluate(_surface(
            [_cell(rupees=notify.HIGH_MIN_RUPEES + 1, cases=30, prior_share=0.90)]
            + _background()), dry_run=True)
        assert not any(a["rule_id"] == "R-HIGH-CONVERGE" for a in out)

    def test_registry_hit_raises_high(self):
        out = notify.evaluate(_surface([_cell(rupees=100.0, cases=1, registry=1)]))
        assert any(a["rule_id"] == "R-HIGH-REGISTRY" for a in out)

    def test_watch_threshold_uses_the_whole_surface(self):
        """surface_p90 is computed over every cell, not the filtered view --
        the backend analogue of TriageFeed.jsx:504."""
        cells = [_cell(f"CELL-{i:03d}", rupees=float(i * 100)) for i in range(1, 21)]
        cands = notify.candidates_from_surface(_surface(cells))
        assert len({c.surface_p90 for c in cands}) == 1
        assert cands[0].surface_p90 > 0

    def test_dry_run_persists_nothing(self):
        before = len(db.list_alerts(limit=500))
        out = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]),
                              dry_run=True)
        assert out
        assert len(db.list_alerts(limit=500)) == before


# ── dedupe ───────────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_a_persistently_hot_cell_raises_once_per_hour(self):
        """Without this a district that stays dangerous for an afternoon buries
        the officer under identical rows, exactly when the inbox matters most."""
        s = _surface([_cell(rupees=notify.CRIT_RUPEES + 1)])
        when = datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc)
        first = notify.evaluate(s, as_of=when)
        second = notify.evaluate(s, as_of=when + timedelta(minutes=20))
        assert first and not second

    def test_the_next_hour_is_a_new_bucket(self):
        s = _surface([_cell(rupees=notify.CRIT_RUPEES + 1)])
        when = datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc)
        assert notify.evaluate(s, as_of=when)
        assert notify.evaluate(s, as_of=when + timedelta(hours=1))


# ── delivery ─────────────────────────────────────────────────────────────────

class TestDelivery:
    def test_an_alert_reaches_national_and_state_recipients(self):
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        assert raised
        rows = db.list_deliveries(raised[0]["id"])
        assert rows, "no delivery attempted"
        assert {r["state"] for r in rows} == {"sent"}
        assert all(r["provider_ref"] for r in rows)

    def test_a_failing_gateway_is_recorded_not_swallowed(self, monkeypatch):
        """COMPLIANCE_AUDIT.md finding 5.5: the old simulated SMS reported
        success unconditionally. A control that cannot fail is untested."""
        monkeypatch.setenv("MULESHIELD_ADAPTER_FAIL_RATE", "1.0")
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        assert raised
        rows = db.list_deliveries(raised[0]["id"])
        assert rows and all(r["state"] == "failed" for r in rows)
        assert all(r["last_error"] for r in rows)
        assert all(r["next_retry_at"] for r in rows), "no backoff scheduled"

    def test_repeated_failure_dead_letters(self, monkeypatch):
        monkeypatch.setenv("MULESHIELD_ADAPTER_FAIL_RATE", "1.0")
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        for _ in range(notify.MAX_ATTEMPTS + 1):
            notify.retry_due(datetime.now(timezone.utc) + timedelta(days=1))
        rows = db.list_deliveries(raised[0]["id"])
        assert all(r["state"] == "dead" for r in rows)
        assert all(r["attempts"] >= notify.MAX_ATTEMPTS for r in rows)

    def test_webhook_payloads_are_labelled_as_proposed(self):
        """The README once claimed schemas it did not have. These say what they
        are: our mapping, not a published contract."""
        from backend.adapters.webhook import cfcfrms_payload, samanvaya_payload
        a = {"id": "ALT-1", "severity": "CRITICAL", "prior_share": 0.1,
             "rupees_at_risk": 1e6, "complaint_ids": ["C-1"], "case_count": 2}
        assert "proposed" in cfcfrms_payload(a)["schema"]
        assert "proposed" in samanvaya_payload(a)["schema"]
        assert cfcfrms_payload(a)["confidence"]["live_share"] == pytest.approx(0.9)


# ── acknowledgement ──────────────────────────────────────────────────────────

class TestAcknowledgement:
    def test_ack_records_actor_and_disposition(self):
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        row = notify.acknowledge(raised[0]["id"], "Duty Officer", "False positive")
        assert row["status"] == "acknowledged"
        assert row["acknowledged_by"] == "Duty Officer"
        assert row["disposition"] == "False positive"

    def test_false_positive_rate_is_reported(self):
        """The outcome capture finding 2.5 asked for. A framework that cannot
        tell whether its alerts were worth raising cannot improve."""
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        notify.acknowledge(raised[0]["id"], "Officer", "False positive")
        s = notify.summary()
        assert s["actioned"] >= 1
        assert s["false_positive_rate"] > 0


# ── the API ──────────────────────────────────────────────────────────────────

class TestAlertEndpoints:
    def test_inbox_requires_authentication(self, anon):
        assert anon.get("/api/v1/alerts").status_code == 401

    def test_evaluate_requires_an_administrator(self, anon):
        assert anon.post("/api/v1/alerts/evaluate").status_code == 401

    def test_inbox_lists_raised_alerts(self, auth_client):
        notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        r = auth_client.get("/api/v1/alerts")
        assert r.status_code == 200
        body = r.json()
        assert body and body[0]["severity"] == "CRITICAL"
        assert "prior_share" in body[0]

    def test_detail_carries_the_delivery_attempts(self, auth_client):
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        r = auth_client.get(f"/api/v1/alerts/{raised[0]['id']}")
        assert r.status_code == 200
        assert r.json()["deliveries"], "detail hides the delivery record"

    def test_unknown_alert_is_404(self, auth_client):
        assert auth_client.get("/api/v1/alerts/ALT-999999").status_code == 404

    def test_ack_requires_a_valid_disposition(self, auth_client):
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        aid = raised[0]["id"]
        bad = auth_client.post(f"/api/v1/alerts/{aid}/ack",
                               json={"disposition": "whatever"})
        assert bad.status_code == 422
        ok = auth_client.post(f"/api/v1/alerts/{aid}/ack",
                              json={"disposition": "Dispatched"})
        assert ok.status_code == 200
        assert ok.json()["disposition"] == "Dispatched"

    def test_ack_is_written_to_the_audit_trail(self, auth_client):
        raised = notify.evaluate(_surface([_cell(rupees=notify.CRIT_RUPEES + 1)]))
        aid = raised[0]["id"]
        auth_client.post(f"/api/v1/alerts/{aid}/ack",
                         json={"disposition": "Monitoring"})
        trail = auth_client.get("/api/v1/audit", params={"limit": 50}).json()
        assert any("Acknowledged alert" in e["action"] and aid in e["object"]
                   for e in trail)

    def test_recipients_are_scoped(self, auth_client):
        """A state query returns that state's force AND the national desk.

        Asserted as a property rather than by name: seed_recipients() only
        populates an empty table, so whichever roster loaded first wins, and a
        test pinned to one name is really testing which fixture ran.

        Both tiers must appear because the problem statement's unit of action is
        "LEAs at the state and local levels, coordinated by I4C" -- if a state
        query dropped the national desk, I4C would stop being coordinated.
        """
        r = auth_client.get("/api/v1/alerts/recipients", params={"state": "Haryana"})
        assert r.status_code == 200
        rows = r.json()
        assert rows, "no recipient in scope for Haryana"
        assert any(x["scope_state"] == "" for x in rows), "national desk dropped"
        assert any(x["scope_state"] == "Haryana" for x in rows), "state force dropped"

        # And a different state must not receive Haryana's units.
        other = auth_client.get("/api/v1/alerts/recipients",
                                params={"state": "Kerala"}).json()
        assert not any(x["scope_state"] == "Haryana" for x in other)

    def test_evaluate_reports_degraded_separately_from_empty(self, auth_client):
        """Zero raised because nothing qualified, and zero raised because there
        were no live cases, are different facts."""
        r = auth_client.post("/api/v1/alerts/evaluate",
                             params={"as_of": "2019-01-01T00:00:00", "dry_run": True})
        assert r.status_code == 200
        assert r.json()["degraded"] is True
