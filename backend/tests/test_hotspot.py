# -*- coding: utf-8 -*-
"""
MuleShield AI -- forward hotspot surface.
SIH26184 | MHA / I4C

    Run: python -m pytest backend/tests/test_hotspot.py -v

Two kinds of test live here. The pure ones exercise engine/hotspot.py directly
and need no server; the API ones drive the router. The pure ones matter most:
the properties they check -- the prior cap, the survival renormalisation, the
declared degradation -- are the claims this component is judged on, and each of
them is a way the component could quietly stop being a forecast and start being
a density map.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(ROOT), str(ROOT / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hotspot as H  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def toy_cells():
    """Three ATMs, far enough apart to land in three separate cells."""
    atms = [
        {"atm_id": "A1", "lat": 19.0760, "long": 72.8777, "district": "Mumbai",
         "state": "Maharashtra", "city": "Mumbai", "historical_fraud_count": 9},
        {"atm_id": "B1", "lat": 28.6139, "long": 77.2090, "district": "Delhi",
         "state": "Delhi", "city": "Delhi", "historical_fraud_count": 4},
        {"atm_id": "C1", "lat": 12.9716, "long": 77.5946, "district": "Bengaluru",
         "state": "Karnataka", "city": "Bengaluru", "historical_fraud_count": 1},
    ]
    return H.build_cells(atms)


def _complaint(cell_id, *, amount=100000.0, median=40.0, age_min=0.0,
               as_of=None, cid="C-1"):
    return {
        "complaint_id": cid,
        "cell_probs": {cell_id: 1.0},
        "m": median, "lo": median * 0.5, "hi": median * 2.0,
        "amount": amount,
        "ts": (as_of or datetime(2026, 1, 1, 12, 0)) - timedelta(minutes=age_min),
        "fraud_type": "UPI Fraud",
    }


# ── clustering ───────────────────────────────────────────────────────────────

class TestCells:
    def test_distant_atms_land_in_separate_cells(self, toy_cells):
        assert len(toy_cells) == 3

    def test_every_atm_is_indexed_exactly_once(self, toy_cells):
        idx = H.atm_to_cell(toy_cells)
        assert sorted(idx) == ["A1", "B1", "C1"]
        assert len(set(idx.values())) == 3

    def test_nearby_atms_share_a_cell(self):
        """Two machines 1 km apart must not be two separate dispatch targets."""
        atms = [
            {"atm_id": "X", "lat": 19.0760, "long": 72.8777, "district": "Mumbai",
             "state": "Maharashtra", "city": "Mumbai", "historical_fraud_count": 5},
            {"atm_id": "Y", "lat": 19.0850, "long": 72.8800, "district": "Mumbai",
             "state": "Maharashtra", "city": "Mumbai", "historical_fraud_count": 1},
        ]
        cells = H.build_cells(atms)
        assert len(cells) == 1
        assert cells["CELL-001"]["atm_count"] == 2

    def test_cell_ids_are_stable_across_runs(self):
        """Alerts persist cell ids into SQLite. If clustering were
        order-dependent, an acknowledged alert would silently re-point at a
        different place after a restart."""
        atms = [
            {"atm_id": f"A{i}", "lat": 19.0 + i * 0.5, "long": 72.8,
             "district": "D", "state": "S", "city": "C",
             "historical_fraud_count": i}
            for i in range(6)
        ]
        first = H.build_cells(atms)
        second = H.build_cells(list(reversed(atms)))
        assert {c: v["atm_ids"] for c, v in first.items()} == \
               {c: v["atm_ids"] for c, v in second.items()}

    def test_rollup_keys_are_present(self, toy_cells):
        """District and state ride on every cell so drill-down is a group-by."""
        for c in toy_cells.values():
            assert c["district"] and c["state"]


# ── temporal kernel ──────────────────────────────────────────────────────────

class TestSurvivalKernel:
    def test_mass_is_a_probability(self):
        for age in (0, 5, 40, 200):
            m = H.window_mass(40, 20, 80, age, 0, 30)
            assert 0.0 <= m <= 1.0

    def test_windows_partition_the_horizon(self):
        total = sum(H.window_mass(40, 20, 80, 10, a, b) for a, b in H.WINDOWS)
        assert total <= 1.0 + 1e-9

    def test_an_overdue_case_concentrates_in_the_nearest_window(self):
        """The survival renormalisation is what makes this a forecast.

        A case already past its median has NOT been cashed out -- that is what
        being open means -- so the remaining mass has to pile into the imminent
        window. A plain (unconditioned) lognormal would instead say the moment
        has passed and the risk is falling, which is exactly backwards.
        """
        fresh = H.window_mass(40, 20, 80, 0, 0, 30)
        overdue = H.window_mass(40, 20, 80, 60, 0, 30)
        assert overdue > fresh

    def test_a_missing_band_does_not_produce_a_point_mass(self):
        """Without the quantile models the countdown must stay uncertain rather
        than claiming a precision the regressor (R^2 0.17) has never shown."""
        spread = H.window_mass(40, None, None, 0, 0, 30)
        assert 0.0 < spread < 1.0


# ── the surface, and the cap ─────────────────────────────────────────────────

class TestSurface:
    def test_conditional_mass_dominates_when_cases_are_open(self, toy_cells):
        prior = {"CELL-001": 0.7, "CELL-002": 0.2, "CELL-003": 0.1}
        now = datetime(2026, 1, 1, 12, 0)
        s = H.build_hotspot_surface([_complaint("CELL-002", as_of=now)],
                                    toy_cells, prior, now)
        assert not s["degraded"]
        assert s["prior_share_national"] <= H.PRIOR_WEIGHT + 1e-9

    def test_prior_never_exceeds_its_cap(self, toy_cells):
        """The anti-Pratibimb guarantee, checked against a deliberately
        overwhelming prior: even when history says one cell is everything, it
        cannot carry more than PRIOR_WEIGHT of the surface."""
        prior = {"CELL-001": 0.99, "CELL-002": 0.005, "CELL-003": 0.005}
        now = datetime(2026, 1, 1, 12, 0)
        s = H.build_hotspot_surface([_complaint("CELL-003", as_of=now)],
                                    toy_cells, prior, now)
        assert s["prior_share_national"] <= H.PRIOR_WEIGHT + 1e-9

    def test_the_live_case_outranks_the_historical_favourite(self, toy_cells):
        """One open complaint pointing at a cold cell must beat a hot-but-quiet
        one. If this inverts, the surface has become a density map."""
        prior = {"CELL-001": 0.98, "CELL-002": 0.01, "CELL-003": 0.01}
        now = datetime(2026, 1, 1, 12, 0)
        s = H.build_hotspot_surface(
            [_complaint("CELL-003", amount=500000.0, as_of=now)],
            toy_cells, prior, now)
        assert s["cells"][0]["cell_id"] == "CELL-003"

    def test_decomposition_is_reported_per_cell(self, toy_cells):
        prior = {"CELL-001": 0.5, "CELL-002": 0.3, "CELL-003": 0.2}
        now = datetime(2026, 1, 1, 12, 0)
        s = H.build_hotspot_surface([_complaint("CELL-001", as_of=now)],
                                    toy_cells, prior, now)
        for c in s["cells"]:
            assert "conditional_rupees" in c and "prior_rupees" in c
            assert abs((c["conditional_rupees"] + c["prior_rupees"]) - c["score"]) < 1e-6

    def test_empty_open_set_is_declared_degraded(self, toy_cells):
        """With nothing open the surface is pure history. Saying so is the
        difference between an honest fallback and passing a density map off as
        a forecast."""
        prior = {"CELL-001": 0.5, "CELL-002": 0.3, "CELL-003": 0.2}
        s = H.build_hotspot_surface([], toy_cells, prior,
                                    datetime(2026, 1, 1, 12, 0))
        assert s["degraded"] is True
        assert s["prior_share_national"] == 1.0
        assert all(c["prior_share"] == 1.0 for c in s["cells"])

    def test_rupees_weight_the_surface(self, toy_cells):
        now = datetime(2026, 1, 1, 12, 0)
        small = H.build_hotspot_surface([_complaint("CELL-001", amount=1000.0, as_of=now)],
                                        toy_cells, {}, now)
        large = H.build_hotspot_surface([_complaint("CELL-001", amount=1000000.0, as_of=now)],
                                        toy_cells, {}, now)
        assert large["total_conditional_rupees"] > small["total_conditional_rupees"]

    def test_a_case_filed_in_the_future_is_not_open(self, toy_cells):
        now = datetime(2026, 1, 1, 12, 0)
        future = _complaint("CELL-001", age_min=-60, as_of=now)
        s = H.build_hotspot_surface([future], toy_cells, {}, now)
        assert s["degraded"] is True

    def test_distinct_complaints_are_counted_once(self, toy_cells):
        """Mirrors the discipline in state.atm_intelligence(): a chain that
        splits and reconverges is ONE case, not four."""
        now = datetime(2026, 1, 1, 12, 0)
        same = [_complaint("CELL-001", as_of=now, cid="C-1") for _ in range(4)]
        s = H.build_hotspot_surface(same, toy_cells, {}, now)
        top = s["cells"][0]
        assert top["case_count"] == 1


# ── the API ──────────────────────────────────────────────────────────────────

class TestHotspotEndpoint:
    def test_requires_authentication(self, anon):
        """Tier A. A forward forecast says where officers are about to be sent."""
        assert anon.get("/api/v1/hotspots/cells").status_code == 401

    def test_returns_a_surface(self, auth_client):
        r = auth_client.get("/api/v1/hotspots/cells")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("as_of", "degraded", "prior_weight", "cells",
                    "prior_share_national", "windows_min"):
            assert key in body

    def test_every_cell_carries_the_decomposition(self, auth_client):
        body = auth_client.get("/api/v1/hotspots/cells").json()
        if not body["cells"]:
            pytest.skip("no cells; run: python scripts/generate_data.py")
        for c in body["cells"][:20]:
            assert "conditional_rupees" in c
            assert "prior_rupees" in c
            assert 0.0 <= c["prior_share"] <= 1.0

    def test_prior_stays_capped_through_the_api(self, auth_client):
        body = auth_client.get("/api/v1/hotspots/cells").json()
        if body["degraded"]:
            assert body["prior_share_national"] == 1.0
        else:
            assert body["prior_share_national"] <= body["prior_weight"] + 1e-6

    def test_as_of_is_echoed_back(self, auth_client):
        r = auth_client.get("/api/v1/hotspots/cells",
                            params={"as_of": "2026-08-29T00:00:00"})
        assert r.status_code == 200
        assert r.json()["as_of"].startswith("2026-08-29")

    def test_window_bounds_are_validated(self, auth_client):
        assert auth_client.get("/api/v1/hotspots/cells",
                               params={"window_start_min": -1}).status_code == 422
        assert auth_client.get("/api/v1/hotspots/cells",
                               params={"window_end_min": 9999}).status_code == 422

    def test_fraud_type_filter_narrows_the_surface(self, auth_client):
        """Deliverable (b) asks for drill-down by crime category."""
        r = auth_client.get("/api/v1/hotspots/cells",
                            params={"fraud_type": "UPI Fraud",
                                    "as_of": "2026-08-29T00:00:00"})
        assert r.status_code == 200
        assert r.json()["filters"]["fraud_type"] == "UPI Fraud"


class TestServingParity:
    """The served posterior must be the measured posterior.

    scripts/evaluate_hotspots.py publishes PAI, hit rate and lead time. Those
    numbers describe the API only if the API computes the same quantity the same
    way. An earlier version of backend.state read predict()'s display-rounded
    `confidence` instead of calling posterior_from_scores -- numerically within
    5e-5, but a second implementation of one quantity, which is how two paths
    drift apart with no test noticing.
    """

    def test_state_posterior_matches_the_evaluation_path_exactly(self):
        import numpy as np
        import backend.state as state
        from feature_builder import FeatureBuilder          # noqa: E402
        from xgb_model import MuleXGBPredictor              # noqa: E402

        if not state.complaints:
            pytest.skip("no corpus; run: python scripts/generate_data.py")
        cid = next((c for c in state._hotspot_cache), None)
        if cid is None:
            pytest.skip("hotspot cache empty")

        entry = state._hotspot_cache[cid]
        fb = state.get_feature_builder()
        predictor = state.get_xgb_predictor()

        terminal = state.get_terminal_accounts(cid)
        assert terminal
        t0 = terminal[0]
        acc = str(t0.get("dst_account", ""))
        lat, lon = float(t0.get("lat")), float(t0.get("long"))
        node = state.get_node_feature(acc) or {}

        # Recompute the way the EVALUATION script does.
        cand_idx, block = fb.candidate_block(
            lat, lon, str(node.get("bank_name", "UNKNOWN")), account=acc)
        ids = [str(fb.atm_df.iloc[int(i)]["atm_id"]) for i in cand_idx]
        raw = predictor.classifier.predict(MuleXGBPredictor.log_features(block))
        expected = H.project_to_cells(H.posterior_from_scores(ids, list(raw)),
                                      state._atm_cell)

        assert set(expected) == set(entry["cell_probs"])
        for k in expected:
            assert expected[k] == pytest.approx(entry["cell_probs"][k], abs=1e-12), (
                "the served posterior diverges from the measured one"
            )

    def test_the_posterior_is_a_distribution(self):
        import backend.state as state
        if not state._hotspot_cache:
            pytest.skip("hotspot cache empty")
        entry = next(iter(state._hotspot_cache.values()))
        total = sum(entry["cell_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"cell probabilities sum to {total}, not 1 -- projection lost mass"
        )
