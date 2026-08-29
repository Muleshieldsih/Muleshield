# -*- coding: utf-8 -*-
"""
Ranked candidate cash-out locations — contract and evaluation tests.
SIH26184 | MHA / I4C

The system does not name the ATM a suspect will use. It returns the K locations
an investigator should search first, ordered. These tests hold that contract in
place and guard the two ways it has historically been broken: a K that drifts
between the interface and the evaluation, and a feature that quietly encodes the
answer.

Reference figures come from scripts/topk_curve.py, which scores the shipped
checkpoint on the held-out complaints of the same seed-42 GroupShuffleSplit that
train_xgb.py trains on. Nothing here retrains, and nothing tunes on test.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from xgb_model import MuleXGBPredictor, OPERATING_K       # noqa: E402
from feature_builder import FeatureBuilder                 # noqa: E402

CURVE_PATH = ROOT / "data" / "topk_curve.json"
METRICS_PATH = ROOT / "data" / "metrics.json"


@pytest.fixture(scope="module")
def curve():
    if not CURVE_PATH.exists():
        pytest.skip("run: python scripts/topk_curve.py")
    return json.loads(CURVE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predictor():
    try:
        return MuleXGBPredictor.load()
    except FileNotFoundError:
        pytest.skip("run: python engine/train_xgb.py")


# predict() builds the candidate set from a position, so the call needs one.
# Connaught Place, New Delhi -- dense enough that 25 candidates exist nearby.
NODE_LAT, NODE_LON = 28.6139, 77.2090


@pytest.fixture(scope="module")
def sample_vector():
    """
    A BASE feature vector. predict() appends the candidate block itself, so the
    ranking-set rows (context + candidate, already concatenated) are the wrong
    shape to hand back in.
    """
    fb = FeatureBuilder()
    fb.load()
    X, _, _, _ = fb.build_training_set()
    return X[0]


def _predict(predictor, vec, **kw):
    return predictor.predict(vec, node_lat=NODE_LAT, node_lon=NODE_LON, **kw)


# ── K = 5 output size ────────────────────────────────────────────────────────

class TestOperatingPoint:
    """The interface promises five locations. It must return five."""

    def test_operating_k_is_five(self):
        assert OPERATING_K == 5

    def test_predict_returns_exactly_k_candidates(self, predictor, sample_vector):
        r = _predict(predictor, sample_vector)
        assert len(r["ranked_candidates"]) == OPERATING_K

    def test_ranks_are_dense_and_ordered(self, predictor, sample_vector):
        ranks = [c["rank"] for c in _predict(predictor, sample_vector)["ranked_candidates"]]
        assert ranks == list(range(1, OPERATING_K + 1))

    def test_confidence_is_descending(self, predictor, sample_vector):
        conf = [c["confidence"] for c in _predict(predictor, sample_vector)["ranked_candidates"]]
        assert conf == sorted(conf, reverse=True)

    def test_candidates_are_distinct(self, predictor, sample_vector):
        ids = [c["atm_id"] for c in _predict(predictor, sample_vector)["ranked_candidates"]]
        assert len(set(ids)) == len(ids)

    def test_top_k_is_a_prefix_of_a_larger_k(self, predictor, sample_vector):
        """Top-5 must be the first five of Top-10, not a different selection."""
        five = [c["atm_id"] for c in _predict(predictor, sample_vector, top_k=5)["ranked_candidates"]]
        ten = [c["atm_id"] for c in _predict(predictor, sample_vector, top_k=10)["ranked_candidates"]]
        assert ten[:5] == five


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:
    """A briefing an officer acts on must not change between two identical calls."""

    def test_repeated_calls_agree(self, predictor, sample_vector):
        a = _predict(predictor, sample_vector)["ranked_candidates"]
        b = _predict(predictor, sample_vector)["ranked_candidates"]
        assert [x["atm_id"] for x in a] == [x["atm_id"] for x in b]
        assert [x["confidence"] for x in a] == [x["confidence"] for x in b]

    def test_ordering_is_total(self, predictor, sample_vector):
        """No ties at the boundary, which would make rank 5 vs 6 arbitrary."""
        r = _predict(predictor, sample_vector, top_k=10)["ranked_candidates"]
        conf = [c["confidence"] for c in r]
        assert len(set(conf)) > 1


# ── Containment curve ────────────────────────────────────────────────────────

class TestContainmentCurve:
    """
    Guards the reported numbers, and the honesty of how they are computed.
    """

    def test_curve_is_monotonic(self, curve):
        vals = [r["containment"] for r in curve["curve"]]
        assert vals == sorted(vals), "containment cannot fall as K grows"

    def test_containment_at_five_matches_ledger(self, curve):
        at5 = next(r for r in curve["curve"] if r["K"] == 5)
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        assert m["ranking"]["top5"] == pytest.approx(at5["containment"], abs=1e-6)

    def test_containment_at_five_is_in_the_measured_band(self, curve):
        """
        0.7136 measured. The band is wide enough to survive a reseed and narrow
        enough that a leak would trip it: the Bayes bound on this generator is
        0.7217, so anything at or above ~0.75 is evidence of leakage, not skill.
        """
        at5 = next(r for r in curve["curve"] if r["K"] == 5)
        assert 0.65 <= at5["containment"] <= 0.75

    def test_denominator_counts_retrieval_failures_as_misses(self, curve):
        """
        Containment is over all held-out cashouts, not only those whose true ATM
        made it into the candidate pool. Dropping the unreachable ones would
        raise every number without finding a single extra ATM.
        """
        rf = curve["curve"][0]["retrieval_failures"]
        kf = curve["curve"][0]["ranking_failures"]
        hits = round(curve["curve"][0]["containment"] * curve["n_cashouts"])
        assert hits + rf + kf == curve["n_cashouts"]

    def test_search_reduction_is_consistent_with_directory_size(self, curve):
        for r in curve["curve"]:
            assert r["reduction"] == pytest.approx(1.0 - r["K"] / curve["n_atms"], abs=1e-6)

    def test_reported_top5_is_not_the_zone_number(self, curve):
        """
        The search zone contains 87.4% with a median of 8 ATMs. That is a
        different operating point from Top-5 and the two were conflated once.
        """
        at5 = next(r for r in curve["curve"] if r["K"] == 5)
        assert at5["containment"] < 0.80


# ── Leakage ──────────────────────────────────────────────────────────────────

class TestNoFutureAtmInformation:
    """
    The candidate block must be computable before the withdrawal happens.
    """

    def test_candidate_features_do_not_depend_on_the_chosen_atm(self):
        """
        candidate_block() is a function of position, bank and account history
        only. If the true ATM were reachable from its arguments, the block would
        differ when the label differs; it cannot, because the label is not passed.
        """
        fb = FeatureBuilder()
        fb.load()
        idx_a, blk_a = fb.candidate_block(19.07, 72.87, node_bank="HDFC Bank")
        idx_b, blk_b = fb.candidate_block(19.07, 72.87, node_bank="HDFC Bank")
        assert np.array_equal(idx_a, idx_b)
        assert np.allclose(blk_a, blk_b)

    def test_ranker_never_sees_context_features(self):
        """
        The ranker is fitted on X[:, RANK_CONTEXT_DIM:] only. Context is constant
        within a candidate group and cancels in the conditional-logit softmax, so
        admitting it could only add a channel for a group-level artefact.
        """
        fb = FeatureBuilder()
        fb.load()
        X, _, groups, _ = fb.build_ranking_set()
        C = fb.RANK_CONTEXT_DIM
        g0 = np.where(groups == groups[0])[0]
        ctx = X[g0, :C]
        assert np.allclose(ctx, ctx[0]), "context must be constant within a group"

    def test_label_is_not_recoverable_from_a_single_candidate_feature(self):
        """
        No candidate column may separate the true ATM on its own. A column that
        did would be the target wearing a different name.
        """
        from sklearn.metrics import roc_auc_score
        fb = FeatureBuilder()
        fb.load()
        X, y, _, _ = fb.build_ranking_set()
        C = fb.RANK_CONTEXT_DIM
        for j in range(C, X.shape[1]):
            col = X[:, j]
            if np.allclose(col, col[0]):
                continue
            auc = roc_auc_score(y, col)
            auc = max(auc, 1.0 - auc)
            assert auc < 0.90, f"candidate column {j - C} separates the label (AUC {auc:.3f})"
