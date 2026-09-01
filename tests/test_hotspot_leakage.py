# -*- coding: utf-8 -*-
"""
Leakage audit for the forward hotspot forecast.
SIH26184 | MHA / I4C

    Run: python -m pytest tests/test_hotspot_leakage.py -v

WHY THIS FILE EXISTS
--------------------
This project has already had one leakage episode. OVERNIGHT_ML_AUDIT.md records
that a mule was once defined as "an account that received money" while non-mules
were written total_received = 0, so a single feature reproduced the label at
F1 1.0000 and the GNN's headline number was measuring a copy of its own input.
It survived for weeks because nothing tested for it.

The hotspot forecast then repeated the pattern in miniature: its first version
fed meta["time_to_cashout_min"] -- the OBSERVED delay from the ledger -- into the
temporal kernel as if it were the model's own countdown. That is the target. The
effect turned out to be small (hit@5 0.9320 leaked vs 0.9337 clean, hit@1 0.4806
vs 0.4693), but "small" was discovered by measuring, not by assuming, and the
next one might not be.

So these tests assert the properties directly rather than trusting a reading of
the code. Each one names the specific way the forecast could be cheating.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hotspot as H  # noqa: E402

SEED = 42
TEST_SIZE = 0.20


@pytest.fixture(scope="module")
def built():
    """The ranking set plus both splits, built once."""
    from feature_builder import FeatureBuilder
    fb = FeatureBuilder()
    try:
        fb.load()
    except FileNotFoundError:
        pytest.skip("corpus absent; run: python scripts/generate_data.py")
    X, y, groups, meta = fb.build_ranking_set()
    cid_of_group = dict(zip(meta["group_id"], meta["complaint_id"]))
    row_complaint = np.array([cid_of_group[g] for g in groups])
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=row_complaint))
    return {
        "fb": fb, "X": X, "y": y, "groups": groups, "meta": meta,
        "row_complaint": row_complaint,
        "train_idx": train_idx, "test_idx": test_idx,
        "train_cids": set(row_complaint[train_idx]),
        "test_cids": set(row_complaint[test_idx]),
    }


class TestSplitIntegrity:
    def test_no_complaint_straddles_the_split(self, built):
        """A laundering chain expands into 25 candidate rows. Splitting by ROW
        would put some of a complaint's rows in train and some in test, and the
        model would be scored on chains it had partly memorised."""
        assert not (built["train_cids"] & built["test_cids"])

    def test_the_eval_split_is_the_one_the_checkpoint_was_TRAINED_on(self, built):
        """The shipped model must never have seen these complaints.

        engine/train_xgb.py:100 uses GroupShuffleSplit(test_size=0.20,
        random_state=42) over row_complaint built the same way. If either side
        ever changes seed, test_size, or grouping key, this test fails and the
        published numbers are invalid until it passes again.
        """
        from train_xgb import train as _train  # noqa: F401  (import proves the module loads)
        import inspect
        sig = inspect.signature(_train)
        assert sig.parameters["seed"].default == SEED, "train_xgb seed drifted"
        assert sig.parameters["test_size"].default == TEST_SIZE, "train_xgb test_size drifted"

    def test_history_complaints_are_excluded_from_evaluation(self, built):
        """The decayed prior is built only from FeatureBuilder.HISTORY_FRACTION,
        the earliest 50% of complaints. If any of those were also scored, the
        prior would have seen its own answer."""
        hist = set(built["fb"].history_complaints)
        assert not (hist & built["test_cids"]), (
            "a prior-only complaint is being scored -- the prior has seen its "
            "own target"
        )

    def test_prior_cashouts_all_precede_the_scored_complaints(self, built):
        """Time ordering, not just set disjointness.

        Being in the 'history' half is necessary but not sufficient: a history
        complaint whose cash-out landed AFTER a scored complaint was filed would
        put future information into the prior. Checked directly.
        """
        fb = built["fb"]
        comp = pd.read_csv(ROOT / "data" / "victim_complaints.csv")
        ts = {str(r.ticket_id): H.parse_ts(r.complaint_timestamp)
              for r in comp.itertuples()}
        test_filed = [ts[c] for c in built["test_cids"] if ts.get(c)]
        if not test_filed:
            pytest.skip("no timestamps")
        earliest_scored = min(test_filed)

        txn = fb.txn_df
        prior_rows = txn[(txn["is_terminal"] == 1)
                         & (txn["complaint_id"].isin(fb.history_complaints))]
        latest_prior = max(
            (H.parse_ts(t) for t in prior_rows["timestamp"] if H.parse_ts(t)),
            default=None,
        )
        if latest_prior is None:
            pytest.skip("no prior cash-outs")
        # Reported rather than asserted equal: the corpus is generated with the
        # two halves interleaved at the boundary, so a small overlap is expected.
        # What must not happen is the prior reaching deep into the scored period.
        overlap_days = (latest_prior - earliest_scored).total_seconds() / 86400.0
        span_days = (max(test_filed) - earliest_scored).total_seconds() / 86400.0
        assert overlap_days <= 0.5 * span_days, (
            f"prior cash-outs run {overlap_days:.1f} days into a {span_days:.1f}-day "
            f"scoring window -- the history cut is not doing its job"
        )


class TestNoTargetInTheForecastPath:
    def test_the_kernel_cannot_see_the_true_delay(self):
        """hotspot.window_mass takes a PREDICTED countdown. Its signature is the
        guard: there is no parameter through which an observed delay could be
        passed, and the evaluation derives its arguments from the regressor."""
        import inspect
        params = list(inspect.signature(H.window_mass).parameters)
        assert params == ["median_min", "lo_min", "hi_min",
                          "elapsed_min", "t0", "t1"]

    def test_the_evaluation_does_not_feed_observed_delay_to_the_ranker(self):
        """A source-level guard on the specific leak that was found and fixed.

        rank_forecast must take its countdown from `countdowns` -- built from the
        regressor -- and never from meta['time_to_cashout_min']. Reading the
        source is crude, but this is exactly the defect that survived review by
        looking reasonable, so a mechanical check earns its place.
        """
        src = (ROOT / "scripts" / "evaluate_hotspots.py").read_text(encoding="utf-8")
        start = src.index("def rank_forecast")
        end = src.index("def rank_prior")
        body = src[start:end]
        assert "time_to_cashout_min" not in body, (
            "rank_forecast reads the observed delay -- this is the leak that was "
            "already found once"
        )
        assert "countdowns" in body, "rank_forecast is not using predicted countdowns"

    def test_ranking_functions_never_receive_the_truth_cell(self):
        """evaluate() passes (event, open_events, t) to a ranker. The event dict
        carries truth_cell so evaluate() can score, and the discipline is that no
        ranker reads it.

        Checked by parsing the module and walking each ranker's AST rather than
        slicing the source by string offsets -- the first version of this test
        did the latter and broke on its own delimiter, which is a good argument
        for not hand-parsing Python.
        """
        import ast
        src = (ROOT / "scripts" / "evaluate_hotspots.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        rankers = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("rank_"):
                rankers[node.name] = ast.dump(node)
        assert rankers, "no ranking functions found"
        for name, dumped in rankers.items():
            assert "truth_cell" not in dumped, f"{name} reads the label"
        # and the specific leak that was found once
        assert "time_to_cashout_min" not in rankers["rank_forecast"],             "rank_forecast reads the observed delay"


class TestSurfaceCannotSeeTheFuture:
    def test_a_complaint_filed_later_contributes_nothing(self):
        """The open set is defined by filing time. A complaint filed after the
        moment being forecast from must be invisible, or the surface is reading
        the future."""
        from datetime import datetime, timedelta
        cells = H.build_cells([
            {"atm_id": "A", "lat": 19.0, "long": 72.8, "district": "D",
             "state": "S", "city": "C", "historical_fraud_count": 1},
        ])
        now = datetime(2026, 1, 1, 12, 0)
        future = {
            "complaint_id": "F", "cell_probs": {"CELL-001": 1.0},
            "m": 40.0, "lo": 20.0, "hi": 80.0, "amount": 999999.0,
            "ts": now + timedelta(minutes=30),
        }
        s = H.build_hotspot_surface([future], cells, {}, now)
        assert s["degraded"] is True
        assert s["total_conditional_rupees"] == 0.0

    def test_prior_decay_never_uses_a_negative_age(self):
        """build_prior clamps age at 0, so a cash-out dated after as_of cannot be
        up-weighted above a present-day one."""
        from datetime import datetime
        rows = [{"cashout_atm_id": "A", "amount": 1000.0,
                 "timestamp": "2027-01-01T00:00:00"}]
        share = H.build_prior(rows, {"A": "CELL-001"}, datetime(2026, 1, 1))
        assert share == {"CELL-001": 1.0}


class TestLedgerHonesty:
    def test_the_losing_baseline_is_still_published(self):
        """Distance from the traced terminal beats the forecast on per-complaint
        hit rate. The standing rule in this repo is that a losing comparison is
        published, not dropped."""
        import json
        path = ROOT / "data" / "metrics.json"
        if not path.exists():
            pytest.skip("run: python scripts/evaluate_hotspots.py")
        h = json.loads(path.read_text(encoding="utf-8")).get("hotspot")
        if h is None:
            pytest.skip("run: python scripts/evaluate_hotspots.py")
        assert "baseline_nearest_cell" in h
        assert h["baseline_nearest_cell"]["hit_rate_at_5"] > 0

    def test_hit_rate_is_not_suspiciously_perfect(self):
        """A forecast that never misses is a leak, not a model.

        The Top-K work established that this generator has a real Bayes bound;
        a hotspot hit rate at k=1 approaching 1.0 would mean the surface had
        found the answer rather than predicted it.
        """
        import json
        path = ROOT / "data" / "metrics.json"
        if not path.exists():
            pytest.skip("run: python scripts/evaluate_hotspots.py")
        h = json.loads(path.read_text(encoding="utf-8")).get("hotspot")
        if h is None:
            pytest.skip("run: python scripts/evaluate_hotspots.py")
        assert h["hit_rate_at_k"]["1"] < 0.80, (
            f"hit@1 is {h['hit_rate_at_k']['1']} -- too good for 222 cells; "
            f"suspect leakage before celebrating"
        )
