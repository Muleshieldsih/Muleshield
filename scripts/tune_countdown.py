# -*- coding: utf-8 -*-
"""
Validation-only hyperparameter search for the countdown regressor.

    python scripts/tune_countdown.py [--trials 40]

WHY THIS SCRIPT EXISTS SEPARATELY FROM train_xgb.py
---------------------------------------------------
The countdown is the one component with measured headroom: 11.86 min MAE against
a 9.38 min ceiling conditioned on observables (OVERNIGHT_ML_AUDIT.md, section 8).
Everything else in the system is at its bound.

THE PROTOCOL, AND WHY IT IS NOT NEGOTIABLE
------------------------------------------
This script reproduces `train_xgb.train()`'s outer GroupShuffleSplit exactly
(seed 42, 20% by complaint) and then **throws the test half away without
scoring it even once**. The search happens on an inner split of the training
half only, grouped by complaint so no laundering chain straddles the boundary.

The test set is not read, not scored, and not used for selection anywhere in
this file. A number chosen because it looked good on test is not a measurement,
and reporting one would be worse than shipping the current model unchanged.

This script therefore DOES NOT WRITE ANY MODEL. It prints what it found on
validation. Promoting a configuration into `engine/train_xgb.py` is a separate,
deliberate act, and the honest test-set number can only be produced once by
retraining with the config frozen.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from feature_builder import FeatureBuilder          # noqa: E402

SEED = 42
TEST_SIZE = 0.20        # identical to train_xgb.train()
INNER_VAL_SIZE = 0.25   # of the training half


def build() -> tuple:
    """Rebuild the ranking set and the training half. The test half is discarded."""
    print("[1/3] Building the candidate-ranking matrix ...")
    fb = FeatureBuilder()
    fb.load()
    X, y, groups, meta = fb.build_ranking_set()

    cid_of_group = dict(zip(meta["group_id"], meta["complaint_id"]))
    row_complaint = np.array([cid_of_group[g] for g in groups])

    # The SAME outer split the trainer uses ...
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=row_complaint))

    # ... and here the test half leaves the program. Nothing below can reach it.
    del test_idx

    Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
    complaint_tr = row_complaint[train_idx]

    time_of_group = dict(zip(meta["group_id"], meta["time_to_cashout_min"]))

    # The countdown is defined per cash-out, on the TRUE ATM's row.
    pos = np.where(ytr == 1)[0]
    Xp = Xtr[pos]
    yp = np.array([time_of_group[int(g)] for g in gtr[pos]], dtype=float)
    cp = complaint_tr[pos]

    print(f"      training half : {len(Xp):,} cash-outs "
          f"across {len(set(cp)):,} complaints")
    return Xp, yp, cp


def split_inner(Xp, yp, cp):
    """Inner train/validation split, grouped by complaint."""
    gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_SIZE, random_state=SEED)
    fit_idx, val_idx = next(gss.split(Xp, yp, groups=cp))
    assert not (set(cp[fit_idx]) & set(cp[val_idx])), "complaint leaked across inner split"

    sc = StandardScaler().fit(Xp[fit_idx])
    print(f"      inner fit     : {len(fit_idx):,} cash-outs")
    print(f"      inner val     : {len(val_idx):,} cash-outs  (complaint-disjoint)")
    return (sc.transform(Xp[fit_idx]), yp[fit_idx],
            sc.transform(Xp[val_idx]), yp[val_idx])


def evaluate(params, Xf, yf, Xv, yv) -> tuple[float, float, int]:
    """Fit one configuration and score it on the inner validation split."""
    reg = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=2000,
        random_state=SEED,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=50,
        **params,
    )
    reg.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    pred = reg.predict(Xv)
    return (mean_absolute_error(yv, pred), r2_score(yv, pred),
            int(reg.best_iteration or reg.n_estimators))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    Xp, yp, cp = build()
    print("\n[2/3] Inner split of the TRAINING half (test never touched) ...")
    Xf, yf, Xv, yv = split_inner(Xp, yp, cp)

    # The shipped configuration, scored on this same inner split so the
    # comparison is like-for-like.
    shipped = {"max_depth": 7, "learning_rate": 0.06, "subsample": 0.85,
               "colsample_bytree": 0.85, "min_child_weight": 2}
    base_mae, base_r2, base_it = evaluate(shipped, Xf, yf, Xv, yv)

    # A mean-prediction baseline on the same validation rows, for scale.
    naive = float(np.mean(np.abs(yv - yf.mean())))

    print(f"\n      mean-guess baseline : {naive:6.3f} min")
    print(f"      shipped config      : {base_mae:6.3f} min   "
          f"(R2 {base_r2:+.3f}, {base_it} trees)")

    print(f"\n[3/3] Random search, {args.trials} configurations ...\n")
    rng = np.random.default_rng(SEED)
    space = {
        "max_depth":        [3, 4, 5, 6, 7, 8, 10],
        "learning_rate":    [0.02, 0.03, 0.05, 0.06, 0.08, 0.12],
        "subsample":        [0.6, 0.7, 0.85, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.85, 1.0],
        "min_child_weight": [1, 2, 5, 10, 20],
        "reg_lambda":       [0.5, 1.0, 3.0, 10.0],
        "reg_alpha":        [0.0, 0.1, 1.0],
    }

    results = []
    t0 = time.time()
    for i in range(args.trials):
        params = {k: rng.choice(v).item() for k, v in space.items()}
        mae, r2, it = evaluate(params, Xf, yf, Xv, yv)
        results.append((mae, r2, it, params))
        flag = "  <-- best so far" if mae == min(r[0] for r in results) else ""
        print(f"  {i + 1:>3}/{args.trials}  val MAE {mae:6.3f}  R2 {r2:+.3f}  "
              f"depth {params['max_depth']:>2} lr {params['learning_rate']:.2f}{flag}")

    results.sort(key=lambda r: r[0])
    best_mae, best_r2, best_it, best_params = results[0]

    print(f"\n  searched {args.trials} configs in {time.time() - t0:.0f}s\n")
    print("=" * 68)
    print("  VALIDATION RESULT  (test set still untouched)")
    print("=" * 68)
    print(f"  mean-guess baseline : {naive:6.3f} min")
    print(f"  shipped config      : {base_mae:6.3f} min")
    print(f"  best found          : {best_mae:6.3f} min   "
          f"({base_mae - best_mae:+.3f} vs shipped)")
    print(f"  best R2             : {best_r2:+.3f}  ({best_it} trees)")
    print("\n  best configuration:")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<18} {v}")

    gain = base_mae - best_mae
    print()
    if gain < 0.10:
        print("  VERDICT: no meaningful gain on validation. The shipped configuration")
        print("  should stay. The remaining error is the unobservable two-regime")
        print("  mixture, not a hyperparameter that was set wrong.")
    else:
        print(f"  VERDICT: {gain:.3f} min improvement on VALIDATION.")
        print("  This is not a test-set number and must not be reported as one.")
        print("  To realise it: promote the config into engine/train_xgb.py, retrain")
        print("  once, and read the test MAE that comes out - whatever it says.")
    print("=" * 68)


if __name__ == "__main__":
    main()
