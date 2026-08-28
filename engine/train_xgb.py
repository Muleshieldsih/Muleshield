# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b: XGBoost Training Script (v2)
SIH26184 | MHA / I4C

Trains two XGBoost models on the 80-dim hybrid feature vector (v2):
  1. XGBClassifier  -- predicts nearest ATM (Top-3 with probabilities)
  2. XGBRegressor   -- predicts time-to-cashout (minutes)

v2 Accuracy Upgrades:
  - 80-dim feature vector (added 3D Cartesian, bearing, multi-ATM distances,
    bank affinity features)
  - Tuned hyperparameters: max_depth=7, n_estimators=160, lr=0.06
  - ATM coordinate/risk arrays saved into the predictor for candidate ranking

Usage:
    python engine/train_xgb.py
    python engine/train_xgb.py --seed 42 --n-estimators 160

Output:
    models/xgb_cashout.pkl  (full MuleXGBPredictor bundle)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from feature_builder import FeatureBuilder, TOTAL_FEATURE_DIM, TABULAR_FEATURE_NAMES
from xgb_model import ConditionalLogitRanker, MuleXGBPredictor

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
XGB_MODEL_PATH = MODELS_DIR / "xgb_cashout.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    seed: int = 42,
    n_estimators: int = 160,
    max_depth: int = 7,
    learning_rate: float = 0.06,
    test_size: float = 0.20,
    verbose: bool = True,
) -> dict:
    """
    Full Phase 2b training pipeline (v2 — 80-dim features, tuned hyperparameters).

    Returns:
        dict with trained predictor and evaluation metrics.
    """
    # ── Step 1: Build the candidate-ranking set ───────────────────────────────
    if verbose:
        print("[1/5] Building candidate-ranking matrix (GNN embedding + spatial + per-ATM)...")
    t0 = time.time()
    fb = FeatureBuilder()
    fb.load()
    X, y, groups, meta_df = fb.build_ranking_set()
    elapsed = (time.time() - t0) * 1000

    k = fb.CANDIDATE_K
    if verbose:
        print(f"      X shape     : {X.shape}  "
              f"({fb.RANK_CONTEXT_DIM} context + {fb.CANDIDATE_FEATURES} candidate)")
        print(f"      Cashouts    : {len(meta_df):,}  x  {k} candidates each")
        reach = meta_df["truth_in_candidates"].mean()
        print(f"      Ground-truth ATM inside candidate set: {reach:.2%}")
        print(f"      Built in    : {elapsed:.0f}ms")

    # ── Step 2: Split by COMPLAINT ────────────────────────────────────────────
    #
    # One complaint yields several terminal cashouts sharing a syndicate, a
    # region and upstream accounts. A row-level split scatters those siblings
    # across train and test, letting the model recognise a chain it has already
    # seen. Grouping keeps every chain wholly on one side.
    if verbose:
        print(f"\n[2/5] Splitting by complaint ({int((1-test_size)*100)}/{int(test_size*100)})...")

    cid_of_group = dict(zip(meta_df["group_id"], meta_df["complaint_id"]))
    row_complaint = np.array([cid_of_group[g] for g in groups])

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=row_complaint))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    if verbose:
        overlap = set(row_complaint[train_idx]) & set(row_complaint[test_idx])
        print(f"      Train: {len(X_train):,} rows / {len(set(g_train)):,} cashouts")
        print(f"      Test : {len(X_test):,} rows / {len(set(g_test)):,} cashouts")
        print(f"      Complaint overlap between splits: {len(overlap)} (must be 0)")

    # ── Step 3: Scale ─────────────────────────────────────────────────────────
    if verbose:
        print("\n[3/5] Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Step 4: Fit the conditional-logit ATM ranker ──────────────────────────
    if verbose:
        print("\n[4/5] Fitting ConditionalLogitRanker (discrete choice over candidates)...")

    # Rows must be contiguous per cashout for the group reshape.
    tr_order = np.argsort(g_train, kind="stable")
    te_order = np.argsort(g_test, kind="stable")
    X_train, y_train, g_train = X_train[tr_order], y_train[tr_order], g_train[tr_order]
    X_test, y_test, g_test = X_test[te_order], y_test[te_order], g_test[te_order]
    X_train_s, X_test_s = X_train_s[tr_order], X_test_s[te_order]

    C = fb.RANK_CONTEXT_DIM
    log_train = MuleXGBPredictor.log_features(X_train[:, C:])
    log_test = MuleXGBPredictor.log_features(X_test[:, C:])

    clf = ConditionalLogitRanker(n_features=log_train.shape[1], k=k)
    t0 = time.time()
    clf.fit(log_train, y_train, g_train)
    clf_time = (time.time() - t0) * 1000

    scores = clf.predict(log_test)

    # ── Rank-based evaluation, per cashout ────────────────────────────────────
    by_group: dict[int, list[tuple[float, int]]] = {}
    for score, gid, label in zip(scores, g_test, y_test):
        by_group.setdefault(int(gid), []).append((float(score), int(label)))

    top1 = top3 = total = 0
    for gid, items in by_group.items():
        items.sort(key=lambda t: t[0], reverse=True)
        total += 1
        if items[0][1] == 1:
            top1 += 1
        if any(lbl == 1 for _, lbl in items[:3]):
            top3 += 1

    atm_acc = top1 / max(1, total)
    top3_acc = top3 / max(1, total)

    # Distance-only baseline over the same candidate sets.
    dist_col = C
    base_top1 = base_top3 = 0
    for gid in by_group:
        rows = np.where(g_test == gid)[0]
        order = rows[np.argsort(X_test[rows, dist_col])]
        labels = y_test[order]
        if len(labels) and labels[0] == 1:
            base_top1 += 1
        if any(labels[:3] == 1):
            base_top3 += 1
    base_top1 /= max(1, total)
    base_top3 /= max(1, total)

    # Ranking model: a thresholded F1 is meaningless, so the exported
    # 'atm_f1_weighted' slot carries Top-1 accuracy. Named honestly below.

    # ── C2: withdrawal-LOCATION forecast (the problem statement's actual ask) ──
    #
    # SIH26184 asks for "likely cash withdrawal locations", and a unit deploys to
    # an area, not to one machine. Exact-ATM Top-3 is the tactical drill-down;
    # the zone hit-rate is the deliverable. Reported against two naive zones so
    # the number means something.
    atm_lat_arr = fb._atm_lats
    atm_lon_arr = fb._atm_lons
    truth_atm_of_group = dict(zip(meta_df["group_id"], meta_df["cashout_atm_id"]))
    node_ll_of_group = dict(zip(meta_df["group_id"], zip(meta_df["lat"], meta_df["lon"])))

    def _hav(alat, alon, blat, blon):
        r = 6371.0
        p1, p2 = np.radians(alat), np.radians(blat)
        dphi, dlam = np.radians(blat - alat), np.radians(blon - alon)
        h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

    ZONE_MASS = 0.80
    RADII = (1.0, 2.0, 5.0)
    model_err, near_err, mule_err = [], [], []
    zone_radius, zone_atms = [], []
    contained, near_contained, mule_contained = [], [], []

    for gid, items in by_group.items():
        rows = np.where(g_test == gid)[0]
        truth_id = truth_atm_of_group[gid]
        t_i = fb.atm_to_idx.get(truth_id)
        if t_i is None:
            continue
        t_lat, t_lon = float(atm_lat_arr[t_i]), float(atm_lon_arr[t_i])

        util = scores[rows]
        order_local = np.argsort(util)[::-1]
        e = np.exp(util - util.max())
        conf = e / e.sum()

        cum = np.cumsum(conf[order_local])
        n_keep = max(1, int(np.searchsorted(cum, ZONE_MASS) + 1))
        keep = order_local[:n_keep]

        # Candidate rows are emitted in ascending-distance order within a group,
        # so a row's rank on the distance column recovers which ATM it is.
        n_lat, n_lon = node_ll_of_group[gid]
        d_all = _hav(float(n_lat), float(n_lon), atm_lat_arr, atm_lon_arr)
        near_order = np.argsort(d_all)[:len(rows)]
        dist_rank = np.argsort(np.argsort(X_test[rows, C]))
        keep_idx = near_order[[int(dist_rank[j]) for j in keep]]

        w = conf[keep] / conf[keep].sum()
        c_lat = float(np.sum(w * atm_lat_arr[keep_idx]))
        c_lon = float(np.sum(w * atm_lon_arr[keep_idx]))
        rad = float(_hav(c_lat, c_lon, atm_lat_arr[keep_idx], atm_lon_arr[keep_idx]).max())
        rad = max(rad, 0.05)

        err = float(_hav(c_lat, c_lon, t_lat, t_lon))
        model_err.append(err)
        zone_radius.append(rad)
        zone_atms.append(int((_hav(c_lat, c_lon, atm_lat_arr, atm_lon_arr) <= rad).sum()))
        contained.append(err <= rad)

        # Baselines are given the SAME radius, so the comparison is at equal
        # search cost - a bigger zone always contains more, and rewarding that
        # would be measuring zone size rather than model skill.
        n3 = near_order[:3]
        ne = float(_hav(float(atm_lat_arr[n3].mean()), float(atm_lon_arr[n3].mean()),
                        t_lat, t_lon))
        me = float(_hav(float(n_lat), float(n_lon), t_lat, t_lon))
        near_err.append(ne)
        mule_err.append(me)
        near_contained.append(ne <= rad)
        mule_contained.append(me <= rad)

    model_err = np.array(model_err)
    near_err = np.array(near_err)
    mule_err = np.array(mule_err)

    zone_hit = float(np.mean(contained))
    near_hit = float(np.mean(near_contained))
    mule_hit = float(np.mean(mule_contained))

    if verbose:
        print()
        print("      -- C2: withdrawal-location forecast ----------------------")
        print("      Does the withdrawal fall inside the predicted search zone?")
        print("      (all three zones given the same radius = equal search cost)")
        print(f'      {"zone centre":<28}{"contains":>10}{"median err":>13}')
        for nm, hit, err in (("model search zone", zone_hit, model_err),
                             ("nearest-3 centroid", near_hit, near_err),
                             ("mule location", mule_hit, mule_err)):
            print(f'      {nm:<28}{hit:>10.3f}{np.median(err):>10.2f} km')
        print(f'      search cost: {np.median(zone_radius):.2f} km radius, '
              f'{np.median(zone_atms):.0f} of {len(fb.atm_ids)} ATMs (median)')

    atm_top1 = atm_acc

    if verbose:
        print(f"      Fitted in   : {clf_time:.0f}ms")
        print(f"      Top-1 Acc   : {atm_acc:.4f}   (distance-only baseline {base_top1:.4f})")
        print(f"      Top-3 Acc   : {top3_acc:.4f}   (distance-only baseline {base_top3:.4f})")
        print(f"      Lift over distance baseline: "
              f"Top-1 {atm_acc - base_top1:+.4f} | Top-3 {top3_acc - base_top3:+.4f}")
        names = ["-dist/5", "log(1+2*risk)", "same_bank", "atm_prior", "crew_prior", "crew_seen"]
        print("      Learned utility weights (interpretable by construction):")
        for nm, wv in zip(names, clf.w):
            print(f"        {nm:<16}{wv:+.4f}")

    # ── Step 5: Train the countdown regressor ─────────────────────────────────
    #
    # Trained on the TRUE cashout row. There is a real train/serve asymmetry here
    # - at inference the regressor is handed the ranker's predicted top-1 row -
    # and training on the predicted row instead was tried and measured:
    #
    #     true row       MAE 6.35 min   R2 0.53
    #     predicted row  MAE 8.65 min   R2 0.14
    #
    # It is worse because the LABEL is defined by the true ATM: the delay depends
    # on travel to the ATM actually used. Feeding the model a different ATM's
    # distance against that label corrupts the target rather than fixing the skew.
    #
    # The durable fix is to make the countdown depend less on exact-ATM distance
    # and more on account behaviour (dwell time, velocity), which is available
    # regardless of which candidate is ranked first - see WS2.
    if verbose:
        print(f"\n[5/5] Training XGBRegressor (time-to-cashout, {n_estimators} trees)...")

    time_of_group = dict(zip(meta_df["group_id"], meta_df["time_to_cashout_min"]))
    pos_train = np.where(y_train == 1)[0]
    pos_test = np.where(y_test == 1)[0]
    yt_train = np.array([time_of_group[int(g)] for g in g_train[pos_train]], dtype=float)
    yt_test = np.array([time_of_group[int(g)] for g in g_test[pos_test]], dtype=float)

    # Hold out a slice of TRAIN for early stopping, so tree count is chosen on
    # data the final score never sees.
    n_pos = len(pos_train)
    rng = np.random.default_rng(seed)
    shuf = rng.permutation(n_pos)
    n_es = max(1, int(0.15 * n_pos))
    es_rows, fit_rows = shuf[:n_es], shuf[n_es:]

    def _make_reg(**extra):
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
            n_jobs=-1,
            **extra,
        )

    # The headline metric is MAE, so optimise absolute error rather than squared
    # error. Squared error chases the heavy Gamma tail the generator injects and
    # pays for it in the median case an officer actually experiences.
    reg = _make_reg(objective="reg:absoluteerror", early_stopping_rounds=25)
    t0 = time.time()
    reg.fit(
        X_train_s[pos_train][fit_rows], yt_train[fit_rows],
        eval_set=[(X_train_s[pos_train][es_rows], yt_train[es_rows])],
        verbose=False,
    )
    reg_time = (time.time() - t0) * 1000

    y_time_pred = reg.predict(X_test_s[pos_test])
    time_mae = mean_absolute_error(yt_test, y_time_pred)
    time_r2 = r2_score(yt_test, y_time_pred)
    baseline_mae = float(np.mean(np.abs(yt_test - yt_train.mean())))

    # ── Prediction interval ───────────────────────────────────────────────────
    # The delay carries irreducible heavy-tailed noise, so a point estimate
    # overstates what is knowable. "Cashout expected in 25-45 min" is both more
    # honest and more useful to a dispatcher than a single number, and it is what
    # the PS phrase "in Advance" actually calls for.
    reg_lo = _make_reg(objective="reg:quantileerror", quantile_alpha=0.05)
    reg_hi = _make_reg(objective="reg:quantileerror", quantile_alpha=0.95)
    reg_lo.fit(X_train_s[pos_train], yt_train, verbose=False)
    reg_hi.fit(X_train_s[pos_train], yt_train, verbose=False)

    lo_pred = reg_lo.predict(X_test_s[pos_test])
    hi_pred = reg_hi.predict(X_test_s[pos_test])
    coverage = float(np.mean((yt_test >= lo_pred) & (yt_test <= hi_pred)))
    interval_width = float(np.mean(hi_pred - lo_pred))

    if verbose:
        print(f"      Trained in  : {reg_time:.0f}ms  (best iter {reg.best_iteration})")
        print(f"      Test MAE    : {time_mae:.2f} min   (mean-baseline {baseline_mae:.2f} min)")
        print(f"      Test R2     : {time_r2:.4f}")
        print(f"      q05-q95 band: {coverage:.1%} empirical coverage, "
              f"{interval_width:.1f} min wide")

    # ── Save ──────────────────────────────────────────────────────────────────
    atm_lats = fb.atm_df["lat"].to_numpy()
    atm_lons = fb.atm_df["long"].to_numpy()
    atm_risk = (fb.atm_df["cashout_risk_score"].to_numpy(dtype=float)
                if "cashout_risk_score" in fb.atm_df.columns
                else np.zeros(len(fb.atm_ids), dtype=float))
    atm_fraud = (fb.atm_df["historical_fraud_count"].to_numpy(dtype=float)
                 if "historical_fraud_count" in fb.atm_df.columns
                 else np.zeros(len(fb.atm_ids), dtype=float))
    atm_banks = (fb.atm_df["bank_name"].to_numpy(dtype=str)
                 if "bank_name" in fb.atm_df.columns
                 else np.array(["UNKNOWN"] * len(fb.atm_ids), dtype=str))

    predictor = MuleXGBPredictor(
        classifier=clf,          # ConditionalLogitRanker
        regressor=reg,
        atm_ids=fb.atm_ids,
        scaler=scaler,
        label_encoder=None,          # ranking model: no class encoding needed
        atm_lats=atm_lats,
        atm_lons=atm_lons,
        atm_risk_scores=atm_risk,
        atm_banks=atm_banks,
        atm_fraud_counts=atm_fraud,
        atm_prior_counts=fb._atm_prior_count,
        acct_atm_hist=fb._acct_atm_hist,
        graph_adj={n: set(fb._graph.neighbors(n)) for n in fb._graph.nodes()},
        behaviour=fb._behaviour_lookup,
        regressor_lo=reg_lo,
        regressor_hi=reg_hi,
        candidate_k=k,
    )
    predictor.save(XGB_MODEL_PATH)
    if verbose:
        print(f"\n      [OK] Saved to {XGB_MODEL_PATH}")

    return {
        "predictor": predictor,
        "atm_accuracy": atm_acc,
        "atm_top1_accuracy": atm_top1,
        "top3_accuracy": top3_acc,
        "zone_containment": zone_hit,
        "zone_containment_baseline_nearest3": near_hit,
        "zone_containment_baseline_mule": mule_hit,
        "zone_median_error_km": float(np.median(model_err)),
        "zone_median_radius_km": float(np.median(zone_radius)),
        "zone_median_atms": float(np.median(zone_atms)),
        "baseline_top1": base_top1,
        "baseline_top3": base_top3,
        "time_mae_minutes": time_mae,
        "time_baseline_mae": baseline_mae,
        "time_r2": time_r2,
        "interval_coverage": coverage,
        "interval_width_min": interval_width,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "meta_df": meta_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END INFERENCE SPEED TEST
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_inference(predictor: MuleXGBPredictor, X_test: np.ndarray,
                        n_runs: int = 100, node_lat: float = 28.6139,
                        node_lon: float = 77.2090):
    """Measure average single-sample inference latency (candidate build + rank)."""
    # Context width only; the candidate block is rebuilt inside predict().
    sample = X_test[0][:FeatureBuilder.RANK_CONTEXT_DIM]
    latencies = []
    for _ in range(n_runs):
        t0 = time.time()
        predictor.predict(sample, node_lat=node_lat, node_lon=node_lon)
        latencies.append((time.time() - t0) * 1000)
    return np.mean(latencies), np.max(latencies)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuleShield AI -- Train XGBoost v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.06)
    args = parser.parse_args()

    print("=" * 60)
    print("  MuleShield AI - XGBoost Training  |  SIH26184")
    print("=" * 60)

    results = train(
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
    )


    # Inference speed check — measured on the real path the API calls, which
    # includes building the candidate set, not just a forward pass.
    print("\n[BENCHMARK] Measuring inference latency...")
    fb = FeatureBuilder().load()
    X, _, _, _ = fb.build_ranking_set()

    mean_ms, max_ms = benchmark_inference(results["predictor"], X)
    print(f"  Mean inference : {mean_ms:.2f}ms")
    print(f"  Max  inference : {max_ms:.2f}ms")
    ac_met = "PASS" if mean_ms < 200 else "FAIL (AC: <200ms)"
    print(f"  AC <200ms      : {ac_met}")

    # Every headline number is printed next to the naive alternative it has to
    # beat. A score without its baseline says nothing about the model.
    print("\n" + "=" * 66)
    print("  RESULTS vs BASELINES")
    print("=" * 66)
    print(f"  Top-1 ATM      : {results['atm_accuracy']:.4f}"
          f"   | distance-only {results['baseline_top1']:.4f}"
          f"   | lift {results['atm_accuracy'] - results['baseline_top1']:+.4f}")
    print(f"  Top-3 ATM      : {results['top3_accuracy']:.4f}"
          f"   | distance-only {results['baseline_top3']:.4f}"
          f"   | lift {results['top3_accuracy'] - results['baseline_top3']:+.4f}")
    print(f"  Countdown MAE  : {results['time_mae_minutes']:.2f} min"
          f" | mean-baseline {results['time_baseline_mae']:.2f} min"
          f" | R2 {results['time_r2']:.4f}")

    beats_dist = results['top3_accuracy'] > results['baseline_top3']
    beats_mean = results['time_mae_minutes'] < results['time_baseline_mae']
    print("-" * 66)
    print(f"  Beats distance-only Top-3 : {'YES' if beats_dist else 'NO'}")
    print(f"  Beats mean-countdown MAE  : {'YES' if beats_mean else 'NO'}")
    print("=" * 66)
