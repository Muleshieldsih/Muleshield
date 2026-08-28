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
  - ATM coordinate arrays saved into predictor for Bayesian spatial reranking

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

    atm_f1 = atm_acc

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
    # Trained on the TRUE candidate row so the delay is conditioned on the ATM
    # actually withdrawn from.
    if verbose:
        print(f"\n[5/5] Training XGBRegressor (time-to-cashout, {n_estimators} trees)...")

    time_of_group = dict(zip(meta_df["group_id"], meta_df["time_to_cashout_min"]))
    pos_train = np.where(y_train == 1)[0]
    pos_test = np.where(y_test == 1)[0]
    yt_train = np.array([time_of_group[int(g)] for g in g_train[pos_train]], dtype=float)
    yt_test = np.array([time_of_group[int(g)] for g in g_test[pos_test]], dtype=float)

    reg = XGBRegressor(
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
    )
    t0 = time.time()
    reg.fit(X_train_s[pos_train], yt_train, verbose=False)
    reg_time = (time.time() - t0) * 1000

    y_time_pred = reg.predict(X_test_s[pos_test])
    time_mae = mean_absolute_error(yt_test, y_time_pred)
    time_r2 = r2_score(yt_test, y_time_pred)
    baseline_mae = float(np.mean(np.abs(yt_test - yt_train.mean())))

    if verbose:
        print(f"      Trained in  : {reg_time:.0f}ms")
        print(f"      Test MAE    : {time_mae:.2f} min   (mean-baseline {baseline_mae:.2f} min)")
        print(f"      Test R2     : {time_r2:.4f}")

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
        candidate_k=k,
    )
    predictor.save(XGB_MODEL_PATH)
    if verbose:
        print(f"\n      [OK] Saved to {XGB_MODEL_PATH}")

    return {
        "predictor": predictor,
        "atm_accuracy": atm_acc,
        "atm_f1_weighted": atm_f1,
        "top3_accuracy": top3_acc,
        "baseline_top1": base_top1,
        "baseline_top3": base_top3,
        "time_mae_minutes": time_mae,
        "time_baseline_mae": baseline_mae,
        "time_r2": time_r2,
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
    sample = X_test[0][:7]      # ranker context only; candidates are built inside
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
