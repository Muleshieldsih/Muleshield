# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b: XGBoost Training Script
SIH26184 | MHA / I4C

Trains two XGBoost models on the 72-dim hybrid feature vector:
  1. XGBClassifier  -- predicts nearest ATM (Top-3 with probabilities)
  2. XGBRegressor   -- predicts time-to-cashout (minutes)

Class imbalance: scale_pos_weight handled internally by XGBoost.

Usage:
    python engine/train_xgb.py
    python engine/train_xgb.py --seed 42 --n-estimators 300

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from feature_builder import FeatureBuilder, TOTAL_FEATURE_DIM, TABULAR_FEATURE_NAMES
from xgb_model import MuleXGBPredictor

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
XGB_MODEL_PATH = MODELS_DIR / "xgb_cashout.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    test_size: float = 0.20,
    verbose: bool = True,
) -> dict:
    """
    Full Phase 2b training pipeline.

    Returns:
        dict with trained predictor and evaluation metrics.
    """
    # ── Step 1: Build features ────────────────────────────────────────────────
    if verbose:
        print("[1/5] Building hybrid feature matrix (GNN embeddings + tabular)...")
    t0 = time.time()
    fb = FeatureBuilder()
    fb.load()
    X, y_atm, y_time, meta_df = fb.build_training_set()
    elapsed = (time.time() - t0) * 1000

    if verbose:
        print(f"      X shape     : {X.shape}  ({TOTAL_FEATURE_DIM} features = 64 GNN + 8 tabular)")
        print(f"      ATM classes : {len(set(y_atm))}")
        print(f"      Time range  : {y_time.min():.1f} -- {y_time.max():.1f} min")
        print(f"      Built in    : {elapsed:.0f}ms")

    assert X.shape[1] == TOTAL_FEATURE_DIM, \
        f"Feature dim mismatch: expected {TOTAL_FEATURE_DIM}, got {X.shape[1]}"

    # ── Step 2: Train/test split ──────────────────────────────────────────────
    if verbose:
        print(f"\n[2/5] Splitting data ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)...")

    # Only stratify if every class has >= 2 members (sklearn requirement)
    class_counts = np.bincount(y_atm)
    can_stratify = bool((class_counts >= 2).all()) and len(set(y_atm)) > 1
    if not can_stratify and verbose:
        rare = int((class_counts < 2).sum())
        print(f"      Note: {rare} ATM class(es) have only 1 sample -- skipping stratification")

    X_train, X_test, y_atm_train, y_atm_test, y_time_train, y_time_test = train_test_split(
        X, y_atm, y_time,
        test_size=test_size,
        random_state=seed,
        stratify=y_atm if can_stratify else None,
    )
    if verbose:
        print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Step 3a: Encode ATM labels to contiguous 0..N-1 for XGBoost ─────────
    # CRITICAL: Fit LabelEncoder on training labels ONLY so they are dense.
    # If we fit on all labels, rare test-only classes create gaps in training.
    from sklearn.preprocessing import LabelEncoder
    le_train = LabelEncoder()
    y_atm_train_enc = le_train.fit_transform(y_atm_train)  # always contiguous
    num_classes = len(le_train.classes_)
    if verbose:
        print(f"      ATM classes in train: {num_classes}")

    # For test evaluation: only keep samples whose label appears in training
    test_mask_known = np.isin(y_atm_test, le_train.classes_)
    n_unknown = int((~test_mask_known).sum())
    if verbose and n_unknown > 0:
        print(f"      Note: {n_unknown} test samples have unseen ATM class -- excluded from metrics")
    y_atm_test_enc = le_train.transform(y_atm_test[test_mask_known])

    # Also build a full LabelEncoder over all labels for inference (saved in predictor)
    le_full = LabelEncoder()
    le_full.fit(y_atm)

    # ── Step 3: Scale features ────────────────────────────────────────────────
    if verbose:
        print("\n[3/5] Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    # Subset of test features for known-class metrics
    X_test_s_known = X_test_s[test_mask_known]

    # ── Step 4: Train ATM Classifier ─────────────────────────────────────────
    if verbose:
        print(f"\n[4/5] Training XGBClassifier (ATM prediction, {n_estimators} trees)...")

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=seed,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
    )

    t0 = time.time()
    clf.fit(X_train_s, y_atm_train_enc)   # encoded labels guaranteed contiguous
    clf_time = (time.time() - t0) * 1000

    # ATM Classifier metrics (only on test samples with known training classes)
    y_atm_pred_enc = clf.predict(X_test_s_known)
    atm_acc = accuracy_score(y_atm_test_enc, y_atm_pred_enc)
    atm_f1  = f1_score(y_atm_test_enc, y_atm_pred_enc, average="weighted", zero_division=0)

    if verbose:
        print(f"      Trained in  : {clf_time:.0f}ms")
        print(f"      Test Acc    : {atm_acc:.4f}")
        print(f"      Test F1 (W) : {atm_f1:.4f}")

    # Top-3 accuracy
    probs = clf.predict_proba(X_test_s_known)
    top3_correct = sum(
        1 for i, true in enumerate(y_atm_test_enc)
        if true in np.argsort(probs[i])[::-1][:3]
    )
    top3_acc = top3_correct / max(1, len(y_atm_test_enc))
    if verbose:
        print(f"      Top-3 Acc   : {top3_acc:.4f}")

    # ── Step 5: Train Time Regressor ─────────────────────────────────────────
    if verbose:
        print(f"\n[5/5] Training XGBRegressor (time-to-cashout, {n_estimators} trees)...")

    reg = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
    )

    t0 = time.time()
    reg.fit(
        X_train_s, y_time_train,
        eval_set=[(X_test_s, y_time_test)],
        verbose=False,
    )
    reg_time = (time.time() - t0) * 1000

    y_time_pred = reg.predict(X_test_s)
    time_mae = mean_absolute_error(y_time_test, y_time_pred)
    time_r2  = r2_score(y_time_test, y_time_pred)

    if verbose:
        print(f"      Trained in  : {reg_time:.0f}ms")
        print(f"      Test MAE    : {time_mae:.2f} min")
        print(f"      Test R2     : {time_r2:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    predictor = MuleXGBPredictor(
        classifier=clf,
        regressor=reg,
        atm_ids=fb.atm_ids,
        scaler=scaler,
        label_encoder=le_train,   # trained-label encoder; maps 0..N-1 -> ATM index
    )
    predictor.save(XGB_MODEL_PATH)
    if verbose:
        print(f"\n      [OK] Saved to {XGB_MODEL_PATH}")

    return {
        "predictor": predictor,
        "atm_accuracy": atm_acc,
        "atm_f1_weighted": atm_f1,
        "top3_accuracy": top3_acc,
        "time_mae_minutes": time_mae,
        "time_r2": time_r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END INFERENCE SPEED TEST
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_inference(predictor: MuleXGBPredictor, X_test: np.ndarray, n_runs: int = 100):
    """Measure average single-sample inference latency."""
    sample = X_test[0]
    latencies = []
    for _ in range(n_runs):
        t0 = time.time()
        predictor.predict(sample)
        latencies.append((time.time() - t0) * 1000)
    return np.mean(latencies), np.max(latencies)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuleShield AI -- Train XGBoost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.08)
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


    # Inference speed check
    print("\n[BENCHMARK] Measuring inference latency...")
    fb = FeatureBuilder().load()
    X, _, _, _ = fb.build_training_set()
    scaler = results["predictor"].scaler
    X_s = scaler.transform(X)

    mean_ms, max_ms = benchmark_inference(results["predictor"], X_s)
    print(f"  Mean inference : {mean_ms:.2f}ms")
    print(f"  Max  inference : {max_ms:.2f}ms")
    ac_met = "PASS" if mean_ms < 200 else "FAIL (AC: <200ms)"
    print(f"  AC <200ms      : {ac_met}")

    print("\n" + "=" * 60)
    print(f"  Top-3 ATM Acc  : {results['top3_accuracy']:.4f}")
    print(f"  Time MAE       : {results['time_mae_minutes']:.2f} min  [AC: <5 min]")
    mae_ac = "PASS" if results['time_mae_minutes'] <= 5.0 else "REVIEW"
    print(f"  Time MAE AC    : {mae_ac}")
    print("=" * 60)
