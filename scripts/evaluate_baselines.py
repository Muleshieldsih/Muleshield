# -*- coding: utf-8 -*-
"""
MuleShield AI — Baseline Comparison
SIH26184 | MHA / I4C

Answers the first question any evaluator should ask: does the model earn its
complexity? A score means nothing without the naive alternatives beside it.

Reports, for mule detection:
  1. Majority class          — predict everything is a mule
  2. Best single feature     — one threshold, swept over every feature
  3. Logistic regression     — linear, same features, no graph
  4. Random forest           — non-linear tabular, same features, no graph
  5. GraphSAGE (this system) — same features PLUS neighbourhood structure

and for ATM prediction:
  1. Nearest ATM             — pure distance, no model
  2. Nearest-3 ATMs          — the distance-only Top-3 baseline
  3. XGBoost v2 (this system)

Run:
    python scripts/evaluate_baselines.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))
DATA = ROOT / "data"

from gnn_model import FEATURE_COLS  # noqa: E402


def _hav(lat, lon, lats, lons):
    r = 6371.0
    p1, p2 = np.radians(lat), np.radians(lats)
    dphi = np.radians(lats - lat)
    dlam = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mule_detection_baselines(seed: int = 42) -> pd.DataFrame:
    node = pd.read_csv(DATA / "node_features.csv")
    X = node[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = node["is_mule_label"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    rows = []

    # 1. Majority class
    pred = np.ones_like(yte)
    rows.append(("Majority class (all mule)", f1_score(yte, pred), float("nan")))

    # 2. Best single feature, threshold swept on TRAIN, scored on TEST
    best_f1, best_col, best_rule = 0.0, None, None
    for j, col in enumerate(FEATURE_COLS):
        v_tr, v_te = Xtr[:, j], Xte[:, j]
        for t in np.unique(np.percentile(v_tr, np.linspace(2, 98, 50))):
            for sign in (1, -1):
                p_tr = ((v_tr > t) if sign == 1 else (v_tr <= t)).astype(int)
                if p_tr.sum() in (0, len(p_tr)):
                    continue
                f1_tr = f1_score(ytr, p_tr)
                if f1_tr > best_f1:
                    best_f1, best_col, best_rule = f1_tr, col, (t, sign)
    t, sign = best_rule
    j = FEATURE_COLS.index(best_col)
    p_te = ((Xte[:, j] > t) if sign == 1 else (Xte[:, j] <= t)).astype(int)
    rows.append((f"Best single feature ({best_col})", f1_score(yte, p_te),
                 roc_auc_score(yte, p_te)))

    # 3. Logistic regression
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, ytr)
    rows.append(("Logistic regression (no graph)",
                 f1_score(yte, lr.predict(Xte_s)),
                 roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1])))

    # 4. Random forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14, random_state=seed,
        class_weight="balanced", n_jobs=-1
    ).fit(Xtr, ytr)
    rows.append(("Random forest (no graph)",
                 f1_score(yte, rf.predict(Xte)),
                 roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])))

    return pd.DataFrame(rows, columns=["model", "f1", "auc"])


def gnn_reported() -> tuple[float, float]:
    """Read the metrics recorded in the trained GraphSAGE checkpoint."""
    import torch
    ckpt = torch.load(ROOT / "models" / "graphsage_mule.pt",
                      map_location="cpu", weights_only=False)
    m = ckpt.get("metrics", {})
    return float(m.get("test_f1", float("nan"))), float(m.get("test_auc", float("nan")))


def atm_baselines() -> pd.DataFrame:
    txn = pd.read_csv(DATA / "transactions.csv", low_memory=False)
    atm = pd.read_csv(DATA / "atm_directory.csv")
    node = pd.read_csv(DATA / "node_features.csv").set_index("account_id")

    term = txn[(txn["is_terminal"] == 1) & txn["cashout_atm_id"].notna()]
    lats, lons = atm["lat"].to_numpy(), atm["long"].to_numpy()
    ids = atm["atm_id"].tolist()

    top1 = top3 = total = 0
    for _, r in term.iterrows():
        acc = r["dst_account"]
        if acc not in node.index:
            continue
        la, lo = float(node.loc[acc, "lat"]), float(node.loc[acc, "long"])
        d = _hav(la, lo, lats, lons)
        order = np.argsort(d)
        truth = r["cashout_atm_id"]
        total += 1
        if ids[int(order[0])] == truth:
            top1 += 1
        if truth in [ids[int(i)] for i in order[:3]]:
            top3 += 1

    return pd.DataFrame([
        ("Nearest ATM (distance only)", top1 / total, np.nan),
        ("Nearest 3 ATMs (distance only)", np.nan, top3 / total),
    ], columns=["model", "top1", "top3"])


def main():
    print("=" * 74)
    print("  MULE DETECTION — does the GNN earn its complexity?")
    print("=" * 74)
    df = mule_detection_baselines()
    gf1, gauc = gnn_reported()
    df.loc[len(df)] = ("GraphSAGE GNN (this system)", gf1, gauc)

    print(f'{"model":<38}{"F1":>10}{"AUC":>10}')
    print("-" * 74)
    for _, r in df.iterrows():
        auc = "—" if not np.isfinite(r["auc"]) else f'{r["auc"]:.4f}'
        print(f'{r["model"]:<38}{r["f1"]:>10.4f}{auc:>10}')

    best_naive = df.iloc[:-1]["f1"].max()
    print("-" * 74)
    print(f'  Lift over the best non-graph baseline: +{gf1 - best_naive:.4f} F1')

    print()
    print("=" * 74)
    print("  ATM PREDICTION — is it more than 'go to the nearest one'?")
    print("=" * 74)
    a = atm_baselines()
    print(f'{"model":<38}{"Top-1":>10}{"Top-3":>10}')
    print("-" * 74)
    for _, r in a.iterrows():
        t1 = "—" if not np.isfinite(r["top1"]) else f'{r["top1"]:.4f}'
        t3 = "—" if not np.isfinite(r["top3"]) else f'{r["top3"]:.4f}'
        print(f'{r["model"]:<38}{t1:>10}{t3:>10}')
    print("-" * 74)
    print("  Compare against the Top-3 accuracy printed by engine/train_xgb.py.")
    print("  Beating the distance-only Top-3 row is the whole claim.")


if __name__ == "__main__":
    main()
