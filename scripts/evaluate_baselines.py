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
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))
DATA = ROOT / "data"

from gnn_model import FEATURE_COLS, derive_features  # noqa: E402


def _hav(lat, lon, lats, lons):
    r = 6371.0
    p1, p2 = np.radians(lat), np.radians(lats)
    dphi = np.radians(lats - lat)
    dlam = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))



def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, ks=(100, 500, 1000)) -> dict:
    """
    Precision within the top-K highest-scoring accounts.

    At realistic prevalence (~1.5%) F1 stops being the operative question. An
    investigation team can work a fixed number of alerts a day, so what matters
    is: of the K accounts we surface, how many are actually mules? Reported
    alongside lift over random, which is what makes the number interpretable -
    Precision@100 of 0.30 is poor at 20% prevalence and excellent at 1.5%.
    """
    order = np.argsort(y_score)[::-1]
    base_rate = float(y_true.mean())
    out = {}
    for k in ks:
        k_eff = min(k, len(order))
        hits = float(y_true[order[:k_eff]].sum())
        prec = hits / max(1, k_eff)
        out[k] = {
            "precision": prec,
            "recall": hits / max(1.0, float(y_true.sum())),
            "lift": prec / base_rate if base_rate > 0 else float("nan"),
        }
    return out


def _load_gnn_checkpoint():
    """Load the GraphSAGE checkpoint: its split, metrics and threshold."""
    import torch
    return torch.load(ROOT / "models" / "graphsage_mule.pt",
                      map_location="cpu", weights_only=False)


def mule_detection_baselines(ckpt) -> pd.DataFrame:
    """
    Score every non-graph baseline on the GNN's OWN test nodes.

    A previous version split `node_features.csv` independently for the baselines
    while reading the GNN's score out of its checkpoint - so the reported "lift
    over the best non-graph baseline" compared two different test sets and meant
    nothing. The split is now persisted at training time and reused verbatim here.
    """
    node = pd.read_csv(DATA / "node_features.csv")
    node = derive_features(node)

    # The checkpoint's split indices refer to `account_ids` order, so align on it.
    order = {a: i for i, a in enumerate(ckpt["account_ids"])}
    node = node[node["account_id"].isin(order)].copy()
    node["_pos"] = node["account_id"].map(order)
    node = node.sort_values("_pos").reset_index(drop=True)

    X = node[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = node["is_mule_label"].to_numpy()

    split = ckpt["split"]
    tr = np.array(split["train_idx"] + split["val_idx"])   # baselines get train+val
    te = np.array(split["test_idx"])

    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    def row(name, y_pred, y_score=None):
        return (
            name,
            f1_score(yte, y_pred, zero_division=0),
            roc_auc_score(yte, y_score) if y_score is not None else float("nan"),
            average_precision_score(yte, y_score) if y_score is not None else float("nan"),
            precision_score(yte, y_pred, zero_division=0),
            recall_score(yte, y_pred, zero_division=0),
        )

    rows = [row("Majority class (all mule)", np.ones_like(yte))]
    scores_by_model: dict[str, np.ndarray] = {}

    # Best single feature: threshold chosen on TRAIN, scored on TEST.
    best_f1, best_col, best_rule = 0.0, None, None
    for j, col in enumerate(FEATURE_COLS):
        v_tr = Xtr[:, j]
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
    score = Xte[:, j] if sign == 1 else -Xte[:, j]
    rows.append(row(f"Best single feature ({best_col})",
                    ((Xte[:, j] > t) if sign == 1 else (Xte[:, j] <= t)).astype(int),
                    score))
    scores_by_model[f"Best single feature ({best_col})"] = score

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, ytr)
    lr_score = lr.predict_proba(Xte_s)[:, 1]
    rows.append(row("Logistic regression (no graph)", lr.predict(Xte_s), lr_score))
    scores_by_model["Logistic regression (no graph)"] = lr_score

    rf = RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42,
                                class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)
    rf_score = rf.predict_proba(Xte)[:, 1]
    rows.append(row("Random forest (no graph)", rf.predict(Xte), rf_score))
    scores_by_model["Random forest (no graph)"] = rf_score

    return (pd.DataFrame(rows, columns=["model", "f1", "auc", "pr_auc",
                                        "precision", "recall"]),
            scores_by_model, yte)


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
    ckpt = _load_gnn_checkpoint()
    m = ckpt["metrics"]

    print("=" * 92)
    print("  MULE DETECTION — does the GNN earn its complexity?")
    print("  (all rows scored on the SAME held-out nodes as the GNN)")
    print("=" * 92)

    df, scores_by_model, yte = mule_detection_baselines(ckpt)
    df.loc[len(df)] = (
        "GraphSAGE GNN (this system)",
        m["test_f1"], m["test_auc"], m.get("test_pr_auc", float("nan")),
        m.get("test_precision", float("nan")), m.get("test_recall", float("nan")),
    )

    hdr = f'{"model":<40}{"F1":>9}{"AUC":>9}{"PR-AUC":>9}{"Prec":>9}{"Recall":>9}'
    print(hdr)
    print("-" * 92)
    for _, r in df.iterrows():
        def fmt(v):
            return "—" if not np.isfinite(v) else f"{v:.4f}"
        print(f'{r["model"]:<40}{fmt(r["f1"]):>9}{fmt(r["auc"]):>9}'
              f'{fmt(r["pr_auc"]):>9}{fmt(r["precision"]):>9}{fmt(r["recall"]):>9}')

    best_naive = df.iloc[:-1]["f1"].max()
    print("-" * 92)
    print(f'  Lift over the best non-graph baseline: {m["test_f1"] - best_naive:+.4f} F1')
    print(f'  Decision threshold (tuned on validation): {m.get("threshold", 0.5):.4f}')

    # ── Precision@K: what an alert budget actually buys ──────────────────────
    base_rate = float(yte.mean())
    print()
    print("=" * 92)
    print("  ALERT BUDGET - of the top K accounts we surface, how many are mules?")
    print(f"  (base rate in this population: {base_rate:.2%})")
    print("=" * 92)
    print(f'{"model":<40}{"P@100":>11}{"P@500":>11}{"P@1000":>11}{"lift@100":>11}')
    print("-" * 92)
    for name, sc in scores_by_model.items():
        pk = precision_at_k(yte, sc)
        print(f'{name:<40}{pk[100]["precision"]:>11.3f}{pk[500]["precision"]:>11.3f}'
              f'{pk[1000]["precision"]:>11.3f}{pk[100]["lift"]:>10.1f}x')
    print("-" * 92)
    print("  The GNN's own scores are not re-derived here; run engine/train_gnn.py")
    print("  for its precision/recall at the tuned threshold.")

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
