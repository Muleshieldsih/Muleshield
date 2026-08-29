# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2a: GraphSAGE Training Script
SIH26184 | MHA / I4C

Trains GraphSAGEMule as binary node classifier (mule=1 / clean=0).
Uses full-batch training on the synthetic graph from Phase 1.

Usage:
    python engine/train_gnn.py
    python engine/train_gnn.py --epochs 200 --lr 0.001 --seed 42

Output:
    models/graphsage_mule.pt  -- best model weights (by val F1)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

# ── Resolve project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from gnn_model import (GraphSAGEMule, FEATURE_COLS, IN_CHANNELS, OUT_CHANNELS,
                       derive_features)
from metrics_io import write_metrics

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "graphsage_mule.pt"


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_pyg_data(
    transactions_path: Path = DATA_DIR / "transactions.csv",
    node_features_path: Path = DATA_DIR / "node_features.csv",
    seed: int = 42,
) -> tuple[Data, list[str], StandardScaler]:
    """
    Build a PyTorch Geometric Data object from Phase 1 CSVs.

    Returns:
        (data, account_ids, scaler)
        - data:         PyG Data with x, edge_index, y, train/val/test masks
        - account_ids:  Ordered list of account IDs (index -> account_id)
        - scaler:       Fitted StandardScaler for inference-time normalization
    """
    node_df = pd.read_csv(node_features_path)
    # Ratios and logs the raw columns cannot express to an axis-aligned split.
    node_df = derive_features(node_df)
    txn_df = pd.read_csv(transactions_path)

    # ── Build account_id -> integer index mapping ─────────────────────────────
    # Include all accounts: dst_accounts (mules) AND src_accounts (victims/clean)
    all_accounts = pd.concat([
        txn_df["src_account"], txn_df["dst_account"]
    ]).unique().tolist()

    # Merge with node_features (some src accounts may not have full node features)
    account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
    N = len(all_accounts)

    # ── Build node feature matrix [N, 7] ─────────────────────────────────────
    node_lookup = node_df.set_index("account_id")
    x_list = []
    y_list = []

    for acc in all_accounts:
        if acc in node_lookup.index:
            row = node_lookup.loc[acc]
            feats = [float(row.get(col, 0.0)) for col in FEATURE_COLS]
            label = int(row.get("is_mule_label", 0))
        else:
            # Account exists in transactions but not in node_features (edge case)
            feats = [0.0] * IN_CHANNELS
            label = 0
        x_list.append(feats)
        y_list.append(label)

    x_raw = np.array(x_list, dtype=np.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float)

    # ── Split FIRST, so normalisation can be fitted on training nodes only ────
    # Fitting the scaler over all N nodes lets validation and test feature
    # distributions bleed into the transform. Stratified so the mule rate is the
    # same in every split - it matters much more once prevalence is realistic.
    N = len(y_list)
    y_arr = np.asarray(y_list)
    idx_all = np.arange(N)
    train_idx, hold_idx = train_test_split(
        idx_all, train_size=0.70, random_state=seed, stratify=y_arr
    )
    val_idx, test_idx = train_test_split(
        hold_idx, train_size=0.50, random_state=seed, stratify=y_arr[hold_idx]
    )

    scaler = StandardScaler()
    scaler.fit(x_raw[train_idx])
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float)

    # ── Build edge_index [2, E] ───────────────────────────────────────────────
    #
    # Both directions. SAGEConv aggregates over a node's INCOMING edges only, so
    # a directed src->dst graph means an account never sees who it paid - only
    # who paid it. Mule behaviour is defined by what an account does with money
    # after receiving it, so the out-neighbourhood is at least as informative as
    # the in-neighbourhood. Reverse edges are appended so message passing reaches
    # both sides, with direction preserved as an edge feature.
    src_indices = [account_to_idx[s] for s in txn_df["src_account"]]
    dst_indices = [account_to_idx[d] for d in txn_df["dst_account"]]

    fwd = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    rev = torch.tensor([dst_indices, src_indices], dtype=torch.long)
    edge_index = torch.cat([fwd, rev], dim=1)

    # ── Train / Val / Test masks (70 / 15 / 15, stratified above) ────────────
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=N,
    )

    return data, all_accounts, scaler, {
        "train": train_idx, "val": val_idx, "test": test_idx
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, data, optimizer, pos_weight: torch.Tensor) -> float:
    """Single training epoch. Returns training loss."""
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index).squeeze(-1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model, data, mask, threshold: float = 0.5,
             pos_weight: torch.Tensor | None = None) -> dict:
    """
    Evaluate on a given mask.

    `pos_weight` must be the SAME tensor used in training, otherwise the reported
    loss is on a different scale from the training loss and cannot be compared to
    it. A previous version recomputed it per-mask as n_pos/n_neg - the reciprocal
    of the training weight - which made the printed val/test loss meaningless.
    """
    model.eval()
    logits = model(data.x, data.edge_index).squeeze(-1)
    y_true = data.y[mask].numpy()
    y_prob = torch.sigmoid(logits[mask]).numpy()
    y_pred = (y_prob >= threshold).astype(int)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = float(loss_fn(logits[mask], data.y[mask]))

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        auc, ap = 0.0, 0.0

    return {"loss": loss, "f1": f1, "acc": acc, "auc": auc,
            "precision": prec, "recall": rec, "pr_auc": ap}


@torch.no_grad()
def best_threshold(model, data, mask) -> float:
    """
    Choose the decision threshold that maximises F1 on the given mask.

    The threshold was previously hardcoded at 0.5, which is only optimal when the
    classes are balanced and the scores are calibrated. Neither holds here, so
    this is the cheapest genuine gain available - and it is selected on
    VALIDATION, never on test.
    """
    model.eval()
    logits = model(data.x, data.edge_index).squeeze(-1)
    y_true = data.y[mask].numpy()
    y_prob = torch.sigmoid(logits[mask]).numpy()

    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99))):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train(
    epochs: int = 200,
    lr: float = 0.001,
    dropout: float = 0.3,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Full GNN training pipeline.

    Returns:
        dict with best val F1, test F1, test AUC
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    if verbose:
        print("[1/4] Loading graph data...")
    data, account_ids, scaler, split_idx = load_pyg_data(seed=seed)

    # Imbalance weighting must be measured on training labels alone; computing
    # it over the full label vector reads validation and test class balance.
    y_train = data.y[data.train_mask]
    n_mule = int(y_train.sum().item())
    n_clean = int(len(y_train) - n_mule)
    if verbose:
        print(f"      Nodes: {len(data.y)} | Train mules: {n_mule} | Train clean: {n_clean}")
        print(f"      Edges: {data.edge_index.shape[1]}")
        print(f"      Train: {data.train_mask.sum()} | Val: {data.val_mask.sum()} | Test: {data.test_mask.sum()}")

    # Class imbalance: weight mule class more
    pos_weight = torch.tensor([n_clean / max(1, n_mule)], dtype=torch.float)
    if verbose:
        print(f"      pos_weight (mule class): {pos_weight.item():.3f}")

    # Model
    if verbose:
        print("\n[2/4] Initializing GraphSAGEMule...")
    model = GraphSAGEMule(dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=20, factor=0.5, min_lr=1e-5
    )

    # Training
    if verbose:
        print(f"\n[3/4] Training for {epochs} epochs...")

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    early_stop_patience = 40

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, data, optimizer, pos_weight)
        val_metrics = evaluate(model, data, data.val_mask, pos_weight=pos_weight)
        scheduler.step(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:>4d} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValF1={val_metrics['f1']:.4f} | "
                f"ValAUC={val_metrics['auc']:.4f}"
            )

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    # Restore best model and evaluate on test set
    if best_state:
        model.load_state_dict(best_state)

    # ── Probability calibration on VALIDATION ─────────────────────────────────
    # The raw sigmoid is badly calibrated: scores saturate, so the console shows
    # "100.0% risk" for every account in a traced chain, which reads as fabricated
    # even when the ranking is correct. Isotonic regression fitted on validation
    # maps scores to frequencies that mean what they say. Fitted on validation
    # only, so it never sees test.
    from sklearn.isotonic import IsotonicRegression
    model.eval()
    with torch.no_grad():
        _p_all = torch.sigmoid(model(data.x, data.edge_index).squeeze(-1)).numpy()
    _va = data.val_mask.numpy()
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(_p_all[_va], data.y.numpy()[_va])

    # Threshold is tuned on validation only, then frozen for the test report.
    tuned_threshold = best_threshold(model, data, data.val_mask)
    test_metrics = evaluate(model, data, data.test_mask,
                            threshold=tuned_threshold, pos_weight=pos_weight)
    test_at_half = evaluate(model, data, data.test_mask,
                            threshold=0.5, pos_weight=pos_weight)
    if verbose:
        print(f"      Tuned threshold (val): {tuned_threshold:.4f}")
        print(f"      Test F1 @0.5         : {test_at_half['f1']:.4f}")
        print(f"      Test F1 @tuned       : {test_metrics['f1']:.4f}")
        print(f"      Test precision/recall: {test_metrics['precision']:.4f}"
              f" / {test_metrics['recall']:.4f}")
        print(f"      Test PR-AUC          : {test_metrics['pr_auc']:.4f}")
    if verbose:
        print(f"\n  Best Val F1   : {best_val_f1:.4f}")
        print(f"  Test F1       : {test_metrics['f1']:.4f}")
        print(f"  Test AUC      : {test_metrics['auc']:.4f}")
        print(f"  Test Accuracy : {test_metrics['acc']:.4f}")

    # Save model
    if verbose:
        print(f"\n[4/4] Saving model to {MODEL_PATH}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "account_ids": account_ids,
        "feature_cols": FEATURE_COLS,
        "hyperparams": {"epochs": epochs, "lr": lr, "dropout": dropout,
                        "seed": seed, "threshold": tuned_threshold},
        # The split is saved so scripts/evaluate_baselines.py can score every
        # baseline on exactly these test nodes. Comparing a model measured on one
        # split against baselines measured on another is not a comparison.
        # Isotonic mapping from raw sigmoid to calibrated probability.
        "calibration": {
            "x": calibrator.X_thresholds_.tolist(),
            "y": calibrator.y_thresholds_.tolist(),
        },
        "split": {
            "train_idx": [int(i) for i in split_idx["train"]],
            "val_idx": [int(i) for i in split_idx["val"]],
            "test_idx": [int(i) for i in split_idx["test"]],
        },
        "metrics": {
            "best_val_f1": best_val_f1,
            "threshold": tuned_threshold,
            "test_f1": test_metrics["f1"],
            "test_f1_at_0.5": test_at_half["f1"],
            "test_auc": test_metrics["auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_accuracy": test_metrics["acc"],
        },
    }, MODEL_PATH)
    if verbose:
        print(f"      [OK] Saved: {MODEL_PATH}")

    return {
        "best_val_f1": best_val_f1,
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "model": model,
        "data": data,
        "account_ids": account_ids,
        "scaler": scaler,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuleShield AI -- Train GraphSAGE")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  MuleShield AI - GraphSAGE Training  |  SIH26184")
    print("=" * 60)

    results = train(
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    f1 = results["test_f1"]
    target_met = "PASS" if f1 >= 0.85 else "FAIL (target: F1 > 0.85)"
    print(f"  Final Test F1: {f1:.4f}  [{target_met}]")
    print("=" * 60)

    # ── Publish to the ledger ────────────────────────────────────────────────
    # The checkpoint already carries these; the ledger is what the deck card and
    # the console read, so they land there too rather than being retyped.
    ckpt_metrics = torch.load(MODEL_PATH, map_location="cpu",
                              weights_only=False)["metrics"]
    write_metrics("detection", ckpt_metrics, source="python engine/train_gnn.py")
    print("\n  [OK] Wrote detection metrics to data/metrics.json")
