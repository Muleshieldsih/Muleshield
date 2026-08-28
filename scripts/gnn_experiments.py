# -*- coding: utf-8 -*-
"""
MuleShield AI — GNN Experiment Matrix
SIH26184 | MHA / I4C

A controlled experiment runner for the mule detector. One variable changes per
experiment; everything is selected on VALIDATION; the test split is touched once,
at the end, by `--final`.

Every run records: id, change, config, val metrics, test metrics (final only),
and is appended to docs/gnn_experiments.csv so results are comparable later.

Run:
    python scripts/gnn_experiments.py --list
    python scripts/gnn_experiments.py --run baseline,selection_prauc,layers_3
    python scripts/gnn_experiments.py --all
    python scripts/gnn_experiments.py --final <experiment_id>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))
DOCS = ROOT / "docs"

from gnn_model import FEATURE_COLS  # noqa: E402
from train_gnn import load_pyg_data  # noqa: E402

LOG_PATH = DOCS / "gnn_experiments.csv"
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class FlexSAGE(nn.Module):
    """GraphSAGE with depth, width, aggregation and skip-connection as knobs."""

    def __init__(self, in_ch, hidden=64, out=64, layers=2, dropout=0.3,
                 aggr="mean", skip=False):
        super().__init__()
        self.dropout, self.skip, self.n_layers = dropout, skip, layers

        dims = [in_ch] + [hidden] * (layers - 1) + [out]
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr=aggr) for i in range(layers)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(dims[i + 1]) for i in range(layers))

        # Concatenating the raw features with the final embedding lets the head
        # use signal the message passing may have smoothed away.
        head_in = out + (in_ch if skip else 0)
        self.classifier = nn.Linear(head_in, 1)

    def encode(self, x, edge_index):
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            if i < self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        if self.skip:
            h = torch.cat([h, x], dim=1)
        return self.classifier(h)


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """Focal loss: down-weights easy negatives, which dominate at 2.8% prevalence."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS  (one variable changed from baseline each time)
# ─────────────────────────────────────────────────────────────────────────────

BASE = dict(
    hidden=64, out=64, layers=2, dropout=0.3, aggr="mean", skip=False,
    lr=0.001, weight_decay=1e-4, epochs=200, patience=40,
    loss="bce", pos_weight_scale=1.0, select="f1@0.5",
)

EXPERIMENTS = {
    "baseline":        ({}, "current configuration"),
    # A. model-selection criterion
    "select_prauc":    (dict(select="pr_auc"), "select checkpoint on PR-AUC, not F1@0.5"),
    "select_f1tuned":  (dict(select="f1@tuned"), "select on F1 at a per-epoch tuned threshold"),
    # B. class weighting
    "posw_half":       (dict(pos_weight_scale=0.5), "halve pos_weight"),
    "posw_double":     (dict(pos_weight_scale=2.0), "double pos_weight"),
    "loss_focal":      (dict(loss="focal"), "focal loss instead of weighted BCE"),
    # C/D/E. optimisation
    "lr_3e3":          (dict(lr=0.003), "higher learning rate"),
    "lr_3e4":          (dict(lr=0.0003), "lower learning rate"),
    "wd_1e3":          (dict(weight_decay=1e-3), "stronger weight decay"),
    "wd_0":            (dict(weight_decay=0.0), "no weight decay"),
    "drop_0":          (dict(dropout=0.0), "no dropout"),
    "drop_5":          (dict(dropout=0.5), "heavier dropout"),
    # F/G/H. capacity and structure
    "hidden_128":      (dict(hidden=128, out=128), "wider: 128 hidden/embedding"),
    "hidden_32":       (dict(hidden=32, out=32), "narrower: 32 hidden/embedding"),
    "layers_1":        (dict(layers=1), "single GraphSAGE layer"),
    "layers_3":        (dict(layers=3), "three layers (oversmoothing check)"),
    "aggr_max":        (dict(aggr="max"), "max aggregation"),
    "skip_concat":     (dict(skip=True), "concatenate raw features with embedding"),
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────

def tune_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.80, 0.999, 80))):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(
        f1=f1_score(y, pred, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
    )


def run_one(name, overrides, data, verbose=True):
    cfg = {**BASE, **overrides}
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_all = data.y.numpy()
    tr, va = data.train_mask.numpy(), data.val_mask.numpy()

    n_pos = int(y_all[tr].sum())
    n_neg = int((~y_all[tr].astype(bool)).sum())
    pos_w = torch.tensor([(n_neg / max(1, n_pos)) * cfg["pos_weight_scale"]],
                         dtype=torch.float)

    model = FlexSAGE(data.x.shape[1], cfg["hidden"], cfg["out"], cfg["layers"],
                     cfg["dropout"], cfg["aggr"], cfg["skip"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    best_score, best_state, best_epoch, best_thr = -1.0, None, 0, 0.5
    since = 0
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index).squeeze(-1)
        if cfg["loss"] == "focal":
            loss = focal_loss(logits[data.train_mask], data.y[data.train_mask])
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_w)(
                logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p_all = torch.sigmoid(model(data.x, data.edge_index).squeeze(-1)).numpy()
        yv, pv = y_all[va], p_all[va]

        if cfg["select"] == "pr_auc":
            score = average_precision_score(yv, pv)
            thr, _ = tune_threshold(yv, pv)
        elif cfg["select"] == "f1@tuned":
            thr, score = tune_threshold(yv, pv)
        else:
            score = f1_score(yv, (pv >= 0.5).astype(int), zero_division=0)
            thr = 0.5

        if score > best_score:
            best_score, best_epoch, best_thr = score, epoch, thr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_all = torch.sigmoid(model(data.x, data.edge_index).squeeze(-1)).numpy()

    # Threshold is always fixed from VALIDATION.
    val_thr, _ = tune_threshold(y_all[va], p_all[va])
    vm = metrics_at(y_all[va], p_all[va], val_thr)

    row = dict(
        experiment=name, change=EXPERIMENTS.get(name, ({}, ""))[1],
        params=sum(p.numel() for p in model.parameters()),
        epochs_run=epoch, best_epoch=best_epoch, threshold=val_thr,
        seconds=round(time.time() - t0, 1),
        val_f1=vm["f1"], val_precision=vm["precision"], val_recall=vm["recall"],
        val_auc=vm["auc"], val_pr_auc=vm["pr_auc"],
        **{f"cfg_{k}": v for k, v in cfg.items()},
    )
    if verbose:
        print(f'  {name:<16} val F1 {vm["f1"]:.4f} | PR-AUC {vm["pr_auc"]:.4f} '
              f'| AUC {vm["auc"]:.4f} | {row["params"]:,}p | {row["seconds"]:.0f}s')
    return row, model, p_all, val_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", type=str, default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--final", type=str, default="")
    args = ap.parse_args()

    if args.list:
        for k, (_, desc) in EXPERIMENTS.items():
            print(f"  {k:<16} {desc}")
        return

    print("Loading graph...")
    data, accounts, scaler, split = load_pyg_data(seed=SEED)
    y = data.y.numpy()
    print(f"  {data.num_nodes:,} nodes | {data.edge_index.shape[1]:,} edges "
          f"| {y.mean():.2%} mules | {data.x.shape[1]} features")
    print()

    if args.final:
        name = args.final
        overrides = EXPERIMENTS[name][0]
        print(f"FINAL RUN — {name} (test split touched once)")
        row, model, p_all, thr = run_one(name, overrides, data)
        te = data.test_mask.numpy()
        tm = metrics_at(y[te], p_all[te], thr)
        print()
        print("  TEST (frozen config, validation-selected threshold)")
        for k, v in tm.items():
            print(f"    {k:<12}{v:.4f}")
        row.update({f"test_{k}": v for k, v in tm.items()})
        _append(row)
        return

    names = list(EXPERIMENTS) if args.all else [n for n in args.run.split(",") if n]
    print(f"Running {len(names)} experiment(s), selecting on VALIDATION only")
    print("-" * 78)
    rows = []
    for n in names:
        if n not in EXPERIMENTS:
            print(f"  {n}: unknown, skipped")
            continue
        row, *_ = run_one(n, EXPERIMENTS[n][0], data)
        rows.append(row)
        _append(row)

    if rows:
        df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
        print()
        print("=" * 78)
        print("  RANKED BY VALIDATION PR-AUC (threshold-free, imbalance-robust)")
        print("=" * 78)
        print(f'{"experiment":<18}{"val F1":>9}{"PR-AUC":>9}{"AUC":>9}{"params":>10}')
        for _, r in df.iterrows():
            print(f'{r["experiment"]:<18}{r["val_f1"]:>9.4f}{r["val_pr_auc"]:>9.4f}'
                  f'{r["val_auc"]:>9.4f}{r["params"]:>10,}')


def _append(row):
    DOCS.mkdir(exist_ok=True)
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    main()
