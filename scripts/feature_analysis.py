# -*- coding: utf-8 -*-
"""
MuleShield AI — Feature Correlation Heatmap & Signal Analysis
SIH26184 | MHA / I4C

Answers "which features actually carry signal, and which are redundant?" so the
feature set is chosen from evidence rather than from habit.

Produces:
  1. Signal ranking      — each feature's association with the mule label
                           (point-biserial r, mutual information, AUC alone)
  2. Correlation heatmap — feature-vs-feature, rendered in the terminal and
                           written to docs/feature_heatmap.png
  3. Redundancy report   — near-duplicate pairs (|r| >= 0.85), where one of the
                           pair can be dropped without losing information
  4. Recommendation      — features to drop, and derived features worth adding

Run:
    python scripts/feature_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))
DATA = ROOT / "data"
DOCS = ROOT / "docs"

from gnn_model import FEATURE_COLS, derive_features  # noqa: E402

REDUNDANCY_THRESHOLD = 0.85
WEAK_SIGNAL_AUC = 0.53          # a feature this close to 0.50 carries ~nothing

# Terminal heatmap ramp, low correlation -> high
RAMP = " .:-=+*#%@"


def shade(v: float) -> str:
    """Map |correlation| in [0,1] to a single ramp character."""
    i = int(np.clip(abs(v), 0, 1) * (len(RAMP) - 1))
    return RAMP[i]


def main() -> None:
    node = pd.read_csv(DATA / "node_features.csv")
    node = derive_features(node)
    X = node[FEATURE_COLS].astype(float)
    y = node["is_mule_label"].to_numpy()

    print("=" * 96)
    print("  FEATURE SIGNAL ANALYSIS")
    print(f"  {len(node):,} accounts | {y.mean():.2%} mules | {len(FEATURE_COLS)} features")
    print("=" * 96)

    # ── 1. Signal ranking ────────────────────────────────────────────────────
    mi = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    rows = []
    for i, col in enumerate(FEATURE_COLS):
        v = X[col].to_numpy()
        r = float(np.corrcoef(v, y)[0, 1]) if v.std() > 0 else 0.0
        try:
            auc = roc_auc_score(y, v)
            auc = max(auc, 1 - auc)          # direction-agnostic
        except ValueError:
            auc = 0.5
        rows.append((col, r, float(mi[i]), auc))

    rows.sort(key=lambda t: t[3], reverse=True)

    print()
    print(f'{"feature":<26}{"corr(r)":>10}{"mutual info":>14}{"AUC alone":>12}   signal')
    print("-" * 96)
    for col, r, m, auc in rows:
        bar = "#" * int((auc - 0.5) * 60)
        flag = "  <- weak" if auc < WEAK_SIGNAL_AUC else ""
        print(f"{col:<26}{r:>+10.3f}{m:>14.4f}{auc:>12.4f}   {bar}{flag}")

    weak = [c for c, _, _, a in rows if a < WEAK_SIGNAL_AUC]

    # ── 2. Correlation heatmap ───────────────────────────────────────────────
    corr = X.corr().to_numpy()
    labels = [c[:14] for c in FEATURE_COLS]

    print()
    print("=" * 96)
    print("  CORRELATION HEATMAP  (blank=0.0  .:-=+*#%@=1.0, absolute value)")
    print("=" * 96)
    width = max(len(l) for l in labels) + 1
    print(" " * width + "".join(f"{i:>3}" for i in range(len(labels))))
    for i, lab in enumerate(labels):
        cells = "".join(f"  {shade(corr[i, j])}" for j in range(len(labels)))
        print(f"{lab:<{width}}{cells}   {i}")

    # ── 3. Redundancy ────────────────────────────────────────────────────────
    print()
    print("=" * 96)
    print(f"  REDUNDANT PAIRS  (|r| >= {REDUNDANCY_THRESHOLD})")
    print("=" * 96)
    dupes = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i + 1, len(FEATURE_COLS)):
            if abs(corr[i, j]) >= REDUNDANCY_THRESHOLD:
                a, b = FEATURE_COLS[i], FEATURE_COLS[j]
                auc_a = next(x[3] for x in rows if x[0] == a)
                auc_b = next(x[3] for x in rows if x[0] == b)
                drop = a if auc_a < auc_b else b
                dupes.append((a, b, corr[i, j], drop))
                print(f"  {a:<26} ~ {b:<26} r={corr[i, j]:+.3f}   drop: {drop}")
    if not dupes:
        print("  none - no feature pair is a near-duplicate of another")

    # ── 4. Recommendation ────────────────────────────────────────────────────
    print()
    print("=" * 96)
    print("  RECOMMENDATION")
    print("=" * 96)
    if weak:
        print(f"  Weak on their own (AUC < {WEAK_SIGNAL_AUC}): {', '.join(weak)}")
        print("    Keep only if they interact - a GNN can use a feature that is")
        print("    uninformative alone but discriminative in a neighbourhood.")
    else:
        print("  Every feature clears the weak-signal bar on its own.")
    if dupes:
        print(f"  Redundant, safe to drop: {', '.join(sorted({d for *_, d in dupes}))}")

    print()
    print("  Derived features worth adding (all bank-observable, none label-derived):")
    print("    in_out_amount_ratio   = total_sent / (total_received + 1)")
    print("    counterparty_ratio    = distinct_senders / (distinct_receivers + 1)")
    print("    avg_amount_per_txn_in = total_received / (in_degree + 1)")
    print("    dwell_log             = log1p(median_dwell_seconds)   [heavy right tail]")
    print("    activity_per_day      = (in_degree+out_degree) / (account_age_days + 1)")

    # ── PNG heatmap for the deck ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        DOCS.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(FEATURE_COLS, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(corr[i, j]) > 0.55 else "black")
        ax.set_title("MuleShield AI — GNN feature correlation\nSIH26184 | MHA / I4C")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        fig.tight_layout()
        out = DOCS / "feature_heatmap.png"
        fig.savefig(out, dpi=150)
        print()
        print(f"  Heatmap written to {out}")
    except ImportError:
        print()
        print("  (matplotlib not installed - PNG heatmap skipped)")


if __name__ == "__main__":
    main()
