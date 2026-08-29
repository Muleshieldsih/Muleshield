# -*- coding: utf-8 -*-
"""
MuleShield AI — Seed Stability
SIH26184 | MHA / I4C

Reports every headline metric as mean ± spread across several random seeds,
rather than as a single number from one lucky split.

This exists because it caught us out. Two consecutive dataset regenerations, with
no model change at all, moved the exact-ATM Top-1 lift from +0.021 to -0.014. A
single-seed "we beat the distance baseline" was therefore not a result — it was
noise being reported as a finding. Any lift smaller than the spread below should
be described as "within noise", not as a win.

Run:
    python scripts/seed_stability.py                # 5 seeds, ranking + countdown
    python scripts/seed_stability.py --seeds 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from train_xgb import train  # noqa: E402


METRICS = [
    ("zone_containment", "Zone containment", "{:.4f}"),
    ("zone_containment_baseline_nearest3", "  baseline: nearest-3 centroid", "{:.4f}"),
    ("atm_accuracy", "Exact-ATM Top-1", "{:.4f}"),
    ("baseline_top1", "  baseline: nearest ATM", "{:.4f}"),
    ("top3_accuracy", "Exact-ATM Top-3", "{:.4f}"),
    ("baseline_top3", "  baseline: nearest 3", "{:.4f}"),
    ("time_mae_minutes", "Countdown MAE (min)", "{:.2f}"),
    ("time_baseline_mae", "  baseline: predict the mean", "{:.2f}"),
    ("time_r2", "Countdown R2", "{:.4f}"),
    ("lead_actionable_rate", "Lead time >= 15 min", "{:.4f}"),
]

# Pairs where the lift is the claim, so the claim must survive the spread.
LIFTS = [
    ("Zone containment", "zone_containment", "zone_containment_baseline_nearest3"),
    ("Exact-ATM Top-1", "atm_accuracy", "baseline_top1"),
    ("Exact-ATM Top-3", "top3_accuracy", "baseline_top3"),
]


def main(n_seeds: int) -> None:
    seeds = [42 + i for i in range(n_seeds)]
    runs = []

    for s in seeds:
        print(f"[seed {s}] training...", flush=True)
        runs.append(train(seed=s, verbose=False))

    print()
    print("=" * 78)
    print(f"  SEED STABILITY over {n_seeds} seeds: {seeds}")
    print("=" * 78)
    print(f'{"metric":<34}{"mean":>10}{"std":>10}{"min":>10}{"max":>10}')
    print("-" * 78)
    for key, label, fmt in METRICS:
        vals = np.array([r[key] for r in runs if key in r], dtype=float)
        if not len(vals):
            continue
        print(f'{label:<34}{fmt.format(vals.mean()):>10}{fmt.format(vals.std()):>10}'
              f'{fmt.format(vals.min()):>10}{fmt.format(vals.max()):>10}')

    print()
    print("=" * 78)
    print("  IS THE LIFT REAL? (a lift smaller than its own spread is not a result)")
    print("=" * 78)
    print(f'{"claim":<30}{"mean lift":>12}{"std":>10}{"verdict":>24}')
    print("-" * 78)
    for label, mk, bk in LIFTS:
        lifts = np.array([r[mk] - r[bk] for r in runs], dtype=float)
        mean, std = lifts.mean(), lifts.std()
        if mean <= 0:
            verdict = "NO - does not beat it"
        elif mean > 2 * std:
            verdict = "YES - clears 2x spread"
        elif mean > std:
            verdict = "WEAK - within 2x spread"
        else:
            verdict = "WITHIN NOISE"
        print(f'{label:<30}{mean:>+12.4f}{std:>10.4f}{verdict:>24}')

    print()
    print("  Report only the claims marked YES as wins. Report the others as")
    print("  measured-but-not-distinguishable, and say so before a judge asks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Seed stability for MuleShield metrics")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    main(args.seeds)
