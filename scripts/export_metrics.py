# -*- coding: utf-8 -*-
"""
MuleShield AI — refresh the metrics ledger from what is already on disk
SIH26184 | MHA / I4C

Rebuilds the parts of data/metrics.json that can be recovered from saved
artifacts without retraining, then projects the ledger into the JSON the
console bundles.

Recoverable here:
  detection            <- models/graphsage_mule.pt (the checkpoint stores it)
  frontend stats       <- projection of the ledger

Needs a real run:
  detection_baselines  <- python scripts/evaluate_baselines.py
  location             <- python engine/train_xgb.py

Run:
    python scripts/export_metrics.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from metrics_io import (METRICS_PATH, MetricsMissing, read_metrics,  # noqa: E402
                        write_frontend_stats, write_metrics)

GNN_CKPT = ROOT / "models" / "graphsage_mule.pt"


def main() -> int:
    print("=" * 66)
    print("  Refreshing data/metrics.json from saved artifacts")
    print("=" * 66)

    if GNN_CKPT.exists():
        ckpt = torch.load(GNN_CKPT, map_location="cpu", weights_only=False)
        write_metrics("detection", ckpt["metrics"],
                      source="python engine/train_gnn.py")
        print(f"  [OK] detection            <- {GNN_CKPT.name} "
              f"(F1 {ckpt['metrics']['test_f1']:.4f})")
    else:
        print(f"  [--] detection            missing {GNN_CKPT.name}; "
              f"run: python engine/train_gnn.py")

    ledger = read_metrics()
    for section, cmd in (("detection_baselines", "python scripts/evaluate_baselines.py"),
                         ("location", "python engine/train_xgb.py")):
        if section in ledger:
            print(f"  [OK] {section:<21} already measured "
                  f"({ledger[section].get('measured_utc', 'unknown time')})")
        else:
            print(f"  [--] {section:<21} not measured; run: {cmd}")

    print()
    try:
        stats = write_frontend_stats()
    except MetricsMissing as exc:
        print(f"  [--] Console stats not written.\n       {exc}")
        print(f"\n  Ledger: {METRICS_PATH}")
        return 1

    print("  [OK] frontend/src/data/model_stats.json written:")
    print(f"       zone containment {stats['zoneContainment']:.3f} "
          f"vs {stats['zoneBaselineNearest3']:.3f} nearest-3")
    print(f"       countdown MAE    {stats['countdownMae']:.2f} min "
          f"vs {stats['countdownBaseline']:.2f} min")
    print(f"       mule F1          {stats['gnnF1']:.4f} "
          f"vs {stats['gnnBaseline']:.4f} ({stats['gnnBaselineModel']})")
    print(f"\n  Ledger: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
