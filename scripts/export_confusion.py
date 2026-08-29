# -*- coding: utf-8 -*-
"""
Exact confusion matrix for the shipped detector, into the metrics ledger.

    python scripts/export_confusion.py

The Model Performance screen states how many accounts the detector got wrong in
each direction. Those counts could be algebraically recovered from the stored
precision, recall and accuracy, but only to within rounding -- and a page whose
whole purpose is to be checkable should not print a reconstructed number as if
it were counted. So this loads the shipped checkpoint, replays its own frozen
test split, and counts.

No retraining, no tuning. The threshold is the one already chosen on validation
and stored in the checkpoint.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from gnn_model import build_model          # noqa: E402
from train_gnn import load_pyg_data        # noqa: E402
from metrics_io import write_metrics       # noqa: E402

CKPT = ROOT / "models" / "graphsage_mule.pt"


def main() -> dict:
    if not CKPT.exists():
        raise SystemExit("no checkpoint — run: python engine/train_gnn.py")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    threshold = float(ckpt["metrics"]["threshold"]) if "metrics" in ckpt else 0.5
    test_idx = np.array(ckpt["split"]["test_idx"])

    data, _ids, _scaler, _meta = load_pyg_data()
    model = build_model(in_channels=data.x.shape[1])
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        prob = torch.sigmoid(model(data.x, data.edge_index).squeeze(-1)).numpy()

    y = data.y.numpy()[test_idx]
    pred = (prob[test_idx] >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    payload = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "n_test_accounts": int(len(test_idx)),
        "actual_mules": tp + fn,
        "threshold": round(threshold, 4),
        # What an alert queue would actually feel like: of everything the model
        # flags, how much is worth an analyst's time.
        "flagged_total": tp + fp,
        "flagged_precision": round(tp / max(1, tp + fp), 4),
        "missed_rate": round(fn / max(1, tp + fn), 4),
    }

    write_metrics("detection_confusion", payload,
                  source="python scripts/export_confusion.py")

    print(f"  threshold        {threshold:.4f}")
    print(f"  test accounts    {len(test_idx):,}")
    print(f"  actual mules     {tp + fn:,}")
    print()
    print(f"  {'':>18}{'pred mule':>12}{'pred clean':>12}")
    print(f"  {'actual mule':>18}{tp:>12,}{fn:>12,}")
    print(f"  {'actual clean':>18}{fp:>12,}{tn:>12,}")
    print()
    print(f"  of {tp + fp:,} flagged, {tp:,} are mules ({payload['flagged_precision']:.1%})")
    print(f"  {fn:,} mules missed ({payload['missed_rate']:.1%} of all mules)")
    return payload


if __name__ == "__main__":
    main()
