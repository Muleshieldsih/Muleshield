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

BASE-RATE SENSITIVITY, AND WHY IT IS HERE
-----------------------------------------
The confusion matrix above is measured at this corpus's mule prevalence, which is
~3%. A real bank population is nowhere near that -- MuleHunter.AI-era reporting
puts mule accounts well under 1% of a book, and a low-risk district is an order of
magnitude below that again.

Precision is not a property of a classifier. It is a property of a classifier AND
a base rate, and it collapses as the base rate falls even though the model has not
changed at all. Sensitivity and specificity are what transfer; precision is what
an officer actually experiences. So this script projects the SAME measured TPR and
FPR onto the prevalences a deployment would meet, and reports what an alert queue
looks like at each: how many accounts get flagged per 100,000, and how many of
those are innocent people.

That last column is the one that matters. At 0.1% prevalence this detector flags
roughly 449 accounts per 100,000 and about 357 of them have done nothing wrong.
No amount of F1 makes that acceptable as an automated freeze, and saying so on the
console is the difference between a demo and a system somebody could deploy.

Nothing here is fitted or assumed: TPR and FPR come from the counts above, and the
prevalences are stated inputs.
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

# Prevalences a deployment would actually meet, and the posture each one earns.
#
# THE ACTIONS ARE A RECOMMENDED POSTURE, NOT A BEHAVIOUR OF THIS BUILD.
# backend/routers/freeze.py has no prevalence gate: it freezes whatever it is
# given. Printing "automated freezes are disabled below 0.5%" as though the code
# enforced it would be exactly the class of claim this project has already had to
# retract once, so the console labels this column as a policy recommendation and
# says it is not enforced. Implementing the gate is a deployment decision that
# needs a real per-bank prevalence estimate to gate on.
DEPLOYMENT_SCENARIOS = (
    # (label, prevalence, recommended action)
    ("Evaluated corpus",   None,    "Automated micro-hold"),   # None = measured
    ("High-risk district", 0.005,   "Dual-officer review"),
    ("National average",   0.001,   "Watchlist alert only"),
    ("Low-risk district",  0.0001,  "Passive audit log"),
)

AUTOMATION_FLOOR = 0.005
"""Prevalence below which no automated irreversible action is recommended.

At 0.5% this detector's precision is ~57%: roughly two in five freezes would hit
an innocent customer. There is no threshold tuning that fixes this -- it is the
base rate, not the model."""


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

    # ── Base-rate sensitivity ────────────────────────────────────────────────
    #
    # Sensitivity and specificity are the properties of the classifier that
    # transfer across populations; precision is not. Project the measured pair
    # onto each deployment prevalence:
    #
    #     precision(p) = TPR*p / (TPR*p + FPR*(1-p))
    #
    # and report the queue it implies per 100,000 accounts screened.
    tpr = tp / max(1, tp + fn)              # measured recall
    fpr = fp / max(1, fp + tn)              # measured false-positive rate
    measured_prevalence = (tp + fn) / max(1, len(test_idx))

    rows = []
    for label, prevalence, action in DEPLOYMENT_SCENARIOS:
        p = measured_prevalence if prevalence is None else float(prevalence)
        flagged = tpr * p + fpr * (1.0 - p)
        precision = (tpr * p) / flagged if flagged > 0 else 0.0
        per_100k = flagged * 100_000
        rows.append({
            "scenario": label,
            "prevalence": round(p, 6),
            "is_measured": prevalence is None,
            "precision": round(precision, 4),
            "flagged_per_100k": round(per_100k, 1),
            "true_mules_per_100k": round(tpr * p * 100_000, 1),
            # The column that decides whether this is deployable. Innocent
            # account holders caught in the queue, per 100k screened.
            "innocent_per_100k": round(fpr * (1.0 - p) * 100_000, 1),
            "recommended_action": action,
            "automation_safe": p >= AUTOMATION_FLOOR,
        })

    prevalence_payload = {
        "tpr": round(tpr, 6),
        "fpr": round(fpr, 6),
        "measured_prevalence": round(measured_prevalence, 6),
        "automation_floor": AUTOMATION_FLOOR,
        "actions_are_enforced": False,
        "_about": ("Precision projected onto deployment prevalences from the "
                   "measured TPR/FPR. recommended_action is a POLICY "
                   "RECOMMENDATION; this build does not enforce it -- "
                   "backend/routers/freeze.py has no prevalence gate."),
        "scenarios": rows,
    }
    write_metrics("detection_prevalence", prevalence_payload,
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

    print()
    print(f"  base-rate sensitivity  (TPR {tpr:.4f}, FPR {fpr:.6f} held fixed)")
    print(f"  {'scenario':<20}{'mule rate':>11}{'precision':>11}"
          f"{'flags/100k':>12}{'innocent':>10}  action")
    for r in rows:
        print(f"  {r['scenario']:<20}{r['prevalence']:>10.2%}"
              f"{r['precision']:>11.1%}{r['flagged_per_100k']:>12,.0f}"
              f"{r['innocent_per_100k']:>10,.0f}  {r['recommended_action']}")
    print()
    print(f"  Recommended automation floor: {AUTOMATION_FLOOR:.1%} prevalence.")
    print("  NOT enforced in this build -- freeze.py has no prevalence gate.")
    return payload


if __name__ == "__main__":
    main()
