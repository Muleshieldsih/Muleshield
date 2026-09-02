# -*- coding: utf-8 -*-
"""
Is the search zone's advantage over the naive zones statistically real?

    python scripts/zone_significance.py

WHY THIS SCRIPT EXISTS
----------------------
`data/metrics.json` records that the model's search zone contains the withdrawal
more often than a nearest-3 centroid given the SAME radius. It does not record
whether that gap could be chance. The console was about to promote the zone
figure to a headline card, and a headline needs to survive the question "how do
you know that is not noise?".

The project has an explicit norm here, from README's Top-K section:

    "the paired significance test that settled the question on the previous
     corpus (McNemar, no p < 0.05 at any K) has NOT been re-run here, so a
     two-point lead is not a claim we are entitled to make."

So the choice is to measure it or to say nothing. This measures it.

McNemar, because the comparison is PAIRED: the same held-out cash-out is scored
by both the model zone and the baseline zone, at the same radius. An unpaired
two-proportion test would throw away that pairing and overstate the variance.
Only the discordant pairs carry information -- the cases where one zone contained
the withdrawal and the other did not -- which is exactly what McNemar uses.

SELF-VERIFYING, AND THAT IS THE POINT
-------------------------------------
The zone geometry is recomputed here rather than imported, because train_xgb.py
computes it inline while fitting. A second implementation of a published quantity
is the defect this project has already had to retract a number for, so this script
refuses to publish anything unless its own recomputed containment matches the
ledger's `location.zone_containment` to 1e-9. If the two ever drift, this fails
loudly instead of reporting a p-value for a different computation.

No retraining. The shipped checkpoint is loaded and scored on the same frozen
GroupShuffleSplit(seed=42) that train_xgb.py used.
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from feature_builder import FeatureBuilder          # noqa: E402
from metrics_io import read_metrics, write_metrics  # noqa: E402
from xgb_model import MuleXGBPredictor              # noqa: E402

SEED = 42
TEST_SIZE = 0.20
ZONE_MASS = 0.80          # must match train_xgb.py
EARTH_R_KM = 6371.0


def _hav(alat, alon, blat, blon):
    p1, p2 = np.radians(alat), np.radians(blat)
    dphi, dlam = np.radians(blat - alat), np.radians(blon - alon)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    b = model contained it, baseline did not.
    c = baseline contained it, model did not.

    Exact rather than the chi-square approximation: the discordant count here is
    in the dozens, where the continuity-corrected chi-square is known to be
    conservative, and an exact binomial costs nothing at this size.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> dict:
    print("[1/4] Rebuilding the candidate-ranking set...")
    fb = FeatureBuilder()
    fb.load()
    X, y, groups, meta_df = fb.build_ranking_set()
    C = fb.RANK_CONTEXT_DIM

    print("[2/4] Reproducing the frozen held-out split (seed=42)...")
    cid_of_group = dict(zip(meta_df["group_id"], meta_df["complaint_id"]))
    row_complaint = np.array([cid_of_group[g] for g in groups])
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    _train_idx, test_idx = next(gss.split(X, y, groups=row_complaint))
    overlap = set(row_complaint[_train_idx]) & set(row_complaint[test_idx])
    assert not overlap, f"complaint leak across split: {len(overlap)}"

    X_test, y_test, g_test = X[test_idx], y[test_idx], groups[test_idx]
    order = np.argsort(g_test, kind="stable")
    X_test, y_test, g_test = X_test[order], y_test[order], g_test[order]

    print("[3/4] Scoring with the SHIPPED checkpoint (no retraining)...")
    pred = MuleXGBPredictor.load()
    scores = pred.classifier.predict(MuleXGBPredictor.log_features(X_test[:, C:]))

    atm_lat, atm_lon = fb._atm_lats, fb._atm_lons
    truth_of = dict(zip(meta_df["group_id"], meta_df["cashout_atm_id"]))
    node_of = dict(zip(meta_df["group_id"], zip(meta_df["lat"], meta_df["lon"])))

    model_hit, near_hit, mule_hit = [], [], []
    for gid in np.unique(g_test):
        rows = np.where(g_test == gid)[0]
        t_i = fb.atm_to_idx.get(truth_of[gid])
        if t_i is None:
            continue
        t_lat, t_lon = float(atm_lat[t_i]), float(atm_lon[t_i])

        util = scores[rows]
        order_local = np.argsort(util)[::-1]
        e = np.exp(util - util.max())
        conf = e / e.sum()

        cum = np.cumsum(conf[order_local])
        n_keep = max(1, int(np.searchsorted(cum, ZONE_MASS) + 1))
        keep = order_local[:n_keep]

        n_lat, n_lon = node_of[gid]
        d_all = _hav(float(n_lat), float(n_lon), atm_lat, atm_lon)
        near_order = np.argsort(d_all)[:len(rows)]
        dist_rank = np.argsort(np.argsort(X_test[rows, C]))
        keep_idx = near_order[[int(dist_rank[j]) for j in keep]]

        w = conf[keep] / conf[keep].sum()
        c_lat = float(np.sum(w * atm_lat[keep_idx]))
        c_lon = float(np.sum(w * atm_lon[keep_idx]))
        rad = max(float(_hav(c_lat, c_lon, atm_lat[keep_idx], atm_lon[keep_idx]).max()), 0.05)

        # All three zones get the SAME radius, so the comparison is at equal
        # search cost. A bigger zone always contains more.
        n3 = near_order[:3]
        model_hit.append(float(_hav(c_lat, c_lon, t_lat, t_lon)) <= rad)
        near_hit.append(float(_hav(float(atm_lat[n3].mean()),
                                   float(atm_lon[n3].mean()), t_lat, t_lon)) <= rad)
        mule_hit.append(float(_hav(float(n_lat), float(n_lon), t_lat, t_lon)) <= rad)

    model_hit = np.array(model_hit)
    near_hit = np.array(near_hit)
    mule_hit = np.array(mule_hit)
    n = len(model_hit)

    # ── The self-check that licenses everything below ────────────────────────
    ledger = read_metrics()
    published = float(ledger["location"]["zone_containment"])
    recomputed = float(model_hit.mean())
    if abs(recomputed - published) > 1e-9:
        raise SystemExit(
            f"REFUSING TO PUBLISH: recomputed zone containment {recomputed:.6f} "
            f"does not match the ledger's {published:.6f}. This script's zone "
            f"geometry has drifted from engine/train_xgb.py; a p-value computed "
            f"here would describe a different quantity than the one published.")
    print(f"      self-check OK: containment {recomputed:.4f} matches the ledger")

    print("[4/4] McNemar over the discordant pairs\n")
    out = {"n_cashouts": int(n), "zone_containment": round(recomputed, 6),
           "alpha": 0.05, "comparisons": []}

    for label, base, base_key in (("nearest-3 ATM centroid", near_hit, "nearest3"),
                                  ("mule location", mule_hit, "mule")):
        b = int(np.sum(model_hit & ~base))     # model only
        c = int(np.sum(~model_hit & base))     # baseline only
        p = exact_mcnemar(b, c)
        row = {
            "baseline": label,
            "baseline_key": base_key,
            "baseline_containment": round(float(base.mean()), 6),
            "model_only": b,
            "baseline_only": c,
            "discordant": b + c,
            # NOT rounded: these p-values run to 1e-27, and round(p, 8) turned
            # the strongest result in the comparison into a literal 0.0 in the
            # console payload. Kept as a float; the screen formats it.
            "p_value": float(p),
            "significant_at_05": bool(p < 0.05),
            "model_better": bool(b > c),
        }
        out["comparisons"].append(row)
        verdict = ("model better, significant" if row["significant_at_05"] and row["model_better"]
                   else "baseline better, significant" if row["significant_at_05"]
                   else "not significant")
        print(f"  vs {label}")
        print(f"     containment      {recomputed:.4f}  vs  {base.mean():.4f}")
        print(f"     discordant pairs {b} model-only / {c} baseline-only")
        print(f"     exact McNemar    p = {p:.3g}   -> {verdict}")
        print()

    write_metrics("zone_significance", out,
                  source="python scripts/zone_significance.py")
    print(f"  n = {n} held-out cash-outs. Written to data/metrics.json "
          f"section 'zone_significance'.")
    return out


if __name__ == "__main__":
    main()
