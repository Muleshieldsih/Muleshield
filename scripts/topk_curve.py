# -*- coding: utf-8 -*-
"""
Top-K candidate containment curve for the shipped ATM ranker.

SIH26184 asks for withdrawal *locations*, and an investigator searches a small
ranked set. This measures the operational quantity directly: given K locations
to search, how often is the true cash-out ATM among them.

NO RETRAINING. The shipped checkpoint is loaded and scored on the held-out
complaints of the SAME GroupShuffleSplit(seed=42, test_size=0.20) that
train_xgb.py used, so these rows were never seen in fitting. Nothing here is
tuned; it is a read-out of a frozen model.

Containment@K counts a retrieval failure (true ATM outside the K=25 candidate
pool) as a miss, matching train_xgb.py's denominator. Inflating the pool to
rescue those would be buying containment with search cost.
"""

import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from feature_builder import FeatureBuilder
from xgb_model import MuleXGBPredictor
from metrics_io import write_metrics

K_GRID = [1, 2, 3, 4, 5, 6, 8, 10]
SEED = 42
TEST_SIZE = 0.20


def main() -> dict:
    print("[1/4] Rebuilding candidate-ranking set...")
    fb = FeatureBuilder()
    fb.load()
    X, y, groups, meta_df = fb.build_ranking_set()
    C = fb.RANK_CONTEXT_DIM
    n_atms = len(fb.atm_df) if hasattr(fb, "atm_df") else 1000
    print(f"      X={X.shape}  cashouts={len(meta_df):,}  candidates/cashout={fb.CANDIDATE_K}")
    print(f"      ATM directory: {n_atms:,}")

    print("\n[2/4] Reproducing the frozen held-out split (seed=42)...")
    cid_of_group = dict(zip(meta_df["group_id"], meta_df["complaint_id"]))
    row_complaint = np.array([cid_of_group[g] for g in groups])
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=row_complaint))
    overlap = set(row_complaint[train_idx]) & set(row_complaint[test_idx])
    assert not overlap, f"complaint leak across split: {len(overlap)}"

    X_test, y_test, g_test = X[test_idx], y[test_idx], groups[test_idx]
    order = np.argsort(g_test, kind="stable")
    X_test, y_test, g_test = X_test[order], y_test[order], g_test[order]
    print(f"      Test rows={len(X_test):,}  cashouts={len(set(g_test)):,}  overlap={len(overlap)}")

    print("\n[3/4] Scoring with the SHIPPED checkpoint (no retraining)...")
    pred = MuleXGBPredictor.load()
    log_test = MuleXGBPredictor.log_features(X_test[:, C:])
    t0 = time.perf_counter()
    scores = pred.classifier.predict(log_test)
    total_ms = (time.perf_counter() - t0) * 1000

    # ── Per-cashout ranking ───────────────────────────────────────────────────
    meta_by_gid = meta_df.set_index("group_id")
    dist_col = C
    per_group = {}
    for gid in np.unique(g_test):
        rows = np.where(g_test == gid)[0]
        o = rows[np.argsort(scores[rows])[::-1]]
        labels = y_test[o]
        hit = np.where(labels == 1)[0]
        rank = int(hit[0]) + 1 if len(hit) else None      # None => not retrieved
        d = rows[np.argsort(X_test[rows, dist_col])]
        dh = np.where(y_test[d] == 1)[0]
        per_group[int(gid)] = {
            "rank": rank,
            "dist_rank": int(dh[0]) + 1 if len(dh) else None,
            "retrieved": bool(len(hit)),
        }

    n = len(per_group)
    retrieved = sum(1 for v in per_group.values() if v["retrieved"])
    latency_per = total_ms / max(1, n)

    print(f"      Cashouts scored: {n:,}   batch {total_ms:.0f}ms  ({latency_per:.3f} ms/cashout)")
    print(f"      Retrieval ceiling (truth inside K=25 pool): {retrieved/n:.4f}")

    print("\n[4/4] Containment curve\n")
    rows_out = []
    hdr = f"{'K':>3} | {'Containment':>11} | {'Baseline':>8} | {'Reduction':>9} | {'Retr.fail':>9} | {'Rank.fail':>9}"
    print(hdr); print("-" * len(hdr))
    for K in K_GRID:
        hits = sum(1 for v in per_group.values() if v["rank"] is not None and v["rank"] <= K)
        base = sum(1 for v in per_group.values() if v["dist_rank"] is not None and v["dist_rank"] <= K)
        rfail = n - retrieved
        kfail = n - hits - rfail
        cont = hits / n
        red = 1.0 - (K / n_atms)
        rows_out.append({
            "K": K, "containment": round(cont, 4), "baseline": round(base / n, 4),
            "reduction": round(red, 4), "avg_candidates": K, "median_candidates": K,
            "retrieval_failures": rfail, "ranking_failures": kfail,
            "latency_ms": round(latency_per, 3),
        })
        print(f"{K:>3} | {cont:>11.4f} | {base/n:>8.4f} | {red:>8.2%} | "
              f"{rfail/n:>8.2%} | {kfail/n:>8.2%}")

    print(f"\n  n = {n:,} held-out cashouts | ATM directory = {n_atms:,}")
    print(f"  Retrieval ceiling = {retrieved/n:.4f} "
          f"(no K can exceed this without enlarging the pool)")

    out = {
        "n_cashouts": n, "n_atms": n_atms,
        "retrieval_ceiling": round(retrieved / n, 4),
        "latency_ms_per_cashout": round(latency_per, 3),
        "curve": rows_out,
    }
    dest = ROOT / "data" / "topk_curve.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Into the ledger, so the console renders a measured Top-5 rather than one
    # someone remembered. K=5 is the operating point the interface exposes.
    at = {r["K"]: r for r in rows_out}
    write_metrics("ranking", {
        "top1": at[1]["containment"],
        "top3": at[3]["containment"],
        "top5": at[5]["containment"],
        "top8": at[8]["containment"],
        "top10": at[10]["containment"],
        "top5_baseline_distance": at[5]["baseline"],
        "top5_search_reduction": at[5]["reduction"],
        "operating_k": 5,
        "n_atms": n_atms,
        "n_test_cashouts": n,
        "retrieval_ceiling": round(retrieved / n, 4),
        "retrieval_failure_rate": round((n - retrieved) / n, 4),
        "latency_ms_per_cashout": round(latency_per, 3),
        "curve": rows_out,
    }, source="python scripts/topk_curve.py")
    print(f"\n  wrote {dest.relative_to(ROOT)}")

    # Rank histogram for failure analysis (task 6).
    ranks = [v["rank"] for v in per_group.values() if v["rank"] is not None]
    np.save(ROOT / "data" / "_topk_ranks.npy", np.array(ranks))
    pd.DataFrame([
        {"group_id": g, **v} for g, v in per_group.items()
    ]).to_csv(ROOT / "data" / "_topk_pergroup.csv", index=False)
    return out


if __name__ == "__main__":
    main()
