# -*- coding: utf-8 -*-
"""
MuleShield AI — Exploratory Data Analysis
SIH26184 | MHA / I4C

A judge's first question about a synthetic dataset is "how do you know it is
realistic, and how do you know your model is not just reading the label?" This
script answers both from the data itself, in one page of output.

Sections:
  1. Population        — accounts, prevalence, archetype mix, activity density
  2. Leakage guard     — can any single feature reproduce the label?
  3. Class overlap     — do mules and legitimate accounts actually overlap?
  4. Graph structure   — is there ring structure for a GNN to exploit?
  5. Cashout geography — is the ATM choice more than "go to the nearest one"?
  6. Timing            — is the countdown target learnable, and how urgent?
  7. Attainable ceiling — what could a PERFECT classifier score here?

Run:
    python scripts/eda_report.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))
DATA = ROOT / "data"

from gnn_model import FEATURE_COLS  # noqa: E402

W = 78


def head(title: str) -> None:
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def hav(alat, alon, blat, blon):
    r = 6371.0
    p1, p2 = np.radians(alat), np.radians(blat)
    dphi, dlam = np.radians(blat - alat), np.radians(blon - alon)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main() -> None:
    node = pd.read_csv(DATA / "node_features.csv")
    txn = pd.read_csv(DATA / "transactions.csv", low_memory=False)
    atm = pd.read_csv(DATA / "atm_directory.csv")
    comp = pd.read_csv(DATA / "victim_complaints.csv")

    y = node["is_mule_label"].to_numpy()
    mule, clean = node[y == 1], node[y == 0]

    print("=" * W)
    print("  MULESHIELD AI — EXPLORATORY DATA ANALYSIS")
    print("  SIH26184 | Ministry of Home Affairs / I4C")
    print("=" * W)

    # ── 1. Population ────────────────────────────────────────────────────────
    head("1. POPULATION")
    activity = node["in_degree"] + node["out_degree"]
    print(f"  accounts                 : {len(node):,}")
    print(f"  labelled mules           : {int(y.sum()):,}  ({y.mean():.2%})")
    print(f"  complaints               : {len(comp):,}")
    print(f"  transactions             : {len(txn):,}"
          f"  ({int((txn['is_fraud'] == 1).sum()):,} fraud /"
          f" {int((txn['is_fraud'] == 0).sum()):,} legitimate)")
    print(f"  ATMs                     : {len(atm):,}")
    print()
    print(f"  transactions per account : mean {activity.mean():.1f},"
          f" median {activity.median():.0f}, p95 {activity.quantile(0.95):.0f}")
    print(f"  accounts with <=2 txns   : {(activity <= 2).mean():.1%}")
    print(f"  accounts never forwarding: {(node['median_dwell_seconds'] > 1e7).mean():.1%}")
    print()
    print("  Real mule prevalence is under 1%; this dataset sits higher so that the")
    print("  positive class is large enough to train and evaluate on. Metrics are")
    print("  therefore reported with Precision@K, not F1 alone.")

    # ── 2. Leakage guard ─────────────────────────────────────────────────────
    head("2. LEAKAGE GUARD — can one feature reproduce the label?")
    worst, worst_col = 0.0, None
    for col in FEATURE_COLS:
        v = node[col].to_numpy(dtype=float)
        for t in np.unique(np.quantile(v, np.linspace(0.02, 0.98, 40))):
            for pred in ((v > t).astype(int), (v <= t).astype(int)):
                if pred.sum() in (0, len(pred)):
                    continue
                f = f1_score(y, pred, zero_division=0)
                if f > worst:
                    worst, worst_col = f, col
    print(f"  best single-feature F1   : {worst:.4f}   (feature: {worst_col})")
    print(f"  guard threshold          : 0.95")
    print(f"  verdict                  : {'PASS' if worst < 0.95 else 'FAIL - LEAKAGE'}")
    print()
    print("  An earlier version of this dataset scored 1.0000 here: the label was")
    print("  defined as 'received money' and non-mules had total_received zeroed,")
    print("  so the label was a literal copy of a feature.")

    # ── 3. Class overlap ─────────────────────────────────────────────────────
    head("3. CLASS OVERLAP — mules vs legitimate accounts (p5-p95)")
    print(f'  {"feature":<24}{"mule":>24}{"legitimate":>24}')
    for col in ["median_dwell_seconds", "passthrough_ratio", "account_age_days",
                "night_txn_ratio", "in_degree", "total_received"]:
        m5, m95 = np.percentile(mule[col], [5, 95])
        c5, c95 = np.percentile(clean[col], [5, 95])
        print(f'  {col:<24}{f"{m5:,.1f} - {m95:,.1f}":>24}{f"{c5:,.1f} - {c95:,.1f}":>24}')
    print()
    print("  The ranges overlap by design. `transit_business` accounts (payment")
    print("  aggregators, trading firms) legitimately sweep almost everything they")
    print("  receive within minutes, exactly like a mule.")

    # ── 4. Graph structure ───────────────────────────────────────────────────
    head("4. GRAPH STRUCTURE — is there anything for a GNN to exploit?")
    import networkx as nx
    e = pd.read_csv(DATA / "graph_edges.csv")
    g = nx.from_pandas_edgelist(e, "src_account", "dst_account",
                                create_using=nx.DiGraph)
    comps = sorted((len(c) for c in nx.weakly_connected_components(g)), reverse=True)
    print(f"  nodes / edges            : {g.number_of_nodes():,} / {g.number_of_edges():,}")
    print(f"  connected components     : {len(comps):,}")
    print(f"  largest component        : {comps[0]:,} nodes ({comps[0]/g.number_of_nodes():.1%})")
    print(f"  mean degree              : {2*g.number_of_edges()/g.number_of_nodes():.2f}")
    print()
    print("  Mule accounts are reused across complaints by syndicate, which is what")
    print("  creates ring structure. Minting fresh accounts per chain left the graph")
    print("  as thousands of disconnected 5-node stubs with nothing to learn from.")

    # ── 5. Cashout geography ─────────────────────────────────────────────────
    head("5. CASHOUT GEOGRAPHY — is it more than 'the nearest ATM'?")
    term = txn[(txn["is_terminal"] == 1) & txn["cashout_atm_id"].notna()]
    lut = node.set_index("account_id")
    lats, lons = atm["lat"].to_numpy(), atm["long"].to_numpy()
    ids = atm["atm_id"].tolist()
    sample = term.sample(min(2000, len(term)), random_state=0)

    ranks, dists = [], []
    for _, r in sample.iterrows():
        acc = r["dst_account"]
        if acc not in lut.index:
            continue
        d = hav(float(lut.loc[acc, "lat"]), float(lut.loc[acc, "long"]), lats, lons)
        order = np.argsort(d)
        try:
            pos = ids.index(r["cashout_atm_id"])
        except ValueError:
            continue
        ranks.append(int(np.where(order == pos)[0][0]) + 1)
        dists.append(float(d[pos]))
    ranks = np.array(ranks)
    print(f"  sampled cashouts         : {len(ranks):,}")
    print(f"  used the NEAREST ATM     : {(ranks == 1).mean():.1%}")
    print(f"  within the nearest 3     : {(ranks <= 3).mean():.1%}")
    print(f"  within the nearest 10    : {(ranks <= 10).mean():.1%}")
    print(f"  median travel distance   : {np.median(dists):.2f} km")
    print()
    print("  If this read 100%, the ATM label would be argmin(distance) and any")
    print("  model given the distance would be solving a tautology.")

    # ── 6. Timing ────────────────────────────────────────────────────────────
    head("6. TIMING — how urgent is the withdrawal?")
    tt = term["time_to_cashout_min"].astype(float)
    print(f"  delay to cashout         : mean {tt.mean():.1f} min, median {tt.median():.1f}")
    print(f"                             p05 {tt.quantile(.05):.1f}, p95 {tt.quantile(.95):.1f}")
    print(f"  under 15 minutes         : {(tt < 15).mean():.1%}  (no time to dispatch)")
    print(f"  under 5 minutes          : {(tt < 5).mean():.1%}")
    print()
    print("  Two regimes: crews with a runner already at the machine cash out in")
    print("  minutes; the rest have to travel. A single-regime model with a ~22 min")
    print("  floor made every case look comfortably interceptable.")

    # ── 7. Attainable ceiling ────────────────────────────────────────────────
    head("7. ATTAINABLE CEILING — what could a PERFECT classifier score?")
    try:
        from generate_data import UNDETECTED_MULE_RATE, FALSE_REPORT_RATE
    except Exception:
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_data import UNDETECTED_MULE_RATE, FALSE_REPORT_RATE

    prev = y.mean()
    tp = prev * (1 - UNDETECTED_MULE_RATE)
    fn = prev * UNDETECTED_MULE_RATE
    fp = (1 - prev) * FALSE_REPORT_RATE
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print(f"  undetected mules         : {UNDETECTED_MULE_RATE:.0%} of mules labelled clean")
    print(f"  false reports            : {FALSE_REPORT_RATE:.1%} of clean labelled mule")
    print(f"  ceiling on precision     : {prec:.3f}")
    print(f"  ceiling on recall        : {rec:.3f}")
    print(f"  ceiling on F1            : {2*prec*rec/(prec+rec):.3f}")
    print()
    print("  Any reported F1 should be read against this number, not against 1.0.")
    print()


if __name__ == "__main__":
    main()
