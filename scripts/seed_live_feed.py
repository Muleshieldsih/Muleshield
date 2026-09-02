# -*- coding: utf-8 -*-
"""
Replay a live 1930 complaint feed into a running backend.

    Run: python scripts/seed_live_feed.py [--api http://127.0.0.1:8000]
                                          [--per-day 8000] [--minutes 120]

WHY THIS EXISTS
---------------
The forward surface aggregates over complaints filed in the last 120 minutes.
That is the design, and it is the right one: a complaint from last Tuesday tells
you nothing about where cash is about to come out tonight.

But the seeded corpus holds 2,500 complaints spread across 120 days -- about 21
a day for the whole of India. In a two-hour window that is under two complaints,
and often zero. So a freshly generated corpus still opens on an empty forward
surface, and REMEDIATION_AUDIT.md was wrong to file this under "regenerate the
dataset before demoing": regeneration is necessary and it is not sufficient. The
arrival rate is the problem, not the clock.

The problem statement names the real rate twice: roughly 8,000 complaints a day
on NCRP. At that rate a 120-minute window holds around 670 open complaints, which
is the condition this system was designed for and the condition under which the
convergence rules were tuned. This script puts the backend in that condition.

WHAT IT IS, AND IS NOT
----------------------
It is a replay of the corpus through the REAL ingestion endpoint -- the same
POST /api/v1/complaint/ingest an NCRP bridge would call, with authentication,
chain synthesis, GNN inference and ATM ranking all running exactly as they do in
production. Nothing is written past the API, and no metric is touched.

It is NOT a way to manufacture a result. Complaints are drawn from
data/victim_complaints.csv, so the city, state, fraud-type and amount
distributions are the corpus's own; only the filing times are new, spread over
the window at a Poisson-ish rate rather than dropped in a single instant. The
surface it produces is the surface the model computes from them.

Every ingested ticket is echoed as an ordinary complaint. If you want the
console empty again, restart the backend: this writes to in-memory state only.
"""

import argparse
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def _login(api: str, user: str, pwd: str) -> dict:
    r = requests.post(f"{api}/api/v1/auth/login",
                      json={"username": user, "password": pwd}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(
            f"login failed ({r.status_code}). Start the backend and pass "
            f"--user/--password, or set MULESHIELD_ADMIN_PASSWORD before booting it."
        )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _surface(api: str, headers: dict) -> dict:
    r = requests.get(f"{api}/api/v1/hotspots/cells", headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> dict:
    ap = argparse.ArgumentParser(description="Replay a live 1930 feed")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="officer")
    ap.add_argument("--password", default="")
    ap.add_argument("--per-day", type=int, default=8000,
                    help="national complaint rate to simulate (NCRP runs ~8,000)")
    ap.add_argument("--minutes", type=int, default=120,
                    help="how far back to spread arrivals; matches the open window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluate", action="store_true",
                    help="run the alert rule pass once the feed is in")
    args = ap.parse_args()

    pwd = args.password or __import__("os").environ.get("MULESHIELD_ADMIN_PASSWORD", "")
    if not pwd:
        raise SystemExit("pass --password or set MULESHIELD_ADMIN_PASSWORD")

    n = max(1, round(args.per_day * args.minutes / 1440.0))
    H = _login(args.api, args.user, pwd)

    before = _surface(args.api, H)
    print(f"  before : {before['n_open_complaints']:>4} open  "
          f"degraded={before['degraded']}  cells={before['n_cells']}")

    src = pd.read_csv(ROOT / "data" / "victim_complaints.csv")
    rng = random.Random(args.seed)
    rows = src.sample(n=n, replace=n > len(src), random_state=args.seed)

    now = datetime.now()
    print(f"  filing {n} complaints over the last {args.minutes} min "
          f"({args.per_day:,}/day, the NCRP rate named in the problem statement)")

    filed = 0
    t0 = time.perf_counter()
    for _, r in rows.iterrows():
        # Uniform over the window is the right null model for a Poisson arrival
        # process observed over a fixed interval, and it keeps the newest
        # complaints genuinely new rather than all landing at t=0.
        offset = rng.uniform(0, args.minutes)
        payload = {
            "victim_name": str(r.victim_name),
            "victim_bank": str(r.victim_bank),
            "victim_account": str(r.victim_account),
            "fraud_type": str(r.fraud_type),
            "stolen_amount": float(r.stolen_amount),
            "city": str(r.city),
            "state": str(r.state),
            "complaint_timestamp": (now - timedelta(minutes=offset)).isoformat(),
        }
        resp = requests.post(f"{args.api}/api/v1/complaint/ingest",
                             json=payload, headers=H, timeout=180)
        if resp.status_code != 200:
            print(f"    ! ingest failed ({resp.status_code}) {resp.text[:120]}")
            continue
        filed += 1
        if filed % 100 == 0:
            print(f"      {filed}/{n}")
    elapsed = time.perf_counter() - t0

    after = _surface(args.api, H)
    print(f"  after  : {after['n_open_complaints']:>4} open  "
          f"degraded={after['degraded']}  cells={after['n_cells']}")
    print(f"  filed {filed} in {elapsed:.1f}s "
          f"({filed / max(elapsed, 1e-9) * 60:,.0f}/min)")
    print(f"  forecast mass: Rs {after['total_conditional_rupees']:,.0f} conditional "
          f"+ Rs {after['total_prior_rupees']:,.0f} prior "
          f"(prior share {after['prior_share_national']:.4f}, "
          f"cap {after['prior_weight']})")

    top = after["cells"][:5]
    if top:
        print()
        hdr = f"  {'cell':<12} {'district':<18} {'cases':>5} {'Rs forecast':>16} {'prior':>7}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for c in top:
            print(f"  {c['cell_id']:<12} {c['district'][:18]:<18} "
                  f"{c['case_count']:>5} {c['score']:>16,.0f} "
                  f"{c['prior_share']:>6.1%}")

    if args.evaluate:
        r = requests.post(f"{args.api}/api/v1/alerts/evaluate", headers=H, timeout=180)
        r.raise_for_status()
        body = r.json()
        print(f"\n  rule pass: raised {body['raised']} alert(s) over "
              f"{body['cells_considered']} cells, {body['open_complaints']} open")

    return {"filed": filed, "open_after": after["n_open_complaints"]}


if __name__ == "__main__":
    main()
