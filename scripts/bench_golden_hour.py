# -*- coding: utf-8 -*-
"""
Golden-hour latency: complaint filed -> alert delivered.

    Run: python scripts/bench_golden_hour.py [--api http://127.0.0.1:8000] [-n 50]

WHY THIS NUMBER
---------------
The 1930 rail exists because the hour after a complaint is filed is when money
can still be stopped. A forecast that arrives after that window is a report, not
an intervention. So the claim this project makes -- "proactive" -- reduces to a
measurable quantity: how long does it take, from a citizen filing at 1930, for a
district-level alert to exist and be dispatched?

COMPLIANCE_AUDIT.md finding 6.1 also noted the problem statement names ~8,000
complaints/day twice and nothing in the repository measured whether the system
could take them. The burst pass at the end answers that: 8,000/day is ~0.09
complaints/second, which is not a demanding rate, but "we have not measured it"
was the honest answer until now and it is a weak one.

WHAT IS MEASURED
----------------
Five spans, each an authenticated HTTP round trip against a RUNNING server, so
what is timed is the deployed path rather than a function call in isolation:

    ingest    POST /api/v1/complaint/ingest
    forecast  GET  /api/v1/predict/cashout/{id}      per-case ATM ranking + countdown
    surface   GET  /api/v1/hotspots/cells            the national forward surface
    rules     POST /api/v1/alerts/evaluate           the rule pass
    alert     GET  /api/v1/alerts                    the alert readable by an officer

p50 and p95 are reported per span and end to end. p95 rather than a mean: a mean
hides the tail, and it is the tail that decides whether the slowest case still
lands inside the hour.

The result is written to data/metrics.json section "pipeline", so the figure on
any slide is one a script produced.
"""

import argparse
import statistics as st
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from metrics_io import write_metrics  # noqa: E402

CITIES = [("Hyderabad", "Telangana"), ("Mumbai", "Maharashtra"),
          ("Jaipur", "Rajasthan"), ("Ranchi", "Jharkhand")]


def _p(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _login(api: str, user: str, pwd: str) -> dict:
    r = requests.post(f"{api}/api/v1/auth/login",
                      json={"username": user, "password": pwd}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(
            f"login failed ({r.status_code}). Start the backend and pass "
            f"--user/--password, or set MULESHIELD_ADMIN_PASSWORD before booting it."
        )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _complaint(i: int) -> dict:
    city, state = CITIES[i % len(CITIES)]
    return {
        "victim_name": f"Bench Subject {i}",
        "victim_bank": "SBI",
        "victim_account": f"BENCH-{uuid.uuid4().hex[:8].upper()}",
        "fraud_type": "UPI Fraud",
        "stolen_amount": 250000.0 + (i * 1000),
        "city": city, "state": state,
    }


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="officer")
    ap.add_argument("--password", default="")
    ap.add_argument("-n", "--iterations", type=int, default=50)
    ap.add_argument("--burst", type=int, default=100)
    args = ap.parse_args()

    pwd = args.password or __import__("os").environ.get("MULESHIELD_ADMIN_PASSWORD", "")
    if not pwd:
        raise SystemExit("pass --password or set MULESHIELD_ADMIN_PASSWORD")

    H = _login(args.api, args.user, pwd)
    print(f"[1/3] Warming the path once (first call pays model load)...")
    warm = requests.post(f"{args.api}/api/v1/complaint/ingest",
                         json=_complaint(0), headers=H, timeout=180)
    warm.raise_for_status()
    wid = warm.json()["ticket_id"]
    requests.get(f"{args.api}/api/v1/predict/cashout/{wid}", headers=H, timeout=180)
    requests.get(f"{args.api}/api/v1/hotspots/cells", headers=H, timeout=180)

    spans = {k: [] for k in ("ingest", "forecast", "surface", "rules", "alert")}
    e2e = []

    print(f"[2/3] Timing {args.iterations} complaints end to end...")
    for i in range(1, args.iterations + 1):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        r = requests.post(f"{args.api}/api/v1/complaint/ingest",
                          json=_complaint(i), headers=H, timeout=180)
        r.raise_for_status()
        cid = r.json()["ticket_id"]
        spans["ingest"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        requests.get(f"{args.api}/api/v1/predict/cashout/{cid}", headers=H, timeout=180)
        spans["forecast"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        requests.get(f"{args.api}/api/v1/hotspots/cells", headers=H, timeout=180)
        spans["surface"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        requests.post(f"{args.api}/api/v1/alerts/evaluate", headers=H, timeout=180)
        spans["rules"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        requests.get(f"{args.api}/api/v1/alerts", params={"limit": 5},
                     headers=H, timeout=60)
        spans["alert"].append((time.perf_counter() - t0) * 1000)

        e2e.append((time.perf_counter() - t_start) * 1000)
        if i % 10 == 0:
            print(f"      {i}/{args.iterations}  last end-to-end {e2e[-1]:.0f} ms")

    print(f"[3/3] Ingestion burst ({args.burst} complaints)...")
    t0 = time.perf_counter()
    for i in range(args.burst):
        requests.post(f"{args.api}/api/v1/complaint/ingest",
                      json=_complaint(10_000 + i), headers=H, timeout=180)
    burst_s = time.perf_counter() - t0
    per_min = args.burst / (burst_s / 60.0)

    print()
    hdr = f"{'span':>10} | {'p50 ms':>9} | {'p95 ms':>9} | {'max ms':>9}"
    print(hdr); print("-" * len(hdr))
    for k, v in spans.items():
        print(f"{k:>10} | {_p(v, 0.5):>9.1f} | {_p(v, 0.95):>9.1f} | {max(v):>9.1f}")
    print("-" * len(hdr))
    print(f"{'END TO END':>10} | {_p(e2e, 0.5):>9.1f} | {_p(e2e, 0.95):>9.1f} | {max(e2e):>9.1f}")

    p95_s = _p(e2e, 0.95) / 1000.0
    print()
    print(f"  Complaint filed -> district-level alert dispatched: "
          f"p95 {p95_s:.2f} s, against a 60-minute golden hour.")
    print(f"  Ingestion throughput: {per_min:,.0f} complaints/min "
          f"= {per_min * 60 * 24:,.0f}/day capacity (NCRP runs ~8,000/day).")
    print(f"  Headroom at the stated national load: "
          f"{(per_min * 60 * 24) / 8000:,.0f}x")

    payload = {
        "iterations": args.iterations,
        "burst_complaints": args.burst,
        "ingest_ms_p50": round(_p(spans["ingest"], 0.5), 2),
        "ingest_ms_p95": round(_p(spans["ingest"], 0.95), 2),
        "forecast_ms_p50": round(_p(spans["forecast"], 0.5), 2),
        "forecast_ms_p95": round(_p(spans["forecast"], 0.95), 2),
        "surface_ms_p50": round(_p(spans["surface"], 0.5), 2),
        "surface_ms_p95": round(_p(spans["surface"], 0.95), 2),
        "rules_ms_p50": round(_p(spans["rules"], 0.5), 2),
        "rules_ms_p95": round(_p(spans["rules"], 0.95), 2),
        "alert_ms_p50": round(_p(spans["alert"], 0.5), 2),
        "alert_ms_p95": round(_p(spans["alert"], 0.95), 2),
        "end_to_end_ms_p50": round(_p(e2e, 0.5), 2),
        "end_to_end_ms_p95": round(_p(e2e, 0.95), 2),
        "end_to_end_seconds_p95": round(p95_s, 3),
        "golden_hour_minutes": 60,
        "throughput_complaints_per_min": round(per_min, 1),
        "implied_daily_capacity": int(per_min * 60 * 24),
        "ncrp_stated_daily_load": 8000,
        "mean_ms": round(st.mean(e2e), 2),
    }
    write_metrics("pipeline", payload, source="python scripts/bench_golden_hour.py")
    print("\n  wrote data/metrics.json section 'pipeline'")
    return payload


if __name__ == "__main__":
    main()
