# -*- coding: utf-8 -*-
"""
Capture console screenshots for the README.

    python scripts/capture_screens.py

Needs both servers up:
    python -m uvicorn backend.main:app --port 8000
    cd frontend && npm run build && npm run preview -- --port 4173

Shots are taken against the real backend on a real complaint, so what lands in
the README is what the console actually renders -- not a mockup. The complaint is
resolved from the live queue rather than hard-coded, because a pinned ticket id
outlives the dataset it points at and every screen then shows "not found".
"""

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screens"
API = "http://127.0.0.1:8000"
APP = "http://127.0.0.1:4173"

VIEWPORT = {"width": 1600, "height": 1000}
# Leaflet tiles and React Flow layout settle well after networkidle.
SETTLE_MS = {"triage": 2500, "map": 7000, "graph": 6000, "intercept": 4000}


def pick_complaint() -> str:
    """Newest complaint that actually returns a prediction."""
    rows = requests.get(f"{API}/api/v1/complaint/list", params={"limit": 25}, timeout=30).json()
    for row in rows:
        cid = row["ticket_id"]
        try:
            r = requests.get(f"{API}/api/v1/predict/cashout/{cid}", timeout=90)
            if r.status_code == 200 and len(r.json().get("ranked_candidates", [])) == 5:
                print(f"  using {cid}  ({row['victim_bank']}, {row['city']})")
                return cid
        except requests.RequestException:
            continue
    raise SystemExit("no complaint produced a 5-candidate prediction")


def main() -> None:
    from playwright.sync_api import sync_playwright

    for url, name in ((API + "/health", "backend"), (APP, "frontend")):
        try:
            requests.get(url, timeout=10).raise_for_status()
        except Exception as e:
            raise SystemExit(f"{name} not reachable at {url}: {e}")

    cid = pick_complaint()
    OUT.mkdir(parents=True, exist_ok=True)

    shots = [
        ("triage", f"/?c={cid}", "01-triage-queue"),
        ("map", f"/map?c={cid}", "02-priority-search-locations"),
        ("graph", f"/graph?c={cid}", "03-money-flow-graph"),
        ("intercept", f"/intercept?c={cid}", "04-interception"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        for key, path, fname in shots:
            page.goto(APP + path, wait_until="networkidle", timeout=90_000)
            page.wait_for_timeout(SETTLE_MS[key])
            dest = OUT / f"{fname}.png"
            page.screenshot(path=str(dest))
            print(f"  {dest.relative_to(ROOT)}  ({dest.stat().st_size // 1024} KB)")
        browser.close()

    print(f"\n{len(shots)} screenshots in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
