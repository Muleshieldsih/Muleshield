# -*- coding: utf-8 -*-
"""
Capture console screenshots for the README.

    python scripts/capture_screens.py

Needs a build and one server:
    cd frontend && npm run build
    python -m uvicorn backend.main:app --port 8000

Shots are taken against the real backend on a real complaint, so what lands in
the README is what the console actually renders -- not a mockup. The complaint is
resolved from the live queue rather than hard-coded, because a pinned ticket id
outlives the dataset it points at and every screen then shows "not found".

Deliberately :8000 and NOT `npm run preview` on :4173. The production bundle
leaves VITE_API_BASE_URL unset and therefore calls the API on *relative* paths,
so under the preview server the console asks :4173 for /api/v1/... and the SPA
fallback answers with index.html. Every fetch then resolves to an object with no
fields, and the case queue renders blank. :8000 is FastAPI serving frontend/dist
-- the same single-origin arrangement the container ships.
"""

import sys
import time
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screens"

# Overridable so the sweep can point at an isolated stack rather than whatever
# happens to be on :8000 with the real credential store behind it.
API = APP = os.environ.get("MULESHIELD_CAPTURE_API", "http://127.0.0.1:8000")
USER = os.environ.get("MULESHIELD_ADMIN_USER", "officer")
PASSWORD = os.environ.get("MULESHIELD_ADMIN_PASSWORD", "")

VIEWPORT = {"width": 1600, "height": 1000}
TOKEN_KEY = "muleshield:token"        # services/auth.js
# Leaflet tiles and React Flow layout settle well after networkidle.
SETTLE_MS = {"triage": 5000, "map": 7000, "graph": 6000, "intercept": 4000,
             "model": 3000, "risk": 7000, "alerts": 4000}

# Text that must appear before a screenshot is worth keeping. Without this the
# script happily saved a 22 KB all-black PNG of the case queue -- the first
# navigation is a cold start, the settle time was short, and nothing checked.
# A capture script that can silently ship a blank image has the same failure
# mode as a build that passes over a broken screen.
MUST_CONTAIN = {
    "triage": "Case queue",
    "graph": "Transaction trail",
    "map": "Ranked locations",
    "intercept": "Intervention",
    "model": "Model performance",
    "risk": "Tactical Risk Forecast",
    "alerts": "Alert Inbox",
}


def sign_in() -> dict:
    """Authenticate, the way every caller of this API now has to.

    This script predated authentication by a day and never had a token. After
    Phase 1 locked the routers, pick_complaint() below would 401 and the browser
    would photograph the sign-in form -- and MUST_CONTAIN would have caught the
    photograph but not explained why. Failing here, with the reason, is better.
    """
    if not PASSWORD:
        raise SystemExit(
            "set MULESHIELD_ADMIN_PASSWORD (the account the backend booted with)")
    r = requests.post(f"{API}/api/v1/auth/login",
                      json={"username": USER, "password": PASSWORD}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"login failed ({r.status_code}) as {USER!r}: {r.text[:200]}")
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def pick_complaint(headers: dict) -> str:
    """Newest complaint that actually returns a prediction."""
    rows = requests.get(f"{API}/api/v1/complaint/list", params={"limit": 25},
                        headers=headers, timeout=30).json()
    for row in rows:
        cid = row["ticket_id"]
        try:
            r = requests.get(f"{API}/api/v1/predict/cashout/{cid}",
                             headers=headers, timeout=90)
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

    headers = sign_in()
    cid = pick_complaint(headers)
    OUT.mkdir(parents=True, exist_ok=True)

    shots = [
        ("triage", f"/?c={cid}", "01-case-queue"),
        ("graph", f"/graph?c={cid}", "02-transaction-trail"),
        ("map", f"/map?c={cid}", "03-cash-out-locations"),
        ("intercept", f"/intercept?c={cid}", "04-intervention"),
        ("model", "/model", "05-model-performance"),
        ("risk", "/risk", "06-risk-heatmap"),
        ("alerts", "/alerts", "07-alert-inbox"),
    ]

    token = headers["Authorization"].split(" ", 1)[1]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        # Seed the session the console reads on boot, so the capture opens on the
        # console rather than on the sign-in screen.
        ctx.add_init_script(
            "window.localStorage.setItem(%s, %s)"
            % (json.dumps(TOKEN_KEY), json.dumps(token)))
        page = ctx.new_page()
        for key, path, fname in shots:
            page.goto(APP + path, wait_until="networkidle", timeout=90_000)
            page.wait_for_timeout(SETTLE_MS[key])

            needle = MUST_CONTAIN[key]
            for attempt in range(4):
                if needle in page.inner_text("body"):
                    break
                page.wait_for_timeout(3000)
            else:
                raise SystemExit(
                    f"{fname}: never rendered {needle!r} - refusing to save a blank "
                    f"screenshot. Is the backend up and the build current?"
                )
            dest = OUT / f"{fname}.png"
            page.screenshot(path=str(dest))
            print(f"  {dest.relative_to(ROOT)}  ({dest.stat().st_size // 1024} KB)")
        browser.close()

    print(f"\n{len(shots)} screenshots in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
