# -*- coding: utf-8 -*-
"""
Drive every interactive control in the console and report what breaks.

    python scripts/smoke_ui.py

Needs both servers up. Walks all four routes, clicks every button and link it
can reach, fills every input, and records what a human click-through misses:

  * console errors / unhandled rejections   (React crashes, undefined access)
  * failed network requests                 (4xx/5xx, aborted, CORS)
  * controls that produce no visible effect (candidates for dead UI)

It prints findings for triage rather than asserting a pass. Destructive controls
are exercised deliberately -- the freeze endpoint is simulated and the point is
to learn whether the button actually works.
"""

import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"
APP = "http://127.0.0.1:4173"
VIEWPORT = {"width": 1600, "height": 1000}

console_errors: list[str] = []
failed_requests: list[str] = []
findings: list[str] = []

IGNORE_CONSOLE = ("Download the React DevTools", "React Router Future Flag", "favicon")
IGNORE_REQ = ("favicon.ico", "tile.openstreetmap", "basemaps", ".png", ".jpg", ".webp")

NAV = ("TRIAGE QUEUE", "TACTICAL MAP", "MONEY FLOW", "INTERCEPTION")
NEWLINE = chr(10)


def note(page_name: str, msg: str) -> None:
    findings.append(f"[{page_name}] {msg}")
    print(f"  ! {page_name}: {msg}")


def wire(page) -> None:
    def on_console(m):
        if m.type == "error" and not any(s in m.text for s in IGNORE_CONSOLE):
            console_errors.append(m.text[:300])

    def on_pageerror(e):
        console_errors.append(f"UNCAUGHT: {str(e)[:300]}")

    def on_response(r):
        if r.status >= 400 and not any(s in r.url for s in IGNORE_REQ):
            failed_requests.append(f"{r.status} {r.request.method} {r.url[:140]}")

    def on_requestfailed(r):
        if not any(s in r.url for s in IGNORE_REQ):
            failed_requests.append(f"FAILED {r.method} {r.url[:140]} ({r.failure})")

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    page.on("response", on_response)
    page.on("requestfailed", on_requestfailed)


def pick_complaint() -> str:
    rows = requests.get(f"{API}/api/v1/complaint/list", params={"limit": 20}, timeout=30).json()
    for row in rows:
        r = requests.get(f"{API}/api/v1/predict/cashout/{row['ticket_id']}", timeout=90)
        if r.status_code == 200 and len(r.json().get("ranked_candidates", [])) == 5:
            return row["ticket_id"]
    raise SystemExit("no usable complaint")


def control_labels(page) -> list[str]:
    """
    Accessible labels of every visible, enabled control.

    Labels, not ElementHandles. React re-renders after almost every click, which
    detaches a handle captured beforehand -- an earlier version of this script
    reported 34 "failures" that were all its own stale handles. A Playwright
    locator re-resolves at action time, so it survives the re-render.
    """
    out, seen = [], set()
    for el in page.query_selector_all("button, a[href], [role=button]"):
        try:
            if not el.is_visible() or not el.is_enabled():
                continue
            raw = (el.inner_text() or el.get_attribute("aria-label")
                   or el.get_attribute("title") or "")
            label = " ".join(raw.split())
            if not label or label in seen:
                continue
            seen.add(label)
            out.append(label)
        except Exception:
            continue
    return out


def walk(page, name: str, path: str) -> None:
    print(f"{NEWLINE}-- {name} --")
    before_err = len(console_errors)
    page.goto(APP + path, wait_until="networkidle", timeout=90_000)
    page.wait_for_timeout(4500 if name in ("map", "graph") else 2500)

    body = page.inner_text("body")
    for bad in ("Prediction unavailable", "Backend unreachable", "Something went wrong",
                "No prediction to display", "complaint not found", "Failed to load"):
        if bad.lower() in body.lower():
            note(name, f"page shows '{bad}' on load")

    labels = control_labels(page)
    print(f"  {len(labels)} interactive controls")

    inert = []
    for label in labels:
        if label.upper().startswith(NAV):
            continue
        try:
            loc = page.get_by_text(label, exact=True).first
            if loc.count() == 0 or not loc.is_visible():
                continue

            n_err, n_req = len(console_errors), len(failed_requests)
            dom_before = len(page.inner_text("body"))
            url_before = page.url

            loc.click(timeout=5000)
            page.wait_for_timeout(1000)

            if len(console_errors) > n_err:
                note(name, f"click '{label[:48]}' -> console error")
            if len(failed_requests) > n_req:
                note(name, f"click '{label[:48]}' -> failed request")

            changed = (page.url != url_before
                       or abs(len(page.inner_text("body")) - dom_before) >= 3
                       or len(failed_requests) != n_req)
            if not changed:
                inert.append(label[:48])

            if page.url != url_before:
                page.goto(APP + path, wait_until="networkidle", timeout=60_000)
                page.wait_for_timeout(2500)

            for close in page.query_selector_all("[aria-label='Close'], button:has-text('Cancel')"):
                if close.is_visible():
                    close.click(timeout=2000)
                    page.wait_for_timeout(300)
                    break
        except Exception as e:
            msg = str(e).split(NEWLINE)[0][:110]
            if "intercepts pointer events" in msg:
                note(name, f"'{label[:40]}' is overlapped by another element")
            elif "Timeout" in msg:
                note(name, f"'{label[:40]}' click timed out (never became actionable)")
            elif "strict mode violation" in msg:
                pass
            else:
                note(name, f"'{label[:40]}' -> {msg}")

    if inert:
        shown = ", ".join(inert[:6]) + (" ..." if len(inert) > 6 else "")
        print(f"  no visible effect ({len(inert)}): {shown}")

    for inp in page.query_selector_all("input[type=text], input:not([type]), input[type=search]"):
        try:
            if inp.is_visible() and inp.is_enabled():
                inp.fill("zzz-no-match")
                page.wait_for_timeout(700)
                inp.fill("")
                page.wait_for_timeout(400)
        except Exception as e:
            note(name, f"input not fillable: {str(e).split(NEWLINE)[0][:80]}")

    for sel in page.query_selector_all("select"):
        try:
            if sel.is_visible():
                opts = [o.get_attribute("value") for o in sel.query_selector_all("option")]
                if len(opts) > 1:
                    sel.select_option(opts[1])
                    page.wait_for_timeout(800)
                    sel.select_option(opts[0])
                    page.wait_for_timeout(400)
        except Exception as e:
            note(name, f"select failed: {str(e).split(NEWLINE)[0][:80]}")

    if len(console_errors) > before_err:
        note(name, f"{len(console_errors) - before_err} console error(s) on this route")


def selection_check(page, cid: str) -> None:
    """
    Selecting a different complaint must actually change the incident panel --
    WITH a ?c= already in the address bar.

    That qualifier is the whole test. A bug where the URL sync overwrote every
    in-app selection made the queue unresponsive in normal use, and an earlier
    version of this script missed it completely by loading a bare "/", where the
    sync short-circuits and the fault cannot appear.
    """
    print(f"{NEWLINE}-- selection --")
    page.goto(f"{APP}/?c={cid}", wait_until="networkidle", timeout=90_000)
    page.wait_for_timeout(3500)

    # Case references render as 1930-NNNNNN. They were TKT-XXXXXXXX until the
    # two id formats in the store were unified behind one display reference.
    def shown():
        m = re.search(r"Selected case.*?(1930-\d{6})",
                      " ".join(page.inner_text("body").split()))
        return m.group(1) if m else None

    rows = [r for r in page.query_selector_all("button")
            if re.search(r"1930-\d{6}", r.inner_text() or "")]
    if len(rows) < 4:
        note("selection", "too few queue rows to test selection")
        return

    seen, first = [], shown()
    for i in (1, 2, 3):
        rows[i].click()
        page.wait_for_timeout(1600)
        seen.append(shown())

    print(f"  panel showed: {first} -> {' -> '.join(str(x) for x in seen)}")
    if len(set(seen)) < len(seen) or any(x == first for x in seen):
        note("selection", "clicking a row did NOT change the incident panel "
                          "(URL/selection are fighting)")


def main() -> None:
    from playwright.sync_api import sync_playwright

    requests.get(f"{API}/health", timeout=10).raise_for_status()
    requests.get(APP, timeout=10).raise_for_status()
    cid = pick_complaint()
    print(f"complaint: {cid}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        wire(page)

        for name, path in (
            ("triage", f"/?c={cid}"),
            ("map", f"/map?c={cid}"),
            ("graph", f"/graph?c={cid}"),
            ("intercept", f"/intercept?c={cid}"),
        ):
            walk(page, name, path)

        selection_check(page, cid)

        print(f"{NEWLINE}-- deep links / edge cases --")
        for name, path in (
            ("no-complaint", "/"),
            ("bad-id", "/map?c=TKT-DOES-NOT-EXIST"),
            ("unknown-route", "/nope"),
        ):
            n = len(console_errors)
            page.goto(APP + path, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(3500)
            txt = " ".join(page.inner_text("body").split())[:120]
            print(f"  {name}: {txt}")
            if len(console_errors) > n:
                note(name, "console error")

        browser.close()

    print(NEWLINE + "=" * 64)
    print(f"CONSOLE ERRORS ({len(console_errors)})")
    for e in dict.fromkeys(console_errors):
        print(f"  - {e}")
    print(f"{NEWLINE}FAILED REQUESTS ({len(failed_requests)})")
    for r in dict.fromkeys(failed_requests):
        print(f"  - {r}")
    print(f"{NEWLINE}FINDINGS ({len(findings)})")
    for f in findings:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
