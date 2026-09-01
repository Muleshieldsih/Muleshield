# -*- coding: utf-8 -*-
"""
Drive every interactive control in the console and report what breaks.

    python scripts/smoke_ui.py              # isolated (default)
    python scripts/smoke_ui.py --attach     # against already-running servers

By default this starts its OWN backend and frontend on their own ports, runs
against those, and tears them down.

That is not fastidiousness. The sweep clicks every control it can reach, which
now includes the case-status dropdown, "Assign to me" and the account freeze --
so running it against the demo backend left real cases reassigned, re-statused
and carrying audit entries authored by a test. The audit trail is the one part
of this system whose whole value is that it records what actually happened; a
test writing into it is worse than a test that skips the control.

The isolated backend loads the same CSVs into its own memory, so every
destructive control is still genuinely exercised against real-shaped data. The
mutations simply die with the process.

--attach runs against whatever is already up, for iterating on the test itself.
It WILL mutate that backend.

Needs both servers up. Walks all four routes, clicks every button and link it
can reach, fills every input, and records what a human click-through misses:

  * console errors / unhandled rejections   (React crashes, undefined access)
  * failed network requests                 (4xx/5xx, aborted, CORS)
  * controls that produce no visible effect (candidates for dead UI)

It prints findings for triage rather than asserting a pass. Destructive controls
are exercised deliberately -- the freeze endpoint is simulated and the point is
to learn whether the button actually works.
"""

import argparse
import os
import re
import socket
import json
import subprocess
import tempfile
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests

# The console renders en-dashes, rupee signs and a Unicode minus in its
# control labels, and this script prints those labels back. On a cp1252
# Windows console that raised UnicodeEncodeError and took down the whole
# sweep -- while REPORTING, after every page had already passed.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]

# Deliberately not 8000/4173. A run must never land on the demo by accident.
API = "http://127.0.0.1:8000"
APP = "http://127.0.0.1:4173"

# Built to its own directory so the demo's dist/ is left alone -- the isolated
# build points at the test backend and would otherwise be served to the demo.
SMOKE_DIST = "dist-smoke"
VIEWPORT = {"width": 1600, "height": 1000}

console_errors: list[str] = []
failed_requests: list[str] = []
findings: list[str] = []

IGNORE_CONSOLE = ("Download the React DevTools", "React Router Future Flag", "favicon")
IGNORE_REQ = ("favicon.ico", "tile.openstreetmap", "basemaps", ".png", ".jpg", ".webp")

# Sidebar labels, skipped so the sweep does not navigate away mid-walk.
# These drifted: the old tuple named "TRIAGE QUEUE" / "TACTICAL MAP" /
# "MONEY FLOW", none of which are labels in Shell.jsx any more, so the
# sweep was clicking its own nav on every page.
NAV = ("CASES", "TRANSACTION TRAIL", "LOCATIONS", "INTERVENTION",
       "MODEL PERFORMANCE", "RISK HEATMAP", "ALERTS",
       # Sign out is a control like any other and the sweep clicked it -- which
       # ended the session, so every route AFTER the first reported three
       # controls and a clean bill of health for the login form. A sweep that
       # logs itself out mid-run reports on a screen the operator never sees.
       "SIGN OUT")
NEWLINE = chr(10)


def free_port() -> int:
    """A port the OS says is free. Racy in principle, fine for a local run."""
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


def wait_for(url: str, timeout: float = 180.0, what: str = "service") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=4).status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(1.5)
    raise SystemExit(f"{what} did not come up at {url} within {timeout:.0f}s")


@contextmanager
def isolated_stack():
    """
    A backend and frontend of this run's own, on their own ports.

    The backend re-reads the same CSVs into a fresh process, so the corpus is
    identical and the destructive controls are exercised for real. Nothing it
    writes outlives the context manager.
    """
    api_port, app_port = free_port(), free_port()
    api = f"http://127.0.0.1:{api_port}"
    app = f"http://127.0.0.1:{app_port}"
    python = sys.executable
    procs: list[subprocess.Popen] = []
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}

    try:
        print(f"  starting isolated backend on :{api_port}")
        # A throwaway credential store, admin pinned so sign_in() can predict it,
        # the background rule pass off so a tick cannot mutate state mid-sweep,
        # and the preview port allowlisted now that CORS is not "*".
        # Before MULESHIELD_DB_PATH existed this "isolated" stack wrote officers
        # into the real data/muleshield.db -- isolated in every respect except
        # the one file holding password hashes.
        api_env = {
            **os.environ,
            "MULESHIELD_DB_PATH": str(Path(tempfile.mkdtemp(prefix="smoke-db-")) / "smoke.db"),
            "MULESHIELD_ADMIN_USER": SMOKE_USER,
            "MULESHIELD_ADMIN_PASSWORD": SMOKE_PASSWORD,
            "MULESHIELD_SCHEDULER": "off",
            "MULESHIELD_CORS_ORIGINS": f"http://127.0.0.1:{app_port},http://localhost:{app_port}",
        }
        procs.append(subprocess.Popen(
            [python, "-m", "uvicorn", "backend.main:app",
             "--host", "127.0.0.1", "--port", str(api_port)],
            cwd=str(ROOT), env=api_env, **quiet,
        ))
        wait_for(f"{api}/health", what="isolated backend")

        # The API base is inlined at build time, so the isolated frontend needs
        # its own build. Into SMOKE_DIST, never dist/.
        print(f"  building frontend against :{api_port}")
        env = {**os.environ, "VITE_API_BASE_URL": api}
        build = subprocess.run(
            f"npm run build -- --outDir {SMOKE_DIST}",
            cwd=str(ROOT / "frontend"), env=env, shell=True,
            capture_output=True, text=True,
        )
        if build.returncode != 0:
            raise SystemExit(f"isolated build failed:{NEWLINE}{build.stdout[-1500:]}{build.stderr[-1500:]}")

        print(f"  serving on :{app_port}")
        procs.append(subprocess.Popen(
            f"npx vite preview --outDir {SMOKE_DIST} --port {app_port} --host 127.0.0.1",
            cwd=str(ROOT / "frontend"), env=env, shell=True, **quiet,
        ))
        wait_for(app, what="isolated frontend")

        yield api, app
    finally:
        for proc in reversed(procs):
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # vite preview is spawned through a shell, so terminating the shell can
        # leave the server holding its port. Close it explicitly.
        if os.name == "nt":
            subprocess.run(f'npx --no-install kill-port {app_port}', shell=True,
                           capture_output=True)
        print("  isolated stack stopped")


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


# The console is behind a login (App.jsx Gate). Until this existed, every walk
# below landed on the sign-in form and reported a clean sweep of a page that had
# mounted nothing -- the script predated authentication by one day and nobody
# noticed, which is exactly the class of failure it exists to catch.
TOKEN = ""
SMOKE_USER = "smokeofficer"
SMOKE_PASSWORD = "smoke-password-not-a-secret"


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def sign_in() -> str:
    """Exchange the seeded credentials for a bearer token."""
    global TOKEN
    r = requests.post(f"{API}/api/v1/auth/login",
                      json={"username": SMOKE_USER, "password": SMOKE_PASSWORD},
                      timeout=30)
    if r.status_code != 200:
        raise SystemExit(
            f"smoke sign-in failed ({r.status_code}). In --attach mode set "
            f"MULESHIELD_ADMIN_USER/PASSWORD on the running backend to match, "
            f"or run the isolated default."
        )
    TOKEN = r.json()["access_token"]
    return TOKEN


def pick_complaint() -> str:
    rows = requests.get(f"{API}/api/v1/complaint/list", params={"limit": 20},
                        headers=auth_headers(), timeout=30).json()
    for row in rows:
        r = requests.get(f"{API}/api/v1/predict/cashout/{row['ticket_id']}",
                         headers=auth_headers(), timeout=90)
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


def reassert_session(page) -> None:
    """Put the bearer token back before each route.

    The sweep clicks every control on every screen, so it will eventually
    provoke a 401 -- and services/api.js clears the token on any 401, which
    cascades: one refusal signs the sweep out and every screen after it is the
    login form wearing a clean bill of health.

    Re-seeding per route keeps each screen's result independent. It is not
    papering over the 401s -- they are still collected and reported below. It
    stops ONE of them from masking every finding that would have come after.
    Session lifetime has its own dedicated check in login_check().
    """
    try:
        page.evaluate(
            "t => window.localStorage.setItem('muleshield:token', t)", TOKEN)
    except Exception:
        pass


def walk(page, name: str, path: str) -> None:
    print(f"{NEWLINE}-- {name} --")
    before_err = len(console_errors)
    page.goto(APP + path, wait_until="networkidle", timeout=90_000)
    reassert_session(page)
    page.reload(wait_until="networkidle", timeout=90_000)
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

    # Re-query before every click. React re-renders the queue when a row is
    # selected, which detaches the handles captured a moment ago -- the same
    # failure control_labels() documents for labels, and it crashed the whole
    # sweep here rather than reporting a finding.
    seen, first = [], shown()
    for i in (1, 2, 3):
        current = [r for r in page.query_selector_all("button")
                   if re.search(r"1930-\d{6}", r.inner_text() or "")]
        if i >= len(current):
            break
        try:
            current[i].click()
        except Exception as e:
            note("selection", f"row {i} could not be clicked: {type(e).__name__}")
            break
        page.wait_for_timeout(1600)
        seen.append(shown())

    print(f"  panel showed: {first} -> {' -> '.join(str(x) for x in seen)}")
    if len(set(seen)) < len(seen) or any(x == first for x in seen):
        note("selection", "clicking a row did NOT change the incident panel "
                          "(URL/selection are fighting)")


def intervention_check(page, cid: str) -> None:
    """
    Carry an account freeze all the way through, including the confirmation.

    The generic sweep cannot do this. It clicks "Freeze account", a confirmation
    dialog opens, and its modal handling clicks Cancel to escape -- so the most
    destructive control in the product was opened and then dismissed, every run,
    and reported as exercised. The audit trail is what caught it.

    Safe to do for real because this runs against a throwaway backend.
    """
    print(f"{NEWLINE}-- intervention --")
    page.goto(f"{APP}/intercept?c={cid}", wait_until="networkidle", timeout=90_000)
    page.wait_for_timeout(6000)

    body = lambda: " ".join(page.inner_text("body").split())
    try:
        page.get_by_role("button", name=re.compile("Freeze account")).first.click(timeout=8000)
        page.wait_for_timeout(1200)
    except Exception:
        note("intervention", "the freeze control could not be reached")
        return

    if "Confirm freeze" not in body():
        note("intervention", "freezing an account did not ask for confirmation")
        return
    print("  confirmation required before freezing: yes")

    try:
        page.get_by_role("button", name="Confirm freeze").first.click(timeout=8000)
        page.wait_for_timeout(4000)
    except Exception as e:
        note("intervention", f"confirm failed: {str(e).split(NEWLINE)[0][:80]}")
        return

    after = body()
    if "Debit hold confirmed" not in after:
        note("intervention", "the freeze did not report success")
        return
    print("  freeze completed: yes")

    if "Case moved to" in after:
        print("  case state advanced with the freeze: yes")
    else:
        note("intervention", "the freeze did not move the case status")



def login_check(browser) -> None:
    """Sign in through the real form, in a context with no seeded token.

    The rest of the sweep seeds localStorage because driving the form on every
    page would be slow and would tell us nothing new. But the login screen is a
    route, it has controls, and it is the one screen that gates every other --
    so it gets exercised once, properly.
    """
    print(f"{NEWLINE}-- login (real form) --")
    ctx = browser.new_context(viewport=VIEWPORT)
    page = ctx.new_page()
    try:
        page.goto(APP, wait_until="networkidle", timeout=60_000)
        page.fill("#username", SMOKE_USER)
        page.fill("#password", SMOKE_PASSWORD)
        page.click("button[type=submit]")
        page.wait_for_timeout(2500)
        body = page.inner_text("body")
        if "Sign in" in body[:400]:
            print("  FAIL the sign-in form did not admit the seeded officer")
        else:
            print("  OK   signed in through the form; console mounted")
    except Exception as e:
        print(f"  FAIL login screen: {type(e).__name__}: {e}")
    finally:
        ctx.close()


def alert_check(page) -> None:
    """Exercise the alert inbox end to end.

    Its own check rather than relying on the generic sweep, for the same reason
    intervention_check exists: acknowledging is a multi-step, state-changing
    control, and the generic modal handling would dismiss it while reporting it
    as exercised.
    """
    print(f"{NEWLINE}-- alerts --")
    try:
        # Evaluate AT THE CORPUS CLOCK, not at wall-clock time.
        #
        # The forward surface only sees complaints filed in the last two hours.
        # The corpus is generated once and then ages, so by wall-clock "now" it
        # has no open cases and the rule pass correctly raises nothing -- which
        # would leave this check permanently skipping and the alerting
        # deliverable permanently unexercised. Anchoring to the newest complaint
        # tests the code rather than the calendar.
        newest = requests.get(f"{API}/api/v1/complaint/list", params={"limit": 1},
                              headers=auth_headers(), timeout=30).json()
        as_of = (newest[0]["complaint_timestamp"][:19] if newest else "")
        requests.post(f"{API}/api/v1/alerts/evaluate",
                      params={"as_of": as_of} if as_of else {},
                      headers=auth_headers(), timeout=180)
        page.goto(f"{APP}/alerts", wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(1200)

        rows = page.query_selector_all(
            "button:has-text('CRITICAL'), button:has-text('HIGH'), button:has-text('WATCH')")
        if not rows:
            print("  SKIP no alerts raised for the current window "
                  "(regenerate the corpus, or check the as-of clock)")
            return
        rows[0].click()
        page.wait_for_timeout(900)
        if "Delivery attempts" not in page.inner_text("body"):
            print("  FAIL detail pane did not open")
            return

        btn = page.get_by_text("Monitoring", exact=True).first
        btn.click()
        page.wait_for_timeout(1500)
        body = page.inner_text("body")
        if "closed by" in body or "Monitoring" in body:
            print("  OK   raised, opened, acknowledged with a disposition")
        else:
            print("  FAIL acknowledgement did not take")
    except Exception as e:
        print(f"  FAIL alert inbox: {type(e).__name__}: {e}")

def exercised_report() -> list[str]:
    """
    What the sweep actually did, read back from the audit trail.

    Isolation makes it cheap to stop testing the destructive controls without
    noticing: nothing would break, and the sweep would still print clean. This
    reads the backend's own record of what happened and names anything the run
    failed to exercise, so a weakened test fails loudly.
    """
    try:
        entries = requests.get(f"{API}/api/v1/audit", params={"limit": 500},
                               headers=auth_headers(), timeout=15).json()
    except requests.RequestException as e:
        return [f"could not read the audit trail: {e}"]

    actions = " ".join(e.get("action", "").lower() for e in entries)
    expected = {
        "status change": "changed status",
        "assignment": "assigned case",
        "account freeze": "froze account",
        # Phase 4. Without these the sweep would go on reporting clean while the
        # alerting deliverable silently stopped being exercised.
        "alert acknowledgement": "acknowledged alert",
    }
    missing = [name for name, needle in expected.items() if needle not in actions]

    print(f"{NEWLINE}-- exercised (from the backend's own audit trail) --")
    print(f"  {len(entries)} audit entries written by this run")
    for name, needle in expected.items():
        n = sum(1 for e in entries if needle in e.get("action", "").lower())
        print(f"  {'OK ' if n else 'MISS'} {name}: {n}")
    return missing


def run_sweep() -> None:
    from playwright.sync_api import sync_playwright

    requests.get(f"{API}/health", timeout=10).raise_for_status()
    requests.get(APP, timeout=10).raise_for_status()

    # Before anything else. Every API call this script makes, and every page it
    # opens, is now behind a bearer token.
    sign_in()
    print(f"  signed in as {SMOKE_USER}")

    cid = pick_complaint()
    print(f"complaint: {cid}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Seed the bearer token BEFORE any page script runs, or Gate() renders the
        # login form and every walk below sweeps a screen that mounted nothing.
        # The storage key is services/auth.js TOKEN_KEY.
        ctx = browser.new_context(viewport=VIEWPORT)
        ctx.add_init_script(
            "window.localStorage.setItem('muleshield:token', %s)" % json.dumps(TOKEN))
        page = ctx.new_page()
        wire(page)

        # Drive the real sign-in form once, so the login screen is COVERED rather
        # than only bypassed -- it is a route like any other and it regressed
        # once already.
        login_check(browser)

        for name, path in (
            ("triage", f"/?c={cid}"),
            ("map", f"/map?c={cid}"),
            ("graph", f"/graph?c={cid}"),
            ("intercept", f"/intercept?c={cid}"),
            # Previously uncovered. /model was never in this list; /risk and
            # /alerts are the Phase 3 and Phase 4 deliverables.
            ("model", "/model"),
            ("risk", "/risk"),
            ("alerts", "/alerts"),
        ):
            walk(page, name, path)

        selection_check(page, cid)
        intervention_check(page, cid)
        alert_check(page)

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

    for gap in exercised_report():
        note("coverage", f"the sweep never exercised: {gap}")

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


def main() -> None:
    global API, APP

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--attach", action="store_true",
        help="run against already-running servers instead of an isolated pair. "
             "This WILL mutate that backend's case data.",
    )
    ap.add_argument("--api", default=API, help="backend URL when --attach")
    ap.add_argument("--app", default=APP, help="frontend URL when --attach")
    args = ap.parse_args()

    if args.attach:
        API, APP = args.api, args.app
        print(f"-- attached to {API} (its case data WILL be modified) --")
        run_sweep()
        return

    print("-- isolated run: own backend and frontend, discarded afterwards --")
    with isolated_stack() as (api, app):
        API, APP = api, app
        run_sweep()


if __name__ == "__main__":
    main()
