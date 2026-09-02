# Remediation audit — every finding, re-checked

**SIH26184 · MHA / I4C** · audited 1 Sep 2026 · branch `sameer`

`COMPLIANCE_AUDIT.md` recorded 26 findings against the published problem
statement: **7 BLOCKER · 11 MAJOR · 8 MINOR**, with three of four named
deliverables substantially incomplete.

This document re-checks each one against the code as it now stands, and records
what was found *during* the remediation and the final build that the original
audit did not catch — including three defects introduced by the remediation
itself (§2) and five more found only by regenerating the corpus and running the
system at the load the problem statement names (§4).

Nothing below is asserted from a reading of the code. Every row names the command
or test that establishes it.

---

## 0 · Verdict

| # | Deliverable | Before | After |
|---|---|---|---|
| **a** | Predictive Analytics Engine | 🟡 ~60% | 🟢 **Met** |
| **b** | Risk Heatmap Dashboard | 🔴 ~25% | 🟢 **Met** |
| **c** | Law Enforcement Interface | 🟡 ~40% | 🟢 **Met** — secure, alerts, and evidence documentation with a chain of custody (§4.6) |
| **d** | Alert & Notification System | 🔴 ~15% | 🟢 **Met**, transport simulated and labelled; alert volume needs one threshold set (§4.4) |

**7 BLOCKER → 0 open. 11 MAJOR → 1 open. 8 MINOR → 2 open.**

**Test suite: 449 passed, 0 failed, 0 skipped** (`python -m pytest -q`, 13m25s),
up from 290 before this work, 397 before the final build, and 402 before evidence
documentation. The graph-build budget test, which skips rather than measure a
contended machine, ran and passed on this run.

Everything below was re-measured after a full regeneration — corpus, GraphSAGE,
embeddings, XGBoost, every evaluation script, the frontend bundle — so no figure
in this document was carried over from a previous corpus.

The one open MAJOR is the complaint arrival rate (§5.1). The previous revision of
this audit called it the dataset *clock* and prescribed regeneration; running that
fix is what proved the diagnosis wrong, and §5.1 now records both.

---

## 1 · The original findings, one by one

### BLOCKERs

| # | Finding | Status | Evidence |
|---|---|---|---|
| 1.1 | Repository stated the wrong problem statement | ✅ **Closed** | `README.md` now carries the official title verbatim plus theme/category |
| 1.2 | Predicts per-complaint candidates, not hotspots | ✅ **Closed** | `engine/hotspot.py` + `GET /api/v1/hotspots/cells`; forward surface over open complaints, not a density map |
| 2.1 | No scheduled or batch analytical job anywhere | ✅ **Closed** | `_tick()` in `backend/main.py` lifespan, gated by `MULESHIELD_SCHEDULER`, cancelled and awaited on shutdown |
| 3.1 | No heatmap in a deliverable named "Risk Heatmap Dashboard" | ✅ **Closed** | `frontend/src/pages/RiskHeatmap.jsx`, `L.circleMarker` vector ramp, no plugin added |
| 3.2 | No national / state / district view | ✅ **Closed** | Score-weighted state centroids → district → cell, four-level breadcrumb |
| 4.1 | **Authentication was a client-side gate; no endpoint required a token** | ✅ **Closed** | 13 endpoints on `Depends(current_user)`; live `curl` returns 401 + `WWW-Authenticate: Bearer` on `POST /bank/micro-freeze` |
| 4.2 | `allow_origins=["*"]` with `allow_credentials=True` | ✅ **Closed** | Env-driven allowlist; foreign `Origin` receives no `Access-Control-Allow-Origin` |
| 4.3 | No alert object existed | ✅ **Closed** | `alerts` table + `/api/v1/alerts` + `AlertInbox.jsx` |
| 4.4 | No intelligence report could be produced | 🟡 **Partial** | Alert detail carries headline, cases, delivery record and disposition; CFCFRMS/Samanvaya payloads are emitted; and the s.63 certificate is a printable, generated document. A general case dossier covering the forecast and the graph is still not built |
| 4.5 | No evidence documentation | ✅ **Closed** | `backend/evidence.py` + `case_evidence` + `/api/v1/evidence/*` + the Evidence panel on the case screen. Hash at collection, chain across the case, no delete, BSA 2023 s.63 certificate. §4.6 |
| 5.1 | Zero of four named channels implemented | ✅ **Closed** | `backend/adapters/{sms,email,webhook}.py` with full delivery contract; transport simulated and labelled on screen |
| 5.2 | No trigger engine — WebSocket only echoed user actions | ✅ **Closed** | `backend/notify.py` `RULES`, fired by the tick with no human present |
| 6.2 | README claimed API schemas that did not exist | ✅ **Closed** | Claim withdrawn; `docs/INTEGRATION_SEAMS.md` publishes the actual mapping, versioned `v1-proposed` |

### MAJORs

| # | Finding | Status | Evidence |
|---|---|---|---|
| 1.3 | "In Advance" met only within one case's countdown | ✅ **Closed** | `hotspot.lead_time_median_min` = **41.5 min**, 84.1% of hits with ≥15 min warning |
| 2.2 | `intel/atms` built and rendered nowhere | ✅ **Closed** | Recurring-machines table on `/risk` |
| 2.3 | No temporal dimension in any aggregate | ✅ **Closed** | Three forecast windows; conditional-lognormal survival kernel |
| 3.3 | No drill-down filters on the map | ✅ **Closed** | Location breadcrumb, window pills, as-of picker, multi-select crime category |
| 3.4 | Data existed and was unused | ✅ **Closed** | District/state roll-up keys on every cell |
| 4.6 | No jurisdiction model | 🟡 **Partial** | `alert_recipients` scopes by state/district with LEA/I4C/BANK roles, so alerts route correctly. The `users` table still has only `is_admin` — officers are not yet scoped |
| 4.7 | Audit trail in memory, dies on restart | ❌ **Open (MAJOR→MINOR)** | Alerts, deliveries and recipients are now in SQLite. The **case** audit trail is still `state.audit_log` in memory |
| 4.8 | Audit omits logins, admin actions, case access | ❌ **Open** | Alert acknowledgement is now audited; logins and case views still are not |
| 4.9 | Audit not tamper-evident | 🟡 **Partial** | Evidence IS hash-chained per case and verified on read (§4.6). The general case audit trail still is not, and the chain has no external anchor — both stated in `backend/db.chain_hash` |
| 6.1 | ~8,000 complaints/day unaddressed | ✅ **Closed** | `scripts/bench_golden_hour.py`: **3,499 complaints/min = 5.04M/day, 630× headroom** |
| 6.3 | Cross-jurisdiction sharing had no mechanism | ✅ **Closed** | Scoped recipients + Samanvaya dissemination payload |

### MINORs

| # | Finding | Status |
|---|---|---|
| 2.4 | No model monitoring / drift detection | ❌ Open, deferred |
| 2.5 | No outcome capture | ✅ **Closed** — required disposition incl. *False positive*; `false_positive_rate` on screen |
| 3.5 | `district` not captured on ingestion | ❌ Open, documented in `INTEGRATION_SEAMS.md` §1.1 |
| 4.10 | Admin roster built and unreachable | ❌ Open — endpoints still render on no screen |
| 4.11 | No separation of duties on reset approval | ❌ Open |
| 5.5 | Simulated SMS reported success unconditionally | ✅ **Closed** — `MULESHIELD_ADAPTER_FAIL_RATE` drives real failure, retry and dead-letter |
| 6.4 | No bank/FI-facing surface | 🟡 Partial — CFCFRMS payload exists; no bank-facing screen |
| 7.x | Hygiene: stale branches, unbuilt Dockerfile, dataset clock | Mixed — see §4 |

---

## 2 · What the remediation itself broke, and how it was caught

Three defects were introduced by this work. All three were caught by
instrumentation rather than by review, which is the point.

### 2.1 A data leak in the hotspot evaluation — introduced and fixed

`scripts/evaluate_hotspots.py` fed `meta_df["time_to_cashout_min"]` — the
**observed delay from the ledger** — into the lognormal survival kernel as though
it were the model's own countdown. That is the target.

| | leaked | clean |
|---|---|---|
| hit@1 | 0.4806 | **0.4693** |
| hit@5 | 0.9320 | **0.9337** |
| PAI@5 | 27.98 | **28.17** |

*(Measured on the pre-regeneration corpus, which is the only way the two columns
are comparable — the point is the difference between them, not the level. §6
carries the current figures.)*

Immaterial, because the window mass sums to ≈1 across the three bands and barely
discriminated — but *immaterial* is a measurement, not an assumption. Fixed to
use the regressor's predicted median and q05/q95, derived exactly as
`MuleXGBPredictor.predict()` derives them. `tests/test_hotspot_leakage.py` now
asserts by AST that `rank_forecast` never reads the observed delay.

### 2.2 A serving/metrics skew — introduced and fixed

`backend.state.hotspot_entry` read `predict()`'s display-rounded `confidence`
while the evaluation called `posterior_from_scores`. Numerically within 5e-5, but
two implementations of one quantity — which is how two paths drift apart with
nothing noticing. Both now funnel through `posterior_from_scores`;
`backend/tests/test_hotspot.py::TestServingParity` asserts they agree to 1e-12.

A first static check *passed this falsely* by matching an `import` rather than a
call. The AST check replaced it.

### 2.3 Alert spam that was Pratibimb in disguise — caught on the first live pass

The rule set as specified raised **66 alerts on one pass**. Of those, **48 had
zero open cases and `prior_share` of exactly 1.0** — pure historical density —
and 57 were under ₹1,000.

`R-WATCH-SCORE` was the only rule with no conditional floor, so it fired on the
capped prior showing through where nothing live was happening. Three quarters of
the inbox was the very thing this component exists not to be.

Fixed at the source: `candidates_from_surface` now admits only cell-windows with
non-zero conditional mass, so no future rule can forget the floor, and
`surface_p90` is computed over live candidates only. **66 → 3 alerts, all 100%
live-driven.**

Both figures are from a surface holding a handful of open complaints, which is
all the corpus could produce at the time. §4.4 is the same subsystem measured at
the load the problem statement actually names, where a different threshold — the
one that survived this fix — turned out to be calibrated just as wrongly. Fixing
an alerting rule against the wrong load is a thing that can be done twice.

---

## 3 · What the repaired smoke sweep found in the existing app

`scripts/smoke_ui.py` had been broken since authentication shipped: 505 lines
with no login handling, so every walk landed on the sign-in form and reported a
clean sweep of a screen that had mounted nothing. The README's "209 controls,
0 console errors" described a run that could no longer happen.

Repaired: it signs in, re-asserts its session per route, isolates its credential
store (it had been writing officers into the **real** `data/muleshield.db`), and
now covers `/model`, `/risk` and `/alerts` alongside a real `login_check` and an
`alert_check`.

On its first working run it immediately found a genuine app bug:

> **`AuthContext.jsx` cleared the session on *any* `/auth/me` failure**, including
> a request aborted by navigating during boot. A user on a flaky connection, or
> who clicked quickly while the console was loading, was silently signed out and
> dropped on the login form mid-shift.

Fixed: only 401/403 — the server actually refusing the token — ends the session.
A network abort or 5xx means *we do not know*, and destroying a valid session on
"we do not know" is the wrong default.

**The 401 that this note used to hedge about has a cause, and it is banal.**
The sign-out control is an icon-only `<button>` with no inner text (`Shell.jsx`).
A sweep that identifies controls by their text clicks it without recognising it,
ends its own session, and then reports a clean bill of health for the login form
on every route after. `smoke_ui.py` already reads `aria-label` as a fallback and
skips it; the independent sweep written for the final build did not, and
reproduced the symptom exactly until it was taught the same trick.

With that closed, the independent sweep is a clean gate: **7 routes, 279 controls
clicked, 3 viewports — 0 uncaught page errors, 0 console errors, 0 failed
requests, 0 4xx/5xx responses**, against a backend carrying 667 open complaints.
It found §4.5 on the way.

---

## 4 · The final build: eight defects found, and the last clause closed

The build was regenerated end to end — corpus, GraphSAGE, embeddings, XGBoost,
every evaluation, the frontend bundle — and then put under the load the problem
statement actually names: **8,000 complaints a day**, replayed through the real
ingestion endpoint. Five defects surfaced, and the last unbuilt clause (§4.6)
was closed on the way through. Four of them were invisible to 397
passing tests, and the reason is the same in every case: **the tests replay
history, and these only appear against a live clock at real load.**

### 4.1 A live complaint could not reach the forward surface

The worst of them.

`backend/routers/complaint.py` stamps an ingested complaint with
`datetime.now(timezone.utc)`. The seeded corpus carries naive **local** time.
`hotspot.parse_ts` reconciled the two by calling `.replace(tzinfo=None)` on the
parsed value — which **discards** the offset rather than applying it. On an IST
machine that read a complaint filed one second ago as **330 minutes old**.
`hotspot_surface` opens a 120-minute window, so:

- every live-ingested complaint was dropped from the forecast at the moment it
  arrived;
- the surface could only ever be driven by the seeded corpus;
- no alert rule could fire on a real complaint, because the rules read that
  surface.

The deliverable this project is built around — *forecast where the cash from
**this** complaint will surface* — was structurally incapable of responding to a
complaint.

**Not a regression from this work.** The same `.replace(tzinfo=None)` sits in
`backend/routers/complaint.py` at `f76e842`, before any of this started. It was
harmless there: its only caller differenced two timestamps *from the same chain*,
so the error cancelled on both sides. Phase 2 lifted the parser into
`engine/hotspot.py` to have one implementation instead of two — the right move —
and made a latent bug load-bearing.

**Why the suite missed it.** Every hotspot test and the whole evaluation replay a
historical epoch with an `as_of` drawn from the same naive-local corpus. The
offset cancels there too. Only a genuinely UTC-stamped complaint meeting a real
wall clock exposes it, and no automated test walked that path.

**Fixed** in `engine/hotspot.py`: an aware value is converted with `.astimezone()`
and then made naive; a naive value is assumed local. **Guarded** by
`TestIngestedComplaintIsLive` in `backend/tests/test_hotspot.py`, which ingests
through the real endpoint at wall clock and asserts the complaint reads as zero
minutes old, carries mass on a cell of the default surface, and is visible to the
rule pass. No `as_of` anywhere in that class, deliberately.

### 4.2 Every sqlite read ran outside the lock

`backend/db.py` guarded **writes** with a module lock. Seventeen read functions
went straight to `conn.execute()` on the same single connection. The background
alert tick runs in a threadpool alongside HTTP handlers, so reads and writes
interleaved on one connection object constantly under load.

The failure mode was not a clean error. The module lost track of which exception
it was raising: a UNIQUE-index violation — which `insert_alert` catches *by
design*, to suppress a duplicate — arrived as a bare `sqlite3.DatabaseError` and
escaped as a **500**. The stress pass produced **43** of those, plus
`InterfaceError: bad parameter or other API misuse` on unrelated reads.

**Fixed:** the lock is now an `RLock` and every access takes it. Three
read-then-write pairs (`approve_reset`, `deny_reset`, `seed_recipients`) were
also split across the boundary and are now single transactions — two
administrators actioning one reset request would both have read "pending" and
both minted a token.

### 4.3 The alert id allocator could silently discard a real alert

`next_alert_id()` was `COUNT(*) + 1`. Two threads allocating at once propose the
same id; and after any deletion, `COUNT(*) + 1` names an id that already exists.
Either way the INSERT hits the PRIMARY KEY — and `insert_alert` catches a UNIQUE
violation to mean *duplicate suppressed*, so it would have **discarded a genuine
alert while reporting normal operation**. Silently dropping an alert is the worst
failure this subsystem has.

**Fixed:** `MAX + 1` over the numeric suffix, read under the same lock as the
INSERT, and `insert_alert` now distinguishes the dedupe index from the primary
key — the first is suppressed, the second is raised.

### 4.4 At the stated national load, one rule pass raised 685 alerts

`R-HIGH-CONVERGE` fired on "three or more open cases point at this cell". That is
a sound rule on a surface holding a dozen complaints, which is the condition it
was tuned against.

Measured at 8,000 complaints/day — the rate the problem statement names twice —
**664 of 666 cell-windows held three or more cases.** The mean was 10.7 and the
maximum 35. One pass raised **685 alerts, 618 of them HIGH**, one concerning
**₹2,452**. An alert that fires on the average is not an alert; an inbox of 685 is
an inbox nobody opens, which is the same failure as raising none.

**Fixed** by making the threshold relative to the surface it reads — the same
discipline `surface_p90` already used. A cell must now carry `CONVERGE_EXCESS`
(2.5) times the mean case count of the live cell-windows, with `CONVERGE_CASES`
(3) as an absolute floor beneath it. At national load that lifts the cut to ~27
cases; on a quiet surface the mean approaches 1 and the rule collapses back to
the original "three or more". A `HIGH_MIN_RUPEES` floor of ₹1 lakh was added,
because `R-HIGH-CONVERGE` had no rupee floor at all, and `WATCH_PERCENTILE` moved
from the top decile to the top 2% — 67 WATCH items an hour is a feed, not a
watchlist.

**Re-measured on an identical surface: 685 → 23** (9 HIGH, 14 WATCH), smallest
₹120,685, every one in a district the surface actually ranks.

**But one pass is not an hour, and the screenshots caught the difference.** Left
running for forty minutes with the 60-second tick and continued ingestion, the
inbox reached **108 alerts in a single dedupe hour — 25 CRITICAL, 51 HIGH, 32
WATCH.** Dedupe suppresses a repeat of the same rule on the same cell and window
within the hour; it does not suppress a *different* cell crossing the line as the
surface moves, and at national load the surface moves constantly.

`R-CRIT-RUPEES` is the reason, and it is the one threshold this fix did not
touch: **₹50 lakh forecast inside the golden hour.** At 8,000 complaints a day
there is on the order of ₹10 crore in flight nationally at any moment, so ₹50
lakh in one 12 km cell is no longer exceptional — the median CRITICAL raised was
₹3.08 crore and the smallest was ₹50.28 lakh.

That is left as an absolute threshold on purpose. Unlike convergence, "₹50 lakh
is about to be withdrawn in the next hour" means the same thing whatever else is
happening, and it is not our call how many of those a desk can work. It is
`MULESHIELD_CRIT_RUPEES`, and **a deployment must set it against the force's own
capacity** — ₹2 crore would bring CRITICAL to a handful an hour on this data.
Stated here rather than quietly tuned to make a screenshot look better.

All four constants are environment-tunable, because an alert budget belongs to
the force running it.

### 4.5 Every toast on the alert inbox threw

`AlertInbox.jsx` did `const { toast } = useToast()`. `useToast()` returns the push
**function itself** — `RiskHeatmap.jsx` and `TriageFeed.jsx` both call it
correctly. Destructuring `{ toast }` off a function yields `undefined`, so every
toast on that screen threw `TypeError: toast is not a function`.

It hid in the happy path because the throw lands inside a promise handler. But it
broke acknowledgement outright: `toast()` is the **first** statement in that
`.then`, so `load()` and `setSelectedId()` after it never ran — **the inbox did
not refresh once an officer closed an alert.** That is deliverable (d) failing
quietly, in the one interaction the deliverable exists for.

Found by an independent frontend sweep clicking every control on every route
against a fully loaded backend. Fixed to `const toast = useToast()`.

**After the fix, that sweep is clean:** 7 routes, 279 controls clicked, 3
viewports — 0 uncaught page errors, 0 console errors, 0 failed requests, 0
4xx/5xx responses.

### 4.6 Evidence documentation — the last named clause, now built

`COMPLIANCE_AUDIT.md` finding 4.5 recorded that the problem statement names
*"evidence documentation"* in deliverable (c) and the repository had no upload
path, no store, no accounting, and not even `python-multipart`. It was carried
as the one named clause with nothing behind it.

**A file uploader would have closed it on paper and been worthless in a
courtroom.** Four properties are what separate an artefact that survives being
produced from an attachment:

| | What it does | Where |
|---|---|---|
| **Hash at collection** | SHA-256 taken from the bytes as they arrive and re-checked on every read. A file that no longer matches is served as a **409**, never as evidence | `evidence.stage`, `evidence.verify_item` |
| **Chain across the case** | Each artefact carries the previous one's `entry_hash`, so the set cannot be added to, removed from or reordered without breaking every link after it | `db.chain_hash`, `evidence.verify_case` |
| **No delete** | Withdrawal is a status with an actor and a stated reason. The artefact, its hash and its position stay on the record | `db.withdraw_evidence` |
| **A certificate** | **BSA 2023 s.63** (which replaced IT Act s.65B in July 2024), generated from the store with a live re-verification, so it cannot describe artefacts the store does not hold | `evidence.certificate` |

**Two decisions that look like paranoia and are not.** Downloads are always
`application/octet-stream` with `nosniff` and an attachment disposition, never
the content type the uploader declared — an uploaded `.html` or `.svg` served
back under its own type, on the same origin as the console, is stored XSS
against the next officer who opens the case. And the supplied filename is
sanitised for display but never used as a path: the stored file is named by the
artefact id.

**The chain walk was wrong on the first attempt, and a test caught it.**
`verify_case` originally carried the *stored* `entry_hash` forward from row to
row. Rewriting one row in sqlite therefore flagged that row and then
resynchronised, so every artefact after it verified clean — a per-row checksum
with extra steps, which would have let somebody alter one artefact and hand over
a report showing a single isolated problem. It now carries the **recomputed**
hash forward, so one broken link invalidates everything downstream, which is the
property the whole construction exists for. `test_a_rewritten_row_breaks_the_chain`
asserts it.

**47 tests**, and the ones that matter are adversarial: they edit the bytes on
disk, delete them, rewrite a row in sqlite and remove a row, because a custody
system that only holds when nobody touches it is not a custody system.

**What it does not claim.** The chain establishes that a case's set of records is
internally consistent. It is not anchored outside the operator's own store, so a
party with write access to the whole table could recompute it end to end. That
limit is written into `db.chain_hash`, printed on the certificate under *Stated
limitations*, and carried in §7 — because a document that overclaims is worse
than no document.

### 4.7 Three defects an independent audit found that this one had missed

`INDEPENDENT_AUDIT.md` re-derived the project's claims rather than reading them,
and turned up three things neither self-audit had recorded. All three are fixed;
they are recorded here because an audit that only lists what its own author found
is an audit with a blind spot the size of its author.

**The first `/hotspots/cells` call took 75 seconds.** `state.hotspot_surface`
filtered complaints in the wrong order: it called `hotspot_entry()` — the feature
builder, the GraphSAGE head and the conditional-logit ranker over 25 candidate
ATMs — for **every complaint in the store**, and only then discarded the result
for falling outside the 120-minute window. On a 2,500-complaint corpus with two
complaints in the window, that is 2,498 full inferences thrown away. Measured
**75.7 s cold against 7.9 ms warm**, and the live API served its first request in
67 s.

Nothing caught it because `bench_golden_hour.py` deliberately warms the path
before timing it, so **every published latency was a warm-cache number**, and the
60-second scheduler tick simply overran its own interval on the first pass. No
measurement was wrong; what it cost was the demo — a judge opening the Risk
Heatmap, the flagship screen, waited over a minute on a blank map.

Fixed by hoisting the age test above the model call: the complaint's timestamp is
already on its record, so the cheap filter runs first and the expensive one only
against what survives. **Re-measured against a freshly booted server: 33 ms cold
on an empty window, and 47 ms median while carrying 667 open complaints at the
problem statement's national rate.** The first call is no longer distinguishable
from the rest, which is the only version of this that a demo survives.

**`python-multipart` was missing from `requirements.txt`.** It is installed in
the local `venv`, so the evidence module and its 47 tests passed here — and a
clean `pip install -r requirements.txt`, which is what the Dockerfile does and
what the README tells a reader to do, would have produced an environment where
the newest deliverable did not work at all. One line, and the highest
severity-per-character finding in either audit.

**`evaluate_hotspots.py` printed one number under another one's name.** It
reported `false_cells_per_hit` — `(k·n − hits)/hits`, the cells visited that hold
nothing — under the label *"cells searched per genuine interception"*, which is
`k/hit_rate`. At k=5 those are **4.25 and 5.25**. §6 of this document quoted 5.25
correctly, so the prose was right and the program was wrong, which is the more
dangerous direction: the program is what someone re-runs. It now prints both,
each under its own name.

A fourth finding — `POST /complaint/{id}/note` singular against `GET .../notes`
plural — is left as it is. The asymmetry is real and costs a 405 to anyone who
guesses, but renaming a shipped route to fix a cosmetic inconsistency is a worse
trade than documenting it.

---

## 5 · Open risks

### 5.1 The corpus arrival rate, not the corpus clock

The previous revision of this audit filed this as *"the dataset clock — the
highest-probability demo failure"* and prescribed regeneration. **That diagnosis
was wrong, and running the fix is what showed it.**

Regeneration was performed. The corpus now ends at wall-clock time. The forward
surface was still empty, and here is why: 2,500 complaints spread across 120 days
is **21 complaints a day for the whole of India**. The open window is 120
minutes, so it holds 1.7 complaints in expectation — and on the freshly generated
corpus it held **zero**. Regeneration fixes the clock. The clock was not the
problem. The arrival rate is.

This is a property of the corpus, not of the system. The problem statement names
the real rate twice: roughly **8,000 complaints a day** on NCRP, which puts about
670 complaints in a 120-minute window. That is the condition the surface was
designed for and the condition the convergence rules were tuned against.

**The fix is `scripts/seed_live_feed.py`**, which replays the corpus through the
real `POST /api/v1/complaint/ingest` at the stated national rate, spread across
the window. Authentication, chain synthesis, GNN inference and ATM ranking all
run exactly as in production; nothing is written past the API and no metric is
touched. It is a load condition, not a result.

**Before demoing:**

```bash
python scripts/generate_data.py          # only if the corpus is days old
python engine/train_gnn.py && python engine/embed.py && python engine/train_xgb.py
python scripts/evaluate_hotspots.py && python scripts/export_metrics.py
cd frontend && npm run build && cd ..
# start the backend, then:
python scripts/seed_live_feed.py --evaluate
```

Regeneration and retraining take about twenty minutes. The live feed takes
seconds, and it is the step that decides whether the flagship screen has anything
on it.

### 5.2 The corpus is not heavy-tailed enough for PAI to shine

Measured: top city 46 complaints against 39 at rank 12 — a 1.2× spread across 79
districts, where real Nuh against a typical district is orders of magnitude. The
*ranking* matches published reality (UP, Maharashtra, Jharkhand lead; Jamtara #2
by city); the *concentration* does not. PAI rewards concentration, so the real
figure would be higher, not lower. Fold a concentration parameter into the
regeneration above — it is nearly free since regeneration is mandatory anyway.

### 5.3 The forecast does not beat distance from the traced terminal

0.9517 against 1.0000 at k=5. Published in the ledger, printed by the evaluation
script, and guarded by a test that fails if the baseline is deleted. The honest
framing: the **trace** earns the value — victim-city-only with no trace scores
0.0193 — and the forecast adds the time dimension, rupee weighting, and
cross-complaint aggregation that a distance rule cannot supply, and which are what
let many complaints aggregate into one national surface at all.

---

## 6 · Headline numbers, and what produced each

| Claim | Value | Written by |
|---|---|---|
| Forecast vs historical density (Pratibimb) | **PAI 26.86 vs 1.09 — 24.6×** | `scripts/evaluate_hotspots.py` |
| Hit rate @5 cells | **0.9517** against 0.0612 for historical density | same |
| Rupees covered @5 | **0.9561** | same |
| Median lead time | **41.5 min**, 84.1% ≥15 min | same |
| Cells searched per interception @5 | **5.25** | same |
| Prior share of the surface | **0.1304** against a 0.15 cap, measured | same |
| Complaint → alert dispatched | **p95 0.25 s** vs a 60-minute golden hour | `scripts/bench_golden_hour.py` |
| Ingestion throughput | **3,499/min = 5.04M/day, 630× NCRP load** | same |
| Mule detection F1 | **0.9050** vs 0.8565 best non-graph | `engine/train_gnn.py` |
| Alerts, one pass at 8,000/day | **23** (9 HIGH, 14 WATCH), from **685** before §4.4 | `scripts/seed_live_feed.py --evaluate` |
| Alerts, one hour of sustained load | **108** (25 CRITICAL, 51 HIGH, 32 WATCH) — see §4.4 on `MULESHIELD_CRIT_RUPEES` | same, left running |

Latency is measured under the load the system is designed for — 667 open
complaints on the surface — rather than against an empty one, which is why p95
moved from 0.11 s to 0.25 s and why that is the more useful number.

Every figure above is in `data/metrics.json` with a `source` and `measured_utc`,
enforced by `tests/test_metrics_ledger.py`, and none is hand-typed into the
console.

---

## 7 · What still is not built

Stated here so nobody has to discover it.

1. **Case dossier export** — the s.63 evidence certificate is generated and
   printable, but there is no general case report covering the forecast, the
   transaction trail and the graph (finding 4.4).
2. **Case audit persistence** — alerts and evidence are in SQLite; the general
   case trail is still in memory (4.7–4.8).
3. **An external anchor for the evidence chain** — it proves the set is
   internally consistent, and a party with write access to the whole table could
   recompute it. Real tamper-evidence needs a signed daily digest, a notary, or
   an append-only log the operator does not own (4.9).
4. **Officer jurisdiction scoping** — recipients are scoped, officers are not (4.6).
5. **Admin roster screen** — endpoints exist, nothing renders them (4.10).
6. **Branch-counter and bulk-payout cash-out** — ATM-only (documented scope boundary).
7. **Model monitoring / drift detection** (2.4).
8. **Real transport** — every channel is a labelled simulation.

---

## 8 · Reproducing this audit

```bash
python -m pytest -q                        # whole suite, both roots
python scripts/evaluate_hotspots.py        # forecast vs three baselines
python -m pytest tests/test_hotspot_leakage.py -v   # the leak guards
python scripts/export_metrics.py
python scripts/bench_golden_hour.py --api http://127.0.0.1:8000 -n 30
python scripts/smoke_ui.py                 # isolated browser sweep

# the two security BLOCKERs, in one line each
curl -i -X POST localhost:8000/api/v1/bank/micro-freeze \
  -H 'Content-Type: application/json' -d '{"account_id":"X","complaint_id":"Y"}'
curl -i -X OPTIONS localhost:8000/api/v1/complaint/list -H 'Origin: https://evil.example'
```

---

## 9 · Every problem-statement clause, re-checked

`COMPLIANCE_AUDIT.md` broke the published statement into testable clauses. This
is the same list, scored against the build as it now stands. **Nothing here is
scored from a reading of the code** — each row names what establishes it.

### (a) Predictive Analytics Engine

> *AI/ML-based system to analyse historical cybercrime and financial data to
> predict potential withdrawal hotspots. Features include pattern detection,
> geospatial risk modelling, and real-time alerts.*

| Clause | Before | Now | Established by |
|---|---|---|---|
| Analyse historical cybercrime **and** financial data | ✅ | ✅ | 2,500 complaints + 622,188 transactions + 1,000 ATMs fused in `engine/feature_builder.py` |
| Pattern detection | ✅ | ✅ | velocity anomaly, 1-to-N splitting, terminal identification; GraphSAGE **F1 0.9050** vs 0.8565 best non-graph |
| Geospatial risk modelling | 🟡 per-case | ✅ | 222 cells at 12 km, forward intensity surface (`engine/hotspot.py`) |
| Predict potential withdrawal **hotspots** | 🔴 | ✅ | **PAI@5 26.86** against 1.09 for historical density — 24.6× |
| **Real-time alerts** | 🔴 | ✅ | `backend/notify.py` fired by the lifespan tick with no human present |
| Scheduled / batch analytical job | 🔴 | ✅ | `_tick()`, gated by `MULESHIELD_SCHEDULER`, cancelled and awaited on shutdown |

### (b) Risk Heatmap Dashboard

> *GIS-enabled dashboard visualizing real-time and potential risk zones with
> drill-down filters by time, location, and crime category etc.*

| Clause | Before | Now | Established by |
|---|---|---|---|
| GIS-enabled | ✅ | ✅ | Leaflet + OSM, offline basemap fallback |
| Visualising risk zones | 🟡 one circle | ✅ | `L.circleMarker` per cell, absolute rupee area scale |
| **Heatmap** | 🔴 | ✅ | Vector intensity ramp, red/orange/amber/zinc; no plugin added, deliberately |
| **Potential** risk zones | 🔴 | ✅ | Forward windows 0–30 / 30–60 / 60–120 min over open complaints |
| Drill-down by **time** | 🔴 | ✅ | Window pills + `as_of` picker |
| Drill-down by **location** | 🔴 | ✅ | National → state → district → cell breadcrumb |
| Drill-down by **crime category** | 🟡 list only | ✅ | Multi-select over the nine fraud types, applied server-side |

### (c) Law Enforcement Interface

> *Secure interface for investigators to access alerts, intelligence reports, and
> evidence documentation.*

| Clause | Before | Now | Established by |
|---|---|---|---|
| Interface for investigators | ✅ | ✅ | Seven screens, case workflow, notes, timeline, audit view |
| **Secure** | 🔴 fails | ✅ | 13 endpoints on `Depends(current_user)`; anonymous `POST /bank/micro-freeze` returns 401 + `WWW-Authenticate: Bearer`; CORS allowlisted |
| Access **alerts** | 🔴 | ✅ | `/alerts` inbox, severity ordering, delivery record, required disposition |
| **Intelligence reports** | 🔴 | 🟡 | Alert detail carries headline, contributing cases, decomposition and delivery trail; CFCFRMS/Samanvaya payloads are emitted; the s.63 certificate is generated and printable. A general case dossier is still not built |
| **Evidence documentation** | 🔴 | ✅ | `POST /api/v1/evidence/{case_id}` with SHA-256 taken at collection and re-checked on every read, a hash chain across the case, withdrawal-not-deletion, and a **BSA 2023 s.63 certificate**. 47 tests, including tampering with the bytes on disk and rewriting a row in sqlite |

### (d) Alert & Notification System

> *Real-time notifications to law enforcements, banks, and I4C officers via SMS,
> email, API, or dashboard triggers.*

| Clause | Before | Now | Established by |
|---|---|---|---|
| **SMS** | 🔴 a string | 🟢 simulated | `backend/adapters/sms.py`, full delivery contract, labelled *simulated* on screen |
| **Email** | 🔴 absent | 🟢 simulated | `backend/adapters/email.py`, same |
| **API (outbound)** | 🔴 absent | 🟢 simulated | `backend/adapters/webhook.py` emits real CFCFRMS and Samanvaya payloads, `v1-proposed` |
| **Dashboard triggers** | 🟡 echo only | ✅ | `ALERT_RAISED` broadcast from the rule pass, not from a user action |
| Recipients: **LEA, banks, I4C** | 🔴 no concept | ✅ | `alert_recipients`, roles LEA/I4C/BANK, scoped by state and district |
| Delivery is recorded, retried, dead-lettered | 🔴 | ✅ | `alert_deliveries`; `MULESHIELD_ADAPTER_FAIL_RATE=1.0` drives real failure → retry → `dead` |
| Alert volume is actionable | — | 🟡 | **23 per pass** at 8,000 complaints/day, from 685 before §4.4 — but **108 per hour** sustained, because `R-CRIT-RUPEES` is still an absolute ₹50 lakh. A deployment must set it. §4.4 |

**The transports are mocked, and that is stated everywhere it could mislead** —
on screen per channel, in `INTEGRATION_SEAMS.md` §3, and here. A live SMS route
into India needs a paid gateway and DLT registration; a live CFCFRMS call needs
credentials only I4C can issue. Everything above the adapter is real, and
swapping in a gateway is one module.

### Background clauses

| Clause | Before | Now | Established by |
|---|---|---|---|
| ~8,000 complaints/day, rising | 🔴 unmeasured | ✅ | **3,499/min = 5.04M/day, 630× headroom**, and the system is now *evaluated* at that load, not just benchmarked at it |
| Golden hour / 1930 rail | 🟡 implied | ✅ | complaint filed → alert dispatched, **p95 0.25 s** against 60 minutes |
| Help banks via **CFCFRMS** | 🔴 claimed, absent | 🟡 | Payload and field mapping published, `v1-proposed`. Not a live integration, and no longer claimed to be |
| Coordination by I4C across state and local LEAs | 🔴 | ✅ | Three recipient tiers; a national desk, state force and district unit all receive a scoped alert |
| "In Advance" | 🟡 one case | ✅ | median lead time **41.5 min**, 84.1% of hits with ≥15 minutes of warning |

### What is honestly still missing

Every clause the problem statement names now has an implementation behind it.
Three things remain, all deliberate, all in §7:

1. **A general case dossier** — the s.63 evidence certificate is generated and
   printable; a report covering the forecast, the transaction trail and the
   graph is not.
2. **Case-audit persistence** — alerts and evidence are in SQLite; the general
   case trail is still in memory.
3. **Officer jurisdiction scoping** — alerts route by state and district, but the
   `users` table still knows only `is_admin`.

None of these is a forecast capability, and none is a named clause. They are
records plumbing, which is the right place for the remaining gap to be.
