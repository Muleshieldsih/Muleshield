# Compliance Audit — MuleShield AI against SIH 26184

**Problem Statement:** 26184 — *Development of a Predictive Analytics Framework for
Cybercrime Complaints to Forecast Likely Cash Withdrawal Locations in Advance,
Enabling Generation of Actionable Intelligence for Timely and Proactive Cybercrime
Intervention.*
**Organisation:** Ministry of Home Affairs · Indian Cyber Crime Coordination Centre (I4C), CIS Division
**Category:** Software
**Audited:** 1 Sep 2026 · branch `feat/auth-dashboard` · commit `f76e842`

This audit reads the **published problem statement text as the specification** and checks
the repository against it clause by clause. It is not an ML audit — the model work is
audited separately in `OVERNIGHT_ML_AUDIT.md` and is not re-litigated here. Every finding
below cites the file that establishes it, so each one is checkable in under a minute.

---

## 0 · Verdict

**The engine is strong. The framework around it is not built.**

MuleShield AI does the hardest technical thing the problem statement asks for — forecast
where cash comes out — and does it with unusual honesty. What it does not do is any of the
things the problem statement asks for *around* that forecast: it has no aggregate hotspot
layer, no heatmap, no alerting, and no path to a bank or another jurisdiction.

Three of the four named deliverables are substantially incomplete.

| # | Deliverable | Status | Assessed |
|---|---|---|---|
| **a** | Predictive Analytics Engine | 🟡 **Partial** | ~60% |
| **b** | Risk Heatmap Dashboard | 🔴 **Largely absent** | ~25% |
| **c** | Law Enforcement Interface | 🟡 **Partial** | ~40% |
| **d** | Alert & Notification System | 🔴 **Not met** | ~15% |

Findings are graded:

- **BLOCKER** — a named deliverable clause has no implementation, or a security defect
  contradicts a claim the project makes about itself.
- **MAJOR** — the clause is partially met in a way a judge will notice.
- **MINOR** — real, cheap, and worth closing before submission.

**7 BLOCKER · 11 MAJOR · 8 MINOR.**

---

## 1 · Scope conformance — the finding that reframes the rest

**Finding 1.1 — BLOCKER. The repository states a different problem statement than the one
it is submitted against.**

`README.md:14` reads:

> **Problem Statement ID:** SIH26184 — *Identification of Money Mule Accounts and ATM
> Geolocation for Cashout Interception*

The official title is *"Development of a **Predictive Analytics Framework for Cybercrime
Complaints** to Forecast Likely Cash Withdrawal Locations in Advance."* Same ID, materially
narrower scope. The README's version describes **per-case interception**; the official
version describes **a framework operating over the national complaint stream**.

This is not a wording quibble. It is the reason the rest of this audit finds what it finds,
because the system was built to the narrower title and is complete against *that*.

**Finding 1.2 — BLOCKER. The system predicts per-complaint candidates, not hotspots.**

Deliverable (a) states the object of prediction explicitly: *"predict potential withdrawal
**hotspots**"*. Deliverable (b) states it again: *"real-time and **potential** risk zones"*.
The description states the operational purpose: *"deploying special teams or alerting local
banks and ATMs in **high-risk areas**"*.

What the system produces is a ranked list of 5 ATMs **for one open complaint**
(`backend/routers/predict.py`, `GET /api/v1/predict/cashout/{complaint_id}`). Every screen
in the console is scoped to a single `complaint_id` — `useActiveComplaint` is imported by
`TacticalMap.jsx`, `ForensicGraph.jsx` and `Interception.jsx` alike.

A per-complaint top-5 cannot support the intervention the problem statement names. A special
team cannot be deployed to a prediction that only exists *after* a complaint is filed and
*only* for that complaint. "High-risk area" is an aggregate over many complaints, forward in
time — a district-hour risk surface — and no such object exists anywhere in the codebase.

The project has already identified this itself. `docs/DISCUSSION_NOTES.md:342` rejects
Hawkes processes for per-case ranking (correctly — the ranker sits at the Bayes bound) and
then writes:

> Where Hawkes *would* genuinely fit is the **strategic layer** — recurring ATM hotspots
> across many cases — which is a different question from per-complaint ranking.

That paragraph is the missing deliverable, correctly diagnosed and not built.

**Finding 1.3 — MAJOR. "In Advance" is met only within one case's countdown.**

The countdown (`time_to_cashout_minutes`, 11.82 min MAE) gives advance warning measured in
*minutes after a complaint arrives*. The problem statement's "in advance" is paired with
"proactive" and with "deploying special teams" — an operational horizon of hours to days,
which is a forecast the system does not produce.

---

## 2 · Deliverable (a) — Predictive Analytics Engine

> *AI/ML-based system to analyse historical cybercrime and financial data to predict
> potential withdrawal hotspots. Features include pattern detection, geospatial risk
> modelling, and real-time alerts.*

Four testable clauses.

| Clause | Status | Evidence |
|---|---|---|
| Analyse historical cybercrime **and** financial data | ✅ **Met** | 2,500 complaints + 622,188 transactions + 1,000 ATMs; `engine/feature_builder.py` fuses both |
| Pattern detection | ✅ **Met** | `engine/graph_engine.py` — velocity anomaly, 1-to-N fund splitting, terminal-node identification; GraphSAGE F1 0.8955 |
| Geospatial risk modelling | 🟡 **Partial** | Real and good, but **per-case**: conditional-logit ATM choice + adaptive search zone (`engine/xgb_model.py`). No area-level risk surface |
| Predict potential withdrawal **hotspots** | 🔴 **Absent** | See 1.2 |
| **Real-time alerts** | 🔴 **Absent** | See §5 |

**Finding 2.1 — BLOCKER. There is no scheduled or batch analytical job of any kind.**

`requirements.txt` contains no scheduler (`apscheduler`, `celery`, `rq` all absent) and no
cron/task entry point exists. Every computation in the system is triggered synchronously by
an HTTP request about one complaint. A "predictive analytics framework" that only computes
when a human opens a case is a lookup service, not a framework.

There is no artefact anywhere that answers *"what is the risk in Pune tomorrow evening?"* —
no table, no endpoint, no file.

**Finding 2.2 — MAJOR. The one cross-case analytic that exists is not reachable by a human.**

`GET /api/v1/intel/atms` (`backend/routers/intel.py`) ranks ATMs by **distinct complaints**
touching them, with city, state, lat/lon, risk score and last-seen — deliberately
deduplicated so a 4-way split converging on one machine counts once
(`backend/state.py:618`). Its client wrapper `listAtmIntel` exists at
`frontend/src/services/api.js:93`.

**No React component imports it.** `grep -rn "listAtmIntel" frontend/src --include="*.jsx"`
returns nothing. This is the closest thing in the repository to a hotspot view, it is
already written and already correct, and it renders on no screen.

**Finding 2.3 — MAJOR. No temporal dimension in any aggregate.**

`atm_intelligence()` returns lifetime counts. There is no decay, no time bucketing, no
hour-of-day or day-of-week profile at area level — despite `hour_of_day` and `day_of_week`
already being per-case model features and despite every complaint carrying
`complaint_timestamp`. Deliverable (b) requires drill-down "by time"; nothing aggregates by
time.

**Finding 2.4 — MINOR. No model monitoring or drift detection.**

`data/metrics.json` is written by training and evaluation scripts and read by the console.
Nothing observes live prediction quality, and nothing would notice if the deployed model
degraded.

**Finding 2.5 — MINOR. No outcome capture, so the framework cannot learn.**

A case ends at status `Resolved` or `Closed` (`backend/models/schemas.py`,
`CaseUpdateRequest`). There is no field recording **what actually happened** — was the
withdrawal intercepted, was the predicted ATM correct, was the money recovered. That is both
the metric I4C would judge the framework by and the label a production system would retrain
on. Its absence means the system cannot improve after deployment.

---

## 3 · Deliverable (b) — Risk Heatmap Dashboard

> *GIS-enabled dashboard visualizing real-time and potential risk zones with drill-down
> filters by time, location, and crime category etc.*

**This is the weakest deliverable and it is a named one.**

| Clause | Status | Evidence |
|---|---|---|
| GIS-enabled | ✅ **Met** | Leaflet + OSM tiles, offline fallback (`TacticalMap.jsx`) |
| Visualising **risk zones** | 🟡 **Partial** | One `L.circle` search zone for one case (`TacticalMap.jsx:145`) |
| **Heatmap** | 🔴 **Absent** | No heat layer exists |
| **Potential** risk zones | 🔴 **Absent** | Only zones derived from an already-filed complaint |
| Drill-down by **time** | 🔴 **Absent** | No time filter anywhere in the console |
| Drill-down by **location** | 🔴 **Absent** | No state/district/city filter |
| Drill-down by **crime category** | 🟡 **Partial** | Exists on the *list*, not the map (`TriageFeed.jsx:777`) |

**Finding 3.1 — BLOCKER. There is no heatmap in a deliverable named "Risk Heatmap Dashboard".**

`frontend/package.json` lists `leaflet` and no heat plugin (`leaflet.heat`, `heatmap.js`,
`leaflet-heatmap` all absent). No choropleth, no hex-bin, no density layer, no
`d3`/`recharts`/`chart.js`. The map renders exactly three things: one terminal-account
marker, up to five ATM pins, and one translucent circle.

**Finding 3.2 — BLOCKER. The map has no national, state or district view.**

`INDIA_CENTER` at `TacticalMap.jsx:11` is used only as the initial camera position before
the map fits to one case's bounds. There is no zoom level at which an officer sees the
country, a state, or a district with risk rendered on it. An I4C officer — the coordinating
role the problem statement names — has no screen.

**Finding 3.3 — MAJOR. No drill-down filters on the map at all.**

`grep -n "filter" frontend/src/pages/TacticalMap.jsx` returns nothing. The filters that do
exist are on the triage list — `statusFilter`, `fraud_type` and a text search
(`TriageFeed.jsx:491-542`) — and even there **there is no time-range filter**, which
deliverable (b) names first.

**Finding 3.4 — MAJOR. The data to build this already exists and is unused.**

- `data/atm_directory.csv` carries `district` and `state` for all 1,000 machines.
- `data/transactions.csv` carries `district`, `state`, `lat`, `long`, `timestamp`.
- `data/victim_complaints.csv` carries `city`, `state`, `fraud_type`, `complaint_timestamp`
  across **2,500 complaints, 9 fraud categories and 25 states**.

Every axis deliverable (b) asks to filter on is already in the corpus. Nothing aggregates
over any of them.

**Finding 3.5 — MINOR. `district` is not captured on ingestion.**

`ComplaintIngestRequest` (`backend/models/schemas.py:17`) takes seven fields —
`victim_name`, `victim_bank`, `victim_account`, `fraud_type`, `stolen_amount`, `city`,
`state`. No district, no coordinates. A live complaint therefore cannot be placed in the
district-level view that deliverable (b) requires, even once that view exists.

---

## 4 · Deliverable (c) — Law Enforcement Interface

> *Secure interface for investigators to access alerts, intelligence reports, and evidence
> documentation.*

| Clause | Status |
|---|---|
| Interface for investigators | ✅ **Met** — 5 screens, case workflow, notes, timeline, audit view |
| **Secure** | 🔴 **Fails** — see 4.1 |
| Access **alerts** | 🔴 **Absent** — no alert object exists |
| **Intelligence reports** | 🔴 **Absent** — nothing generates or exports a report |
| **Evidence documentation** | 🔴 **Absent** — no attachment capability |

**Finding 4.1 — BLOCKER. Authentication is a client-side gate. No API endpoint requires a token.**

`backend/auth.py` is well designed — it distinguishes `current_user` (401s) from
`optional_user` (returns `None`), and documents why. But across the eight routers:

- `current_user` / `admin_user` appear **only inside `backend/routers/auth.py`**.
- `optional_user` appears in exactly three places: `complaint.py:120`, `complaint.py:159`,
  `freeze.py:35`.
- `predict.py`, `graph.py`, `embeddings.py`, `audit.py`, `intel.py` have **no auth
  dependency at all**.

Consequences, each reproducible with a single `curl` and no credentials:

| Endpoint | Exposes |
|---|---|
| `POST /api/v1/bank/micro-freeze` | **Freezes an account.** The most destructive action in the product, unauthenticated |
| `GET /api/v1/complaint/{id}` | Victim name, bank, account number, amount, city |
| `GET /api/v1/graph/{id}` | The full mule chain — every account number in it |
| `GET /api/v1/audit` | The complete audit trail, including who froze what |
| `GET /api/v1/predict/cashout/{id}` | The forecast itself |

`frontend/src/App.jsx` gates `<Layout />` behind `<Gate />` so the *browser* shows a login
form. Nothing enforces it on the server. The system is exactly as secure as it was before
authentication was added, for any client that is not the React app.

`freeze.py:49` reasons carefully about *"who gets the blame for an irreversible action
against a person's account"* — and then accepts the request from an anonymous caller.

**Finding 4.2 — BLOCKER. `allow_origins=["*"]` together with `allow_credentials=True`.**

`backend/main.py`. This combination is rejected outright by browsers and is an automatic
finding on any security review. Already flagged in `PROJECT_NOTES.md` §1 and still open.

**Finding 4.3 — BLOCKER. There is no alert object, so "access alerts" cannot be met.**

No `Alert` model in `backend/models/schemas.py`, no alerts router, no alert table, no alert
inbox screen. The word "alert" in the codebase refers to WhatsApp message text
(`Interception.jsx`) and to `AlertTriangle` icons. An investigator cannot open the console
and see what needs attention; they can only see a queue of complaints sorted by severity.

**Finding 4.4 — BLOCKER. No intelligence report can be produced.**

There is no case dossier, no printable brief, no CSV, no PDF, no share link — a repo-wide
search for report generation or export turns up nothing. An investigator who needs to attach
the system's intelligence to a chargesheet, or hand it to another jurisdiction, has no
artefact to hand over.

Note the Background section of the problem statement also states that the portal currently
supports *"reports/graphs being pulled on daily basis"* — so a reporting capability is
assumed baseline, not aspirational.

**Finding 4.5 — BLOCKER. No evidence documentation.**

No file upload endpoint, no attachment schema, no storage path, no `python-multipart` in
`requirements.txt` (FastAPI cannot accept a file upload without it). The only investigator
input is free text via `POST /api/v1/complaint/{id}/notes`. "Evidence documentation" is a
named clause and has zero implementation.

**Finding 4.6 — MAJOR. No jurisdiction model, in a problem statement built around jurisdictions.**

The `users` table (`backend/db.py:63`) is:

```
id · username · display_name · password_hash · is_admin · failed_logins · locked_until
created_at · last_login
```

One boolean is the entire authorisation model. There is no state, no district, no agency, no
unit, no role beyond admin. Consequently:

- Every signed-in officer sees every case in the country. No least-privilege scoping.
- The problem statement's *"LEAs at the state and local levels, coordinated by I4C"* — three
  distinct tiers — cannot be represented.
- *"Real-time actionable intelligence sharing **across jurisdictions**"* is not implementable,
  because there are no jurisdictions to share across.
- There is no bank / FI role at all, despite banks being a named consumer of the intelligence.

**Finding 4.7 — MAJOR. The audit trail is in memory and dies on restart.**

`backend/state.py:519` — `audit_log: list[dict] = []`, capped by `_AUDIT_LIMIT` and trimmed
from the front. `backend/routers/audit.py` states this plainly rather than hiding it, which
is to the project's credit:

> The trail is in-memory and does not survive a restart… what it gives is a truthful record
> of the running session, not a compliance-grade archive.

But SQLite is **already initialised in the same process** for users, sessions and password
resets (`backend/db.py`, `db.init()` called from `main.py`'s lifespan). The persistence
mechanism exists and the audit trail does not use it.

**Finding 4.8 — MAJOR. The audit trail omits the events that matter most for compliance.**

Recorded: status transitions, assignment, notes, freezes. **Not recorded anywhere queryable:**

- successful and failed logins, lockouts, logouts
- administrator actions — user creation, unlock, reset approval/denial
- **case access** — who viewed which victim's data
- prediction requests
- data exports (once they exist)

These go to Python `logging` only (`logger.info("[AUTH] %s signed in", …)`) — not queryable,
not retained, not exportable. For a system that can freeze a citizen's account, *"who read
this victim's file"* is as auditable as *"who froze it"*, and the current design cannot
answer it.

**Finding 4.9 — MAJOR. The audit trail is not tamper-evident.**

A plain Python list with sequential IDs (`AUD-000001`). No hash chain, no signature, no
append-only guarantee, no integrity check. Anything derived from it cannot be certified
under **BSA 2023 s.63 / IT Act s.65B**, which is the statutory form Indian courts require for
electronic records. For an MHA problem statement this is a live gap, and a hash-chained log
with a certificate export is roughly an afternoon's work.

**Finding 4.10 — MINOR. The administrator interface is built and unreachable.**

`GET/POST /api/v1/auth/users`, `POST /users/{id}/unlock`, `GET /reset-requests`,
`POST /reset-requests/{id}/approve|deny` all exist and are correctly protected. Their client
wrappers exist (`api.js:82-90`). **No screen renders any of them** — `App.jsx` routes are
`/`, `/map`, `/graph`, `/intercept`, `/model` and nothing else. The password-reset flow is
therefore *designed* around an administrator who has no way to action a request.

**Finding 4.11 — MINOR. No separation of duties on reset approval.**

`approve_reset(request_id, admin["display_name"])` (`backend/db.py:465`) does not prevent an
administrator raising a reset for another officer and approving it themselves. Low impact
given the roster size; trivially closed with one check.

---

## 5 · Deliverable (d) — Alert & Notification System

> *Real-time notifications to law enforcements, banks, and I4C officers via SMS, email, API,
> or dashboard triggers.*

Four named channels and three named recipient classes. **No channel is implemented and no
recipient class exists.**

| Channel | Status | Evidence |
|---|---|---|
| **SMS** | 🔴 Simulated | `Interception.jsx:161` — `setDispatchStatus('SMS queued … (simulated).')`. Sets a string. Sends nothing |
| **Email** | 🔴 **Does not exist** | No `smtplib`, no SMTP config, no mail library in `requirements.txt`. Zero occurrences repo-wide |
| **API (outbound)** | 🔴 Absent | No webhook, no callback, no outbound client, no delivery contract. `micro-freeze` is an internal simulator returning `FRZ-xxxxxxxx` |
| **Dashboard triggers** | 🟡 Partial | WebSocket exists but only *echoes actions a user just took* |

**Finding 5.1 — BLOCKER. The problem statement names four channels; the repository implements zero.**

The nearest thing to a real dispatch is `Interception.jsx:154`, which opens
`https://api.whatsapp.com/send?phone=91…` — a deep link into **the operator's own WhatsApp
client**. It is a browser navigation, not a server-side notification: it requires a human at
a keyboard, has no delivery record, no retry, no acknowledgement, and cannot reach anyone the
operator has not personally messaged.

**Finding 5.2 — BLOCKER. There is no trigger engine, so nothing is "real-time" in the sense meant.**

The four WebSocket events are `NEW_COMPLAINT` (`complaint.py:54`), `CASE_UPDATED`
(`complaint.py:144`), `PREDICTION_READY` (`predict.py:171`) and `FREEZE_EXECUTED`
(`freeze.py:90`). Every one is emitted **as a consequence of a user action in the console**
and broadcast to every connected socket indiscriminately.

There is no rule engine — nothing of the form *"if predicted risk in district D exceeds T for
window W, notify the recipients responsible for D"*. That conditional is what makes a
notification proactive rather than an echo, and it is the whole point of the deliverable.

**Finding 5.3 — MAJOR. No recipient model.**

The problem statement names three recipient classes — **LEAs, banks, and I4C officers**. The
system has one user table with an `is_admin` flag (see 4.6). There is no bank contact, no ATM
operator, no I4C desk, no escalation path, no routing rule mapping an area to the people
responsible for it. Even with a working SMS gateway, there is no answer to *"send it to
whom"*.

**Finding 5.4 — MAJOR. No delivery record, and that is itself an audit gap.**

Whether an alert was sent, delivered, read and acknowledged is evidence — it establishes that
an LEA was warned. Nothing persists it. `dispatchStatus` is React state that vanishes on
navigation.

**Finding 5.5 — MINOR. The simulated SMS reports success unconditionally.**

`handleSMS` (`Interception.jsx:161`) sets `dispatchOk(true)` with no condition. The WhatsApp
path at least validates a 10-digit number first. A demo control that always reports success
is the kind of thing a judge presses twice.

---

## 6 · Background clauses — the operating context

The Background section is specification too. Four clauses are testable.

**Finding 6.1 — MAJOR. The ~8,000 complaints/day figure is unaddressed.**

The problem statement states the load explicitly and says it will rise. The system:

- loads **60** complaints into the console (`QUEUE_LIMIT = 60`, `App.jsx`)
- holds all case state, notes and audit in **process globals** (`backend/state.py`)
- must run as a **single process** — `PROJECT_NOTES.md` §5 states this is a hard constraint,
  because two instances behind a load balancer would disagree
- has **no ingestion benchmark** anywhere

8,000/day is only ~0.09 requests/second, so the system would very likely cope — but nothing
in the repository demonstrates it, and the architecture explicitly cannot scale horizontally.
Given that the problem statement raises the number twice, a judge will ask, and the honest
answer today is *"we have not measured it."*

**Finding 6.2 — BLOCKER. No CFCFRMS integration path exists, and the README claims one.**

The description states the intelligence should help banks *"through the Citizen Financial
Cyber Fraud Reporting and Management System, enabling faster fund blocking"*. CFCFRMS is
named in `README.md` prose and appears in **no schema, no adapter, no endpoint, no
document**. `README.md` claims:

> **Enterprise Integration:** Standardized API schemas ready for live integration with the
> National Cybercrime Reporting Portal (NCRP / 1930) and NPCI Switch.

There is no such schema in the repository. This is a checkable claim that does not hold, and
it is more damaging than the gap it papers over — `docs/DISCUSSION_NOTES.md:169` handles the
same limitation honestly and well, and the README should match that standard.

**Finding 6.3 — MAJOR. "Sharing across jurisdictions" has no mechanism.**

Follows from 4.6. No jurisdiction entity, no transfer, no share, no export (4.4), no
cross-agency notification (5.3). Four independent gaps close the same clause.

**Finding 6.4 — MINOR. No bank- or FI-facing surface.**

Banks are named twice as consumers of the output. The console is entirely LEA-facing. A bank
receives nothing from this system except a simulated freeze confirmation the LEA triggers.

---

## 7 · Hygiene items carried over

Verified still open at commit `f76e842`.

| # | Item | Severity |
|---|---|---|
| 7.1 | Phase 1–6 work sits on `sameer` / `feat/auth-dashboard`; `main` is behind | MINOR |
| 7.2 | `Dockerfile` written 30 Aug, **never built** (`PROJECT_NOTES.md` §1) | MAJOR |
| 7.3 | Dataset generated 29 Aug — complaint timestamps are relative to generation, so the queue now reads **"5d ago"** and drifts daily. **Regenerate before demoing** | MAJOR |
| 7.4 | 3-minute video absent. Deck exists in `deck/` but is **untracked** (`git status`) | MAJOR |
| 7.5 | Deck title slide still missing Team Name / Team ID (`DISCUSSION_NOTES.md` §To-do) | MINOR |
| 7.6 | `case_notes` and `audit_log` are not cleared by `state.load_all()` while every other store is | MINOR |

Resolved since the last review, confirmed: the `is_terminal` header corruption in
`data/transactions.csv` is fixed — the header now reads correctly.

---

## 8 · Built and unreachable

Work that is already done, already correct, and earns the project nothing because no screen
renders it. This is the cheapest credit available anywhere in the repository.

| Capability | Backend | Client wrapper | Screen |
|---|---|---|---|
| Cross-case recurring ATMs | ✅ `intel.py` | ✅ `listAtmIntel` | ❌ **none** |
| Officer roster | ✅ `auth.py` | ✅ `listUsers` | ❌ **none** |
| Reset request queue | ✅ `auth.py` | ✅ `listResetRequests` | ❌ **none** |
| Reset approve / deny | ✅ `auth.py` | ✅ | ❌ **none** |

`intel/atms` in particular is most of a hotspot table already: it returns atm_id, distinct
complaints, total amount, city, state, lat/lon, risk score and last-seen, correctly
deduplicated. Rendering it is a few hours and converts Finding 2.2 from absent to partial.

---

## 9 · Remediation, ordered by credit per hour

| # | Work | Closes | Effort |
|---|---|---|---|
| 1 | **Require `current_user` on every mutating and PII-returning endpoint; narrow CORS to the console origin** | 4.1, 4.2 | ~4 h |
| 2 | **Hotspot forecast** — decayed complaint/cash-out intensity per district × hour-band, written to a table on a schedule; baselined honestly like everything else in this repo | 1.2, 2.1, 2.3 | ~1 d |
| 3 | **Risk heatmap screen** — national → state → district choropleth over that table, with time / category / location drill-down; fed by `intel/atms` + the new aggregate | 3.1, 3.2, 3.3, 2.2 | ~1 d |
| 4 | **Notification service** — one module, pluggable SMS / email / webhook adapters (mocked adapters are fine and honest), a persisted `alerts` table with delivery state and acknowledgement, and a threshold rule engine over the hotspot table | 5.1, 5.2, 5.4 | ~1 d |
| 5 | **Jurisdiction + role on `users`** — state, district, agency, role ∈ {LEA, I4C, BANK}; scope the queue and route alerts by it | 4.6, 5.3, 6.3 | ~4 h |
| 6 | **Audit → SQLite, hash-chained, with a s.65B certificate export**; extend coverage to logins, admin actions and case access | 4.7, 4.8, 4.9 | ~5 h |
| 7 | **Case dossier export (PDF/HTML) + file attachments** on a case | 4.4, 4.5 | ~4 h |
| 8 | **Alert inbox screen** over the alerts table | 4.3 | ~3 h |
| 9 | **Wire the admin roster screen** | 4.10 | ~1 h |
| 10 | **Correct the PS title in `README.md`; delete or substantiate the "standardized API schemas" claim; publish a CFCFRMS/NCRP field-mapping document** | 1.1, 6.2 | ~2 h |
| 11 | **Ingestion benchmark at 8,000/day**, plus `district` on the ingest schema | 6.1, 3.5 | ~2 h |
| 12 | **Outcome field on case closure** (intercepted / not / recovered amount) | 2.5 | ~1 h |

Items 1, 10 and 12 are hours and close findings graded BLOCKER. Items 2–4 are the substance:
they convert three partially-met deliverables into defensible ones.

---

## 10 · What not to do

Two warnings, because this project's failure mode is over-investing in the part that is
already finished.

**Do not tune the models further.** `OVERNIGHT_ML_AUDIT.md` establishes that Top-1 sits
exactly on the Bayes bound for this generator and Top-5 is 0.8 points under it. There is
nothing left to win, and a number that improves from here is more likely to be a leak than a
gain. `PROJECT_NOTES.md` §1 already froze scope on exactly this reasoning; that decision was
right and this audit does not reopen it.

**Do not weaken the honest disclosures to look more complete.** The retracted-metrics table
(`PROJECT_NOTES.md` §2.4), the "we do not claim the ranker beats distance" paragraph, and the
live-complaint gap in `DISCUSSION_NOTES.md` §8 are the strongest material in this repository
and will land well with technical judges. The gaps this audit finds are of *unbuilt scope*,
not of overclaiming — with the single exception of Finding 6.2, where the README claims an
integration that does not exist. Fix that one by deleting the claim, not by softening
anything else.

---

## Appendix · Evidence index

| Finding | File |
|---|---|
| 1.1 | `README.md:14` |
| 1.2 | `backend/routers/predict.py`; `frontend/src/hooks/useActiveComplaint.js`; `docs/DISCUSSION_NOTES.md:342` |
| 2.1 | `requirements.txt` |
| 2.2 | `backend/routers/intel.py`; `backend/state.py:618`; `frontend/src/services/api.js:93` |
| 2.5 | `backend/models/schemas.py` (`CaseUpdateRequest`) |
| 3.1 | `frontend/package.json`; `frontend/src/pages/TacticalMap.jsx:145` |
| 3.2 | `frontend/src/pages/TacticalMap.jsx:11` |
| 3.3 | `frontend/src/pages/TriageFeed.jsx:491-542,777` |
| 3.4 | `data/atm_directory.csv`; `data/transactions.csv`; `data/victim_complaints.csv` |
| 3.5 | `backend/models/schemas.py:17` |
| 4.1 | `backend/auth.py`; `backend/routers/complaint.py:120,159`; `backend/routers/freeze.py:35`; `frontend/src/App.jsx` |
| 4.2 | `backend/main.py` |
| 4.6 | `backend/db.py:63` |
| 4.7 | `backend/state.py:519`; `backend/routers/audit.py`; `backend/db.py` |
| 4.10 | `backend/routers/auth.py`; `frontend/src/App.jsx` |
| 4.11 | `backend/db.py:465` |
| 5.1 | `frontend/src/pages/Interception.jsx:154,161-163` |
| 5.2 | `backend/routers/complaint.py:54,144`; `predict.py:171`; `freeze.py:90` |
| 6.1 | `frontend/src/App.jsx` (`QUEUE_LIMIT`); `backend/state.py`; `PROJECT_NOTES.md` §5 |
| 6.2 | `README.md` §Ethics & Compliance |
| 7.x | `PROJECT_NOTES.md` §1; `git status`; `git log` |
