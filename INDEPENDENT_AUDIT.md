# Independent Audit — MuleShield AI against SIH26184

**Problem Statement:** SIH26184 — *Development of a Predictive Analytics Framework for
Cybercrime Complaints to Forecast Likely Cash Withdrawal Locations in Advance, Enabling
Generation of Actionable Intelligence for Timely and Proactive Cybercrime Intervention.*
**Organisation:** Ministry of Home Affairs · Indian Cyber Crime Coordination Centre (I4C)
**Audited:** 1 September 2026 · branch `sameer` · working tree (uncommitted evidence module included)
**Auditor:** independent re-execution, not a reading of the existing audits
**Test suite at audit close:** 449 passed · 0 failed · 0 skipped (11m35s)

> **The working tree moved during this audit.** The evidence module landed mid-session,
> and three of the defects recorded in §4 were fixed while §7 was being written. Every
> finding below states whether it is **open** or **closed on re-verification**, with the
> measurement that settles it. Nothing is reported as open that is not open now.

> **SUPERSEDED IN PART — 2 September 2026.** After this audit was written, two of its
> findings were acted on and the corpus was regenerated:
>
> - **§5.2 (the largest evidence gap) is closed.** `scripts/generate_data.py` gained
>   `--mule-concentration`; mule accounts are now drawn toward the recruitment districts
>   (Nuh, Jamtara, Alwar, Bharatpur) at α=0.8, while victims and the ATM directory stay
>   uniform. Top mule district vs rank 12 moved **1.15× → 7.3×**; cash-out districts
>   **7.4×**. Victims remain flat at 1.16× across 79 cities.
> - **§7.1's remaining gap is closed.** A police intelligence dossier
>   (`backend/dossier.py`, `GET /api/v1/dossier/{case_id}`) closes deliverable (c)'s
>   *"intelligence reports"* clause.
>
> **Every figure in §2 below was measured on the PRE-concentration corpus and no longer
> matches `data/metrics.json`.** The reproduction *method* still holds and was re-run;
> the current figures are in `README.md`. §4, §5.1, §5.3 and §6 are unaffected.
> Test suite is now **481 passed**.

---

## 0 · How this audit differs from the two already in the repository

`COMPLIANCE_AUDIT.md` and `REMEDIATION_AUDIT.md` are self-audits. This one assumes
nothing they say is true and re-derives it. Specifically:

- Every headline metric was **recomputed from the shipped model artefacts** in a
  script written for this audit, not read from `data/metrics.json`.
- `scripts/topk_curve.py` and `scripts/evaluate_hotspots.py` were **re-executed** and
  the regenerated ledger diffed field-by-field against the shipped one.
- The API was **driven live** through a real FastAPI app instance — authentication,
  ingestion, forecast, surface, rule pass, freeze, audit, logout.
- The full suite was **executed twice**: 402/402 at session start, and 449/449 on the
  current tree after the uncommitted evidence module landed mid-audit. Zero failures,
  zero skips, both times.
- The corpus was **re-measured** for label leakage, choice-model stochasticity,
  arrival rate and geographic concentration.

Three defects surfaced that neither existing audit recorded. All three were fixed
during the audit and the fixes independently re-verified (§4).

**Score: 85 / 100.** Breakdown in §7.

---

## 1 · Verdict

**The measurement discipline is the best thing in this project, and it holds up under
independent re-execution.** Every published number reproduced. Two of them reproduced
*exactly* — bit-for-bit — from the checkpoints that ship in `models/`.

The engine answers the problem statement. The framework around it — the thing
`COMPLIANCE_AUDIT.md` found missing and `REMEDIATION_AUDIT.md` claims to have built —
is genuinely built and genuinely works; I raised a live alert through it.

What holds the score at 85 rather than higher is not overclaiming. It is:

1. a **corpus that is not concentrated**, in a problem whose entire premise is
   concentration — the single largest evidence gap (§5.2);
2. **documentation drift** — one honesty claim still contradicts the ledger on the same
   page, and the flagship forecast has no numbers in the README's benchmark section (§6);
3. residual engineering debt — the case audit trail is still in memory and not
   hash-chained, next to an evidence chain that is (§7.3).

The two defects that would have cost most — a 75-second cold path on the flagship screen
and a missing dependency that broke a clean install — were **found, measured, fixed and
re-verified within this session** (§4.1, §4.2).

| # | Deliverable | Self-assessed | This audit |
|---|---|---|---|
| **a** | Predictive Analytics Engine | 🟢 Met | 🟢 **Met** — verified live and by re-execution |
| **b** | Risk Heatmap Dashboard | 🟢 Met | 🟢 **Met** — API and data verified; UI assessed from code + captures, not driven |
| **c** | Law Enforcement Interface | 🟡 Mostly met | 🟢 **Met** — evidence module found built (uncommitted), closing the last open clause |
| **d** | Alert & Notification System | 🟢 Met (simulated) | 🟢 **Met**, transports simulated and labelled |

---

## 2 · What reproduced, and to what precision

### 2.1 GraphSAGE mule detection — reproduced exactly

An audit script loaded `models/graphsage_mule.pt`, rebuilt the feature matrix and edge
index from the CSVs, ran the forward pass and scored the checkpoint's own test indices.

| Metric | Ledger | Recomputed | |
|---|---|---|---|
| F1 | 0.9050 | **0.9050** | ✅ |
| Precision | 0.9050 | **0.9050** | ✅ |
| Recall | 0.9050 | **0.9050** | ✅ |
| ROC-AUC | 0.9637 | **0.9637** | ✅ |
| PR-AUC | 0.8515 | **0.8515** | ✅ |
| Accuracy | 0.9944 | **0.9944** | ✅ |
| Confusion (TN/FP/FN/TP) | 7258 / 21 / 21 / 200 | **7258 / 21 / 21 / 200** | ✅ |

Also verified: `ckpt["feature_cols"]` matches `gnn_model.FEATURE_COLS` exactly (no
train/serve feature drift), and the three split index sets are **pairwise disjoint and
cover all 49,999 accounts** — no node is scored that was also trained on.

### 2.2 Top-K containment — reproduced exactly

`scripts/topk_curve.py` re-executed against the shipped `xgb_cashout.pkl`:

| K | Ledger | Re-run | Distance baseline |
|---|---|---|---|
| 1 | 0.2641 | **0.2641** | 0.2786 |
| 3 | 0.5781 | **0.5781** | 0.5539 |
| **5** | **0.7359** | **0.7359** | 0.7150 |
| 8 | 0.8647 | **0.8647** | 0.8647 |
| 10 | 0.9227 | **0.9227** | 0.9163 |

The split guard fired as designed: `overlap=0` complaints across train/test, and
retrieval ceiling 1.0000 on all 621 held-out cash-outs.

### 2.3 Countdown regressor — reproduced exactly

Recomputed from `data/countdown_eval.npz` (n = 621):

| Metric | Ledger | Recomputed |
|---|---|---|
| MAE | 11.6591 | **11.6591** |
| Mean-prediction baseline | 14.7285 | **14.7285** |
| R² | 0.1776 | **0.1776** |

### 2.4 Forward hotspot surface — reproduced exactly

`scripts/evaluate_hotspots.py` re-executed end to end:

| Metric @ k=5 | Ledger | Re-run |
|---|---|---|
| Hit rate | 0.9517 | **0.9517** |
| PAI | 26.86 | **26.86** |
| Rupees covered | 0.9561 | **0.9561** |
| Prior share of surface | 0.1304 (cap 0.15) | **0.1304** |
| Median lead time | 41.5 min | **41.5 min** |

### 2.5 Ledger determinism

After re-running both evaluation scripts, `data/metrics.json` was diffed field-by-field
against the shipped copy. **Two differences, both in the same field:**
`ranking.latency_ms_per_cashout` moved 0.0 → 0.004 ms and the per-K `latency_ms`
followed. Those are wall-clock measurements of a sub-millisecond operation. **Every
substantive figure is bit-identical.**

This is the strongest single result in this audit. A ledger that regenerates to itself
means the published numbers are not hand-typed, not stale, and not from a different
corpus than the one in `data/`.

---

## 3 · The system, driven live

A real `TestClient` app instance, a fresh credential store, scheduler off, all calls over HTTP.

### 3.1 Authentication — the security findings are genuinely closed

| Request (no token) | Result |
|---|---|
| `GET /api/v1/complaint/list` | **401** |
| `POST /api/v1/bank/micro-freeze` | **401** |
| `GET /api/v1/hotspots/cells` | **401** |
| `GET /api/v1/alerts` | **401** |
| `GET /api/v1/audit` | **401** |
| `GET /api/v1/intel/atms` | 200 — open by documented policy (aggregate ATM counts, no PII) |
| `GET /health` | 200 |

Login 326 ms (PBKDF2-HMAC-SHA256, 600k iterations — OWASP floor, and the cost is the
point). Wrong password → 401. **Logout revokes: reusing the token after logout → 401.**
The Tier A / Tier B / Tier C policy in `backend/auth.py` is not a comment; it is what
the app does.

### 3.2 The pipeline, per stage

| Stage | Measured | Claim |
|---|---|---|
| Per-complaint graph build | **2.18 ms** (7 nodes, 8 edges) | ~2 ms ✅ |
| GNN embeddings endpoint | 64-d vectors, per-node risk scores | ✅ |
| Cash-out prediction (API round trip) | **27.8 – 33.6 ms** over 5 calls | model inference ~3.9 ms; API adds ~25 ms |
| Live complaint ingest | **5 ms** | ✅ |
| Forecast on the new complaint | **35 ms** | ✅ |
| Surface rebuild (warm) | **52 ms** | p95 92 ms ✅ |
| Rule pass | **54 ms** | ✅ |
| **Ingest → forecast → surface → alert** | **145 ms** | p95 0.25 s ✅ |

The forecast returned five ranked candidates with confidences (0.2495, 0.2362, …), a
search zone carrying `probability_mass: 0.8237`, and a countdown with a q05–q95 band.

### 3.3 The forward surface behaves as documented

- With no open complaints: `degraded: true`, and the console is built to say *"no live
  cases — this is history only"* rather than present a prior as a prediction. **Verified.**
- After ingesting one live complaint: `degraded: false`, 8 cells carrying live
  conditional mass.
- The top live cell reported `prior_share: 0.00077` against the 0.15 cap — the surface
  was driven **99.92% by the live case**, exactly as the design claims.
- Every cell carried its own `conditional_rupees` / `prior_rupees` split, so a reader
  can check which half is doing the work without trusting a summary.

### 3.4 The alert is a real intelligence product

The rule pass raised `ALT-000001` (`R-WATCH-SCORE`) and emitted two payloads. The
CFCFRMS-shaped one carries `human_in_the_loop: {acknowledged: false, disposition: null}`.
The dispatch payload carries:

> *"Ranked forecast, not a confirmed location. Deploy to observe; do not treat presence
> in this cell as grounds for detention."*

That sentence is the difference between a forecasting tool and a pretext for a stop, and
it is in the machine-readable payload rather than only in a slide.

### 3.5 Freeze and audit

`POST /bank/micro-freeze` returned `FROZEN` with a reference number, and the action
appeared in the audit trail attributed to the **verified session identity** (`Duty
Officer`), not to a caller-supplied `officer_id`. Three audit rows, newest first,
correctly scoped to the case.

---

## 4 · Defects this audit found — all now closed and re-verified

Three defects surfaced that neither `COMPLIANCE_AUDIT.md` nor `REMEDIATION_AUDIT.md`
records. All three were fixed in the working tree during this session. Each is recorded
below with the measurement that found it **and** the measurement that confirms it gone —
because a finding whose fix is not re-measured is only half an audit.

### 4.1 MAJOR (closed) — the first `/hotspots/cells` call took 75 seconds

**Found.** `backend/state.py` filtered complaints in this order:

```python
for comp in get_all_complaints():
    ...
    entry = hotspot_entry(cid)          # full GNN + ranker inference
    if entry is None or entry["ts"] is None:
        continue
    age = (now - entry["ts"]).total_seconds() / 60.0
    if 0 <= age <= open_minutes:        # the cheap test, applied last
        open_set.append(entry)
```

`hotspot_entry()` runs the feature builder, the GraphSAGE head and the conditional-logit
ranker over a 25-ATM candidate set. It was invoked for **every complaint in the store**,
and only then was the result discarded for falling outside the 120-minute window.

Measured on this corpus:

```
complaints in store:                                  2500
complaints actually inside the 120-min window:           2
wasted model invocations per cold call:               2498

COLD hotspot_surface():    75.73 s
WARM hotspot_surface():     7.92 ms
```

The live API run showed the same shape independently: **67.2 s** for the first
`/hotspots/cells` request, **52 ms** for the next.

**Why nothing in the repository caught it.** `scripts/bench_golden_hour.py` warms the
path before timing and passes `timeout=180` on that warm-up call — a timeout that is
itself an artefact of the cost being warmed away. Every published latency
(`surface_ms_p95: 92.46`, `end_to_end_seconds_p95: 0.248`) is therefore a warm-cache
number. Worse, `_tick()` in `backend/main.py` runs every 60 s and calls
`state.hotspot_surface()`: **the first tick overran its own interval by 15 seconds.**

The consequence that mattered: a judge who signed in and clicked **Risk Heatmap** — the
flagship screen, built to close the headline BLOCKER — would have waited over a minute
on a blank map.

**Closed.** The age test now precedes the model call, with a comment recording the
measurement and the reasoning. Re-verified:

```
COLD hotspot_surface():     0.01 s      (was 75.73 s)
WARM hotspot_surface():     8.25 ms
```

A ~7,500× improvement on the cold path, same output (220 cells, 2 open complaints,
`degraded=False`). `backend/tests/test_hotspot.py`, `tests/test_hotspot_leakage.py`,
`tests/test_metrics_ledger.py` and `backend/tests/test_alerts.py` — **88 tests — pass
against the fix.**

Note this never invalidated a published claim: 75 s is still four orders of magnitude
inside the 60-minute golden hour. It was a demo and scheduler defect, not a measurement one.

### 4.2 MAJOR (closed) — `python-multipart` was missing from `requirements.txt`

**Found.** `backend/routers/evidence.py` declares `file: UploadFile = File(...)` and the
router is wired unconditionally in `backend/main.py`. FastAPI requires `python-multipart`
for any `File`/`Form` parameter. The package was present in `venv/` (0.0.32), so
everything passed locally — and **absent from `requirements.txt`**. A fresh
`pip install -r requirements.txt`, which is exactly what the Dockerfile does and what the
README tells a judge to do, would have produced an environment where the newest
deliverable did not work.

This was the highest-severity-per-character finding in the audit: one line, and without
it a clean-room reproduction fails on the module built to close the last open
problem-statement clause.

**Closed.** `requirements.txt:23` now reads
`python-multipart>=0.0.9   # multipart uploads for evidence documentation`.

### 4.3 MINOR (closed) — `evaluate_hotspots.py` mislabelled its own output

**Found.** The script printed *"At k=5: 4.25 cells searched per genuine interception"*
using `false_cells_per_hit` = `(k·n − hits)/hits`. Cells *searched* per interception is
`k / hit_rate` = `5 / 0.9517` = **5.25**; 4.25 is the number of *false* cells per hit.
`REMEDIATION_AUDIT.md` §6 quoted 5.25 correctly, so the document was right and the
program was wrong — the more dangerous direction, because the program is what a judge
would re-run.

**Closed.** Both quantities are now computed and named separately, with a comment
recording which was printed under the other's name.

### 4.4 MINOR (open, cosmetic) — asymmetric note routes

`POST /api/v1/complaint/{id}/note` (singular) creates; `GET .../notes` (plural) lists.
Both work; the asymmetry costs a 405 to anyone who guesses. Not weighted in the score.

---

## 5 · The corpus, re-measured

### 5.1 What is sound

| Property | Measured | Assessment |
|---|---|---|
| Accounts | 49,999 | |
| Mule prevalence | **2.95%** (1,476) | Above real-world (<1%); disclosed in README |
| Transactions | 622,188 (22,201 fraud / 599,987 legitimate) | Class overlap is real, not decorative |
| Terminal cash-outs | 6,426 | |
| Complaints | 2,500, all unique ticket ids | ₹38.91 Cr total |
| ATM directory | 1,000 | |

**Label integrity — verified independently.** I swept every numeric feature in the
model's feature set for a single-threshold rule reproducing the label. The best is
`burst_out_5min` at **F1 0.4970** with the threshold chosen on train and scored on test,
which is what the ledger reports. The label is not a copy of a feature.

**The choice model is genuinely stochastic — verified.** Over 3,000 sampled cash-outs,
`P(true ATM == nearest ATM) = 0.2623`, `P(∈ nearest 3) = 0.5663`, `P(∈ nearest 25) =
0.9977`. If the label were `argmin(distance)` — the defect the project documents having
had and fixed — the first figure would be 1.0. It is not. The fix is real.

**One nuance worth recording.** `hop_depth` sits in `node_features.csv` and the rule
`hop_depth > 0` scores **F1 0.9274**, above the GNN's 0.9050. It is *correctly excluded*
from the model — `gnn_model.py` argues, rightly, that it is non-zero only for accounts
already known to be in a traced chain, so using it assumes the answer. But two things
follow that the README does not say. First, its 0.9274 is an empirical confirmation of
the claimed 0.927 label-noise ceiling (I measured 216 label/chain disagreements in
49,999 rows), which is a stronger result than the project claims for itself. Second, the
README's blanket sentence *"The best single feature now reaches only F1 0.80"* (line 376)
is wrong twice over — it is 0.4970 within the model's feature set per the ledger on the
same page, and 0.9274 across the file as a whole.

### 5.2 What is weak — and it is the single largest evidence gap

The problem statement's premise is **concentration**: stolen money surfaces
disproportionately in a small number of places, which is why forecasting where is worth
doing at all. This corpus is close to uniform.

| Property | Measured | Reality |
|---|---|---|
| Complaint arrival rate | **21/day** | ~8,000/day, named twice in the statement |
| Complaints in the last 120 min of the corpus | **2** | ~670 at the stated rate |
| Top city vs 12th city | **46 vs 40** (1.15×) | Nuh / Jamtara vs a typical district: orders of magnitude |
| Fraud-type mix | 9 types, **267–292 each** | NCRP is dominated by UPI and online financial fraud |

`REMEDIATION_AUDIT.md` §5.1 and §5.2 record both honestly, and correctly argue PAI would
be *higher* under real concentration, not lower. That is probably true. But it means the
national-scale evidence rests on a corpus with no national-scale structure, and the fix
is cheap — a concentration parameter in a generator that must be re-run before a demo
anyway. Leaving it is the clearest missed point in the project.

The arrival rate has a second consequence the audit understates: the flagship screen is
**empty by default**. Without `scripts/seed_live_feed.py`, the 120-minute window holds
two complaints, and the surface reports `degraded`. The demo depends on a seeding step
that a judge running the README from top to bottom will not know to perform until step 5.

### 5.3 Synthesis is defended properly

`docs/DATA_PROVENANCE.md` makes the right argument and makes it with citations: PaySim
and AMLSim are synthetic for the same legal reason, and no public dataset carries the
label this problem needs (*which physical ATM*, and *how long after*). The limitation is
real; the defence is the strongest available.

---

## 6 · Documentation accuracy

The prose is unusually careful — but it has drifted. Two of the six items below were
corrected during this session; four remain, and every one is checkable in under a minute
by someone trying to find fault.

| # | Location | Status | Problem |
|---|---|---|---|
| 1 | `README.md:9` | 🟡 **Partly closed** | The body figures were reconciled to 449 during this session (`:248`, `:501`, `:603`), and `:657` correctly historicises the old 402. The **badge still reads "448 passed · 1 skipped"**. Measured, twice: **449 passed, 0 skipped**. The skip the badge describes did not occur in either run. |
| 2 | `README.md:405` | 🔴 **Open** | *"The best single feature now reaches only F1 0.80"* contradicts the ledger and the table on the same page (0.4970). See §5.1 — it is wrong in both directions. |
| 3 | `engine/xgb_model.py:56-58` | 🔴 **Open** | The `OPERATING_K` docstring cites containment 0.7136 @ K=5 and 0.5615 @ K=3. Current, verified: **0.7359** and **0.5781**. Stale from a previous corpus. |
| 4 | `README.md` §Benchmark | 🔴 **Open** | **The forward hotspot forecast has no numbers anywhere in the benchmark section.** PAI 26.86, hit@5 0.9517 and the 41.5-minute median lead time — the figures for the capability that closed the headline BLOCKER — appear only in `REMEDIATION_AUDIT.md`. All five benchmark subsections are per-case. |
| 5 | `PROJECT_NOTES.md:32-34` | 🔴 **Open** | Marked *"Last reviewed 30 Aug"* and still lists *"No authentication on any endpoint, including freeze"* and *"CORS is `allow_origins=['*']`"* as open items. Both are closed; I verified both live (§3.1). A reader who opens this file first concludes the security findings are outstanding. |
| 6 | `REMEDIATION_AUDIT.md` | ✅ **Closed** | Previously listed evidence documentation as *"the one named clause with no implementation at all"*. Updated during this session; the module is implemented, wired and covered by 47 passing tests. |

None of these is an overclaim in the model's favour. Items 4 and 5 actively **undersell**
the project to anyone reading in order — item 4 hides its best new result, item 5 tells a
reader the security work is unfinished when it is done.

---

## 7 · Score

### 7.1 Problem-statement coverage — **27 / 30**

All four deliverables are met. The forward surface, the national → state → district
drill-down, the rule engine, the four notification channels, the recipient model with
retry and dead-lettering, the required disposition on close, and — as of this session —
evidence documentation with hash-chained custody and a BSA 2023 s.63 certificate.

Deductions: the transports are simulated (largely unavoidable — a live SMS route needs
DLT registration and CFCFRMS needs credentials only I4C can issue, and this is labelled
on every channel); no printable case dossier; officer jurisdiction scoping is absent
while recipient scoping exists.

### 7.2 ML and evaluation rigour — **22 / 25**

The best part of the project, and it survives adversarial re-execution.

Earning it: leak-free splits verified independently (GNN indices pairwise disjoint and
covering all 49,999 accounts; ranking split `overlap=0` by complaint); ATM priors
computed only from the earliest 50% of complaints, which are then excluded from training
and evaluation entirely; a baseline beside **every** figure; negative results published
in the ledger as machine-readable fields (`beats_nearest_cell: false`); a label-noise
ceiling quoted and, per §5.1, empirically correct; and tests that fail the build on
overclaiming — `test_label_is_not_a_copy_of_a_feature`,
`test_hotspot_reports_the_baseline_it_does_not_beat`,
`test_hit_rate_is_not_suspiciously_perfect`,
`test_console_constants_carry_no_literal_figures`. I have not previously seen a project
ship a test whose purpose is to stop its own headline number from migrating into a slot
where it would look better.

Deductions, all disclosed by the project itself:

- **The ranker and the generator share a functional form.** `assign_cashouts()` samples
  from `exp(−d/λ) · (1 + w·risk) · bank_boost`; `ConditionalLogitRanker` fits
  `softmax(w · log_features)`. Recovering weights of +0.99 / +0.79 / +0.60 against truth
  1.00 / 1.00 / 0.69 is a correct and genuinely impressive result, but it demonstrates
  Bayes-optimality against a known simulator, not skill against reality. This is the
  ceiling on what any metric here can mean, and no amount of methodological care lifts it.
- **The forecast loses to a trivial baseline.** Nearest cell to the traced terminal
  scores 1.0000 at k=5 against the model's 0.9517. Published, tested and correctly
  framed — the *trace* earns the value, since victim-city-only scores 0.0193 — but it
  means the model's contribution over plain distance is the time dimension and the
  rupee-weighted cross-case aggregation, not the geography.
- **Countdown R² 0.18.** Honestly bounded at ~0.45 by the deliberate two-regime design
  and reported with 78.9% empirical band coverage rather than as a point estimate. Still
  the weakest component, and the project says so.

### 7.3 Backend engineering — **18 / 20**

Earning it: a considered auth policy rather than a bolted-on one — the Tier A/B/C
reasoning that *retrospective aggregates stay open, forward operational intelligence
closes* is the right distinction and is exactly what the code does; PBKDF2 at 600k with
lockout, expiry and verified token revocation; the sync-vs-async dependency choice made
deliberately and documented with the 354 ms measurement behind it; a lifespan task
cancelled **and awaited**; delivery rows with retry, exponential backoff and a
dead-letter state; hash-chained evidence whose docstring states plainly what the chain
does *not* prove.

Both major defects found in §4 are fixed and re-verified, so they cost nothing here
beyond the fact that neither was caught by 449 tests and a benchmark suite.

Remaining deductions: the case audit trail is still in memory and not hash-chained,
sitting directly beside an evidence chain that is; single-process deployment with the
whole corpus resident; no model monitoring or drift detection.

### 7.4 Data realism and evidence — **10 / 15**

Earning it: provenance argued with citations rather than asserted; label integrity
verified independently; the choice model verified stochastic; and the leakage defects the
project had are documented as *reversals* with the wrong number named — far more
convincing than never having erred.

Deductions, and this is where the real gap is: §5.2 — the corpus is not concentrated, in
a problem about concentration; the arrival rate is 380× below the stated national load;
fraud types are uniform to within ±5%; and live-ingested chains are synthesised at
ingestion (disclosed), so the forward surface for a live case is downstream of fabricated
topology.

### 7.5 Delivery and reproducibility — **8 / 10**

Earning it: seven captured screens from a live run plus two new evidence screens, five
analysis notebooks, a built deck and PDF, a Dockerfile, and an integration-seams document
that withdraws an earlier false claim and replaces it with a versioned `v1-proposed`
mapping. `requirements.txt` is now complete (§4.2), so a clean install reproduces.

Deductions: the four open items in §6 — most importantly that the README's benchmark
section still omits the flagship hotspot numbers, and that `PROJECT_NOTES.md` still tells
a reader the security findings are open; the Dockerfile is written but by the team's own
note **never built**; and the flagship screen is empty without a seeding step buried at
step 5 of the quickstart.

### 7.6 Total

| Dimension | Weight | Score |
|---|---|---|
| Problem-statement coverage | 30 | **27** |
| ML and evaluation rigour | 25 | **22** |
| Backend engineering | 20 | **18** |
| Data realism and evidence | 15 | **10** |
| Delivery and reproducibility | 10 | **8** |
| **Total** | **100** | **85** |

**85 / 100.** A project whose evaluation you can trust, whose engine does the hard thing
the statement asks for, and whose remaining gaps sit in the corpus and the documentation
rather than in the claims.

For calibration: the 15 points not awarded are almost entirely **evidence** points, not
**capability** points. The system does what it says. What it cannot yet show is that the
thing it does would work on a population shaped like India's rather than a population
shaped like a uniform draw.

---

## 8 · What to fix, in order of return per hour

Items 1–3 of the original list were fixed during this session and are struck through.
What remains:

1. **Add a concentration parameter to the generator** (§5.2). This is now the single
   highest-value change in the project. Regeneration is mandatory before a demo anyway,
   so it is nearly free, and it is the difference between national-scale claims and
   national-scale evidence. A corpus where the top district carries 20× the twelfth,
   rather than 1.15×, would raise PAI, make the heatmap look like the phenomenon it
   models, and remove the most obvious question a judge can ask.
2. **Put the hotspot numbers in the README benchmark section** (§6 item 4). PAI 26.86,
   hit@5 0.9517, median lead time 41.5 min — beside the nearest-cell baseline of 1.0000
   that the ledger already records as unbeaten. The project's most defensible new
   capability is currently invisible in its front door. Thirty minutes.
3. **Fix `README.md:405`** (§6 item 2) — the F1 0.80 sentence contradicts the table
   above it. One line, and it is on the *Honest Evaluation* page, which is the worst
   possible place to carry a wrong number.
4. **Refresh or delete the stale open items in `PROJECT_NOTES.md`** (§6 item 5). It
   currently tells a reader the security findings are open when they are closed and
   verified. One paragraph.
5. **Correct the test badge** to `449 passed` (§6 item 1) — it still asserts a skip that
   did not occur in either of my runs.
6. **Refresh the `OPERATING_K` docstring** in `engine/xgb_model.py` (§6 item 3).
7. **Build the Dockerfile once.** The team's own note says it has never been built. The
   untested parts are the CPU-only torch index URL and `libgomp1` for XGBoost — both are
   the kind of thing that fails only on a clean machine, which is the only kind a judge has.
8. **Hash-chain the case audit trail**, or say in the README why the evidence chain is
   hash-chained and the case trail is not. The asymmetry currently reads as an oversight
   rather than a decision.

~~Add `python-multipart` to `requirements.txt`~~ — done during this session (§4.2).
~~Hoist the age filter above `hotspot_entry()`~~ — done and re-verified, 75.73 s → 0.01 s (§4.1).
~~Fix the `false_cells_per_hit` print label~~ — done (§4.3).

Items 2–6 total well under two hours and all five are things a judge can find. Item 1 is
the one that changes what the project can honestly claim.

---

## 9 · Reproducing this audit

```bash
# the whole suite, both roots  ->  449 passed, 0 failed, 0 skipped (11m35s)
python -m pytest -q

# re-derive the two ledger sections from the SHIPPED checkpoints, no retraining
python scripts/topk_curve.py
python scripts/evaluate_hotspots.py
git diff data/metrics.json        # expect latency-only differences

# the cold-path defect and its fix (§4.1)
python -c "
import sys,time; sys.path[:0]=['.','engine']
import backend.state as state; state.load_all()
t=time.perf_counter(); state.hotspot_surface(); print('cold', time.perf_counter()-t)
t=time.perf_counter(); state.hotspot_surface(); print('warm', time.perf_counter()-t)"
# before the fix: cold 75.73 s / warm 0.0079 s
# after  the fix: cold  0.01 s / warm 0.0083 s

# the dependency defect and its fix (§4.2)
grep -i multipart requirements.txt
grep -n 'UploadFile' backend/routers/evidence.py

# auth enforcement, live
curl -i -X POST localhost:8000/api/v1/bank/micro-freeze   -H 'Content-Type: application/json' -d '{"account_id":"X","complaint_id":"Y"}'
curl -i -X OPTIONS localhost:8000/api/v1/complaint/list -H 'Origin: https://evil.example'
```

The GNN reproduction in §2.1 requires a short script: load `models/graphsage_mule.pt`,
apply `ckpt["scaler_mean"]`/`scaler_scale` to `node_features.csv` ordered by
`ckpt["account_ids"]`, build the edge index from `graph_edges.csv` and symmetrise it,
run the forward pass, and score `ckpt["split"]["test_idx"]` at
`ckpt["hyperparams"]["threshold"]`. Every figure in that table falls out.

---

## 10 · Closing note

Two audits already existed in this repository before this one, and both were harder on
the project than an outside reader would have been. The disposition that produced them —
naming the wrong number rather than quietly replacing it, recording reversals, shipping a
test whose job is to stop a figure being quoted in a slot where it would flatter — is why
this audit could verify so much so quickly, and it is worth more than any single metric
in the ledger.

The three defects in §4 are exactly the kind that discipline does not catch. One was
invisible because the benchmark warms past it. One was invisible because the developer's
environment already had the package. One was a print statement that the accompanying
document described correctly. None was a failure of honesty; all three were failures of
a self-audit's blind spot, which is the assumption it is built on.

The remaining gap is not technical. The system forecasts where cash will surface, and it
does so with an evaluation you can re-run. What it has not yet shown is that it does so
on a country shaped like India — a corpus of 21 complaints a day, spread evenly over 79
cities, is not the phenomenon the problem statement describes. Fixing that costs one
generator parameter and a twenty-minute retrain, and it is worth more than any further
work on the model.
