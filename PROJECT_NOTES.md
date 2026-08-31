# MuleShield AI — running checklist and things to keep in mind

SIH26184 · MHA / I4C · working branch `sameer`

Two kinds of thing live here. **Section 1** is work still to do. **Sections 2–4**
are decisions already made deliberately — if someone reports one of them as a
bug, the answer is in here, not in the code.

Last reviewed: 30 Aug 2026.

---

## 1. To do

### Blocking a real deployment

- [ ] **Merge `sameer` into `main`.** Six commits of Phase 1–6 work live only on
      the branch. Nothing is lost; it is a decision, not a task.
- [x] ~~No Dockerfile~~ — **written 30 Aug, but NOT BUILT.** Docker Desktop was
      not running, so it is reviewed and unverified. Build it before trusting it;
      the untested parts are the CPU-only torch index URL, `libgomp1` for XGBoost,
      and how long `generate_data.py` takes inside the image.
- [ ] **A fresh clone still needs `python scripts/generate_data.py` first.**
      `data/transactions.csv` (114 MB) and `data/graph_edges.csv` (40 MB) exceed
      GitHub's limit and are gitignored. The Dockerfile handles this at build
      time; a local clone does not.
- [x] ~~`frontend/.env` committed with a localhost API base~~ — **fixed 30 Aug.**
      It was baking `localhost:8000` into every production build, so a deployed
      console would have asked the *viewer's own machine* for the API. `.env` is
      deleted; the bundle now uses relative URLs. `wsUrl()` had the same defect
      and would have opened an insecure `ws://` socket on an https deployment.
- [ ] **No authentication on any endpoint, including freeze.** Acceptable for a
      judged demo; make it a conscious decision before anything is public.
- [ ] CORS is `allow_origins=["*"]` (`backend/main.py`).

### For the SIH submission

- [ ] **7-slide deck and 3-minute video.** Entirely untouched. Largest remaining
      gap for the submission itself.
- [ ] **Regenerate the dataset shortly before demoing.** Complaint timestamps are
      relative to generation time — the queue currently reads "3d ago" and drifts
      further every day.

### Scope is frozen (30 Aug)

The prototype is done. Further UI or model polish has close to zero marginal
return, and the thing that actually decides SIH — the deck and the video — is
not in this repo. **Before adding anything here, check it beats spending the
same hour on the pitch.**

In particular: do not chase the model further. Top-1 sits exactly on the Bayes
bound for this generator (see 2.3). There is nothing left to win, and a number
that improves is more likely to be a leak than a gain.

### Worth doing if there is time

- [ ] Persist the audit trail (see 2.1). In-memory is honest but thin.
- [ ] Decide whether the ranker should use ATM opening hours (see 2.2). Needs its
      own held-out measurement, not a UI change.
- [ ] `case_notes` and `audit_log` are not cleared by `state.load_all()` while
      every other store is. No live effect — both are module globals that reset
      with the process — but it is an inconsistency in the clearing logic.

---

## 2. Deliberate decisions — do not "fix" these

### 2.1 The audit trail is in-memory

It dies with the backend process. Stated plainly in `backend/routers/audit.py`
rather than papered over. **If you demo across a restart, the case history is
gone.** Restart the backend *before* a demo, not during one.

### 2.2 Three of the top five ATMs are closed at the predicted time

Including rank 1, on the case used for testing. The directory carries opening
hours for all 1,000 machines; the **console flags it**, the **model does not use
it**.

That gap is deliberate. Re-ranking on opening hours through the UI would mean the
shipped system no longer matches the model whose evaluation is published on the
Model Performance screen. Whether the ranker *should* use the feature is a real
question — it needs its own held-out measurement, not a display change. The
overnight audit already listed opening hours as an unused signal.

This is worth a slide. It shows the console catching something the model misses.

### 2.3 The ranker does not beat distance-sorting at K=5

**0.7136 against 0.7217**, and the difference is not statistically significant at
any K (paired McNemar over 618 held-out cash-outs). Top-1 sits exactly on the
Bayes bound for this generator.

The defensible claim is **the narrowing — 1,000 ATMs to 5** — not that the model
outperforms a distance rule. The Model Performance screen states this unprompted.
Better a judge reads it there than finds it themselves.

### 2.4 Retracted metrics must stay retracted

The figures below were invalidated by the leakage audit and must never reappear:

| Retracted | Actual |
|---|---|
| 98.5% Top-3 | 0.5615 |
| 0.9996 F1 | 0.8955 |
| 1.2 s countdown MAE | 11.86 min |

They survived for weeks because they were hand-typed in the frontend. Every
figure on screen is now generated from `data/metrics.json`, which only the
training and evaluation scripts write. **Do not hand-edit that file.**

### 2.5 87.4% is the search zone, not Top-5

Two different operating points. The zone contains 87.4% with a median of 8 ATMs;
Top-5 containment is 0.7136. A test asserts Top-5 < 0.80 so the zone number
cannot migrate into the Top-5 slot.

---

## 3. Testing — read this before trusting a green build

**A passing build has caught none of the four broken screens shipped this
session.** Vite does not catch undefined identifiers, so a page can build
cleanly and render completely blank. Every one was found by the interaction
test. Run it before any demo.

```bash
python scripts/smoke_ui.py          # isolated: starts its own backend + frontend
python scripts/smoke_ui.py --attach # against running servers — WILL mutate them
```

Notes:

- **Isolation is the default and matters.** The sweep clicks the status dropdown,
  "Assign to me" and the account freeze. Run against the demo backend it leaves
  real cases reassigned and writes into the audit trail — the one record whose
  entire value is that it reflects what actually happened.
- The sweep reads back the backend's own audit trail and **names any destructive
  control it failed to exercise**. That check caught the Phase 3 confirmation
  dialog silently breaking freeze coverage: the sweep opened the dialog, the
  generic modal handling clicked Cancel, and the most destructive control in the
  product was dismissed every run while reporting as exercised.
- **`test_full_graph_build_within_budget` no longer flakes** (fixed 30 Aug). It
  calibrates the machine first and skips with a stated reason when the box is too
  contended to measure. A skip there is not a failure — it means "not measured".
  It still fails on a genuine regression.

Full suite: `python -m pytest tests/ backend/tests/ -q` → **290 tests**.

---

## 4. Operational gotchas

- **Killing the backend leaves the console looking broken, not offline.** A dead
  backend produces a spinner on the case summary and `System status: Offline`.
  That is the degraded state working; check the backend before debugging the UI.
- **A stale uvicorn holds port 8000 silently.** `nohup … &` will not report the
  bind failure, so you end up talking to old code while your changes sit on disk.
  Symptom: a field you just added comes back `null`. Kill the PID on 8000 first.
- **Timestamps carry two formats.** Seed complaints are naive local time
  (`2026-05-27T19:43:13`); ingested ones carry `+00:00`. JS resolves offset-less
  strings against the *browser's* zone, so on a machine in a different timezone a
  months-old case could drift into the golden hour and start a fake countdown.
  `hasTrustedClock()` gates the countdown on the timestamp declaring its zone.
- **`frontend/dist-smoke/`** is the isolated test build, pointed at a throwaway
  port. Gitignored. Never serve it.

---

## 5. Deployment

### What it actually needs

Measured on 30 Aug, backend fully loaded:

| | |
|---|---|
| Resident memory | **1.53 GB** |
| Prediction latency | 2 ms |
| Committed models and artifacts | 45 MB |
| Generated data (not in the repo) | 154 MB |
| Python environment | 1.3 GB, almost all PyTorch |

Three properties decide where this can go:

1. **State is in memory.** Ingested cases, notes and the audit trail live in
   process globals. Two instances behind a load balancer would disagree with
   each other, so this runs as a **single process**.
2. **There is a WebSocket** (`/ws/feed`), so the host must hold long-lived
   connections.
3. **1.5 GB resident**, before any headroom.

Together those rule out serverless — Vercel Functions, Netlify, Lambda,
Cloudflare Workers. Not "would be awkward on": cannot work on. Do not spend an
evening trying.

### Recommended

**For the pitch itself: run it locally.** Venue wifi is the single biggest
avoidable risk on the day, and the whole system runs on one laptop with no
network dependency. This is not a fallback; it is the primary plan.

**A public URL is a nice-to-have, and it is not free.** Worth deciding
deliberately rather than assuming — the requirement is 1.45 GB resident plus a
persistent process plus WebSockets, and that combination is precisely what free
tiers exclude.

| Option | RAM | Cost | Verdict |
|---|---|---|---|
| Render free, Koyeb free | 512 MB | free | **OOMs on the first prediction** |
| Hugging Face Spaces (Docker) | 16 GB | **PRO only** | free tier is Static-only now |
| Fly.io | 2 GB | ~$5/mo, card required | least friction that works |
| Render Standard | 2 GB | ~$25/mo | works |
| Oracle Cloud Always Free | 24 GB, 4 ARM cores | free, forever | only genuinely free fit |

Two traps in that table:

- **512 MB tiers do not fail at deploy.** They boot, serve the queue, and die the
  moment someone opens a case — the feature builder lazy-loads on the first
  prediction, taking the process from ~10 MB to 1.45 GB. It looks fine until a
  judge touches it, which is the worst failure mode available.
- **Oracle is ARM64.** The Dockerfile pulls the x86 CPU torch wheel; ARM needs a
  different index. It also wants a card for identity checks and rejects some
  signups without explanation.

**A note on how this file got it wrong:** it previously recommended Hugging Face
Spaces on the free tier. That was based on stale knowledge — Docker Spaces now
require PRO. Check a platform's current free tier before committing to it; this
one changed without the docs I was working from changing.

### Split frontend and backend (Vercel + Render, or similar)

Perfectly reasonable, and the code already supports it: set `VITE_API_BASE_URL`
at frontend build time and `wsUrl()` correctly derives `wss://` from that base
rather than from the page origin.

Two things to change if you go this way:

1. `allow_origins=["*"]` in `backend/main.py` must be narrowed to the frontend
   domain, or the browser blocks credentialed cross-origin requests.
2. The backend still needs ≥2 GB, so the RAM table above still decides it. The
   split does not make the free tier viable — it just moves the frontend off it.

### Before any of that

The three deployment blockers in section 1 still stand: no Dockerfile exists,
`frontend/.env` is committed pointing at localhost, and there is no
authentication on any endpoint — including the freeze.

---

## 6. Where things are

| | |
|---|---|
| UI audit (pre-refinement baseline) | `docs/UI_AUDIT.md` |
| ML leakage audit, Bayes bounds, Top-K curve | `OVERNIGHT_ML_AUDIT.md` |
| Every measured figure | `data/metrics.json` (generated) |
| Console screenshots | `docs/screens/` (regenerate: `scripts/capture_screens.py`) |
| Interaction sweep | `scripts/smoke_ui.py` |
| Top-K containment | `scripts/topk_curve.py` |
| Confusion matrix | `scripts/export_confusion.py` |

**Run order from a clean clone:**

```bash
python scripts/generate_data.py                     # required — not in the repo
python -m uvicorn backend.main:app --port 8000
cd frontend && npm install && npm run dev           # console on :5173
```

### Which URL to open (this changed on 30 Aug)

| What you want | Open | Notes |
|---|---|---|
| Develop, with hot reload | `npm run dev` → **:5173** | Vite proxies `/api` and `/ws` to :8000 |
| See the production build | **:8000** | FastAPI serves `frontend/dist` after `npm run build` |
| `npm run preview` → :4173 | **needs an env var** | see below |

`npm run preview` no longer works on its own. The production bundle now uses
**relative** API URLs — correct, because the deployed container serves the
console from the same process that answers the API — so a preview on :4173 asks
:4173 for `/api`, and nothing is there. Vite's proxy config applies to the dev
server, not `preview`.

Two ways round it, and the first is better:

```bash
npm run build && open http://127.0.0.1:8000      # exactly what the container does
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run build && npm run preview
```

The smoke test already sets that variable for its isolated build, so it is
unaffected.

Models are committed, so training is optional:

```bash
python engine/train_gnn.py && python engine/embed.py && python engine/train_xgb.py
```
