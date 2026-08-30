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
- [ ] **A fresh clone cannot run.** `data/transactions.csv` (114 MB) and
      `data/graph_edges.csv` (40 MB) exceed GitHub's limit and are gitignored, so
      `python scripts/generate_data.py` must run first. Documented in the README,
      but there is still no Dockerfile, Procfile or host config — nothing a
      platform can build against.
- [ ] **`frontend/.env` is committed with `VITE_API_BASE_URL=http://localhost:8000`.**
      A production build would ship pointing at localhost. Read it from the host's
      environment at build time instead.
- [ ] **No authentication on any endpoint, including freeze.** Acceptable for a
      judged demo; make it a conscious decision before anything is public.
- [ ] CORS is `allow_origins=["*"]` (`backend/main.py`).

### For the SIH submission

- [ ] **7-slide deck and 3-minute video.** Entirely untouched. Largest remaining
      gap for the submission itself.
- [ ] **Regenerate the dataset shortly before demoing.** Complaint timestamps are
      relative to generation time — the queue currently reads "3d ago" and drifts
      further every day.

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

## 5. Where things are

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
cd frontend && npm install && npm run dev
```

Models are committed, so training is optional:

```bash
python engine/train_gnn.py && python engine/embed.py && python engine/train_xgb.py
```
