# 🛡️ MuleShield AI: Real-Time Money Mule Detection & ATM Interception Intelligence

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/React-19.0.0-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-GraphSAGE-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20v2-green.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-275%2F275%20Passed-brightgreen.svg)]()
[![SIH 2026](https://img.shields.io/badge/SIH-2026%20Problem%20ID%3A%20SIH26184-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Smart India Hackathon (SIH 2026)** | Ministry of Home Affairs (MHA) / Indian Cybercrime Coordination Centre (I4C)  
> **Problem Statement ID:** SIH26184 — *Identification of Money Mule Accounts and ATM Geolocation for Cashout Interception*

---

## 🖥️ Tactical Command Center Dashboard

Four screens, captured from a **live run** against the real backend — `scripts/capture_screens.py`
drives a headless browser over the running console, so what is below is what the system renders,
not a mockup. Every figure on screen is read from `data/metrics.json`, which only the training and
evaluation scripts write.

<div align="center">
  <h3>1. 1930 Helpline Triage &amp; Intake Queue</h3>
  <img src="docs/screens/01-triage-queue.png" alt="MuleShield AI — 1930 triage queue" width="100%" />
  <p><em>Severity-ranked complaint queue with Golden-Hour badges, live pipeline status read from
  <code>/health</code>, and one-click ingestion. The intake badge reads <strong>connected</strong>,
  not "live" — the socket being open is not the same as complaints arriving, and offline there is no
  NCRP feed pushing them.</em></p>
</div>

<br/>

<div align="center">
  <h3>2. Tactical GIS — Priority Search Locations</h3>
  <img src="docs/screens/02-priority-search-locations.png" alt="MuleShield AI — priority search locations" width="100%" />
  <p><em>The <strong>Top-5 ranked candidate cash-out locations</strong> for the traced terminal
  account, each with its relative score, distance from the mule and prior-incident count, plus the
  aggregated search zone (here 7.72 km covering 5 ATMs at 84% probability mass). These are
  <strong>prioritized candidates, not a predicted ATM</strong>.</em></p>
</div>

<br/>

<div align="center">
  <h3>3. Forensic Money-Flow Graph (DAG)</h3>
  <img src="docs/screens/03-money-flow-graph.png" alt="MuleShield AI — forensic money-flow graph" width="100%" />
  <p><em>Victim ➔ layering mules ➔ terminal cash-out, with a per-node GraphSAGE risk inspector.
  The mule probabilities shown are <code>sigmoid(Wh + b)</code> over the account's cached 64-d
  embedding, passed through isotonic calibration — verified against held-out labels, see
  <a href="OVERNIGHT_ML_AUDIT.md">§10b of the audit</a>.</em></p>
</div>

<br/>

<div align="center">
  <h3>4. Interception Control &amp; Police Dispatch</h3>
  <img src="docs/screens/04-interception.png" alt="MuleShield AI — interception control" width="100%" />
  <p><em>All five ranked candidates, the live cash-out countdown, 1-click bank micro-freeze and PCR
  dispatch. The status badge tracks real inference state (ARMED / COMPUTING / STANDBY) rather than
  being permanently lit, and the dispatch alert carries "a ranked candidate, not a confirmed
  location".</em></p>
</div>

> **Regenerate:** start the backend and `npm run preview`, then
> `python scripts/capture_screens.py`. The script resolves a complaint from the live queue rather
> than pinning a ticket id, because a pinned id outlives the dataset it points at.

---

## 📌 Executive Summary

Cyber fraud incidents reported on the National Cybercrime Reporting Portal (**1930 Helpline**) often involve rapid fund laundering through multi-layered **money mule account networks** within minutes. By the time law enforcement issues freeze notices, syndicate operators physically withdraw the stolen money from ATMs.

**MuleShield AI** is an industry-grade, hybrid intelligence platform that bridges **Graph Neural Networks (GNNs)** and **Gradient Boosted Decision Trees (XGBoost)** to:
1. **Trace multi-hop fund dispersal** in real time from victim complaint origins in $<185\text{ ms}$.
2. **Detect fraud rings, fund-splitting, and velocity anomalies** using graph topology.
3. **Generate 64-dimensional structural risk embeddings** via **GraphSAGE** (capturing complex neighborhood relationships).
4. **Narrow 1,000 ATMs to a search zone containing the withdrawal 87.4% of the time** (a median of 8 machines), with a **countdown and prediction band**, in **under 5 ms** end-to-end.

```
[ 1930 Victim Complaint ]
           │
           ▼
[ NetworkX Directed Graph ] ──► (BFS Traversal, Velocity & Fund-Splitting Detection)
           │
           ▼
[ GraphSAGE GNN (PyG) ]    ──► (64-dim Graph Structural Risk Embeddings)
           │
           ▼
[ 80-dim Hybrid Feature ]  ──► (64 GNN Risk Dims + 16 Geospatial 3D & Temporal Dims)
           │
           ▼
[ XGBoost Classifier & Regressor v2 ]
  ├── 📍 Search Zone (87.4% containment; 1,000 ATMs -> a median of 8)
  ├── 📍 Top-3 ATM Ranking (Conditional Logit over 25 reachable candidates)
  ├── ⏱️ Time-to-Cashout Countdown (MAE: 11.8 min, R² 0.17, q05-q95 band)
  └── 🔒 Real-time Micro-Freeze Action Recommendation (<25ms latency)
```

---

## 🔬 Core AI / ML Architecture

### 1. Graph Intelligence Engine (`engine/graph_engine.py`)
- Constructs directed multi-hop transaction networks: `Victim Account ➔ Layer-1 Mule ➔ Layer-2 Mule ➔ Terminal Cashout Node`.
- **BFS Traversal:** Instant sub-graph extraction for any complaint ID or victim account.
- **Velocity Anomaly Detection:** Flags accounts with $>2$ high-value outgoing transactions within a 5-minute window.
- **Fund-Splitting Detection:** Detects rapid 1-to-$N$ dispersal ($N \ge 3$) where split amounts have variance $<30\%$ of the mean.
- **Terminal Node Identification:** Pinpoints leaf nodes with $100\%$ precision for cashout risk evaluation.

### 2. GraphSAGE Embedding Engine (`engine/gnn_model.py`, `engine/train_gnn.py`)
- **2-Layer GraphSAGE** (Mean Neighborhood Aggregation) built on PyTorch Geometric.
- Encodes node features: geographical coordinates (`lat`, `long`), net transaction volumes (`total_received`, `total_sent`), transaction velocity (`txn_count_24h`, `avg_txn_amount`), and layer depth (`hop_depth`).
- Binary classification head trained with `BCEWithLogitsLoss` and inverse class-frequency weighting (`pos_weight = 0.142`).
- **Output:** 64-dimensional dense risk representation vector ($h_v \in \mathbb{R}^{64}$) for each bank account node.

### 3. Withdrawal-Location Forecasting Engine (`engine/feature_builder.py`, `engine/xgb_model.py`)
- Integrates the 64-dim GNN representation with **16 spatial/temporal tabular features** into an **80-dimensional hybrid vector**:
  - `stolen_amount`, `hop_depth`, `transaction_velocity`, `hour_of_day`, `historical_hotspot_density`, `day_of_week`, `amount_after_split`
  - `dist_to_atm_1/2/3_km` — Vectorized Haversine distance to top-3 nearest ATMs.
  - `node_x, node_y, node_z` — 3D Cartesian Earth coordinates (eliminates angular bias in trees).
  - `bearing_to_atm_1_deg` — Compass bearing to nearest ATM (0°–360°).
  - `is_nearest_same_bank`, `nearest_same_bank_atm_dist` — Bank affiliation preference features.
- **Models:**
  - **`ConditionalLogitRanker`**: ranks the 25 reachable ATMs per cashout — **Top-3 0.5615** vs a 0.5485 distance-only baseline, and aggregated into a **search zone with 87.4% containment** vs 78.5% for a nearest-3 centroid. Where a cashout happens is a *discrete choice among alternatives*, and the drivers compose multiplicatively, so in log space the choice is linear — which is exactly a conditional logit. A 953-way softmax over the national ATM directory saw ~5 examples per class and scored *below* a nearest-ATM rule; a gradient-boosted ranker had to approximate products with axis-aligned steps and also lost.
  - **`XGBRegressor`**: Estimates countdown minutes — **11.82 min MAE** ($R^2 = 0.17$) against a 14.98 min mean-prediction baseline, with a q05-q95 band at 77.2% coverage. The observable-conditioned ceiling is 9.38 min / $R^2$ 0.445 — the cashout regime is not fully knowable.
- **Interpretable utility weights** (recovered from data, checkable against the generator):

  | Term | Learned | True |
  |---|---|---|
  | `-distance/5` | +0.99 | 1.00 |
  | `log(1 + 2·risk)` | +0.79 | 1.00 |
  | `same_bank` | +0.60 | log 2 = 0.69 |
  | `crew_prior` | +0.29 | — |

---

## 📊 Benchmark & Validation Results

SIH26184 asks for one thing: *"Forecast Likely Cash **Withdrawal Locations** in
Advance."* That is what the headline metric measures. Every figure is reported **next
to the naive baseline it has to beat** — a score without its baseline says nothing
about a model, and an accuracy that looks too good usually is (see
[Honest Evaluation](#-honest-evaluation)).

Validated on a Pan-India dataset of **49,999 accounts**, **622,188 transactions**
(22,201 laundering + 599,987 legitimate) and **1,000 ATMs**, with **275/275 tests passing**.

<div align="center">
  <img src="docs/sih_performance_matrix_slide.png" alt="MuleShield AI - validated performance summary" width="100%" />
</div>

<br/>

<div align="center">
  <img src="docs/model_matrix_full.png" alt="MuleShield AI - six-panel model performance matrix" width="100%" />
  <p><em>Figure: evaluation matrix computed directly from the trained models by
  <code>scripts/generate_model_matrix.py</code> (matplotlib + scikit-learn + PyTorch + XGBoost).
  Every panel carries its baseline, and ceilings are drawn where one exists.</em></p>
</div>

### 1. Withdrawal-location forecast — the deliverable

A patrol is dispatched to an *area*, not to one machine. The model collapses its
distribution over reachable ATMs into a search zone; the question is whether the
withdrawal happens inside it.

| Zone centre | Contains the withdrawal | Median error |
|---|---|---|
| Centre on the mule's location | 75.2% | 6.39 km |
| Nearest-3 ATM centroid | 78.5% | 5.68 km |
| **Model search zone** | **87.4%** | **5.49 km** |

*All three given the same radius, so the comparison is at equal search cost — a
bigger zone always contains more, and rewarding that would measure zone size rather
than skill.*

**Search cost: 1,000 ATMs narrowed to a median of 8, inside a 10.0 km radius, in ~4.5 ms.**

### 2. Time to cashout — "in Advance"

| Model | MAE | R² |
|---|---|---|
| Predict the mean | 14.98 min | 0.00 |
| **XGBoost regressor** | **11.82 min** | **0.17** |

Reported with a **q05–q95 band at 77.2% empirical coverage**. The delay is a
two-regime mixture — a crew either has a runner already at the machine or has to
travel — and which regime applies is a draw no model can observe, even knowing the
crew's speed exactly. Conditioned on what *is* observable the ceiling is **9.38 min
MAE / R² 0.445**, not 1.0, so a point estimate alone would overstate what is
knowable and "expected in 25–45 min" is the honest form. This is the one component
with meaningful headroom left.

### 3. Ranked candidate locations — the Top-K operating point

The system returns the **K locations an investigator should search first**, ordered.
Measured by `scripts/topk_curve.py` against the **shipped** checkpoint on the held-out
complaints of the same seed-42 `GroupShuffleSplit` that training fits on — no retraining,
no tuning, and a retrieval failure counts as a miss.

| K | Containment | Distance-only | Search reduction |
|---|---|---|---|
| 1 | 0.2654 | 0.2476 | 99.90% |
| 3 | 0.5615 | 0.5485 | 99.70% |
| **5 (operating point)** | **0.7136** | 0.7217 | **99.50%** |
| 8 | 0.8689 | 0.8689 | 99.20% |
| 10 | 0.9175 | 0.9142 | 99.00% |

*n = 618 held-out cash-outs · 1,000-ATM directory · 0.003 ms to rank.*

> **Top-5 containment is 0.7136 — it is not 87.4%.** That figure is the adaptive search
> zone at a median of 8 ATMs, a different operating point. A test asserts Top-5 < 0.80 so
> the zone number cannot migrate into the Top-5 slot.

**Retrieval never fails.** The true ATM is inside the 25-candidate pool in 618 of 618
cases, so widening the pool cannot help and every miss is a ranking miss. But of the 177
misses at K=5, **94.9% are unavoidable** — the true ATM sat outside the top-5 of the *true
generative posterior*, i.e. the offender made a low-probability choice. Only **9 of 618
(1.5%)** are genuine model error.

**We do not claim the ranker beats distance.** It is statistically indistinguishable from
distance-sorting at every K (paired McNemar over the same 618 cash-outs, no p < 0.05; at
K=5 distance is ahead by 0.008). That is what a ceiling looks like, not a defect: Top-1
equals the Bayes bound exactly and Top-5 sits 0.8 points under it. The defensible claim is
the **pipeline narrowing 1,000 ATMs to 5**, not that the ranker outperforms a distance
heuristic — because on this generator it does not.

The ranker's learned weights are interpretable by construction and can be checked
against expectation:

| Utility term | Learned | Generative truth |
|---|---|---|
| `−distance / 5` | +0.99 | 1.00 |
| `log(1 + 2·risk)` | +0.79 | 1.00 |
| `same_bank` | +0.60 | log 2 = 0.69 |
  | `crew_prior` | +0.29 | — |
| `crew_prior` | +0.29 | — |

### 4. Mule detection — supporting machinery

Not a deliverable of SIH26184; it earns its place by identifying the crew (the
strongest non-distance signal in the location model) and by naming freeze targets.
All rows scored on the **same held-out nodes**:

| Model | F1 | AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Majority class | 0.0574 | — | — | 0.0295 | 1.0000 |
| Best single feature (`burst_out_5min`) | 0.5303 | 0.8936 | 0.3815 | 0.4922 | 0.5747 |
| Logistic regression (no graph) | 0.8322 | 0.9519 | 0.8149 | 0.8458 | 0.8190 |
| XGBoost (tabular, no graph) | 0.8462 | 0.9504 | 0.8285 | 0.8462 | 0.8462 |
| Random forest (no graph) | 0.8463 | 0.9580 | 0.8391 | 0.8333 | 0.8597 |
| **GraphSAGE GNN** | **0.8955** | **0.9639** | **0.8529** | **0.8995** | **0.8914** |

**+0.049 F1 over the best non-graph model on identical features**, all scored on the same
held-out nodes. The label-noise ceiling is **0.927**, so this sits at 97% of what is
attainable. FPR 0.0031, FNR 0.1086, 10,561 parameters.

### 5. System performance

| Requirement | Target | Achieved |
|---|---|---|
| Per-complaint graph build | < 500 ms | **~2 ms** ✅ |
| Full national graph build (startup) | < 1200 ms | **~670 ms** ✅ |
| End-to-end inference | < 200 ms | **~4.5 ms** ✅ |
| Automated test coverage | 100% | **275 / 275** ✅ |

Reproduce with:

```bash
python scripts/evaluate_baselines.py       # baseline tables
python engine/train_xgb.py                 # zone, ranking and countdown metrics
python scripts/generate_model_matrix.py    # regenerates the two figures above
python scripts/eda_report.py               # data-integrity evidence
python scripts/feature_analysis.py         # feature signal ranking + heatmap
```

---

## 🔍 Honest Evaluation

Three properties of the dataset are load-bearing, each fixing a way an earlier version
of this project was measuring nothing:

**1. The label is not derivable from any feature.** Mule status is assigned to an
account *before any transaction exists*, from a behavioural archetype; features are
then measured from simulated activity. Previously a mule was defined as "an account
that received money" and non-mules were written `total_received = 0`, so the rule
`total_received > 0` scored **F1 = 1.0000** — the GNN's reported 0.9996 was measuring a
copy of its own input. `tests/test_phase1.py::test_label_is_not_a_copy_of_a_feature`
now fails the build if any single feature reproduces the label above F1 0.95.

**2. The classes deliberately overlap.** The ledger contains ordinary banking traffic —
salary credits, merchant settlements, remittances — so legitimate accounts receive
money too. The `transit_business` archetype (payment aggregators, trading firms)
forwards almost everything it receives within minutes, exactly like a mule. The best
single feature now reaches only F1 0.80.

**3. Ground truth lives in the data, not in the feature builder.** The cashout ATM is
*sampled* from a behavioural choice model (distance decay × surveillance risk × bank
affinity × the syndicate's established cashout points) and written to the ledger.
Previously the label was `argmin(distance)` while distance was feature #67 — the model
was asked to find the nearest ATM while holding the distance to it.

Additionally: models are split **by complaint**, never by row, so no laundering chain
straddles train and test; ATM priors are computed only from the earliest 50% of
complaints, which are then excluded from training and evaluation; baselines are scored
on the **same held-out nodes** as the GNN; and 2% label noise reflects imperfect bank
reporting.

### Known limitations

- All data is **synthetically generated**. The behavioural archetypes are informed by
  published mule typologies, not fitted to real bank data.
- **Mule prevalence in this dataset is 2.95%; in a real bank population it is under 1%.**
  At realistic prevalence, Precision@K against a fixed daily alert budget is the operative
  metric rather than F1.
- Micro-freeze and the SMS gateway are **simulated**; NPCI/CBS integration is a
  deployment step. WhatsApp dispatch opens a real message.
- A live-ingested complaint has its laundering chain **synthesised at ingestion** from
  real graph accounts in the victim's city. The GNN embeddings, ATM directory and
  inference path are genuine; only the bank/NPCI transaction feed is simulated.
- We have **not** run a field trial, so we claim no fund-recovery-rate improvement.

---

> **Working on this project?** Read [`PROJECT_NOTES.md`](PROJECT_NOTES.md)
> first. It carries the outstanding checklist, the decisions that were made
> deliberately and should not be "fixed", and the testing gotchas — including why
> a passing build is not evidence the console works.

## 📁 Repository Structure

```
SIH2026/
├── backend/                            # FastAPI Real-Time REST & WebSocket Backend
│   ├── main.py                         # Application entry point & lifespan manager
│   ├── state.py                        # In-memory high-speed O(1) state store
│   ├── websocket.py                    # Real-time WebSocket connection broadcaster
│   ├── models/                         # Pydantic v2 schemas
│   │   └── schemas.py                  # Request/response data models
│   ├── routers/                        # API endpoint routers
│   │   ├── complaint.py                # 1930 Complaint ingestion & listing
│   │   ├── graph.py                    # React Flow money-flow DAG builder
│   │   ├── embeddings.py               # GNN risk score ranking
│   │   ├── predict.py                  # XGBoost Top-3 ATM + countdown inference
│   │   └── freeze.py                   # 1-Click Bank micro-freeze simulator
│   └── tests/                          # Phase 3 backend test suite (51/51 passed)
│
├── frontend/                           # React 19 + Tailwind Tactical Command UI
│   ├── src/
│   │   ├── pages/
│   │   │   ├── TriageFeed.jsx          # Screen 1: Live 1930 Triage Queue
│   │   │   ├── TacticalMap.jsx         # Screen 2: Tactical GIS Map (Leaflet)
│   │   │   ├── ForensicGraph.jsx       # Screen 3: Money Flow Graph (React Flow)
│   │   │   └── Interception.jsx        # Screen 4: 1-Click Freeze & Dispatch
│   │   ├── components/                 # Reusable UI shells & navigation bars
│   │   └── services/                   # Axios API & WebSocket connector
│   └── package.json
│
├── data/                               # Pan-India Banking & ATM Dataset (65+ Cities)
│   ├── victim_complaints.csv           # 2,500 National 1930 Cybercrime complaints
│   ├── transactions.csv                # 22,864 Multi-hop transactions (hops 1–4)
│   ├── atm_directory.csv               # 1,000 ATMs across 65+ Indian cities with GPS
│   ├── graph_edges.csv                 # 22,864 Directed edges (NetworkX / PyG)
│   └── node_features.csv               # 20,468 Nodes with behavioral & geographical stats
│
├── engine/                             # Core Hybrid AI Engines
│   ├── graph_engine.py                 # NetworkX directed graph builder & BFS anomalies
│   ├── gnn_model.py                    # 2-Layer GraphSAGE model (PyTorch Geometric)
│   ├── train_gnn.py                    # GNN offline training script
│   ├── embed.py                        # 64-dim GraphSAGE embedding extractor & cache
│   ├── feature_builder.py              # 80-dim Hybrid Feature Matrix Builder (v2)
│   ├── xgb_model.py                    # MuleXGBPredictor (Top-3 ATM + countdown regressor)
│   └── train_xgb.py                    # XGBoost training & latency benchmarking
│
├── models/                             # Serialized Trained Model Checkpoints
│   ├── graphsage_mule.pt               # Trained GraphSAGE PyTorch checkpoint (20,468 nodes)
│   └── xgb_cashout.pkl                 # Trained XGBoost predictor bundle
│
├── embeddings/                         # Cached Node Embeddings
│   └── node_embeddings.pkl             # 20,468 x 64-dim pre-computed risk vectors
│
├── tests/                              # Comprehensive Pytest Regression Suites (184 tests)
│   ├── test_phase1.py                  # Phase 1 tests (84 tests - Data generation & schema)
│   ├── test_phase2a.py                 # Phase 2a tests (47 tests - GraphSAGE & graph engine)
│   └── test_phase2b.py                 # Phase 2b tests (53 tests - XGBoost & 80-dim inference)
│
├── prd.md                              # Product Requirements Document
├── phases.md                           # SIH development phases & milestone tracking
└── README.md                           # Project documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone & Setup Python Backend
```bash
git clone https://github.com/hotshot0104/SIH2026.git
cd SIH2026

# Virtual Environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Generate the dataset  *(required — the repo does not ship it)*
```bash
python scripts/generate_data.py
```
`data/transactions.csv` (~114 MB) and `data/graph_edges.csv` (~40 MB) exceed GitHub's
file limit, so they are gitignored rather than committed. Generation is deterministic
at seed 42 — the same corpus every time — and takes a few minutes. **Skip this and the
backend starts, but every screen renders empty.**

Pre-trained models are committed, so training is optional. To reproduce them:
```bash
python engine/train_gnn.py && python engine/embed.py && python engine/train_xgb.py
```

### 3. Start the FastAPI Backend
```bash
python -m uvicorn backend.main:app --port 8000 --reload
# API Documentation (Swagger UI): http://localhost:8000/docs
```

### 4. Start the React Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
# Command Center UI: http://localhost:5173
```

### 5. Run the Full Test Suite (275 Tests)
```bash
# Run AI engine unit tests (224 tests)
python -m pytest tests/ -v

# Run FastAPI backend tests (51 tests)
python -m pytest backend/tests/ -v
```

**UI smoke test** — drives every button, link, input and select in the console
against the live backend, reporting console errors, failed requests and controls
with no visible effect. Needs both servers running:
```bash
python scripts/smoke_ui.py
```
Last run: **209 controls across 4 routes, 0 console errors, 0 failed requests**
in normal operation. The only 404 is the deliberate existence-check that discards
a stale complaint id.

---

## 🗺️ Roadmap & Phase Completion

- [x] **Phase 1: Synthetic Dataset Generator** *(84/84 Tests Passing)*
  - 2,500 complaints, 22,864 multi-hop transactions, 1,000 ATMs across 65+ Indian cities.
  - Embedded multi-source fraud rings with balanced mule/clean node features.
- [x] **Phase 2a: Graph Intelligence & GraphSAGE Engine** *(49/49 Tests Passing)*
  - NetworkX directed graph builder with BFS traversal (<185ms).
  - 2-layer GraphSAGE classifier on 15 behavioural features (F1 0.8955).
  - Real-time sub-graph embedding extractor (<0.39s).
- [x] **Phase 2b: XGBoost ATM Prediction Engine v2** *(53/53 Tests Passing)*
  - 80-dim hybrid vector for the GNN stage; 19-dim context+candidate vector for the ATM ranker.
  - Search zone (87.4% containment), Top-3 ATM ranking (0.5615), countdown (11.82 min MAE).
  - Single inference latency: 25.8 ms.
- [x] **Phase 3: Real-Time FastAPI Backend** *(51/51 Tests Passing)*
  - REST endpoints for complaint ingestion, graph exploration, GNN embeddings, and ATM predictions.
  - WebSocket broadcaster (`/ws/feed`) for live event push.
- [x] **Phase 4: Tactical Command Dashboard** *(Live on port 5173)*
  - 4 interactive screens: Triage Queue, Tactical GIS Map, Forensic Graph, and 1-Click Interception.
- [x] **Phase 4.5: Hardening Pass** *(275/275 Tests Passing)*
  - Live-ingested complaints now run the full GNN + XGBoost pipeline (chain grounded on real graph accounts).
  - Velocity / fund-splitting detections surfaced from `graph_engine.py` instead of static placeholders.
  - Node risk switched to the trained GraphSAGE classification head — `sigmoid(Wh + b)`.
  - Bank-affinity features repaired (were constant), model retrained; ATM addresses aligned to their own city.
  - Self-hosted fonts + Leaflet CSS and basemap-failure fallback for offline venues.
- [x] **Phase 4.6: Leakage Removal & Honest Re-baselining** *(275/275 Tests Passing)*
  - Rebuilt the generator so mule status is fixed before any transaction exists.
  - Added 29,998 legitimate transactions so the classes genuinely overlap.
  - Cashout ATM + delay sampled by a choice model and stored as ground truth.
  - Complaint-level splits, time-separated priors, 2% label noise.
  - Every metric now reported against its naive baseline (`scripts/evaluate_baselines.py`).
- [ ] **Phase 5: Live Simulation Demo & SIH Presentation Pitch**

---

## ⚖️ Ethics & Compliance (DPDP Act 2023)

All datasets used in this repository are **100% synthetically generated** in strict compliance with Indian cyber laws:
- **Digital Personal Data Protection (DPDP) Act, 2023:** Zero scraped personal data. Account numbers are pseudonymized masks conforming to Indian banking norms (`ACC-XXXXXXXX`).
- **IT Act, 2000 (Section 43 & 66):** Zero unauthorized network access.
- **Enterprise Integration:** Standardized API schemas ready for live integration with the **National Cybercrime Reporting Portal (NCRP / 1930)** and **NPCI Switch**.

---

## 👥 Team
- **Team HACKSTERS** — Smart India Hackathon 2026 (Problem Statement: `SIH26184`)
