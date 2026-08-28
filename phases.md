# 🗓️ MuleShield AI — Development Phases (`SIH26184`)

> Milestone-by-milestone execution plan for SIH 2026.
> Status: 🔴 Not Started | 🟡 In Progress | 🟢 Complete

---

## Overview

```
Phase 1 ──► Phase 2a ──► Phase 2b ──► Phase 3 ──► Phase 4 ──► Phase 5
 Data        GraphSAGE    XGBoost      Backend     Frontend    SIH PPT
 Generator   Embeddings   Classifier   API         Dashboard   & Demo
```

> **AI Pipeline:** `Transaction Graph → NetworkX (BFS) → GraphSAGE (embeddings) → XGBoost (real-time) → ATM Prediction`

| Phase | Name | Status | ETA |
|---|---|---|---|
| 1 | Realistic Indian Banking Dataset Generator | 🟢 Complete | — |
| 2 | Core AI & Graph Intelligence Engine | 🟢 Complete | — |
| 3 | Backend API & Real-Time Engine | 🟢 Complete | — |
| 4 | Frontend Command Dashboard | 🟢 Complete | — |
| 5 | SIH PPT Presentation & Video Demo | 🔴 Not Started | — |

---

## 📍 Phase 1 — Realistic Indian Banking & Cybercrime Dataset Generator

**Goal:** Create a 100% synthetic but realistic dataset that mirrors real 1930 complaint and banking switch feeds. Uses **gen-fraud-graph** (SantanderAI) to generate realistic fraud ring graphs, augmented with Indian banking context.

### Reference Repos
- [`SantanderAI/gen-fraud-graph`](https://github.com/SantanderAI/gen-fraud-graph) — fraud ring graph generator (adapt for Indian data)
- [`sohamvjadhav/Mule-Hunt`](https://github.com/sohamvjadhav/Mule-Hunt) — UPI mule detection reference (study synthetic data approach)

### Deliverables

| File | Description |
|---|---|
| `data/victim_complaints.csv` | 1930-style complaint tickets |
| `data/transactions.csv` | Multi-hop IMPS/UPI transaction ledger with fraud ring structure |
| `data/atm_directory.csv` | Geo-coded ATM/CSP database (Indian lat/long) |
| `data/graph_edges.csv` | Edge list format for direct PyG/NetworkX ingestion |
| `data/node_features.csv` | Per-account feature matrix for GNN training |
| `scripts/generate_data.py` | Python generator script |

### Dataset Schema

**`victim_complaints.csv`**
```
ticket_id, victim_name, victim_bank, victim_account, fraud_type,
stolen_amount, complaint_timestamp, city, state
```
- `fraud_type`: Digital Arrest | Job Scam | UPI Fraud | Investment Scam | Romance Scam

**`transactions.csv`** (edges)
```
txn_id, complaint_id, src_account, dst_account, bank_name, ifsc_code,
city, district, state, lat, long, amount, timestamp, hop_depth, is_terminal
```

**`node_features.csv`** (for GNN input)
```
account_id, bank_name, city, lat, long, total_received, total_sent,
txn_count_24h, avg_txn_amount, is_mule_label, hop_depth
```
- `is_mule_label`: 0 = clean, 1 = confirmed mule (synthetic ground truth)

**`atm_directory.csv`**
```
atm_id, bank_name, address, city, district, state, lat, long,
opening_time, closing_time, cashout_risk_score, historical_fraud_count
```

### Tools & Libraries
- `faker` — Indian names, addresses, phone numbers
- `pandas`, `numpy` — data assembly
- `networkx` — fraud ring graph generation logic
- `geopy` — distance calculations
- `gen-fraud-graph` (adapted) — realistic fraud ring embedding into transaction chains

### Acceptance Criteria
- [ ] 500+ victim complaint records generated
- [ ] 2000+ transaction records (multi-hop chains of depth 1–4, with ≥3 fraud rings embedded)
- [ ] 200+ ATM/CSP entries across 10+ Indian cities
- [ ] `node_features.csv` and `graph_edges.csv` ready for PyG ingestion
- [ ] All lat/long within valid Indian geographic bounds (lat 8–37, lon 68–97.5)
- [ ] CSV files saved to `data/` folder

---

## 📍 Phase 2a — Graph Intelligence & GraphSAGE Embedding Engine

**Goal:** Build the structural intelligence layer — NetworkX for graph analysis + GraphSAGE to learn fraud-ring embeddings.

> **Why GraphSAGE?** It aggregates neighbourhood information for each account node. A mule buried 3 hops deep will have embeddings influenced by its fraudulent neighbours — something pure XGBoost can never detect.

### Reference Repos
- [`sohamvjadhav/Mule-Hunt`](https://github.com/sohamvjadhav/Mule-Hunt) — **primary reference** for GraphSAGE on UPI mule graphs
- [`issacchan26/AntiMoneyLaunderingDetectionWithGNN`](https://github.com/issacchan26/AntiMoneyLaunderingDetectionWithGNN) — GAT reference for comparison

### Deliverables

| File | Description |
|---|---|
| `engine/graph_engine.py` | NetworkX graph builder + BFS traversal + anomaly detection |
| `engine/gnn_model.py` | GraphSAGE model definition (PyTorch Geometric) |
| `engine/train_gnn.py` | Offline GNN training script |
| `engine/embed.py` | Generates and caches node embeddings for all accounts |
| `models/graphsage_mule.pt` | Trained GraphSAGE model weights |
| `embeddings/node_embeddings.pkl` | Pre-computed 64-dim embeddings per account |

### Graph Engine (`graph_engine.py`)
- Build directed graph: `victim_account → mule_L1 → mule_L2 → mule_L3`
- **BFS traversal** to enumerate all reachable mule nodes from victim
- **Velocity detection**: flag accounts with >2 outgoing transactions within 5 minutes
- **Fund-splitting detection**: single source splits into 3+ destinations of similar amounts
- **Terminal node identification**: leaf nodes (cashout candidates)
- Output: JSON node-link format for React Flow

### GraphSAGE Model (`gnn_model.py`)
```python
# Architecture: 2-layer GraphSAGE
GraphSAGE(
    in_channels  = 11,          # node_features columns
    hidden_channels = 64,
    out_channels = 64,          # embedding dimension
    num_layers = 2,
    aggr = 'mean'               # neighbourhood aggregation
)
# Trained as: binary node classifier (mule=1 / clean=0)
# Embedding layer output (before final linear) = 64-dim risk vector
```

**Training:**
- Loss: `BCEWithLogitsLoss` (binary: mule vs clean)
- Optimizer: `Adam`, lr=`0.001`, epochs=`200`
- Split: 70% train, 15% val, 15% test on synthetic graph
- Metric: F1-score, AUC-ROC (target: F1 > 0.88 on synthetic)

### Acceptance Criteria
- [ ] Graph builds from CSV in < 500ms for 500-node datasets
- [ ] Velocity + fund-splitting detection working correctly
- [ ] Terminal nodes correctly identified with ≥90% accuracy (synthetic ground truth)
- [ ] GraphSAGE trains to F1 > 0.85 on synthetic dataset
- [ ] Node embeddings (64-dim) saved to `embeddings/node_embeddings.pkl`
- [ ] `embed.py` can generate embeddings for a new complaint graph in < 2s

---

## 📍 Phase 2b — XGBoost ATM Prediction Engine

**Goal:** Use GraphSAGE embeddings + tabular features to predict the ATM, time-to-cashout, and confidence score in real-time.

### Deliverables

| File | Description |
|---|---|
| `engine/feature_builder.py` | Combines GNN embeddings + tabular features into XGBoost input |
| `engine/xgb_model.py` | XGBoost ATM predictor |
| `engine/train_xgb.py` | XGBoost training script |
| `models/xgb_cashout.pkl` | Trained XGBoost model |

### Hybrid Feature Vector
```python
features = [
    # ── GNN Embeddings (64 dims) — captures fraud ring structure ──
    *graphsage_embedding,         # 64-dim risk vector from GraphSAGE

    # ── Tabular Features (8 dims) ──
    "stolen_amount",              # Original fraud amount
    "hop_depth",                  # Layer depth of terminal mule
    "transaction_velocity",       # Txns per minute from terminal account
    "branch_distance_to_atm",     # km from mule bank to nearest ATM
    "hour_of_day",                # 0–23 (cashout peaks at night)
    "historical_hotspot_density", # ATM fraud incidents within 5km
    "day_of_week",                # 0=Mon, 6=Sun
    "amount_after_split",         # Amount at terminal after splitting
]
# Total: 80 features (64 GNN + 16 spatial/temporal tabular)
# Reduced for ATM ranking to 24 columns per candidate row
#   = 12 case-context (incl. 5 account-behaviour) + 12 per-candidate
```

**Targets:**
- `search_zone` — centroid, radius and ATM count: the PS deliverable
- `predicted_atm_id` — Top-3 ranked ATMs inside the zone (conditional logit)
- `time_to_cashout_minutes` — countdown, with a q05-q95 band
- `interception_confidence` — choice probability of the top candidate (0.0-1.0)

**Class Imbalance Handling:** `scale_pos_weight` in XGBoost + SMOTE-ENN on training set

### Acceptance Criteria
- [ ] `feature_builder.py` correctly concatenates GNN embeddings + tabular features
- [ ] XGBoost trained and saved as `models/xgb_cashout.pkl`
- [x] Search zone returned (centroid, radius, ATM count) — **86.8% containment**
      vs 76.5% for a nearest-3 centroid at equal search cost
- [x] Top-3 ATM prediction returned with confidence scores — **0.5658** vs 0.5596
      distance-only
- [x] Countdown beats a mean-prediction baseline — **6.14 min MAE** vs 9.41 min
      (the original ±5 min target was only reachable when the label was a
      closed-form function of two input features; see README → Honest Evaluation)
- [x] End-to-end inference in < 200ms — **~15 ms**

---

## 📍 Phase 3 — Backend API & Real-Time Engine (FastAPI)

**Goal:** Expose graph + GNN + XGBoost engine via REST API and WebSocket for real-time dashboard.

### Deliverables

| File | Description |
|---|---|
| `backend/main.py` | FastAPI application entry point |
| `backend/routers/complaint.py` | Complaint ingestion router |
| `backend/routers/graph.py` | Graph data router |
| `backend/routers/embeddings.py` | **NEW:** GNN embedding metadata router |
| `backend/routers/predict.py` | Prediction router (XGBoost inference) |
| `backend/routers/freeze.py` | Micro-freeze simulation router |
| `backend/websocket.py` | WebSocket real-time broadcast manager |
| `backend/models/schemas.py` | Pydantic request/response schemas |

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/complaint/ingest` | Ingest a new 1930 complaint |
| `GET` | `/api/v1/complaint/list` | List all active complaints |
| `GET` | `/api/v1/graph/{complaint_id}` | Return node-link graph data (React Flow format) |
| `GET` | `/api/v1/embeddings/{complaint_id}` | **NEW:** Return top mule nodes + GNN risk scores |
| `GET` | `/api/v1/predict/cashout/{complaint_id}` | ATM prediction + countdown (XGBoost) |
| `POST` | `/api/v1/bank/micro-freeze` | Simulate card freeze on target account |
| `WS` | `/ws/feed` | Real-time complaint stream to dashboard |

### Acceptance Criteria
- [ ] All endpoints return correct JSON within < 200ms
- [ ] WebSocket pushes new complaint events to connected clients
- [ ] Micro-freeze endpoint returns `{ status: "FROZEN", account: "...", timestamp: "..." }`
- [ ] API docs auto-generated at `/docs` (FastAPI Swagger UI)
- [ ] CORS enabled for React frontend origin

---

## 📍 Phase 4 — Frontend Command Dashboard (React + Vite + Tailwind + Leaflet)

**Goal:** Build the 4-screen tactical command dashboard for law enforcement.

### Deliverables

| Path | Description |
|---|---|
| `frontend/src/pages/TriageFeed.jsx` | Screen 1: Live 1930 Complaint Feed |
| `frontend/src/pages/TacticalMap.jsx` | Screen 2: GIS Police Map |
| `frontend/src/pages/ForensicGraph.jsx` | Screen 3: Interactive Money Flow Graph |
| `frontend/src/pages/Interception.jsx` | Screen 4: 1-Click Freeze & Dispatch |
| `frontend/src/components/` | Reusable UI components |

### Screen Specifications

**Screen 1 — Live 1930 Triage Feed**
- WebSocket-connected live stream of incoming complaints
- Each card: Ticket ID, Victim Name, Amount, Fraud Type, Time Elapsed
- Color coded: 🔴 Critical (<15 min) | 🟡 Warning (15–30 min) | 🟢 Monitoring (>30 min)
- Click a complaint → triggers graph build + prediction → routes to Screen 2

**Screen 2 — Tactical Police GIS Map**
- Leaflet.js base map (India, OpenStreetMap tiles)
- Blinking red radius circles around predicted ATM targets
- Countdown clock overlay on each ATM marker
- Police station markers with route lines to target ATM
- Satellite toggle for last-mile navigation

**Screen 3 — Interactive Forensic Graph**
- React Flow / Cytoscape.js directed graph
- Victim node (blue) → Layer-1 mules (orange) → Layer-2 (red) → Terminal (blinking red)
- Click any node → side panel shows: Account No, Bank, IFSC, Amount, Timestamp
- Edge labels: transaction amount + time delta
- Export graph as PNG button

**Screen 4 — 1-Click Interception Panel**
- Shows Top-3 ATM predictions with confidence bars
- **"Emergency Card Freeze" button** → calls `/api/v1/bank/micro-freeze`
- **"Dispatch Patrol Unit" button** → generates WhatsApp/SMS alert popup
- Freeze confirmation: animated lock icon + timestamp

### Acceptance Criteria
- [ ] All 4 screens navigable via sidebar
- [ ] Real-time WebSocket feed updates Screen 1 without page refresh
- [ ] Map loads with correct ATM markers and countdown timers
- [ ] Graph renders with correct node/edge structure from API
- [ ] Freeze and Dispatch buttons trigger correct API calls and show confirmation

---

## 📍 Phase 5 — SIH PPT Presentation & Video Demo

**Goal:** Deliver a compelling 7-slide SIH official presentation and a 3-minute demo video.

### PPT Structure (7 Slides)

| Slide | Title | Content |
|---|---|---|
| 1 | Title Slide | MuleShield AI, SIH26184, Team Name, Institute |
| 2 | Problem Statement | 1930 complaints, Golden Hour, recovery rate statistics |
| 3 | Solution Overview | 4-layer architecture diagram |
| 4 | Tech Stack & Innovations | AI/ML, Graph Intelligence, Spatio-Temporal Prediction |
| 5 | Live Demo Screenshots | All 4 dashboard screens |
| 6 | Impact & Scalability | Recovery rate improvement, deployment plan |
| 7 | Team & Acknowledgements | Team members, mentor, institute |

### Demo Video Script (3 minutes)

```
00:00–00:20  Intro: "₹50,000 was stolen. You have 35 minutes."
00:20–00:50  Victim calls 1930 → complaint ingested → graph builds in <1s
00:50–01:30  Money flow graph appears → mule chain visualized → terminal node blinks
01:30–02:10  Target ATM appears on map → countdown clock starts → 92.4% confidence
02:10–02:40  Operator clicks "Emergency Card Freeze" → card locked → bank notified
02:40–03:00  "₹50,000 SAVED." → MuleShield AI tagline
```

### Acceptance Criteria
- [ ] PPT follows SIH official template guidelines
- [ ] Video is ≤ 3 minutes, 1080p minimum
- [ ] All 4 dashboard screens featured in demo
- [ ] Demo scenario uses realistic Indian names, cities, and amounts

---

*Last updated: 2026-08-28*
