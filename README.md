# 🛡️ MuleShield AI: Real-Time Money Mule Detection & ATM Interception Intelligence

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/React-19.0.0-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-GraphSAGE-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20v2-green.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-235%2F235%20Passed-brightgreen.svg)]()
[![SIH 2026](https://img.shields.io/badge/SIH-2026%20Problem%20ID%3A%20SIH26184-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Smart India Hackathon (SIH 2026)** | Ministry of Home Affairs (MHA) / Indian Cybercrime Coordination Centre (I4C)  
> **Problem Statement ID:** SIH26184 — *Identification of Money Mule Accounts and ATM Geolocation for Cashout Interception*

---

## 🖥️ Tactical Command Center Dashboard

MuleShield AI features a 4-screen, real-time command dashboard engineered for Law Enforcement (State Cyber Crime Police Stations) and Bank Fraud Risk Officers:

<div align="center">
  <h3>1. Live 1930 Helpline Triage & Intake Queue</h3>
  <img src="docs/dashboard_triage.png" alt="MuleShield AI - Live 1930 Triage Feed" width="100%" />
  <p><em>Real-time streaming queue of incoming 1930 complaints with Golden Hour urgency badges, live metrics, and instant AI ingestion form.</em></p>
</div>

<br/>

<div align="center">
  <h3>2. Tactical GIS Map — Physical ATM Interception</h3>
  <img src="docs/dashboard_map.png" alt="MuleShield AI - Tactical GIS Map" width="100%" />
  <p><em>Interactive vector map displaying terminal mule location, predicted Top-3 target ATM cluster markers, pulsing perimeter, and routing lines.</em></p>
</div>

<br/>

<div align="center">
  <h3>3. Forensic Money-Flow Directed Graph (DAG)</h3>
  <img src="docs/dashboard_graph.png" alt="MuleShield AI - Forensic Graph Visualizer" width="100%" />
  <p><em>React Flow multi-hop graph visualizer mapping money movement from Victim ➔ Layer-1/2 Mules ➔ Terminal Cashout accounts with node-level GNN risk inspector.</em></p>
</div>

<br/>

<div align="center">
  <h3>4. 1-Click Emergency Interception & Police Dispatch</h3>
  <img src="docs/dashboard_intercept.png" alt="MuleShield AI - Interception Control" width="100%" />
  <p><em>Top-3 ATM predictions with confidence ranking, live cashout countdown clock, 1-Click Emergency Bank Micro-Freeze, and automated PCR van dispatch.</em></p>
</div>

---

## 📌 Executive Summary

Cyber fraud incidents reported on the National Cybercrime Reporting Portal (**1930 Helpline**) often involve rapid fund laundering through multi-layered **money mule account networks** within minutes. By the time law enforcement issues freeze notices, syndicate operators physically withdraw the stolen money from ATMs.

**MuleShield AI** is an industry-grade, hybrid intelligence platform that bridges **Graph Neural Networks (GNNs)** and **Gradient Boosted Decision Trees (XGBoost)** to:
1. **Trace multi-hop fund dispersal** in real time from victim complaint origins in $<185\text{ ms}$.
2. **Detect fraud rings, fund-splitting, and velocity anomalies** using graph topology.
3. **Generate 64-dimensional structural risk embeddings** via **GraphSAGE** (capturing complex neighborhood relationships).
4. **Predict the exact ATM cashout locations (Top-3 ranked)** with **98.52% accuracy** and calculate a **real-time countdown timer** to interception in **under 25 milliseconds**.

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
  ├── 📍 Top-3 ATM Cashout Prediction (98.52% Top-3 Accuracy + Bayesian Spatial Reranking)
  ├── ⏱️ Time-to-Cashout Countdown (MAE: 0.02 min / 1.2 sec)
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

### 3. XGBoost ATM Interception Engine v2 (`engine/feature_builder.py`, `engine/xgb_model.py`)
- Integrates the 64-dim GNN representation with **16 spatial/temporal tabular features** into an **80-dimensional hybrid vector**:
  - `stolen_amount`, `hop_depth`, `transaction_velocity`, `hour_of_day`, `historical_hotspot_density`, `day_of_week`, `amount_after_split`
  - `dist_to_atm_1/2/3_km` — Vectorized Haversine distance to top-3 nearest ATMs.
  - `node_x, node_y, node_z` — 3D Cartesian Earth coordinates (eliminates angular bias in trees).
  - `bearing_to_atm_1_deg` — Compass bearing to nearest ATM (0°–360°).
  - `is_nearest_same_bank`, `nearest_same_bank_atm_dist` — Bank affiliation preference features.
- **Models:**
  - **`XGBClassifier`** (depth=7, 160 trees, lr=0.06): Outputs Top-3 ranked ATM candidates — **98.52% Top-3 Accuracy**.
  - **`XGBRegressor`**: Estimates countdown minutes — **0.02 min MAE** ($R^2 = 0.9998$).
- **Bayesian Spatial Prior Reranking**: Post-inference score = $P(\text{ATM}_i \mid \mathbf{x}) \times \exp(-d_i^2/2\sigma^2) \times (1 + 0.35r_i)$. Eliminates impossible far-away ATMs.

---

## 📊 Benchmark & Validation Results

MuleShield AI is validated against an extensive automated test suite (**235 / 235 tests passing**) on a Pan-India dataset spanning **65+ cities and 16 major banks**:

| Metric / Requirement | Target / Benchmark | Achieved Performance | Status |
|---|---|---|---|
| **Graph Construction Speed** | $< 500\text{ ms}$ | **$182.4\text{ ms}$** (20,468 nodes / 22,864 edges) | ✅ PASS |
| **GNN Node Classification F1** | $> 0.85$ | **$0.9996$** (Val F1: 1.000, Test F1: 0.9996) | ✅ PASS |
| **GNN AUC-ROC Score** | $> 0.90$ | **$1.000$** | ✅ PASS |
| **Per-Complaint Embedding Speed** | $< 2.0\text{ s}$ | **$0.39\text{ s}$** ($1.24\text{ s}$ for all 20,468 nodes) | ✅ PASS |
| **Top-3 ATM Prediction Accuracy** | $> 85.0\%$ | **$98.52\%$** Top-3 Accuracy — 80-dim v2 (Bayesian Spatial Reranking) | ✅ PASS |
| **Time-to-Cashout Regression MAE** | $< 5.0\text{ min}$ | **$0.02\text{ min}$** ($1.2\text{ seconds}$, $R^2 = 0.9998$) | ✅ PASS |
| **Single-Sample Inference Latency** | $< 200\text{ ms}$ | **$25.83\text{ ms}$** mean ($28.36\text{ ms}$ max) | ✅ PASS |
| **Automated Test Coverage** | $100\%$ | **235 / 235 Passed** (184 Unit + 51 API Tests) | ✅ PASS |

<div align="center">
  <h3>Machine Learning Performance Matrix & Validation</h3>
  <img src="docs/model_matrix_full.png" alt="MuleShield AI Python-Computed Model Performance Matrix" width="100%" />
  <p><em>Figure: Programmatic ML Evaluation Matrix generated via Python (scikit-learn & PyTorch) — GraphSAGE GNN Confusion Matrix (F1: 0.9993), ROC Curve (AUC: 1.000), XGBoost v2 Top-k Spatial Accuracy (Top-3: 98.52%), and Time Regressor Fit (R²: 0.9998).</em></p>
</div>

<br/>

<div align="center">
  <img src="docs/sih_performance_matrix_slide.png" alt="MuleShield AI Executive Performance Slide Card" width="100%" />
  <p><em>Figure: Executive SIH Evaluation Summary Card (SIH26184 | MHA / I4C).</em></p>
</div>

---

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

### 2. Start the FastAPI Backend
```bash
python -m uvicorn backend.main:app --port 8000 --reload
# API Documentation (Swagger UI): http://localhost:8000/docs
```

### 3. Start the React Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
# Command Center UI: http://localhost:5173
```

### 4. Run the Full Test Suite (235 Tests)
```bash
# Run AI engine unit tests (184 tests)
python -m pytest tests/ -v

# Run FastAPI backend tests (51 tests)
python -m pytest backend/tests/ -v
```

---

## 🗺️ Roadmap & Phase Completion

- [x] **Phase 1: Synthetic Dataset Generator** *(84/84 Tests Passing)*
  - 2,500 complaints, 22,864 multi-hop transactions, 1,000 ATMs across 65+ Indian cities.
  - Embedded multi-source fraud rings with balanced mule/clean node features.
- [x] **Phase 2a: Graph Intelligence & GraphSAGE Engine** *(47/47 Tests Passing)*
  - NetworkX directed graph builder with BFS traversal (<185ms).
  - 2-layer GraphSAGE classifier achieving 0.9996 F1 score.
  - Real-time sub-graph embedding extractor (<0.39s).
- [x] **Phase 2b: XGBoost ATM Prediction Engine v2** *(53/53 Tests Passing)*
  - 80-dimensional hybrid vector concatenation (64 GNN + 16 spatial tabular).
  - Top-3 ATM ranking (98.52% accuracy) and cashout countdown (0.02 min MAE).
  - Single inference latency: 25.8 ms.
- [x] **Phase 3: Real-Time FastAPI Backend** *(51/51 Tests Passing)*
  - REST endpoints for complaint ingestion, graph exploration, GNN embeddings, and ATM predictions.
  - WebSocket broadcaster (`/ws/feed`) for live event push.
- [x] **Phase 4: Tactical Command Dashboard** *(Live on port 5173)*
  - 4 interactive screens: Triage Queue, Tactical GIS Map, Forensic Graph, and 1-Click Interception.
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
