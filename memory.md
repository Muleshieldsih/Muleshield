# 🧠 MuleShield AI — Project Memory (`SIH26184`)

> **Living Document** — Updated as decisions are made, features are built, and context evolves.
> Always read this file before resuming work on this project.

---

## 🪪 Project Identity

| Field | Value |
|---|---|
| **Problem Statement ID** | SIH26184 |
| **Project Name** | MuleShield AI |
| **Tagline** | AI-Powered Spatio-Temporal Cashout Interdiction & Mule Graph Intelligence for Indian Law Enforcement |
| **Ministry / Org** | Ministry of Home Affairs (MHA) / Indian Cyber Crime Coordination Centre (I4C) |
| **Theme** | Blockchain & Cybersecurity |
| **Category** | Software |
| **Hackathon** | Smart India Hackathon 2026 (SIH 2026) |
| **Workspace Folder** | `d:\SIH 2026\SIHPROJECT2\` |
| **Stage** | 🔴 Active Development |

---

## 🎯 Core Problem Being Solved

When Indian citizens report financial cyber-fraud via Helpline **1930** or **NCRP**:
- Stolen funds traverse **3–5 mule account hops** within **10–20 minutes**
- Money reaches **ATMs / Micro-ATMs / CSP kiosks** for physical cashout within **30–60 minutes**
- Current freeze mechanisms are **reactive** — taking **2–4+ hours**
- National fund recovery rate is **under 8%**

**MuleShield AI closes this gap** by operating within the **"Golden Hour"** — predicting the exact ATM, time window, and interception probability BEFORE the cash is withdrawn.

---

## 🏛️ Key Architectural Decisions

> **Architecture: Hybrid GNN + XGBoost (Approach 2)** — upgraded from plain XGBoost on 2026-08-28.

| Decision | Choice | Reason |
|---|---|---|
| Graph Library | NetworkX | Multi-hop BFS/DFS, graph feature extraction, SIH-demo friendly |
| GNN Model | **GraphSAGE** (PyTorch Geometric) | Learns structural fraud-ring embeddings offline; industry-grade |
| Final Classifier | **XGBoost** | Takes GNN embeddings + tabular features → fast real-time inference |
| Synthetic Graph Gen | **gen-fraud-graph** (SantanderAI) | Realistic fraud ring graphs with embedded mule patterns |
| Reference GNN Code | **Mule-Hunt** (sohamvjadhav) | GNN pipeline for UPI mule detection — adapting for Indian context |
| Backend | FastAPI + WebSockets | Async real-time push, minimal boilerplate, Python-native |
| Frontend | React (Vite) + Tailwind CSS | Fast build, SIH-demo compatible, rich ecosystem |
| Maps | Leaflet.js / React-Leaflet | Offline-capable, OpenStreetMap tiles, no API key needed for demo |
| Graph Viz | **React Flow** (decided) | Interactive node-link diagram; better DX than Cytoscape.js |
| Data | Faker + Pandas + gen-fraud-graph | Realistic synthetic fraud graphs with Indian banking context |

---

## 📦 Key Modules (What Has Been Decided)

### AI Pipeline: `Transaction Graph → GraphSAGE → XGBoost → ATM Prediction`

1. **Synthetic Data Generator** — Faker + Pandas + gen-fraud-graph:
   - Victim complaint logs (1930-style)
   - Multi-hop transaction ledger (IMPS/UPI) with embedded fraud rings
   - Geocoded ATM/CSP database (Indian GPS coordinates)

2. **Graph Intelligence Engine** — NetworkX + PyTorch Geometric:
   - **Step A — NetworkX:** BFS/DFS traversal, velocity detection, fund-splitting, terminal node flagging
   - **Step B — GraphSAGE (offline):** Learns node embeddings (risk vectors) from the transaction graph structure — captures fraud RINGS that tabular models miss
   - Output: 64-dim embedding per account node, saved to `embeddings/` cache

3. **Hybrid AI Engine — XGBoost (real-time):**
   - Features: GNN embeddings (64d) + tabular features `[amount, hop_depth, velocity, branch_distance_to_atm, hour_of_day, hotspot_density]`
   - Outputs: Top-3 ATM candidates + confidence score + countdown minutes
   - Inference time: < 50ms per complaint

4. **FastAPI Backend** — REST + WebSocket:
   - `POST /api/v1/complaint/ingest`
   - `GET /api/v1/graph/{complaint_id}`
   - `GET /api/v1/embeddings/{complaint_id}` ← new: returns GNN embedding metadata
   - `GET /api/v1/predict/cashout/{complaint_id}`
   - `POST /api/v1/bank/micro-freeze`

5. **React Dashboard** — 4 screens:
   - Live 1930 Triage Feed
   - Tactical Police GIS Map
   - Interactive Forensic Money-Flow Graph (React Flow)
   - 1-Click Interception Panel

---

## 🗺️ Indian Banking Domain Glossary

| Term | Meaning |
|---|---|
| IMPS | Immediate Payment Service (real-time, 24×7) |
| UPI | Unified Payments Interface (most common fraud vector) |
| AEPS | Aadhaar-Enabled Payment System (Micro-ATM fraud) |
| CSP | Customer Service Point (rural banking correspondent) |
| IFSC | Indian Financial System Code (identifies a bank branch) |
| 1930 | National Cybercrime Helpline Number (India) |
| NCRP | National Cybercrime Reporting Portal |
| I4C | Indian Cyber Crime Coordination Centre |
| Mule Account | Bank account used to launder stolen funds (often rented/sold) |

---

## 🔑 Key Demo Numbers

| Metric | Value |
|---|---|
| Fraud detection window | "Golden Hour" = first 60 minutes |
| Mule graph traversal time | < 1 second |
| ATM prediction confidence | ~92.4% |
| Cashout countdown (demo) | ~35 minutes |
| Amount saved in demo | ₹50,000 |
| Current national recovery rate | < 8% (baseline we beat) |

---

## 📋 Open Decisions

- [x] ~~Decide: Cytoscape.js vs React Flow~~ → **React Flow chosen**
- [x] ~~ML Model choice~~ → **Hybrid GraphSAGE + XGBoost chosen**
- [ ] Finalize city scope for demo dataset (suggest: Delhi NCR + Mumbai)
- [ ] Decide: SMS alert mock (Twilio sandbox) vs static popup in UI
- [ ] Confirm if offline map tiles are needed for demo venue
- [ ] PPT template style: MHA official theme vs. modern dark SIH theme?
- [ ] GraphSAGE: use 2-layer or 3-layer neighbourhood aggregation?
- [ ] Embedding dimension: 64 or 128? (64 recommended for demo speed)

---

## 📅 Session Log

| Date | What Was Done |
|---|---|
| 2026-08-28 | **Project Initiated & Architecture Upgraded:** Defined 5-phase SIH roadmap. Upgraded architecture to Hybrid GraphSAGE (PyTorch Geometric) + XGBoost predictor. |
| 2026-08-28 | **Phase 1 Complete (84/84 tests):** Implemented `generate_data.py` (Pan-India national dataset generator with 65+ cities, 16 major banks, 9 fraud modalities). |
| 2026-08-28 | **Phase 2a Complete (47/47 tests):** Implemented `graph_engine.py` (NetworkX BFS & anomaly detection) and `gnn_model.py` / `train_gnn.py` (GraphSAGE 2-layer, 99.96% F1, 1.000 AUC). Implemented `embed.py` generating 20,468 64-dim embeddings in 1.2s. |
| 2026-08-28 | **Phase 2b Complete (53/53 tests):** Implemented `feature_builder.py` (O(1) indexed 72-dim hybrid vectors), `xgb_model.py` (MuleXGBPredictor), and `train_xgb.py` (91.48% Top-3 ATM accuracy, 0.08 min MAE, 18.5ms latency). |
| 2026-08-28 | **Pan-India Scale & GitHub Push:** Scaled dataset to 2,500 complaints, 22,864 multi-hop transactions, 1,000 ATMs across 65+ cities, and 20,468 account nodes. All 184 tests pass in 12.55s. Generated ML Architecture and Performance Matrix visual infographics, embedded in `README.md`, and pushed to `https://github.com/hotshot0104/SIH2026`. |

---

## 🎯 Next Session Starting Point: **Phase 3 — FastAPI Real-Time Backend**

1. Create `backend/` package (`backend/main.py`, `backend/routers/`, `backend/schemas/`).
2. Build REST Endpoints:
   - `POST /api/v1/complaint/ingest` — Ingest 1930 incident tickets.
   - `GET /api/v1/graph/{complaint_id}` — Return multi-hop graph nodes, edges, anomalies, and terminal accounts.
   - `GET /api/v1/predict/cashout/{complaint_id}` — Run GNN embedding + XGBoost inference (Top-3 ATM ranking + countdown timer).
   - `POST /api/v1/bank/micro-freeze` — Simulated NPCI / Banking freeze action.
3. Implement WebSocket live alert channel (`/ws/alerts`) for real-time countdown pushes and law enforcement dispatch.
4. Write `tests/test_phase3.py` and verify with pytest.

---

*Last updated: 2026-08-28 (Phase 1, 2a, 2b fully complete | 184/184 tests passed | Pushed to GitHub)*

