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
| **Stage** | 🟢 Feature-complete — Phase 5 (pitch) remaining |

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
- [x] ~~Finalize city scope~~ → **Pan-India, 65+ cities** (78 in node features)
- [ ] Decide: SMS alert mock (Twilio sandbox) vs static popup in UI
- [ ] Confirm if offline map tiles are needed for demo venue
- [ ] PPT template style: MHA official theme vs. modern dark SIH theme?
- [x] ~~GraphSAGE layers~~ → **2-layer** (F1 0.9996)
- [x] ~~Embedding dimension~~ → **64-dim**

---

## 📅 Session Log

| Date | What Was Done |
|---|---|
| 2026-08-28 | **Project Initiated & Architecture Upgraded:** Defined 5-phase SIH roadmap. Upgraded architecture to Hybrid GraphSAGE (PyTorch Geometric) + XGBoost predictor. |
| 2026-08-28 | **Phase 1 Complete (84/84 tests):** Implemented `generate_data.py` (Pan-India national dataset generator with 65+ cities, 16 major banks, 9 fraud modalities). |
| 2026-08-28 | **Phase 2a Complete (47/47 tests):** Implemented `graph_engine.py` (NetworkX BFS & anomaly detection) and `gnn_model.py` / `train_gnn.py` (GraphSAGE 2-layer, 99.96% F1, 1.000 AUC). Implemented `embed.py` generating 20,468 64-dim embeddings in 1.2s. |
| 2026-08-28 | **Phase 2b Complete (53/53 tests):** Implemented `feature_builder.py` (O(1) indexed 72-dim hybrid vectors), `xgb_model.py` (MuleXGBPredictor), and `train_xgb.py` (91.48% Top-3 ATM accuracy, 0.08 min MAE, 18.5ms latency). |
| 2026-08-28 | **Pan-India Scale & GitHub Push:** Scaled dataset to 2,500 complaints, 22,864 multi-hop transactions, 1,000 ATMs across 65+ cities, and 20,468 account nodes. All 184 tests pass in 12.55s. Generated ML Architecture and Performance Matrix visual infographics, embedded in `README.md`, and pushed to `https://github.com/hotshot0104/SIH2026`. |
| 2026-08-28 | **Phase 3 + 4 Complete (51/51 API tests):** FastAPI REST + WebSocket backend and the 4-screen React command console. |
| 2026-08-28 | **Hardening Pass (235/235 tests):** Fixed the demo-critical defects — live-ingested complaints now build a real mule chain and run the full pipeline (were 404/422); micro-freeze POST was failing schema validation on every click and faking success client-side; velocity/fund-split panels were hardcoded and now read real `graph_engine` detections; node risk switched from embedding mean (~0) to the trained GraphSAGE head `sigmoid(Wh+b)`; the two bank-affinity features were dead constants, repaired and XGBoost retrained (98.46% Top-3); ATM addresses named random unrelated cities; removed `react-leaflet` (broke `npm install` on React 19); self-hosted fonts + basemap fallback for offline venues. |
| 2026-08-29 | **Leakage Removal & Honest Re-baselining (237/237 tests):** Peer feedback that the model was "too accurate to be true" was correct. `total_received > 0` reproduced the mule label with F1=1.0000 - the label was a copy of a feature, and the 0.9996 GNN F1 measured nothing. The ATM label was `argmin(distance)` while distance was feature #67, and the countdown target was a closed-form line in two inputs (R2=0.9998). Rebuilt the generator: mule status fixed before any transaction exists, 29,998 legitimate transactions so classes overlap, cashout ATM sampled from a choice model, complaint-level splits, time-separated priors, 2% label noise. Honest results: GNN F1 0.9386 (vs RF 0.9031), ATM Top-3 0.5658 (vs distance 0.5526), countdown MAE 6.35 min (vs mean 9.41). Added `scripts/evaluate_baselines.py`. |
| 2026-08-29 | **Re-aimed at the problem statement (237/237 tests):** Audited SIH26184 and found we were optimising the wrong task - the PS asks to "Forecast Likely Cash Withdrawal Locations", and mule detection appears nowhere in the ministry's wording. Added a SEARCH ZONE as the primary output: 86.8% containment vs 76.5% for a nearest-3 centroid at equal search cost, narrowing 1,000 ATMs to a median of 7. Fixed correctness bugs: crew_prior expanded 3 hops in training but 2 at serving; baselines were scored on a different split from the GNN (headline lift was not apples-to-apples, now +0.032 F1); removed dead "Bayesian Spatial Reranking" that was documented but never invoked; scaler and pos_weight were fitted over all nodes pre-split; threshold hardcoded at 0.5. Fed the countdown the account-behaviour features it was blind to (MAE 6.38 -> 6.14, R2 0.52 -> 0.55) and added a q05-q95 band at 85.7% coverage. Restated every published target in prd.md/product.md to one we actually meet. |

---

## 🎯 Next Session Starting Point: **Phase 5 — Demo Script, PPT & Video**

Phases 1–4 complete, plus a hardening pass and a full leakage-removal rebuild.
237/237 tests pass.

1. **Rehearse the live demo path**: Ingest Complaint → Forensic Graph (real
   anomalies) → Tactical Map (Top-3 ATMs) → Interception (freeze + dispatch).
2. Build the 7-slide deck. **Lead with the baseline table, not the accuracy.**
3. Record the 3-minute demo video.
4. From a clean clone: `python -m pytest tests/ backend/tests/ -q` and
   `cd frontend && npm run build`.

### How to answer "is your model too accurate?"

It was, and we fixed it. Say this plainly:

- The first version reported 0.9996 GNN F1. `total_received > 0` scored
  **F1 = 1.0000** on the same data — the label was a copy of a feature.
- We rebuilt the generator so mule status is assigned **before any transaction
  exists**, added ~30k legitimate transactions so the classes overlap, and made
  the cashout ATM a **sampled choice** rather than `argmin(distance)`.
- `tests/test_phase1.py::test_label_is_not_a_copy_of_a_feature` fails the build
  if any single feature reproduces the label above F1 0.95.

Run `python scripts/evaluate_baselines.py` live if challenged.

### Lead the pitch with THIS, not with F1

SIH26184 asks to *"Forecast Likely Cash **Withdrawal Locations** in Advance."*
Location forecasting is the deliverable; mule detection is internal machinery the
ministry never asked for. Order the deck accordingly.

| Task | Ours | Best naive baseline |
|---|---|---|
| **Withdrawal-zone containment** | **86.8%** | 76.5% (nearest-3 centroid) |
| Search cost | **7 of 1,000 ATMs**, 9.4 km | — |
| Countdown MAE | **6.14 min** | 9.41 min (predict the mean) |
| Mule detection F1 | 0.9386 | 0.9031 (random forest, no graph) |
| Top-3 exact ATM | 0.5658 | 0.5596 (nearest 3 by distance) |

If asked why exact-ATM Top-3 is only marginally above the distance baseline: because
distance genuinely dominates which machine is used — the Bayes-optimal ranker on this
data reaches only ~0.58. That is exactly why the zone is the committed deliverable.
Volunteering this is stronger than being caught by it.

### Known limitations to state before being asked
- All data is synthetic; archetypes are informed by published mule typologies,
  not fitted to real bank data.
- Micro-freeze and SMS are simulated; WhatsApp dispatch is real.
- A live-ingested complaint gets its chain synthesised from real graph accounts
  in the victim's city — embeddings, ATM directory and inference are genuine,
  only the bank/NPCI feed is simulated.
- ATM Top-3 beats distance by ~1.3 points. Distance genuinely dominates where a
  mule withdraws; the Bayes-optimal ranker only reaches ~0.58. The operational
  value is narrowing 1,000 ATMs to 3 in under 15 ms.

---

*Last updated: 2026-08-29 (Phases 1-4 + hardening + leakage removal + PS re-aim | 237/237 tests passed)*

