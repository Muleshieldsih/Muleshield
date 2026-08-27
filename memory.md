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
| 2026-08-28 | Project initiated. Architecture, tech stack, and 5-milestone roadmap defined. All planning docs created in `SIHPROJECT2`. |
| 2026-08-28 | **Architecture upgraded** from plain XGBoost → Hybrid GraphSAGE + XGBoost (Approach 2). All 5 planning docs updated. React Flow chosen for graph viz. Mule-Hunt and gen-fraud-graph repos identified as reference implementations. |

---

*Last updated: 2026-08-28*
