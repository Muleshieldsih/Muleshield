# 📄 MuleShield AI — Product Requirements Document (PRD)

> **Document Version:** `3.0.0`  
> **Problem Statement ID:** `SIH26184`  
> **Sponsoring Body:** Ministry of Home Affairs (MHA) / Indian Cybercrime Coordination Centre (I4C)  
> **Status:** Production-Ready / Final Build  
> **Last Updated:** August 28, 2026  

---

## 1. Executive Summary & Product Vision

### 1.1 Executive Summary
**MuleShield AI** is an AI-powered, real-time cybercrime interdiction and mule account tracking platform engineered for Indian Law Enforcement Agencies (State Cyber Cells, I4C) and Banking Institutions. It compresses the time between initial fraud reporting and physical cashout interception from **hours to less than 1 minute**, turning the reactive 1930 Cyber Helpline workflow into a proactive physical interdiction system.

### 1.2 The "Golden Hour" Vision
When a citizen reports financial cyber fraud (UPI scams, Digital Arrest, Job Scams) to the 1930 helpline, the stolen funds undergo multi-hop layering through intermediate mule accounts before a human mule withdraws physical currency from an ATM or CSP. 

MuleShield AI targets the **Golden Hour (0–60 minutes)** by:
1. Reconstructing the multi-hop transaction chain in $<185\text{ ms}$.
2. Identifying terminal mule accounts using 64-dimensional GraphSAGE GNN embeddings.
3. Predicting the **Top-3 probable ATM destinations** and **cashout countdown timer** as a ranked search zone.
4. Enabling **1-Click Emergency Micro-Freezes** and **Automated Police PCR Van GPS Dispatch**.

---

## 2. Target Personas & User Roles

| Persona | Primary Role | Key Pain Point | MuleShield AI Solution |
|---|---|---|---|
| **Cyber Police Officer (IO)** | First responder at State Cyber Cell / Police Station | Lacks physical location of suspect during live crime | Real-time map with target ATM pin, confidence rank, and PCR van route |
| **Bank Fraud Risk Officer** | Manages bank fraud queue & debit blocks | Inter-bank freeze notices take 2–6 hours of paperwork | 1-Click Micro-Freeze API triggering instant debit hold at switch |
| **State Cyber Cell Supervisor** | Allocates field resources & oversees active cases | No macro visibility of multi-bank mule syndicates | Live triage dashboard with active fraud rings across state |
| **I4C National Analyst** | Macro intelligence & syndicate dismantling | Cannot detect cross-bank layering rings | GNN topological risk export & national hotspot heatmaps |

---

## 3. Core Product Architecture & Tech Stack

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               MULESHIELD AI ARCHITECTURE                                │
├───────────────────────────┬─────────────────────────────────────────────────────────────┤
│ 1. Ingestion Layer        │ 1930 Helpline Intake (REST / WebSockets)                    │
│ 2. Graph Intelligence     │ NetworkX Multi-Hop BFS Traversal (<185ms)                   │
│ 3. Deep Graph Learning    │ PyTorch Geometric GraphSAGE (64-dim Inductive Embeddings)   │
│ 4. Spatio-Temporal Model  │ Conditional-Logit ATM Choice + XGBoost Countdown            │
│ 5. Backend Engine         │ FastAPI + Uvicorn ASGI + WebSocket Feed                     │
│ 6. Tactical Command UI    │ React 19 + TailwindCSS + React Flow + Leaflet.js            │
└───────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 4. Functional Requirements (FR)

### FR-1: Real-Time 1930 Complaint Ingestion & Triage
- **FR-1.1:** System shall accept complaint payloads via `POST /api/v1/complaint/ingest` containing victim details, bank, stolen amount, fraud type, city, state, and timestamp.
- **FR-1.2:** Ingested complaints must be assigned a unique ticket identifier (`TKT-XXXXXXXX`).
- **FR-1.3:** Incoming complaints must broadcast immediately via WebSocket (`/ws/feed`) to all connected command center dashboards without requiring page refresh.
- **FR-1.4:** Complaints must be prioritized into three visual categories:
  - 🔴 **Critical:** $< 15\text{ mins}$ elapsed since incident.
  - 🟡 **Warning:** $15–30\text{ mins}$ elapsed.
  - 🟢 **Monitoring:** $> 30\text{ mins}$ elapsed.

### FR-2: Multi-Hop Transaction Graph Reconstruction
- **FR-2.1:** System shall parse the transaction ledger for a given complaint ID and reconstruct a Directed Acyclic Graph (DAG) in $<200\text{ ms}$.
- **FR-2.2:** Graph nodes must categorize into:
  - **Victim Node** (Hop 0, Blue)
  - **Layer-1 & Layer-2 Mules** (Hop 1–3, Orange)
  - **Terminal Mule** (Hop 4+, Red / Blinking)
- **FR-2.3:** Edges must specify transaction amount, timestamp, and time interval between hops.
- **FR-2.4:** API endpoint `GET /api/v1/graph/{complaint_id}` shall output React Flow node-link JSON format.

### FR-3: Deep Graph Neural Network (GNN) Risk Scoring
- **FR-3.1:** Inductive GraphSAGE GNN must generate a 64-dimensional dense risk representation vector ($h_v \in \mathbb{R}^{64}$) for all accounts.
- **FR-3.2:** System shall expose `GET /api/v1/embeddings/{complaint_id}` returning top-N highest-risk mule accounts ranked by GNN L2-norm centrality.

### FR-4: Spatio-Temporal ATM Cashout Prediction
- **FR-4.1:** System shall construct an 80-dimensional hybrid feature vector combining:
  - 64-dim GraphSAGE embedding
  - 16 spatial/temporal tabular features (3D Cartesian coordinates $X, Y, Z$, compass bearing, multi-ATM distance triplets, bank affinity).
- **FR-4.2:** Model must output a **search zone** — centroid, radius, and the number
  of ATMs inside it — as the primary location forecast, plus **Top-3 ranked ATM
  candidates** with confidence percentages as the tactical drill-down.
- **FR-4.3:** Model must output a **time-to-cashout countdown** in minutes, with a
  q05–q95 prediction band.
- **FR-4.4:** ATM ranking must use a **conditional-logit choice model** over the $K$
  reachable candidates, with utility linear in log space:
  $$P(\text{ATM}_i \mid \text{candidates}) = \text{softmax}_i\left(\mathbf{w} \cdot \log \mathbf{f}_i\right)$$
  where $\log \mathbf{f}_i$ carries distance decay, surveillance risk, bank affinity and
  the crew's prior cashout history. Learned weights are reported so an evaluator can
  check them against expectation.

  > A "Bayesian Gaussian spatial prior reranking" step was specified here in earlier
  > issues. It was never reachable in code and has been removed rather than left as a
  > documented feature that does not run.
- **FR-4.5:** API endpoint `GET /api/v1/predict/cashout/{complaint_id}` must respond in $<200\text{ ms}$ (typical $<30\text{ ms}$).

### FR-5: 1-Click Emergency Bank Micro-Freeze
- **FR-5.1:** System shall provide a secure endpoint `POST /api/v1/bank/micro-freeze` accepting `account_id`, `complaint_id`, and `officer_id`.
- **FR-5.2:** Endpoint must simulate an instantaneous card/IMPS debit block and return a unique freeze reference (`FRZ-XXXXXXXX`) with an audited timestamp.
- **FR-5.3:** Broadcast a `FREEZE_EXECUTED` event across WebSockets to update all active dashboard sessions.

### FR-6: Tactical Police GIS Dispatch
- **FR-6.1:** System shall render target ATM coordinates on a tactical map with a glowing perimeter circle.
- **FR-6.2:** Display live floating countdown timer above the target pin.
- **FR-6.3:** Calculate route and ETA for the nearest police patrol unit to the predicted ATM.

---

## 5. Non-Functional Requirements (NFR)

### NFR-1: Latency & Performance SLA
- **Per-Complaint Graph Build:** $< 500\text{ ms}$ (Achieved: **$\approx 2\text{ ms}$**).
- **Full National Graph Build (startup):** $< 1200\text{ ms}$ (Achieved: **$\approx 670\text{ ms}$**).
- **End-to-End Prediction:** $< 200\text{ ms}$ (Achieved: **$\approx 15\text{ ms}$**).
- **Test Suite Pass Rate:** $100\%$ automated passing tests (237/237).

### NFR-2: Accuracy & Model Precision SLA

Every target below is stated **with the naive baseline it must beat**. A score
without its baseline says nothing about a model.

**Primary — withdrawal-location forecast (the problem statement's ask):**
- **Search-zone containment:** $> 80\%$ — the withdrawal falls inside the predicted
  zone (Achieved: **$86.8\%$**, vs $76.5\%$ for a nearest-3 centroid and $72.1\%$ for
  centring on the mule, all at equal search cost).
- **Search cost:** median zone of $9.4\text{ km}$ radius covering **7 of 1,000 ATMs**.

**Supporting:**
- **Top-3 ATM ranking:** $0.5658$ (vs $0.5596$ distance-only). Distance genuinely
  dominates the exact-machine choice — the Bayes-optimal ranker on this data only
  reaches $\approx 0.58$ — which is precisely why the zone, not the machine, is the
  committed deliverable.
- **Cashout countdown (MAE):** $< 8.0\text{ min}$ (Achieved: **$6.14\text{ min}$**, vs a
  $9.41\text{ min}$ mean-prediction baseline; $R^2 = 0.55$ against a noise-imposed
  ceiling of $0.72$). Reported with a q05–q95 band at $85.7\%$ empirical coverage.
- **GNN mule detection F1:** $> 0.85$ (Achieved: **$0.9386$**, vs $0.9031$ for a random
  forest on identical features, scored on the same held-out nodes).

> **Revision note.** Earlier issues of this PRD reported $0.9996$ GNN F1, $98.52\%$
> Top-3 ATM accuracy and $0.02\text{ min}$ MAE as achieved. Those figures were
> artefacts of label leakage: the mule label was a copy of the `total_received`
> feature, the ATM label was `argmin(distance)` while distance was an input, and
> the countdown target was a closed-form line in two of its own inputs. The
> dataset and the evaluation were rebuilt; the numbers above are what the system
> measures now. See `README.md` → *Honest Evaluation*.

### NFR-3: Security & Statutory Privacy Compliance
- **Digital Personal Data Protection (DPDP) Act, 2023:**
  - Zero scraped PII. All training data 100% synthetically generated via agent-based graph models.
  - Masked synthetic account tokens (`ACC-XXXXXXXX`).
- **Information Technology (IT) Act, 2000 (Section 43 & 66):**
  - Zero unauthorized network access or scraping of citizen/banking portals.
- **Offline Venue Resilience:**
  - Standalone in-memory operation without mandatory external internet dependencies during live hackathon demos.

---

## 6. API Specification Summary

| Method | Endpoint | Description | SLA |
|---|---|---|---|
| `POST` | `/api/v1/complaint/ingest` | Ingests new 1930 complaint & broadcasts via WS | $< 50\text{ ms}$ |
| `GET` | `/api/v1/complaint/list` | Returns all active complaints (newest first) | $< 30\text{ ms}$ |
| `GET` | `/api/v1/complaint/{id}` | Fetches individual complaint record | $< 20\text{ ms}$ |
| `GET` | `/api/v1/graph/{complaint_id}` | Returns React Flow money-flow DAG | $< 185\text{ ms}$ |
| `GET` | `/api/v1/embeddings/{complaint_id}`| Returns top-N mule nodes with GNN risk scores | $< 40\text{ ms}$ |
| `GET` | `/api/v1/predict/cashout/{complaint_id}` | XGBoost Top-3 ATM predictions + countdown | $< 30\text{ ms}$ |
| `POST` | `/api/v1/bank/micro-freeze` | Executes 1-click debit freeze on target account | $< 20\text{ ms}$ |
| `WS` | `/ws/feed` | Real-time event stream (`NEW_COMPLAINT`, etc.) | Instant |
| `GET` | `/health` | Service uptime and loaded assets health check | $< 10\text{ ms}$ |

---

## 7. Product Release Roadmap

```
Phase 1 (Complete)  ──► Phase 2 (Complete)  ──► Phase 3 (Complete)  ──► Phase 4 (Next)     ──► Phase 5 (Final)
Data Generator          AI Engine (GNN+XGB)     FastAPI Backend         React Command Center    PPT & Live Pitch
100% Synthetic Pan-India 86.8% Zone Containment  REST + WebSockets       4 Tactical Screens      Jury Presentation
```

---

*Authored by Team MuleShield AI (Problem Statement: SIH26184 | Smart India Hackathon 2026)*
