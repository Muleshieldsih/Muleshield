# 📦 MuleShield AI — Product Document (`SIH26184`)

> Describes **what** MuleShield AI is, **who** it serves, and **how** it works from a product perspective.

---

## 🧩 Product Overview

**MuleShield AI** is a real-time, AI-powered cybercrime interdiction platform built for Indian Law Enforcement and Banking Institutions. It transforms the reactive 1930 cybercrime helpline workflow into a **proactive, intelligence-driven interception system** by predicting *where* and *when* stolen money will be cashed out — before it happens.

**Core AI Pipeline:** `Transaction Graph → NetworkX (graph traversal) → GraphSAGE (fraud-ring embeddings) → XGBoost (real-time ATM prediction)`

This **Hybrid GNN + XGBoost architecture** is the same approach used by NVIDIA and AWS for production financial fraud systems — GraphSAGE detects coordinated mule rings that pure tabular models miss; XGBoost delivers sub-50ms real-time predictions.

---

## 👥 Primary Users

| User | Role | Primary Need |
|---|---|---|
| **Cyber Police Officer** | Reviews incoming 1930 complaints, dispatches units | Real-time ATM target with GPS + countdown |
| **Bank Fraud Officer** | Monitors flagged accounts, initiates freezes | One-click micro-freeze on mule accounts |
| **State Cyber Cell Supervisor** | Oversees active cases, resource allocation | Live dashboard overview of all active threats |
| **I4C Analyst** | National-level pattern analysis | Mule graph data export, hotspot heatmaps |

---

## 🔑 Core Value Propositions

### 1. 🕐 Golden Hour Intelligence
MuleShield operates within the 60-minute window between fraud complaint and physical cashout. It compresses the police response from **hours → minutes**.

### 2. 🕸️ Mule Graph Reconstruction
Instantly reconstructs the full transaction chain from victim account to every downstream mule in a directed graph — in **< 1 second**, no matter how many hops.

### 3. 📍 Spatio-Temporal ATM Prediction
Predicts the **Top-3 most probable ATMs** where cash will be withdrawn, with:
- GPS coordinates (Lat/Long)
- Confidence score (e.g., 92.4%)
- Estimated time remaining (e.g., 35 minutes)

### 4. 🔒 Pre-Emptive Micro-Freeze
Enables one-click automated webhook to **lock ATM withdrawal limits** on target mule accounts *before* the mule reaches the ATM.

### 5. 🚨 Automated Police Dispatch
Auto-generates GPS navigation alerts via SMS/WhatsApp to the nearest patrol unit with:
- Target ATM address
- Suspect account details
- Countdown remaining

---

## 🖥️ Product Screens

### Screen 1 — Live 1930 Triage Feed
```
┌─────────────────────────────────────────────────┐
│ 🚨 LIVE COMPLAINT FEED                          │
├─────────────────────────────────────────────────┤
│ [🔴 CRITICAL] TKT-001 | Rohan Sharma | ₹1,20,000│
│ UPI Fraud | Delhi | 8 min ago                   │
├─────────────────────────────────────────────────┤
│ [🟡 WARNING]  TKT-002 | Priya Nair  | ₹45,000   │
│ Job Scam  | Mumbai | 22 min ago                 │
├─────────────────────────────────────────────────┤
│ [🟢 MONITOR]  TKT-003 | Amit Yadav  | ₹80,000   │
│ Digital Arrest | Pune | 41 min ago              │
└─────────────────────────────────────────────────┘
```

### Screen 2 — Tactical GIS Police Map
```
┌─────────────────────────────────────────────────┐
│ 🗺️ TACTICAL MAP — Delhi NCR                     │
│                                                 │
│   [🔴 ATM-SBI-004] ⏱️ 27:43 remaining           │
│   Confidence: 92.4%                             │
│   📍 28.6139° N, 77.2090° E                     │
│                                                 │
│   [🟡 ATM-PNB-011] ⏱️ 34:12 remaining           │
│   Confidence: 71.8%                             │
│   📍 28.6279° N, 77.2192° E                     │
│                                                 │
│   [🚔 Nearest Station: Connaught Place PS]      │
│   Route: 2.3 km | ETA: 6 min                    │
└─────────────────────────────────────────────────┘
```

### Screen 3 — Interactive Forensic Money-Flow Graph
```
[VICTIM: Rohan Sharma]
        │ ₹1,20,000 @ 02:14:03
        ▼
[L1: HDFC-XXXXXX4821] ─────► [L1: SBI-XXXXXX9932]
        │ ₹60,000 @ 02:14:47         │ ₹60,000 @ 02:14:47
        ▼                            ▼
[L2: Kotak-XXXXXX2211]      [L2: Axis-XXXXXX7743]
        │ ₹29,500 @ 02:15:12         │ ₹29,500 @ 02:15:19
        ▼                            ▼
[⚡ TERMINAL: PNB-XXXXXX0041] [⚡ TERMINAL: UCO-XXXXXX8812]
   GNN Risk Score: 0.94 🔴       GNN Risk Score: 0.87 🟠
   ← ATM Cashout Predicted →
```
- **GNN Risk Score overlay** on each node (0.0–1.0) — from GraphSAGE embeddings
- High-risk nodes (>0.8) glow red; medium-risk (0.5–0.8) glow orange
- Click any node → side panel: Account No, Bank, IFSC, Amount, GNN embedding visualisation

### Screen 4 — 1-Click Interception Panel
```
┌─────────────────────────────────────────────────┐
│ ⚡ INTERCEPTION CONTROL                          │
│                                                 │
│ Top ATM Prediction:                             │
│ 🔴 SBI ATM — Connaught Place, Delhi  [92.4%]   │
│ ████████████████████░░░░                        │
│                                                 │
│ ┌─────────────────────┐  ┌────────────────────┐ │
│ │ 🔒 EMERGENCY FREEZE  │  │ 🚨 DISPATCH PATROL │ │
│ │ Card: PNB-XXXX0041  │  │ Unit: CP-03        │ │
│ └─────────────────────┘  └────────────────────┘ │
│                                                 │
│ ✅ FROZEN at 02:18:44 — PNB-XXXX0041           │
└─────────────────────────────────────────────────┘
```

---

## 🔄 Core User Journey

```
1. Citizen dials 1930 → Complaint registered
         │
         ▼
2. MuleShield ingests complaint (POST /api/v1/complaint/ingest)
         │
         ▼
3. NetworkX builds mule transaction tree in <1s
         │
         ▼
4. [NEW] GraphSAGE generates 64-dim risk embeddings per mule node
         │   (captures fraud RING structure, not just individual features)
         ▼
5. XGBoost uses embeddings + tabular features → Top-3 ATMs + countdown
         │
         ▼
6. Dashboard updates: Map blinking, GNN risk scores on graph, countdown starts
         │
         ▼
7. Officer clicks "Emergency Freeze" → mule card locked
         │
         ▼
8. Officer clicks "Dispatch Patrol" → GPS alert sent to nearest unit
         │
         ▼
9. Physical interception at ATM OR cashout blocked by freeze
         │
         ▼
10. ₹ Recovered / Crime prevented ✅
```

---

## 📊 Success Metrics

| Metric | Current Baseline | MuleShield Target |
|---|---|---|
| Fund recovery rate | < 8% | > 40% (demo scenario) |
| Police response time | 2–4 hours | < 15 minutes |
| Mule graph build time | Manual, hours | < 1 second (automated) |
| **GNN mule detection F1** | N/A | **> 0.88** (GraphSAGE on synthetic) |
| ATM prediction accuracy | N/A (none exists) | > 85% Top-1, > 95% Top-3 |
| Account freeze time | 2–4 hours (inter-bank) | < 30 seconds (micro-freeze) |
| End-to-end inference | N/A | < 200ms (embed → XGBoost) |

---

## 🌐 Scalability & Deployment Notes

- **Phase 1 (Demo/SIH):** Synthetic data, single-city (Delhi NCR), local deployment
- **Phase 2 (Pilot):** Integration with state Cyber Cell, 1 actual NPCI data feed
- **Phase 3 (National):** Multi-state deployment, NPCI switch integration, I4C dashboard
- **Cloud:** Deployable on NIC Cloud / MeitY-approved infrastructure (not AWS/GCP for govt)
- **Offline Mode:** Leaflet.js allows offline map tiles for demo venues without internet

---

*Last updated: 2026-08-28*
