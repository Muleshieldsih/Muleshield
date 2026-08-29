# 📋 MuleShield AI — Requirements Document (`SIH26184`)

> Defines all **functional**, **non-functional**, and **data** requirements for the SIH 2026 prototype.

---

## 1. Functional Requirements

### FR-01 — Data Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-01.1 | System shall ingest a new 1930 complaint via `POST /api/v1/complaint/ingest` with fields: victim name, bank, account number, fraud type, stolen amount, and timestamp | 🔴 Must Have |
| FR-01.2 | System shall validate all incoming complaint fields and reject malformed requests with a 422 response | 🟡 Should Have |
| FR-01.3 | System shall assign a unique `complaint_id` (UUID) to every ingested complaint | 🔴 Must Have |
| FR-01.4 | System shall broadcast new complaints to all connected WebSocket clients on `/ws/feed` within 500ms of ingestion | 🔴 Must Have |
| FR-01.5 | System shall load the synthetic transaction ledger and ATM directory from CSV on startup | 🔴 Must Have |

---

### FR-02 — Graph Intelligence Engine

| ID | Requirement | Priority |
|---|---|---|
| FR-02.1 | System shall build a directed transaction graph from victim account to all reachable mule nodes via BFS/DFS traversal | 🔴 Must Have |
| FR-02.2 | System shall support multi-hop traversal up to depth 5 (Layer-1 through Layer-5 mules) | 🔴 Must Have |
| FR-02.3 | System shall detect fund-splitting: a single source account disbursing to 3+ destinations within 2 minutes | 🟡 Should Have |
| FR-02.4 | System shall detect velocity anomalies: accounts with >2 outgoing transactions within a 5-minute window | 🟡 Should Have |
| FR-02.5 | System shall identify terminal nodes: leaf nodes with no further outgoing transactions (cashout candidates) | 🔴 Must Have |
| FR-02.6 | System shall return graph data in JSON node-link format via `GET /api/v1/graph/{complaint_id}` | 🔴 Must Have |
| FR-02.7 | Graph traversal shall complete within 1 second for datasets of up to 500 nodes | 🔴 Must Have |

---

### FR-02B — GraphSAGE Embedding Engine (**NEW — Hybrid GNN Architecture**)

| ID | Requirement | Priority |
|---|---|---|
| FR-02B.1 | System shall use a **2-layer GraphSAGE** model (PyTorch Geometric) trained offline on the synthetic transaction graph | 🔴 Must Have |
| FR-02B.2 | GraphSAGE shall produce a **64-dimensional embedding vector** per account node representing its structural fraud-ring risk | 🔴 Must Have |
| FR-02B.3 | GNN training shall use node features: `[lat, long, total_received, total_sent, txn_count_24h, avg_txn_amount, is_mule_label, hop_depth, bank_encoded, city_encoded, district_encoded]` | 🔴 Must Have |
| FR-02B.4 | GraphSAGE shall be trained as a binary node classifier (mule=1 / clean=0) with F1-score target ≥ 0.85 on the synthetic dataset | 🔴 Must Have |
| FR-02B.5 | Pre-computed node embeddings shall be cached in `embeddings/node_embeddings.pkl` and loaded at backend startup | 🔴 Must Have |
| FR-02B.6 | For a new complaint, the system shall generate embeddings for newly added nodes in < 2 seconds via `embed.py` | 🔴 Must Have |
| FR-02B.7 | GNN risk scores (0.0–1.0) per mule node shall be returned via `GET /api/v1/embeddings/{complaint_id}` | 🟡 Should Have |
| FR-02B.8 | Frontend graph (Screen 3) shall display GNN risk score as a colour overlay on each node | 🟡 Should Have |

---

### FR-03 — Hybrid AI Prediction (GraphSAGE + XGBoost)

| ID | Requirement | Priority |
|---|---|---|
| FR-03.1 | System shall predict a **search zone** (centroid, radius, ATMs inside) as the primary withdrawal-location forecast, and the **Top-3 most probable ATMs** inside it as the tactical drill-down | 🔴 Must Have |
| FR-03.2 | Each ATM prediction shall include: ATM ID, bank name, GPS coordinates (lat/long), address, city | 🔴 Must Have |
| FR-03.3 | System shall compute an interception confidence score (0.0–1.0) per predicted ATM | 🔴 Must Have |
| FR-03.4 | System shall compute estimated time-to-cashout in minutes (regression output) | 🔴 Must Have |
| FR-03.5 | The location forecast shall use an **80-dimensional hybrid feature vector** (**64 GNN embedding dims** from GraphSAGE + **16 spatial/temporal tabular dims**), reduced for ranking to a **24-column** row per ATM candidate (12 case-context + 12 candidate features). | 🔴 Must Have |
| FR-03.6 | Prediction results shall be returned via `GET /api/v1/predict/cashout/{complaint_id}` | 🔴 Must Have |
| FR-03.7 | XGBoost model inference time (after embeddings are loaded) shall not exceed 50ms per complaint | 🔴 Must Have |
| FR-03.8 | End-to-end inference time (embedding lookup + XGBoost predict) shall not exceed 200ms | 🔴 Must Have |
| FR-03.9 | Training shall use SMOTE-ENN oversampling + `scale_pos_weight` to handle class imbalance | 🟡 Should Have |

---

### FR-04 — Micro-Freeze Simulation

| ID | Requirement | Priority |
|---|---|---|
| FR-04.1 | System shall expose `POST /api/v1/bank/micro-freeze` accepting: account number, bank name, reason, officer ID | 🔴 Must Have |
| FR-04.2 | Endpoint shall return: `{ status: "FROZEN", account: "...", bank: "...", frozen_at: "<ISO timestamp>", officer_id: "..." }` | 🔴 Must Have |
| FR-04.3 | Freeze operation shall complete and respond within 500ms (simulated) | 🔴 Must Have |
| FR-04.4 | System shall log all freeze events with timestamp and officer ID | 🟡 Should Have |
| FR-04.5 | System shall prevent duplicate freeze requests for the same account within 30 seconds | 🟡 Should Have |

---

### FR-05 — Frontend Dashboard

| ID | Requirement | Priority |
|---|---|---|
| FR-05.1 | Dashboard shall display a live complaint feed (Screen 1) connected via WebSocket, updating without page refresh | 🔴 Must Have |
| FR-05.2 | Each complaint card shall show: Ticket ID, Victim Name, Amount (₹), Fraud Type, Time Elapsed | 🔴 Must Have |
| FR-05.3 | Complaints shall be color-coded: 🔴 <15 min, 🟡 15–30 min, 🟢 >30 min | 🟡 Should Have |
| FR-05.4 | Dashboard shall display a Leaflet.js map (Screen 2) with blinking markers at predicted ATM locations | 🔴 Must Have |
| FR-05.5 | Each ATM marker shall display a live countdown timer (minutes:seconds) | 🔴 Must Have |
| FR-05.6 | Map shall show route lines from nearest police station to target ATM | 🟡 Should Have |
| FR-05.7 | Dashboard shall render an interactive money-flow graph (Screen 3) with clickable nodes | 🔴 Must Have |
| FR-05.8 | Clicking a node in Screen 3 shall display: Account No, Bank, IFSC, Amount, Timestamp in a side panel | 🟡 Should Have |
| FR-05.9 | Interception panel (Screen 4) shall show a "Emergency Card Freeze" button that calls the freeze API and displays confirmation | 🔴 Must Have |
| FR-05.10 | Interception panel shall show a "Dispatch Patrol Unit" button that generates a simulated WhatsApp/SMS alert popup | 🔴 Must Have |

---

## 2. Non-Functional Requirements

### NFR-01 — Performance

| ID | Requirement | Target |
|---|---|---|
| NFR-01.1 | End-to-end complaint ingestion to first ATM prediction | < 2 seconds |
| NFR-01.2 | Graph traversal for up to 500 nodes | < 1 second |
| NFR-01.3 | API response time for all GET endpoints | < 200ms |
| NFR-01.4 | Dashboard initial load time | < 3 seconds |
| NFR-01.5 | WebSocket message delivery latency | < 500ms |

---

### NFR-02 — Reliability

| ID | Requirement |
|---|---|
| NFR-02.1 | System shall gracefully handle missing ATM matches by returning nearest available ATMs |
| NFR-02.2 | System shall not crash on malformed complaint inputs; return appropriate error codes |
| NFR-02.3 | WebSocket server shall auto-reconnect disconnected clients |

---

### NFR-03 — Security (SIH Demo Scope)

| ID | Requirement |
|---|---|
| NFR-03.1 | All API endpoints shall validate input using Pydantic schemas |
| NFR-03.2 | No real PII (Aadhaar, actual bank account numbers) shall appear in any dataset or demo |
| NFR-03.3 | All synthetic account numbers shall follow the format `XXXX-XXXX-XXXX` (masked) |
| NFR-03.4 | Officer actions (freeze, dispatch) shall be logged with timestamp |

---

### NFR-04 — Usability

| ID | Requirement |
|---|---|
| NFR-04.1 | Dashboard shall be usable on a 1080p display without horizontal scrolling |
| NFR-04.2 | All critical actions shall be reachable within 2 clicks from the main dashboard |
| NFR-04.3 | Countdown timers shall update every second without page refresh |
| NFR-04.4 | Color-coding shall follow accessibility contrast standards (WCAG AA) |

---

### NFR-05 — Portability (SIH Demo)

| ID | Requirement |
|---|---|
| NFR-05.1 | System shall run fully on a laptop without external internet (offline-capable) |
| NFR-05.2 | Map tiles shall be cacheable for offline use via Leaflet tile caching |
| NFR-05.3 | Setup shall be documented and completable in < 10 minutes via `README.md` |

---

## 3. Data Requirements

### DR-01 — Synthetic Victim Complaint Dataset

| Field | Type | Constraints |
|---|---|---|
| `ticket_id` | String | UUID, unique |
| `victim_name` | String | Realistic Indian name (Faker) |
| `victim_bank` | String | From approved list of Indian banks |
| `victim_account` | String | Format: `XXXX-XXXX-XXXX` (masked) |
| `fraud_type` | Enum | Digital Arrest \| Job Scam \| UPI Fraud \| Investment Scam \| Romance Scam |
| `stolen_amount` | Float | Range: ₹5,000 – ₹5,00,000 |
| `complaint_timestamp` | DateTime | Within last 90 days, random |
| `city` | String | From approved Indian city list |
| `state` | String | Corresponding Indian state |

**Volume:** Minimum 500 records

---

### DR-02 — Synthetic Transaction Ledger

| Field | Type | Constraints |
|---|---|---|
| `txn_id` | String | UUID |
| `complaint_id` | String | FK to victim complaints |
| `src_account` | String | Masked format |
| `dst_account` | String | Masked format |
| `bank_name` | String | Indian bank |
| `ifsc_code` | String | Valid IFSC format (`XXXX0XXXXXX`) |
| `city` | String | Indian city |
| `lat` | Float | Indian bounds: 8.0–37.0 |
| `long` | Float | Indian bounds: 68.0–97.5 |
| `amount` | Float | > 0, ≤ parent transaction amount |
| `timestamp` | DateTime | Strictly after parent transaction |
| `hop_depth` | Int | 1–5 |
| `is_terminal` | Boolean | True if leaf node |

**Volume:** Minimum 2000 records

---

### DR-03 — ATM / CSP Directory

| Field | Type | Constraints |
|---|---|---|
| `atm_id` | String | Unique ID |
| `bank_name` | String | Indian bank |
| `address` | String | Indian address |
| `city` | String | Indian city |
| `district` | String | Indian district |
| `state` | String | Indian state |
| `lat` | Float | Indian bounds |
| `long` | Float | Indian bounds |
| `opening_time` | Time | 24h format |
| `closing_time` | Time | 24h format |
| `cashout_risk_score` | Float | 0.0–1.0 |
| `historical_fraud_count` | Int | ≥ 0 |

**Volume:** Minimum 200 records across 10+ Indian cities

---

### DR-04 — Approved Indian Bank List (for synthetic data)

```
State Bank of India (SBI), HDFC Bank, ICICI Bank, Axis Bank,
Punjab National Bank (PNB), Bank of Baroda (BoB), Canara Bank,
Kotak Mahindra Bank, Union Bank of India, UCO Bank,
IndusInd Bank, IDBI Bank, YES Bank, Federal Bank, South Indian Bank
```

---

### DR-05 — Approved City Scope for Demo Dataset

```
Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata,
Pune, Ahmedabad, Jaipur, Lucknow, Patna, Bhopal,
Chandigarh, Noida, Gurgaon
```

---

*Last updated: 2026-08-28*
