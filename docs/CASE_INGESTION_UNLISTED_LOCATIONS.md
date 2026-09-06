# Handling Unlisted & Small-Town Case Ingestion in MuleShield AI
**Document Reference:** `docs/CASE_INGESTION_UNLISTED_LOCATIONS.md`  
**System Version:** MuleShield AI v2.4 (Phase 3 & Hotspot Engine)  
**Target Scenario:** An operator ingests a 1930 cybercrime case originating from a city, town, or district not present in the pre-seeded database (e.g., Kohima/Dimapur in Nagaland, or remote rural districts).

---

## 1. Executive Summary

When a cybercrime incident is reported from an unlisted location:
1. **Frontend:** The system accepts any free-text city name and allows manual state assignment without restriction.
2. **Backend Intake:** The FastAPI ingestion pipeline accepts open string inputs for `city` and `state`, persisting the complaint with a unique ticket ID (`TKT-...`).
3. **Money Trail Synthesis:** The routing engine automatically activates **Preference 3: National Syndicate Fallback**, drawing authentic graph accounts from active inter-state mule syndicates across India.
4. **AI & ATM Cash-Out Prediction:** The GraphSAGE GNN and XGBoost models rank cash-out points based on the **terminal mule account's bank branch and geolocation**, ensuring accurate prediction regardless of how remote the victim's location is.
5. **Operational Parity:** Case triage, officer assignment, live WebSocket broadcast, Section 91 CrPC freeze notices, and 65B-compliant legal dossiers function identically to pre-seeded metro cases.

---

## 2. Ingestion Flow (Frontend Layer)

### File: `frontend/src/pages/TriageFeed.jsx`

- **Free-Text with Autocomplete:** The City input uses `<input list="cities" ... />`. While it suggests the major 12 pre-seeded hub cities (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Pune, etc.) for convenience, **it does not lock or restrict user input**.
- **Dynamic State Behavior:**
  - When an operator selects a known city (e.g., `Pune`), `CITY_STATE_MAP` automatically sets `state = "Maharashtra"`, preventing invalid combinations (e.g., "Pune, Uttar Pradesh").
  - When an operator types an unlisted location (e.g., `Kohima` or `Tuensang`), the map returns no match and leaves the State field open for manual entry (`Nagaland`).
- **Payload Sent:**
  ```json
  {
    "fraud_type": "UPI Fraud",
    "stolen_amount": 250000,
    "city": "Kohima",
    "state": "Nagaland",
    "victim_phone": "+91-9876543210",
    "victim_account": "ACC-KOHIMA-01",
    "victim_bank": "State Bank of India"
  }
  ```

---

## 3. Storage & Syndicate Routing (Backend Layer)

### File: `backend/state.py` (`_pick_chain_accounts` & `synthesize_mule_chain`)

When `POST /api/v1/complaint/ingest` executes, `state.add_complaint(data)` creates the complaint record and synthesises a multi-hop laundering chain anchored to real graph nodes.

```python
def _pick_chain_accounts(city: str, count: int) -> list[str]:
    # Preference 1: Mule-labelled accounts in the complaint's city
    pool = list(mules_by_city.get(key, []))
    
    # Preference 2: Any accounts in that city
    if len(pool) < count:
        pool += [a for a in accounts_by_city.get(key, []) if a not in pool]
        
    # Preference 3: National Syndicate Fallback
    if len(pool) < count:
        for city_mules in mules_by_city.values():
            pool += [a for a in city_mules if a not in pool]
            if len(pool) >= count * 4:
                break
```

### Why "National Syndicate Fallback" Matches Ground Reality:
In real cybercrime operations (e.g., 1930 helpline reports), victims in smaller towns or remote states transfer funds under deception, but the illicit banking infrastructure (layering and terminal mule accounts) is operated by organized syndicates in regional financial hubs or known fraud clusters (e.g., Jamtara, Mewat, Kolkata, Mumbai, NCR). The fallback correctly simulates this real-world topology:
- **Victim Node:** Kohima, Nagaland (Origin of funds).
- **Hop 1 (L1 Mule):** Rapid fund absorption.
- **Hop 2 (L2 Mules):** Velocity and fund-splitting rule triggers (3 destination accounts within 5 minutes).
- **Hop 3 (Terminal Mules):** Final cash-out accounts with linked debit cards/ATMs.

---

## 4. AI Predictions & Interception Hotspots

### File: `engine/hotspot.py` & `backend/state.py:hotspot_entry`

1. **Feature Vector Parity:**
   - Each mule account in the synthesised chain is a real node carrying pre-computed 64-dimensional GraphSAGE GNN embeddings.
   - The XGBoost candidate ranker evaluates feature vectors based on account velocity, out-degree, and transaction volume.
2. **Terminal ATM Grounding:**
   - The ML prediction tracks the **terminal mule account's bank branch and location**, predicting the high-probability ATMs where cash withdrawal is imminent.
   - The victim's remote location does not degrade the predictive quality of the ATM cash-out ranker because cash-out occurs where the syndicate operates the terminal accounts.
3. **Centroid Anomaly Safeguard:**
   - If an account lacks resolvable coordinates and defaults to India's centroid `(20.5937, 78.9629)`, `_hotspot_dropped_centroid` filters it out, preventing false cluster visualizations in central India.

---

## 5. Downstream Modules & Operations

| Module | Behavior for Unlisted Town Cases |
|---|---|
| **Live Triage Feed** | Displays case card with custom badge (e.g., `Kohima, Nagaland`), priority score, and alert level. |
| **WebSocket Broadcast** | Dispatches `NEW_COMPLAINT` event across all connected officer terminals instantly. |
| **Forensic Graph** | Renders full interactive money trail from the victim account through L1, L2, and Terminal nodes. |
| **Legal Dossier / FIR** | Automatically compiles Section 65B Indian Evidence Act certificates citing the victim's location and inter-state IFSC codes. |
| **Account Freeze (Sec 91)** | Pre-fills freeze notices for recipient banks with reference IDs, branch IFSCs, and officer badges. |
