# Integration seams — what this system would attach to, and how

**SIH26184 · MHA / I4C**

This document exists because `COMPLIANCE_AUDIT.md` finding 6.2 caught the README
claiming *"standardized API schemas ready for live integration with the National
Cybercrime Reporting Portal (NCRP / 1930) and NPCI Switch"* when no such schema
existed anywhere in the repository. That is a claim a judge can check in thirty
seconds, and it did not hold.

The honest version is more useful anyway: here is the shape we would send, here
is the field mapping, here is what we would need from the system owner, and here
is what we have not built.

---

## 0 · Position: we consume these rails, we do not replace them

Four systems already exist and are operating at national scale. Anything we
claim has to sit beside them, not on top of them.

| System | Owner | What it already does | What it does not answer |
|---|---|---|---|
| **Samanvaya** | I4C | LEA coordination platform and data repository; ~18.43 lakh suspect identifiers, ~24.67 lakh mule accounts tracked, 16,840 arrests | Where the cash will surface, and when |
| **Pratibimb** | I4C | Maps criminals and cybercrime infrastructure **geographically** | Same — it maps what has happened |
| **MuleHunter.AI** | RBI Innovation Hub | ML over 19 mule behaviour patterns; ~20,000 mule accounts/month, 23 banks live (Dec 2025); MHA has directed all FIs to integrate by Dec 2026 | Geography and timing |
| **CFCFRMS / 1930** | I4C + banks | Golden-hour reporting and fund blocking; ~₹11,158 crore saved across ~5.3M complaints | Powerless once cash leaves the banking rails |

**The gap we occupy is the phrase "in Advance" in the problem statement title.**
Pratibimb maps where fraud *has* occurred. MuleHunter flags accounts that *look*
mule-like. Neither answers:

> "This ₹4.2 lakh, reported six minutes ago, will be pulled from an ATM in this
> 3 km cluster within the next ninety minutes."

Everything below is built so that the answer can be delivered into the rails that
already exist, rather than into a new portal nobody opens.

---

## 1 · Inbound — what we would consume

### 1.1 NCRP / 1930 complaint intake

**Have today.** `POST /api/v1/complaint/ingest` takes the seven fields the 1930
intake form actually carries: victim name, bank, account, fraud type, amount,
city, state. This is already the real shape.

**Gap.** The intake carries no `district` and no coordinates
(`backend/models/schemas.py`, `ComplaintIngestRequest`). District is the natural
roll-up for the risk dashboard, and today it is only reachable via the ATM
directory. A production intake should carry district, or pincode.

### 1.2 Inter-bank transaction trace — the honest dependency

**This is the load-bearing assumption of the whole system and it is not ours to
satisfy.**

A victim never knows the mule chain. At 2:30 am they know they were defrauded of
an amount, and nothing about where it went. The chain has to come from the
banking side: bank-to-bank tracing across institutions, which is precisely what
RBI/NPCI is building CFCFRMS infrastructure to do.

- **What we assume:** given a complaint, a trace of the onward transfers exists
  or can be requested.
- **What the demo does instead:** `backend/state.py:synthesize_mule_chain`
  fabricates a plausible chain at ingestion using **real accounts from the graph
  in the victim's city**, reproducing the 1-to-N split pattern the anomaly
  detectors look for. The GNN embeddings, the ATM directory and the whole
  inference path downstream are genuine; only this one input is simulated.
- **The line for a reviewer:** *we assume the inter-bank tracing feed exists,
  because RBI/NPCI is already building it, and we solve the part downstream of
  it — once you know the chain, where does the cash come out and when.*

### 1.3 Suspect registry (MuleHunter.AI / Samanvaya)

**Seam built, empty by design.** `data/suspect_registry.csv` is loaded at
startup and surfaces as `suspect_registry_hits` on a hotspot cell, which feeds
rule `R-HIGH-REGISTRY` in `backend/notify.py`.

The file ships with a single placeholder row and **no real identifiers**. We
consume these systems as a *routing signal* — an account already flagged
elsewhere raises the severity of a forecast — and we do not reproduce their
contents.

**Not done:** the registry is not a model feature. Adding it to
`FeatureBuilder` would require retraining, and both `OVERNIGHT_ML_AUDIT.md` and
the compliance audit close that door: the ranker sits on the Bayes bound for this
generator, so a retrain is a way to lose, not gain. The exact claim is *"we
consume it as a routing signal, not as a model feature yet."*

---

## 2 · Outbound — what we would emit

Two payload builders live in `backend/adapters/webhook.py`. Both are versioned
**`v1-proposed`**, and a test asserts that label stays.

> **These are our proposed mapping. They are NOT the published contracts for
> CFCFRMS or Samanvaya, which are not public and which we do not have.**

### 2.1 CFCFRMS — precautionary fund hold

`muleshield.cfcfrms.block-request/v1-proposed`

| Field | Source | Why it is there |
|---|---|---|
| `action` | constant `PRECAUTIONARY_HOLD` | Names the request as precautionary, not an adjudicated freeze |
| `urgency` | `alerts.severity` | CRITICAL / HIGH / WATCH |
| `complaint_references` | `alerts.complaint_ids` | Lets the bank tie the hold to live NCRP tickets |
| `amount_at_risk_inr` | `alerts.rupees_at_risk` | The sum the forecast is about |
| `predicted_cashout.*` | cell, district, state, window | Where and when |
| `confidence.live_share` | `1 - alerts.prior_share` | **How much of this is live forecast rather than historical pattern** |
| `confidence.contributing_cases` | `alerts.case_count` | How many open complaints drive it |
| `human_in_the_loop.*` | ack fields | Whether an officer has actually signed off |

The confidence block is the part that matters. A bank asked to hold a customer's
funds is entitled to know how strong the signal is, and a request that cannot be
interrogated is one a compliance officer is right to refuse.

### 2.2 Samanvaya — intelligence dissemination

`muleshield.samanvaya.dissemination/v1-proposed`

Scoped by `{state, district}` because the problem statement's unit of action is
*"LEAs at the state and local levels, coordinated by I4C"* — three tiers, which a
national broadcast cannot represent. Carries a standing advisory:

> *Ranked forecast, not a confirmed location. Deploy to observe; do not treat
> presence in this cell as grounds for detention.*

### 2.3 What we would need from the owners

1. Endpoint, auth model, and mTLS/allowlist requirements.
2. The real field names, so our mapping becomes a translation layer rather than a
   proposal.
3. Rate limits and idempotency semantics — we already dedupe one alert per rule
   per cell per window per hour (`idx_alerts_dedupe`), and would align that to
   whatever the receiver expects.
4. An acknowledgement callback, so `alert_deliveries.state` can reflect what the
   receiver did rather than only what we sent.

---

## 3 · Transport: real interfaces, mocked wire

Every adapter in `backend/adapters/` implements the full contract — recipient,
message, success/failure, provider reference — and **sends nothing**.

That is a stated boundary, not an unfinished edge. A live SMS route into India
needs a paid gateway and DLT template registration; a live CFCFRMS call needs
credentials that only I4C can issue. What is real:

- the `alerts` / `alert_deliveries` / `alert_recipients` tables;
- the rule engine and the scheduled pass that fires without a human;
- queueing, exponential backoff, dead-lettering after 4 attempts;
- acknowledgement with a required disposition, including *False positive*.

`MULESHIELD_ADAPTER_FAIL_RATE=1.0` exercises the failure and retry paths for
real. The console labels every channel *simulated* on screen, because an officer
who believes a force was warned when it was not is worse off than one who knows
it was not.

**Swapping in a real gateway is one module.** Nothing above the adapter changes.

---

## 4 · What a deployment needs that this build does not have

Stated here so it is not discovered later.

| Gap | Consequence | Note |
|---|---|---|
| **Exposure-corrected training** | Forecasting Nuh sends officers to Nuh, which produces more Nuh detections, which raises the forecast for Nuh | Mitigated structurally today — the historical prior is capped at 15% and every alert requires live evidence — but a real deployment should weight historical cash-outs by inverse patrol presence. See the feedback-loop section of `engine/hotspot.py` |
| **Branch-counter and bulk-payout cash-out** | ATM-only coverage | Nuh's 2025 figures name 1,400+ ATM IDs **and 75 cheque branches**. A cell is defined as a set of *cash-out points*, so a branch counter is a new point type, not a new model |
| **Real DPDP controls** | Retention, purpose limitation, subject rights | The corpus is synthetic so nothing is at stake today. Production needs a retention schedule and access-purpose logging |
| **Tamper-evident audit** | The trail is queryable but not hash-chained | An evidentiary export under BSA 2023 s.63 / IT Act s.65B would need one |
| **An alert budget agreed with the force** | The four rule thresholds are ours, not theirs | `R-HIGH-CONVERGE` fires on cells carrying 2.5x the current mean case count, `R-WATCH-SCORE` on the top 2%, with a Rs 1 lakh floor on HIGH and Rs 50 lakh on CRITICAL. At 8,000 complaints/day that yields ~23 alerts a pass. All four are environment-tunable (`MULESHIELD_CONVERGE_EXCESS`, `MULESHIELD_WATCH_PERCENTILE`, `MULESHIELD_HIGH_MIN_RUPEES`, `MULESHIELD_CRIT_RUPEES`) because how many alerts a shift can action is a fact about the force, not about the model |
| **Horizontal scale** | Single process, in-memory case state | `PROJECT_NOTES.md` §5 explains why; alerts and credentials are already in SQLite, case workflow is not |

---

## 5 · Where to look in the code

| Seam | File |
|---|---|
| Complaint intake | `backend/routers/complaint.py` |
| Chain synthesis (the simulated input) | `backend/state.py` · `synthesize_mule_chain` |
| Suspect registry load | `data/suspect_registry.csv` |
| Forecast surface | `engine/hotspot.py` · `backend/routers/hotspot.py` |
| Rules and dispatch | `backend/notify.py` |
| CFCFRMS / Samanvaya payloads | `backend/adapters/webhook.py` |
| Recipient roster | `data/alert_recipients.csv` · table `alert_recipients` |
| Delivery record | table `alert_deliveries` |
