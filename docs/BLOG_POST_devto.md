---
title: "MuleShield AI: Intercepting Cybercrime Cash-Outs Before the Trail Goes Cold"
published: true
description: "We built a three-model AI pipeline — GraphSAGE + conditional logit + XGBoost — that forecasts which ATM stolen money will be withdrawn from and how many minutes are left. Here's the architecture and the numbers."
tags: machinelearning, python, datascience, security
cover_image:
canonical_url:
---

# MuleShield AI: Intercepting Cybercrime Cash-Outs Before the Trail Goes Cold

*How a three-model AI pipeline helps Indian law enforcement get ahead of digital financial fraud — and the engineering decisions behind it.*

---

India loses over **₹1.8 lakh crore annually** to cybercrime. Behind nearly every UPI scam, fake loan fraud, and phishing attack is the same final step: stolen money is split across a chain of mule accounts and quietly withdrawn as untraceable cash from an ATM — often within 90 minutes of the original crime.

That last step is where investigations end. Once money becomes physical cash, there is no transaction to reverse, no account to freeze, and nothing left to trace.

**MuleShield AI** was built to interrupt that step.

---

## The Problem: Detection Alone Is Not Enough

India already has robust mule-account detection infrastructure. The Reserve Bank of India's **MuleHunter.AI** (live since December 2024, deployed at Canara Bank, PNB, Bank of India, Bank of Baroda, and 20+ others) identifies suspicious accounts using 19 distinct behaviour patterns, reporting 95% accuracy. The I4C Suspect Registry has shared **32 lakh flagged mule accounts** with banks and declined **₹25,698 crore** in transactions.

These systems are excellent at what they do — but they operate exclusively inside the banking rails. They can flag an account, decline a transfer, or freeze a balance. Every one of them stops working the moment cash leaves an ATM.

No existing system answers the question law enforcement actually needs answered: **which ATM will the money come out of, and how much time is left?**

MuleShield AI starts exactly where existing solutions stop.

---

## System Architecture: Three Models, Three Distinct Problems

We deliberately avoided building a single monolithic model. The task decomposes into three structurally different problems, each requiring a different approach.

```
Complaint filed → 1930 helpline
        │
        ▼
[1] BFS Graph Trace ──────────► identifies the terminal mule account
        │                        (deterministic graph traversal, no ML)
        ▼
[2] GraphSAGE Classifier ─────► scores accounts: mule probability
        │
        ▼
[3] Conditional Logit Ranker ─► ranks 25 nearby ATMs → top 5 + search zone
        │
        ▼
[4] XGBoost Regressor ────────► time-to-withdrawal with confidence band
```

This separation is an architectural choice with concrete benefits: each component can be tested, audited, and improved independently. If the classifier is retrained, the ranker is unaffected. If real transaction data becomes available (via RBI/NPCI inter-bank tracing infrastructure), only the trace layer needs updating.

---

## Component 1: Graph-Based Mule Detection

A mule account — receiving ₹40,000 and forwarding ₹39,500 four minutes later — looks unremarkable in isolation. Embedded in a transaction graph where eight unrelated senders converge on one receiver whose neighbours are doing the same thing simultaneously, the pattern is obvious.

We quantified the isolated contribution of graph structure by holding all other variables constant:

| Model | Test F1 |
|---|---|
| Majority class baseline | 0.057 |
| Best single feature | 0.497 |
| Logistic regression (no graph) | 0.684 |
| Random forest (no graph) | 0.842 |
| **GraphSAGE (graph + features)** | **0.895** |

The graph structure contributes **+0.053 F1** over the best feature-based model — a measurable, reproducible improvement, not a claim. The GNN itself is 10,561 parameters and runs in under 10 ms per inference.

---

## Component 2: ATM Location Prediction via Discrete Choice Theory

Predicting *which* ATM a criminal will use is a discrete-choice problem: one option is selected from a finite set of alternatives based on a weighted combination of factors (distance, surveillance risk, bank familiarity, crew habit patterns).

Standard gradient-boosted rankers model additive relationships. But criminal ATM selection is **multiplicative** — proximity × surveillance risk × bank affinity compose as a product, not a sum. A log transformation converts a product into a sum, making a linear model structurally correct where a tree-based model is structurally misspecified.

The solution is the **conditional logit** — McFadden's 1974 discrete-choice model (awarded the 2000 Nobel Prize in Economics). Applied here:

```python
# Utility of each candidate ATM:
U(atm_i) = w₁·log(1/distance_i) + w₂·log(1 + risk_i) + w₃·same_bank_i

# Probability:
P(atm_i | candidates) = softmax_i( U(atm_i) )
```

Because our simulation has known generative parameters, we can verify model correctness directly:

| Coefficient | Learned | Ground truth |
|---|---|---|
| Distance weight | 0.9887 | 1.000 |
| Surveillance risk | 0.7558 | 1.000 |
| Same-bank bonus | 0.6847 | log 2 = 0.693 |

The model recovered the real generative structure to within 1.5% — confirming the structural choice was correct, not just empirically competitive.

---

## Component 3: Time-to-Withdrawal Forecasting

An XGBoost regressor estimates minutes until cash-out, trained on features including: network velocity, account age, transaction pattern recency, and mule archetype cluster.

| Metric | Value |
|---|---|
| MAE | 11.86 minutes |
| Mean-guess baseline MAE | 14.98 minutes |
| Median lead time delivered | **42 minutes** |
| Cases actionable (≥15 min lead time) | **85.4%** |

42 minutes of median lead time is enough for coordinated police response. 85.4% actionability means the system is useful in the overwhelming majority of cases, not just edge cases.

---

## End-to-End Performance

The deliverable for law enforcement is not a ranked list of ATMs — it is a **geographic search zone** derived from the ranked probability distribution:

| Strategy | Withdrawal contained | Median zone error |
|---|---|---|
| Mule's own location | 75.2% | 6.39 km |
| Nearest-3 ATM centroid | 78.5% | 5.68 km |
| **MuleShield AI search zone** | **87.4%** | **5.49 km** |

87.4% containment at 5.49 km median error — reducing a district of 1,000+ ATMs down to a **median of 8 candidates** within a ~10 km search radius. Combined with the 42-minute lead time, this gives field teams a real and specific target.

The full pipeline inference time: **7.5 ms on a single CPU, ~1 GB RAM, no GPU required.** The system is designed to run on standard government server infrastructure.

---

## Interpretability as a Deployment Requirement

A system whose output directs police to a physical location must be explainable — not as a design preference, but as a practical requirement for operational and legal credibility.

The conditional logit ranker has three interpretable coefficients, readable out loud. GraphSAGE decisions can be traced back to specific neighbourhood structure. The trace layer is deterministic BFS — correct by construction, regardless of what the probabilistic components do.

This interpretability is a first-class engineering constraint, not an afterthought.

---

## Honest Assessment of Current Limitations

**On data**: MuleShield AI is trained on synthetic data. This is a deliberate, necessary constraint — real NCRP transaction data is legally restricted. The synthetic generator was calibrated against known mule behaviour patterns from published I4C and CERT-In reports, and includes 2% label noise and overlapping class distributions to prevent overfit to clean simulated data.

**On the intake dependency**: A live complaint arrives at the 1930 helpline with no transaction chain attached. The victim knows what they lost; they don't know which mule accounts it passed through. That chain must come from the banking side via inter-bank tracing infrastructure (currently being developed by RBI and NPCI). MuleShield AI's demo synthesises this input honestly — on real graph accounts with real transaction patterns — and performs genuine inference on everything downstream.

These limitations are documented, not hidden. A system that overstates its deployment-readiness is more dangerous than one with honest boundaries.

---

## Alignment with National Cybercrime Infrastructure

MuleShield AI is designed as a complement to, not a replacement for, existing government systems:

- **MuleHunter.AI** (RBI): detects mule accounts inside banking rails → MuleShield AI picks up where it stops
- **I4C Suspect Registry**: flags known mules → feeds the GraphSAGE feature layer as prior signal
- **NCRP / 1930 Helpline**: complaint intake → the system's operational trigger point
- **NPCI / inter-bank tracing (in development)**: real transaction chain data → future input layer upgrade path

The architecture was designed with this integration roadmap in mind.

---

## Technical Stack

| Component | Technology |
|---|---|
| Graph construction | NetworkX, custom BFS engine |
| GNN classifier | PyTorch Geometric — GraphSAGE (2-layer, mean aggregation) |
| ATM ranker | Conditional logit via `statsmodels` |
| Time regressor | XGBoost |
| API layer | FastAPI |
| Frontend | React + TypeScript |
| Test suite | 291 tests, including regression guards on all published figures |

---

*MuleShield AI was developed for Smart India Hackathon 2026, Problem Statement SIH26184, issued by the Ministry of Home Affairs / Indian Cybercrime Coordination Centre (I4C). All figures are held-out measurements on data not seen during training. The full implementation and methodology documentation are available in the project repository.*
