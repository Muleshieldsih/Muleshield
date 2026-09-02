---
title: "MuleShield AI: How We Beat Historical Crime Density by 24.6x to Predict Cybercrime Cash-Outs"
published: true
description: "A deep dive into building MuleShield AI — PyTorch GraphSAGE, McFadden's Conditional Logit, Quantile XGBoost, and a forward survival hotspot engine that forecasts ATM cash-outs in advance."
tags: machinelearning, python, architecture, datascience
cover_image:
canonical_url:
---

# MuleShield AI: Intercepting Cybercrime Cash-Outs Before the Trail Goes Cold

*How a hybrid AI pipeline + forward survival engine helps law enforcement predict ATM cashouts 41.5 minutes in advance.*

---

India loses over **₹1.8 lakh crore annually** to financial cybercrime. Behind nearly every UPI scam, fake investment scheme, and phishing attack is the exact same finale: stolen money is bounced across a chain of mule accounts and withdrawn as untraceable cash from an ATM — usually within 60 to 90 minutes of the original fraud.

**That last step is where cases die.**

Once money exits an ATM as paper currency, the digital trail goes cold. There is no transfer to reverse, no bank account left to freeze, and no digital transaction to trace.

We built **MuleShield AI** (for the Ministry of Home Affairs / I4C under Smart India Hackathon 2026) to solve this exact problem: forecasting **which ATM cluster** will be targeted and **how many minutes** police have before cash is pulled out.

Here is the full technical deep dive into the architecture, the machine learning models, and how we audited our own code.

---

## 1. The Strategic Gap: Starting Where Digital Freezing Ends

India already has robust digital account-detection infrastructure:
* **MuleHunter.AI** (RBI Innovation Hub) operates across 23+ major banks (Canara, PNB, BoI, BoB), analyzing 19 mule behavioral patterns with 95% reported accuracy.
* **I4C's Suspect Registry** has shared **32 lakh flagged mule accounts** with banks, declining **₹25,698 crore** in fraudulent transfers.

These platforms are vital, but they operate exclusively **inside digital banking rails**. The moment a criminal puts a card into an ATM, digital tools are out of the fight.

**The unanswered question:** *Which ATM will the cash come out of, and within what time window?*

MuleShield AI bridges this gap between digital fund blocking and physical ground interception.

---

## 2. The 3-Stage ML Decision Architecture

We deliberately avoided a single monolithic neural network. We decomposed the problem into three targeted sub-problems matching the physical generating process:

```
[Complaint Ingested (NCRP / 1930)]
               │
               ▼
[1] Deterministic BFS Graph Trace ───► Finds the terminal cash-out mule node
               │
               ▼
[2] PyTorch GraphSAGE (GNN) ─────────► Relational mule classification (F1: 0.9050, AUC: 0.9637)
               │
               ▼
[3] Conditional Logit Choice Model ──► McFadden's Discrete Choice Theory (Top-5 ATMs)
               │
               ▼
[4] Quantile XGBoost Regressor ──────► Uncertainty countdown window (MAE: 11.66 min, q05–q95)
               │
               ▼
[5] Forward Hotspot Surface ─────────► 12 km grid survival intensity (PAI@5: 26.86, Hit@5: 95.2%)
```

---

## 3. Model Deep Dive

### Model 1: GraphSAGE for Mule Account Detection
* **Why Graph Neural Networks?** A single mule account looks normal in a tabular database. GraphSAGE (PyTorch Geometric, 10,561 parameters) aggregates topological signals: 1-to-N fund splitting, velocity anomalies, and neighborhood behavior across 622,000 transactions.
* **Performance**: Reaches **0.9050 F1** and **0.9637 AUC**, outperforming tabular Random Forest by $+0.0485$ F1.

### Model 2: McFadden's Conditional Logit for ATM Ranking
* **Why Discrete Choice Theory?** When picking an ATM out of 25 nearby candidates, criminals evaluate multiplicative trade-offs (Distance $\times$ Bank Affinity $\times$ CCTV Risk $\times$ Gang Habit). 
* Taking logarithms converts this product into a linear sum of utilities.
* **Result**: **73.59% Top-5 accuracy on the exact machine**, exactly reaching the theoretical **Bayes Bound (optimal mathematical limit)**.

### Model 3: Quantile XGBoost for the Countdown Window
* **Why Quantiles?** Officers deploying patrol cars need a calibrated operational band (e.g. *"within 20 to 65 minutes"*), not a false static timestamp.
* **Performance**: Achieves **11.66 min MAE** with 78.9% interval coverage over a 49.2-minute band.

---

## 4. The Breakthrough: The Forward Hotspot Forecasting Surface

Rather than looking at single complaints in isolation, [`engine/hotspot.py`](file:///d:/SIH%202026/SIHPROJECT2/engine/hotspot.py) aggregates all active complaints in the golden-hour window ($\Delta t \le 120\text{ min}$) onto a **12 km hexagonal grid (222 cells)** across India.

It applies a **conditional lognormal survival kernel** across three forward operational horizons (`0–30 min`, `30–60 min`, `60–120 min`):

| Framework | Hit Rate @ 5 Cells | PAI (Prediction Accuracy Index) | Stolen Rupees Captured |
| :--- | :---: | :---: | :---: |
| **Historical Crime Density (Pratibimb)** | 6.12% | 1.09 | 8.09% |
| **Static Crime Count Baseline** | 6.44% | 1.01 | 5.98% |
| **MuleShield AI (Forward Survival Surface)** | **95.17%** | **26.86** | **95.61%** |
| **Performance Multiplier** | **15.5× Higher** | **24.6× Higher** | **11.8× Higher** |

---

## 5. Engineering Hardening & High-Throughput Benchmarks

1. **National Throughput**: Benchmarked at **3,499 complaints/min = 5.04 Million/day (630× headroom)** over India's ~8,000/day national volume.
2. **Sub-Second Latency**: Ingestion to alert dispatch clocked in at **0.25s (p95)**.
3. **Automated Proactive Alerting**: [`backend/notify.py`](file:///d:/SIH%202026/SIHPROJECT2/backend/notify.py) runs every 60 seconds to auto-dispatch SMS, Email, and CFCFRMS Webhook payloads to district police stations and bank nodal officers.
4. **Test Suite**: Backed by **449 passing automated tests** (0 failed, 0 skipped) with complete thread-safe SQLite concurrency (`RLock`).

---

## 6. What We Learned

1. **Understand incumbent systems first**: Don't rebuild what RBI and I4C have already deployed. Build the bridge between digital rails and ground police intervention.
2. **Respect the mathematical generating process**: Criminal discrete choice is multiplicative. Conditional logit beats deep nets and random forests because it matches how humans make choices.
3. **Compute your Bayes Bound**: Measuring the mathematical ceiling of your problem tells you when to stop tweaking model hyperparameters and start building operational software.

---

*MuleShield AI is an open, auditable predictive analytics framework engineered for SIH 26184 (Ministry of Home Affairs / I4C).*
