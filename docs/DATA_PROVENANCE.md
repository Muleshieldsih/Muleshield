# Data Provenance — where the dataset comes from and what backs it

**MuleShield AI · SIH26184 (MHA / I4C)**

Written to answer one question directly: *"You generated your own data. Why
should anyone believe it resembles reality?"*

The short answer is in three parts. Synthetic data is the **norm** in this field,
not a shortcut. The **structure** of our generator matches an IBM-published
standard and empirical criminology. The **parameter values** are chosen rather
than estimated, and that limits exactly what we may claim.

---

## 1 · Why the data is synthetic

Real NCRP / CFCFRMS transaction data is legally restricted. No competing team
has it either. There is no public Indian mule-transaction corpus with labelled
cash-out ATMs and timings, because publishing one would expose account-level
financial records of fraud victims.

**This constraint is universal in financial-crime ML, and the field's response is
synthetic data.** The single most-used public fraud benchmark in the world,
**PaySim**, is synthetic: Lopez-Rojas et al. (2016) built it by aggregating a
private African mobile-money dataset and re-generating it, precisely because *"the
intrinsically private nature of financial transactions leads to no publicly
available datasets."* It has ~6.3M transactions at 0.13% fraud prevalence and is
the default benchmark in the literature.

So "your data is synthetic" is not an objection to our project specifically. It
is a description of how this entire research area works.

---

## 2 · What already exists publicly, and why we did not simply use it

| Dataset | What it is | Why it does not solve SIH26184 |
|---|---|---|
| **IBM AMLSim** | Multi-agent simulator generating synthetic bank transactions with embedded laundering patterns | **Synthetic too.** Has no geography, no ATMs, no cash-out event, no timing-to-withdrawal |
| **Elliptic / Elliptic2** | Real Bitcoin transaction graph, ~203k nodes, expert-labelled illicit/licit | Bitcoin, not bank accounts. No physical cash-out location — the whole point of our problem |
| **PaySim** | Synthetic mobile-money log, 6.3M transactions, `CASH_OUT` transaction type | Has a cash-out *type* but **no location**. No ATM identity, no coordinates |
| **IEEE-CIS Fraud** | Real card-transaction fraud labels | Card-present fraud, not mule chains. No graph, no cash-out forecasting |

**The common gap: not one of them contains the label our problem requires** —
*which physical ATM* the cash came out of, and *how long after* the transfer.
That label does not exist in any public dataset because it lives in police
investigation files, not in bank feeds.

That is also the honest reason the problem statement is open: the data to solve
it directly is not available to anyone outside I4C.

---

## 3 · What backs the generator's structure

This is where the defence is strongest. We did not invent the topology.

### 3.1 · Chain patterns match IBM's enumerated set

IBM's **AMLSim** — the reference simulator in AML research — models eight
canonical laundering patterns: *fan-in, fan-out, bipartite, stack, random, cycle,
scatter-gather, gather-scatter*.

Our generator produces three of these, by construction:

| Our mechanism | AMLSim pattern | Where in code |
|---|---|---|
| Victim splits to 1–3 accounts, each splitting again | **fan-out / scatter** | `_build_chain`, hops 1–2 |
| 3–5 accounts converge on one terminal (55% of chains) | **gather / gather-scatter** | `_build_chain`, pooling branch |
| Sequential multi-hop forwarding | **stack** | `_build_chain`, deep-chain branch |

We are not claiming novelty in the topology. We are claiming it matches the
patterns an IBM research team enumerated independently.

### 3.2 · The distance-decay constant is in line with empirical criminology

The ranker's first term is `exp(-d / λ)` with **λ = 5.0 km**
(`ATM_DISTANCE_DECAY_KM`).

Journey-to-crime research — one of the most replicated findings in environmental
criminology — establishes that offence probability falls off with distance from
an offender's anchor point, and that **the average distance travelled to a crime
location is about three miles (≈ 4.8 km)**.

Our λ of 5.0 km sits directly on that empirical figure. The resulting median
journey-to-cash-out in the generated corpus is ~5.5 km (measured in
`notebooks/01_exploratory_data_analysis.ipynb`), which is the same order.

That is a genuine external check, and it was not reverse-engineered: the constant
was set first and the distribution measured after.

**Caveat, stated plainly:** the journey-to-crime literature's validity is itself
debated. Recent work reconstructing 449 theft-related offences found the
traditional measure valid for robbery and general theft but **not** for burglary
or motor-vehicle theft, and argues 50+ observations are needed to characterise
one offender's travel distribution. We cite distance decay as a *documented
regularity*, not a law.

### 3.3 · Mule behavioural patterns are enumerable — RBI says so

RBI's **MuleHunter.AI** was built by analysing **19 distinct behavioural patterns
of mule accounts** in partnership with banks, and is deployed at 23+ banks with
Canara reporting 95% accuracy.

We do not have their 19 patterns. But their existence is evidence that mule
behaviour *is* pattern-based and enumerable — which is the assumption our feature
set rests on (rapid pass-through, burst outflows, young accounts, night activity).

### 3.4 · Layering and smurfing are documented typology

Fund-splitting into near-equal amounts across 3+ destinations (our
`SPLIT_DESTINATIONS = 3`, `SPLIT_AMOUNT_TOLERANCE = 0.30`) is *smurfing*, a
standard FATF-documented laundering typology, and RBI's Master Directions on
Fraud Risk Management cover mule-account controls in commercial banks.

---

## 4 · The generative formulas

Two equations carry the load.

**Which ATM** (`assign_cashouts`, `scripts/generate_data.py`) — candidates within
60 km (minimum 25), then **sampled**, not argmax:

```
score_i  ∝  exp(−d_i / 5.0)              distance decay,      λ = 5 km
          × (1 + 2.0 · risk_i)            surveillance risk,   w = 2.0
          × 2.0   if same bank            bank affinity
          × 9.0   if a crew-preferred ATM 4 points per syndicate

P(atm_i) = score_i / Σ_j score_j
```

Sampling rather than taking the argmax is what makes the task non-trivial: the
nearest ATM is the single most likely choice but far from certain, so recovering
it requires combining distance with risk and bank affiliation.

**When** — a two-regime mixture:

```
p_immediate = clip(0.28 × 2.2 × exp(−dwell_hours / 1.5), 0.02, 0.75)

runner already in position:   3.0 + 0.25·km + Gamma(1.6, 2.1) − 3.4
has to travel:               22.0 + 1.9·km + night_penalty
                                  + min(dwell × 2.2, 25) + Gamma(2.2, 4.6) − 10.1

delay = clip(base + noise, 2, 180)  minutes
```

The regime is **tied to the account's own dwell behaviour rather than drawn at
random**. Drawing it randomly made the regime genuinely unobservable and the
countdown regressor collapsed to predicting the mixture mean (MAE 16.2, R² 0.06).
Tying it to `median_dwell_seconds` — a quantity measurable from any bank ledger —
keeps the bimodality realistic *and* leaves it learnable.

**Label noise is deliberately asymmetric.** `UNDETECTED_MULE_RATE` is high and
`FALSE_REPORT_RATE` is very low, because mules routinely go unreported while
banks do not file suspicious-activity reports on random innocent customers. An
earlier symmetric noise model capped F1 at **0.82 for a perfect classifier** — an
artefact of the noise model rather than of the problem.

---

## 5 · What we found wrong in our own data

Three construction artefacts were found and removed. The headline F1 was
re-measured **downward** from an invalid 0.9386 to 0.8955 as a result.

| # | Artefact | Effect |
|---|---|---|
| A1 | Mules excluded from ordinary banking activity — 1.68 legitimate transactions vs 12.35 for clean accounts | "Almost no normal banking history" separated the classes at **AUC 0.96 alone** |
| A2 | Account-age bands near-disjoint (mules 4–320 days, salaried from 400) | Half the clean population could not overlap any mule |
| A3 | `activity_per_day` divided by account age, degenerating to `1/age` beyond the 120-day window | An age proxy wearing an activity costume (r = 0.92 with `1/age`) |

`notebooks/01_exploratory_data_analysis.ipynb` re-runs the check every time it is
executed, against the documented **0.96 artefact level** rather than an arbitrary
threshold. Current strongest single feature: `txn_count_24h` at **0.9021 AUC**
but only **0.47 F1** — it ranks well and classifies poorly, which at 2.95%
prevalence is exactly the expected regime.

---

## 6 · What this licenses us to claim

**Transfers to real data:**
- the architecture (BFS trace → GNN → conditional logit → GBM countdown)
- the relative comparisons (graph beats non-graph by +0.053 F1 on identical splits)
- the finding that a conditional logit beats boosted trees on multiplicative choice
- the code, unchanged — the pipeline retrains on real I4C data without modification

**Does NOT transfer:**
- the absolute numbers. 71.4% top-5 containment is a real measurement *on this
  generative process*. It is not a forecast of what the system scores in Delhi.

**The sentence to say before anyone forces it out of you:**

> The synthetic corpus validates the method, not the performance. Its structure
> matches IBM's AMLSim pattern set and the empirical journey-to-crime literature.
> Its parameter values are ours, and they are published as constants an evaluator
> can change and re-run.

---

## 7 · Migration path to real data

Nothing in the pipeline is coupled to the generator.

| Component | Real-data substitute | Change required |
|---|---|---|
| Transaction ledger | Bank / NPCI inter-bank feed | Column mapping only |
| Mule labels | I4C Suspect Registry (32 lakh accounts already shared with banks) | None — same binary label |
| ATM directory | RBI's ATM master (real, already public) | Drop-in replacement |
| Cash-out ATM + delay | Police investigation records | This is the only genuinely scarce label |

The last row is the honest bottleneck. Detection could retrain on real data
tomorrow; **location forecasting needs cash-out outcomes to be recorded**, which
is an I4C process change, not an engineering one.

---

## Sources

- **PaySim** — Lopez-Rojas et al. (2016), synthetic mobile-money benchmark: [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) · [simulator source](https://github.com/EdgarLopezPhD/PaySim)
- **IBM AMLSim** — multi-agent AML simulator, eight laundering patterns
- **Elliptic / Elliptic2** — Bitcoin AML graph benchmark: [arXiv 2404.19109](https://arxiv.org/html/2404.19109v1)
- **GNN fraud detection review** — [arXiv 2411.05815](https://arxiv.org/pdf/2411.05815)
- **Journey-to-crime distance decay** — [Modeling Criminal Distance Decay (HUD)](https://www.huduser.gov/periodicals/cityscpe/vol13num3/Cityscape_Nov2011_Modelling_Criminal.pdf) · [Rengert, Distance Decay Reexamined, *Criminology* 1999](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-9125.1999.tb00492.x) · [validity critique, *J. Crim. Justice* 2023](https://www.sciencedirect.com/science/article/abs/pii/S0047235223000235)
- **MuleHunter.AI (RBIH/RBI)** — [Business Standard](https://www.business-standard.com/finance/news/what-is-mulehunter-ai-rbi-s-latest-tool-against-financial-fraud-explained-124120600865_1.html) · [23 banks, RTI](https://www.medianama.com/2025/12/223-rti-23-banks-mulehunter-mule-accounts/)
- **RBI Master Directions on Fraud Risk Management** — [rbi.org.in](https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx)
