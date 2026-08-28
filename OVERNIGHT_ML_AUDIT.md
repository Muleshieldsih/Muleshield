# Overnight ML Audit — MuleShield AI (SIH26184)

**Scope:** forensic leakage audit, evaluation repair, clean re-baselining, and ceiling
determination for all three ML components plus the end-to-end search-zone deliverable.

# Executive Summary

Three **construction artefacts** were found in the synthetic data, each of which made mule
detection look far easier than the real problem. All three are fixed. The headline mule F1
did not collapse — it improved — but for a different and legitimate reason: a separate
fix to the label-noise model raised the attainable ceiling more than removing the shortcuts
cost.

The **exact-ATM ranker is at its Bayes bound** and should not be optimised further.
The **search-zone deliverable is strong** (87.4% containment vs 78.5% best naive).
The **countdown is the only component with meaningful remaining headroom**.

| Component | Status |
|---|---|
| Data generator | 🔴 **FIXED** — 3 artefacts removed |
| GNN mule detector | 🟢 **KEEP** — 0.8955 F1, beats best tabular by +0.049, 97% of ceiling |
| Exact-ATM ranker | ⚪ **STOP OPTIMIZING** — at the Bayes bound |
| Countdown regressor | 🟡 **INVESTIGATE** — 11.82 min vs 9.38 achievable |
| Search-zone (deliverable) | 🟢 **KEEP** — 87.4% containment |

---

## 1. What was wrong

Ranked by severity.

**A1 — Mules were excluded from ordinary banking activity.**
`generate_legitimate_activity()` drew only from non-mule accounts, so mules averaged **1.68**
legitimate transactions against **12.35** for clean accounts. "This account has almost no
normal banking history" separated the classes at **AUC 0.96** on its own. A model trained on
that is detecting a hole in the simulator, not a mule.

**A2 — Account-age bands were near-disjoint.**
Mules were assigned `age_days = (4, 320)` while `salaried` began at 400 and `senior` at 900.
Roughly half the clean population could not overlap any mule. `account_age_days` alone scored
**AUC 0.9316**, and the derived `activity_per_day` (corr **0.917** with `1/age`) re-encoded it.

**A3 — `activity_per_day` was mis-specified (my own feature-engineering error).**
Defined as `activity / account_age_days`, it degenerates to `1/age` for any account older than
the 120-day observation window, because transaction counts are measured over that window and
do not grow with account age. It was an age proxy, not an activity measure.

**A4 — Evaluation reporting bug.**
Training logged "Best Val F1" at threshold **0.5** while the final test number used a
validation-tuned threshold. With compressed scores (median 0.29) this produced an apparent
val 0.6262 / test 0.8707 gap that looked like leakage and was not.

---

## 2. Exact leakage mechanism

For A1, the information pathway was:

```
generate_legitimate_activity() excludes mules
        ↓
mules receive ~1.7 ordinary transactions, clean accounts ~12.4
        ↓
in_degree / total_received / out_degree carry a 7x class gap
        ↓
any model reads "low ordinary volume" => mule           (AUC 0.96)
```

Classification: **D — synthetic-data construction artefact** (per the brief's taxonomy). Not
target leakage (no feature is computed from the label) and not temporal leakage (no future
information). It is a distributional shortcut created by the generator's population design.

**Temporal audit result: no temporal leakage found.**
- The GNN is transductive over one static ledger. Its framing is "given this book, which
  accounts are mules" — a bank scoring its portfolio nightly — not "which will become mules".
- The terminal account for a live complaint is identified by **tracing the chain**, not by the
  classifier, so the classifier's framing does not affect the deliverable.
- The ATM ranker **does** enforce a temporal boundary: cashout priors come only from the
  earliest 50% of complaints (`FeatureBuilder.HISTORY_FRACTION`), and those complaints are
  excluded from training and evaluation entirely.
- Mules now carry 54% ordinary traffic (was 9.9%), so their features describe general
  behaviour rather than the single fraud event.

### Feature leakage table (post-fix)

| Feature | Source | Window | Available at T? | Target dependency | Leakage? |
|---|---|---|---|---|---|
| `total_received` / `total_sent` | ledger | 120 d | Yes | none | No |
| `txn_count_24h`, `distinct_receivers` | ledger | 120 d | Yes | none | No |
| `avg_txn_amount` | ledger | 120 d | Yes | none | No |
| `median_dwell_seconds`, `dwell_log` | ledger | 120 d | Yes | none | No |
| `passthrough_ratio` | ledger | 120 d | Yes | none | No |
| `account_age_days` | KYC record | static | Yes | none | No (fixed in A2) |
| `night_txn_ratio`, `burst_out_5min` | ledger | 120 d | Yes | none | No |
| `activity_per_day` | derived | 120 d | Yes | none | No (fixed in A3) |
| `in_out_amount_ratio`, `avg_amount_per_credit`, `counterparty_concentration` | derived | 120 d | Yes | none | No |
| `hop_depth` | traced chain | — | — | **encodes chain membership** | **EXCLUDED** |
| `lat` / `long` | KYC | static | Yes | none | **DROPPED** (AUC 0.51, noise) |
| chain hop timing (countdown only) | traced chain | pre-cashout | Yes | none | No |

---

## 3. Which old metrics were invalidated

| Claim | Verdict |
|---|---|
| GNN F1 **0.9386** | **INVALIDATED** — measured on the A1/A2/A3 dataset |
| GNN F1 0.7210 / 0.7197 | **INVALIDATED** — measured on the broken symmetric-noise dataset (ceiling 0.823) |
| GNN F1 0.8707 with val 0.6262 | **Explained, not leakage** — A4 reporting bug; F1@tuned was consistent (train .889 / val .895 / test .871) |
| "+0.036 F1 lift over best baseline" | **INVALIDATED** — GNN and baselines were scored on different splits |
| Ceiling F1 ≈ 0.823 | **SUPERSEDED** — that ceiling came from a mis-specified symmetric noise model; corrected to **0.927** |

The earlier symmetric noise model flipped 2% of *all* accounts, which at ~5% prevalence turned
~40% of the positive class into unavoidable false positives and capped precision at 0.709. Real
labelling error is asymmetric: mules evade detection often, innocent customers are rarely
reported. Replaced with **8% undetected mules / 0.2% false reports**.

---

## 4. Clean data & evaluation methodology

- **Population:** 49,999 accounts, **2.95% mules**, 622,188 transactions (22,201 laundering /
  599,987 legitimate), 1,000 ATMs. 24.9 transactions per account; 0% near-empty.
- **Split:** 70/15/15, **stratified**, seed 42. Scaler fitted on **train only**. `pos_weight`
  computed on **train only**.
- **Threshold:** always tuned on **validation**, then frozen and applied to test.
- **Baselines** are scored on the **GNN's own persisted test node indices** (stored in the
  checkpoint), so all rows are directly comparable.
- **ATM ranker** splits by **complaint** (`GroupShuffleSplit`), never by row.
- Test split touched once per configuration.

---

## 5. Clean GNN results

All rows on identical held-out nodes, threshold tuned on validation.

| Model | Test F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Best single feature (`burst_out_5min`) | 0.5303 | 0.4922 | 0.5747 | 0.8936 | 0.3815 |
| Logistic regression (no graph) | 0.8322 | 0.8458 | 0.8190 | 0.9519 | 0.8149 |
| XGBoost (tabular, no graph) | 0.8462 | 0.8462 | 0.8462 | 0.9504 | 0.8285 |
| Random forest (no graph) | 0.8463 | 0.8333 | 0.8597 | 0.9580 | 0.8391 |
| **GraphSAGE (this system)** | **0.8955** | **0.8995** | **0.8914** | **0.9639** | **0.8529** |

- **Lift over best non-graph model: +0.0492 F1, +0.0138 PR-AUC.** The graph contribution is
  real and larger than it was on the artefact-laden data.
- FPR = **0.0031**, FNR = **0.1086**.
- Parameters: **10,561**. Inference: full-graph forward pass, ~1.7 s for all 50k nodes
  (embeddings are precomputed and cached; per-request cost is a dictionary lookup).

---

## 6. GNN experiments

Selection on validation only. Full log: `docs/gnn_experiments.csv`; runner:
`scripts/gnn_experiments.py`.

| ID | Change | Val F1 | Val PR-AUC | Verdict |
|---|---|---|---|---|
| `baseline` | current config | 0.8971 | 0.8471 | reference |
| `select_prauc` | select checkpoint on PR-AUC | 0.8873 | **0.8483** | marginal, not adopted |
| `select_f1tuned` | select on tuned-threshold F1 | 0.8922 | 0.8472 | no gain |
| `posw_half` | halve `pos_weight` | 0.8971 | 0.8482 | no gain |
| `posw_double` | double `pos_weight` | 0.8971 | 0.8467 | no gain |

Interpretation: the model is **insensitive to class weighting and to checkpoint criterion** —
it is neither underfit nor mis-weighted. The remaining error is label noise and genuine
behavioural overlap, which no hyperparameter reaches. Depth/width/aggregation variants were
queued but deprioritised once the ceiling analysis (§8) showed only 3 points of headroom.

**Threshold tuning is the one change that matters**, and it matters far more at realistic
prevalence than it did at 20%: F1 at 0.5 is 0.63, at the tuned threshold 0.90.

---

## 7. ATM ranking — at the Bayes bound, stop optimizing

| Model | Top-1 | Top-3 | Top-5 | MRR | NDCG |
|---|---|---|---|---|---|
| Distance only | 0.2476 | 0.5485 | 0.7217 | 0.4452 | 0.5741 |
| **Conditional logit (ours)** | **0.2654** | **0.5615** | 0.7136 | **0.4603** | **0.5857** |
| Bayes (true generative utility) | 0.2654 | 0.5744 | 0.7217 | 0.4629 | 0.5878 |

**Our Top-1 equals the Bayes bound exactly. MRR is within 0.003 of it.**

The learned utility weights recover the generative parameters, which an evaluator can check:

| Term | Learned | Generative truth |
|---|---|---|
| `−distance/5` | +0.9887 | 1.000 |
| `log(1 + 2·risk)` | +0.7558 | 1.000 |
| `same_bank` | +0.6847 | log 2 = 0.693 |

> **Further optimization of exact-ATM accuracy has low expected value because the task is
> information-limited.** The residual gap to Bayes is crew identification, not ranking.

---

## 8. Countdown

| Metric | Value |
|---|---|
| Test MAE | **11.82 min** |
| Mean-prediction baseline | 14.98 min |
| R² | 0.167 |
| q05–q95 coverage | 77.2% (band 49.8 min) |
| Lead time ≥15 min | 85.4% of cases, "go" call precision 85.4% |

**Ceiling conditioned on observables: MAE 9.38 min, R² 0.445.** The delay is a two-regime
mixture — a crew either has a runner already at the machine or has to travel — and which
regime applies is a draw the model cannot observe even knowing the crew's speed exactly.

Adding **observed intra-chain hop timing** (median gap, fastest gap, elapsed span — all visible
the moment a complaint is traced) moved MAE 12.04 → 11.82 and R² 0.155 → 0.167. Real but small:
the chain's timing reflects upstream accounts, while the cashout delay is driven by the
terminal account's own operating speed, which is diluted in its measured statistics now that
mules carry ordinary traffic.

---

## 9. End-to-end search zone — the actual SIH deliverable

All three zones given the **same radius**, so the comparison is at equal search cost.

| Zone centre | Contains the withdrawal | Median error |
|---|---|---|
| Mule location | 75.2% | 6.39 km |
| Nearest-3 ATM centroid | 78.5% | 5.68 km |
| **Model search zone** | **87.4%** | **5.49 km** |

- **Search cost: 1,000 ATMs → median 8**, inside a 9.98 km radius.
- **Latency: 4.5 ms** mean (SLA < 200 ms).
- **+8.9 points over the best naive zone.**

**Component contribution to the deliverable:** the zone is built by aggregating the ranker's
calibrated candidate distribution. Since the ranker is at its Bayes bound, and the GNN's mule
score is *structurally inert* for ranking (any feature constant within a candidate group
cancels in a conditional-logit softmax), **improving the GNN cannot improve zone containment**.
The GNN earns its place by naming freeze targets and by supplying crew structure to
`crew_prior`, not by moving the location metric.

---

## 10. Before vs After

| Metric | OLD (artefact data) | CLEAN | BASELINE | CEILING | Status |
|---|---|---|---|---|---|
| GNN F1 | 0.9386 *(invalid)* | **0.8955** | 0.8463 (RF) | 0.927 | 🟢 97% of ceiling |
| GNN PR-AUC | — | **0.8529** | 0.8391 (RF) | — | 🟢 |
| Legit-only shortcut AUC | 0.95–0.97 | **0.55–0.67** | — | ~0.50 | 🟢 closed |
| ATM Top-3 | 0.5658 | **0.5615** | 0.5485 | 0.5744 | ⚪ at bound |
| ATM MRR | — | **0.4603** | 0.4452 | 0.4629 | ⚪ at bound |
| Countdown MAE | 6.14 *(different target)* | **11.82 min** | 14.98 min | 9.38 min | 🟡 gap remains |
| Zone containment | 86.8% | **87.4%** | 78.5% | — | 🟢 |
| Search cost | 7 ATMs | **8 of 1,000** | — | — | 🟢 |
| Latency | ~15 ms | **4.5 ms** | — | < 200 ms | 🟢 |

---

## 10b. Known open issue — risk-score calibration

Isotonic calibration was fitted on the validation split and stored in the checkpoint
(`train_gnn.py`), and the API serves through it (`backend/state.gnn_risk_score`). The curve
itself is sound, but it does **not** transfer cleanly to the cached-embedding path:

| Calibrated band | Accounts | Actual mule rate |
|---|---|---|
| [0.00, 0.10) | 1,615 | 0.001 |
| [0.40, 0.70) | 106 | 0.009 |
| **[0.70, 0.95)** | **2,089** | **0.005** |
| [0.95, 1.00] | 112 | 0.866 |

The [0.70, 0.95) band should hold ~80% mules and holds 0.5%. The likely cause is a small
mismatch between the training forward pass (`model(x, edge_index)`) and the served path
(classifier head applied to embeddings cached by `embed.py`). The **ordering** is unaffected —
F1/AUC/PR-AUC in §5 are computed from the training path and are correct — but the number shown
in the Node Inspector should not be presented as a probability until this is resolved.

**Recommendation:** either recompute calibration against the cached-embedding scores, or show
the risk as a rank/percentile in the UI until it is fixed. Do not claim calibrated
probabilities to judges in the meantime.

## 11. Remaining bottlenecks

1. **Countdown observability** — the terminal account's operating speed is the dominant driver
   and is diluted in its measured statistics. MAE 11.82 vs 9.38 achievable.
2. **Crew identification** — `crew_prior` weight collapsed to +0.06 (from +0.29). The 3-hop
   neighbourhood is a weak proxy for syndicate membership; community detection (Louvain) on
   the mule subgraph would be the principled replacement.
3. **Label noise floor** — 8%/0.2% caps F1 at 0.927. We are at 0.8955.
4. **Calibration transfer** — see §10b; ordering is correct, displayed probabilities are not.
5. **Prevalence** — 2.95% is far more realistic than the original 19.8%, but real prevalence
   is under 1%. Precision@K would be the right headline there.

## 12. What to optimize next

1. Countdown: richer terminal-account speed features (dwell measured over the account's
   *outgoing* legs only, rather than all traffic).
2. `crew_prior` via community detection instead of k-hop.
3. Fix calibration transfer (§10b) — small, and it removes a demo-visible credibility risk.
4. Precision@K reporting at a realistic alert budget.

## 13. What to STOP optimizing

- ⚪ **Exact-ATM Top-1/Top-3/MRR** — at the Bayes bound; the gap is unobservable crew identity.
- ⚪ **GNN class weighting / checkpoint criterion** — measured insensitive.
- ⚪ **GNN capacity** — 10.5k parameters already reach 97% of the label-noise ceiling; more
  capacity fits noise.
- ⚪ **Pushing zone containment past ~87%** without first enlarging the radius, which would be
  gaming the metric rather than improving the forecast.

---

## 14. SIH judge-facing metrics

**The three numbers to show:**

1. **87.4% search-zone containment vs 78.5% for the best naive zone** — 1,000 ATMs narrowed to
   8, in 4.5 ms. This is the problem statement's actual ask.
2. **GNN F1 0.8955 vs 0.8463 for the best non-graph model on identical features and the same
   held-out nodes** — the graph earns its complexity, measured honestly.
3. **Exact-ATM Top-1 0.2654 = the Bayes-optimal bound.** Presenting a modest number *with its
   proven ceiling* is stronger than presenting a large one without.

## 15. Technical risks judges may challenge

| Challenge | Evidence prepared |
|---|---|
| "Synthetic data is too easy" | `scripts/eda_report.py` §2 leakage guard, §3 class overlap, §7 ceiling; three artefacts found and fixed, documented here |
| "Your accuracy is suspiciously high" | Ceiling analysis: 0.8955 against a 0.927 label-noise ceiling; baseline table on identical splits |
| "Top-3 of 56% is weak" | Bayes bound is 57.4%; our Top-1 *equals* Bayes. The deliverable is the zone, not the machine |
| "Does the GNN actually help?" | +0.049 F1 over RF/XGB/LR on the same nodes; and we state plainly that it does *not* help the location metric |
| "Is prevalence realistic?" | 2.95% vs real <1%; stated as a known limitation with Precision@K as the correct alternative framing |
| "Countdown R² is low" | Observable-conditioned ceiling is 0.445, not 1.0; the regime draw is unknowable |

## 16. Recommended final claims

- ✅ "We narrow 1,000 ATMs to a median of 8, and the withdrawal falls inside our predicted zone
  87.4% of the time — against 78.5% for the best naive zone at equal search cost."
- ✅ "Our graph model beats the best non-graph model by 4.9 F1 points on identical features and
  the same held-out accounts."
- ✅ "Our ATM ranker is at the Bayes-optimal bound for this task; we can prove the ceiling."
- ❌ Do **not** claim F1 > 0.93, 98% Top-3, or any fund-recovery-rate improvement.

## 17. Files changed

| File | Change |
|---|---|
| `scripts/generate_data.py` | mules participate in ordinary banking (A1); age-scaled participation; rented mules inherit civilian ages (A2); session-clustered legitimate timestamps; class-conditional label noise; atomic all-or-nothing dataset write |
| `engine/gnn_model.py` | feature set selected from measured signal; dropped 3 exact duplicates + `lat`/`long`; added 5 derived features; `derive_features()` |
| `engine/train_gnn.py` | stratified split; scaler and `pos_weight` on train only; corrected evaluation `pos_weight`; validation threshold tuning; split + full metrics persisted to checkpoint; reverse edges |
| `engine/feature_builder.py` | `CREW_HOPS` shared with serving; behaviour block; observed chain timing |
| `engine/xgb_model.py` | dead "Bayesian spatial reranking" removed; conditional-logit ranker; search-zone aggregation; prediction interval |
| `engine/train_xgb.py` | complaint-level split; zone containment metrics with baselines; lead-time metric; MAE objective + early stopping + quantile band |
| `scripts/evaluate_baselines.py` | baselines scored on the GNN's own split; Precision@K |
| `scripts/eda_report.py` | **new** — 7-section EDA |
| `scripts/feature_analysis.py` | **new** — correlation heatmap, signal ranking, redundancy |
| `scripts/gnn_experiments.py` | **new** — controlled experiment matrix |
| `scripts/seed_stability.py` | **new** — multi-seed spread |
| `backend/routers/predict.py` | search zone + interval + chain timing surfaced |
| `backend/models/schemas.py` | `SearchZone`, interval fields |
| `frontend/src/pages/TacticalMap.jsx` | draws the real search zone |

## 18. Reproduce

```bash
python scripts/generate_data.py          # ~8 min
python engine/train_gnn.py               # ~4 min
python engine/embed.py
python engine/train_xgb.py               # zone, ranking, countdown, lead time
python scripts/evaluate_baselines.py     # baseline table on the GNN's own split
python scripts/eda_report.py             # data-integrity evidence
python scripts/feature_analysis.py       # signal ranking + heatmap
python scripts/gnn_experiments.py --all  # experiment matrix
python -m pytest tests/ backend/tests/ -q
```
