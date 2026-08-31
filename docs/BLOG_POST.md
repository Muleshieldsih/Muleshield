# Our Model Tied With "Just Pick the Nearest One" — And We Shipped It Anyway

*Building a system to forecast where stolen money gets withdrawn, and what happened when we audited our own results.*

---

Someone gets scammed at 2 AM. They call India's 1930 cybercrime helpline at 2:30. By the time a police station has the complaint in front of them, the money has already moved through four bank accounts and is being withdrawn as cash from an ATM three cities away.

That last step is where the case dies. Once money becomes physical cash, there is no transaction to reverse, no account to freeze, and nothing left to trace. Everything that happens after is paperwork.

We spent a few months building **MuleShield AI**, a system that tries to get ahead of that moment — forecasting *which ATMs* stolen money is likely to be withdrawn from, and *how many minutes* are left before it happens.

This post is about what we built, and more usefully, about the three times we found out we were wrong.

---

## The gap we were actually filling

Our first instinct was the obvious one: build a model that detects money mule accounts.

That instinct was wrong, and it took reading the landscape to figure out why. India's Reserve Bank already deployed exactly that. **MuleHunter.AI**, built in-house by the RBI Innovation Hub, has been live since December 2024 — it analyses 19 distinct mule behaviour patterns and is running at Canara Bank, PNB, Bank of India, Bank of Baroda, and 20+ others. Canara reports 95% accuracy with it. Alongside it, I4C's Suspect Registry has already shared **32 lakh mule accounts** with banks and declined **₹25,698 crore** in transactions.

So mule detection isn't an open problem. It's a solved, deployed, government-backed one.

But look at what all of those systems have in common: **they operate inside the banking rails.** They flag an account, decline a transfer, freeze a balance. Every one of them stops working at the exact moment cash comes out of a machine.

Nobody was answering the next question: *where does it come out, and when?*

That reframing changed the whole project. We weren't competing with RBI's detector. We were starting where it stops.

---

## Three models, three different problems

The thing we got most right, architecturally, was refusing to make one model do everything.

```
Complaint filed
      │
      ▼
[1] BFS trace ────────────► finds the terminal account
      │                     (plain graph traversal — no ML)
      ▼
[2] GraphSAGE ────────────► scores every account: mule or not
      │
      ▼
[3] Conditional logit ────► ranks 25 nearby ATMs → top 5 + a search zone
      │
      ▼
[4] XGBoost ──────────────► minutes until withdrawal, with a confidence band
```

**Tracing isn't machine learning, and that matters.** Finding the last account in a laundering chain is breadth-first search over the transaction graph — start at the victim, walk the edges outward, stop at accounts with no outgoing transfer. It's deterministic. If our classifier were wrong about every probability it produced, the trace would still be correct. That's a property worth designing for, not an accident.

**GraphSAGE earns its place by reading the neighbourhood.** An account that receives ₹40,000 and forwards ₹39,500 four minutes later looks unremarkable on its own row. Surrounded by eight unrelated senders, one receiver, and neighbours doing the same thing within the hour, it's obvious. Two layers of mean-aggregation is literally "summarise your neighbours, concatenate with yourself, repeat."

We measured what the graph was worth, holding features and splits constant:

| Model | Test F1 |
|---|---|
| Majority class | 0.057 |
| Best single feature | 0.497 |
| Logistic regression (no graph) | 0.684 |
| Random forest (no graph) | 0.842 |
| **GraphSAGE** | **0.895** |

**+0.053 F1** is the graph's isolated contribution. Not a huge number. But it's an honest one, and we know exactly where it came from.

---

## The model that lost to "walk to the nearest ATM"

Here's where it gets interesting.

Our first attempt at predicting the cash-out location was a gradient-boosted ranker — XGBoost's `XGBRanker`, the sensible default. It **lost to a plain "pick the nearest ATM" rule.**

So did our second attempt: a 953-way softmax over the entire national ATM directory. With a few thousand training rows, that's about five examples per class. It also lost to nearest-ATM.

The problem wasn't capacity. It was **shape**.

Where a criminal cashes out is a *discrete choice among alternatives*, and the factors compose **multiplicatively** — proximity × surveillance risk × bank affinity × the crew's established habits. A decision tree approximates a product using axis-aligned splits, and does it badly. But take the logarithm of a product and it becomes a sum:

```
P(atm_i | candidates) = softmax_i( w · log(features_i) )
```

That's a conditional logit — McFadden's 1974 discrete-choice model. (McFadden received the 2000 Nobel in economics for developing exactly this theory of how people choose among alternatives.) It trains in under a second, and unlike the boosted ranker, **its weights are readable**.

That last property turned out to be the most valuable thing about it. Because our data is synthetic, we know the true generative parameters, so we could check whether the model recovered them:

| Term | Learned | Ground truth |
|---|---|---|
| `−distance/5` | 0.9887 | 1.000 |
| `log(1 + 2·risk)` | 0.7558 | 1.000 |
| `same_bank` | 0.6847 | log 2 = 0.693 |

It found the real structure. Not approximately — nearly exactly.

**The lesson we'd actually pass on:** we didn't beat the baseline by reaching for a bigger model. We beat it by matching the model to how the choice is actually made.

---

## Then we audited ourselves, and our results got worse

This is the part we're proudest of, which is a strange thing to say about deleting your own best number.

Our first build reported an F1 of **0.9386**. Great number. We went looking for why it was so good, and found three construction artefacts in our own data generator:

**1. Mules were excluded from ordinary banking activity.** Our generator drew "normal" transactions only from non-mule accounts. So mules averaged 1.68 ordinary transactions against 12.35 for clean accounts. The feature "this account has almost no normal banking history" separated the classes at **AUC 0.96 on its own.** We weren't detecting mules. We were detecting a hole in our own simulator.

**2. Account-age bands were near-disjoint.** Mules got ages 4–320 days; salaried archetypes started at 400. Half the clean population couldn't overlap any mule.

**3. A derived feature was mis-specified** — my own error. `activity_per_day` divided by account age, which for any account older than the 120-day observation window degenerates to `1/age`. It was an age proxy wearing an activity costume (correlation 0.92 with `1/age`).

We fixed all three, retrained, and **published the corrected number: 0.8955.**

The headline went down. That's what an honest audit looks like. We also added a regression test that fails the build if a retracted figure ever reappears in the codebase — because the most dangerous number is the one you've already put on a slide.

---

## We stopped optimising because we proved we couldn't do better

Our ATM ranker achieves **71.4% top-5 containment**. Sorting by distance alone achieves **72.2%**.

We do not beat the baseline at K=5. A paired McNemar test across 618 held-out cash-outs finds no significant difference at *any* K.

For a while this felt like failure. Then we computed the **Bayes bound** — the score a theoretically perfect model would get, using the true generative probabilities:

| | Top-1 | Top-5 | MRR |
|---|---|---|---|
| Distance only | 0.2476 | 0.7217 | 0.4452 |
| **Ours** | **0.2654** | 0.7136 | **0.4603** |
| **Bayes optimum** | **0.2654** | 0.7217 | 0.4629 |

**Our top-1 accuracy equals the theoretical optimum exactly.** MRR is within 0.003 of it.

We then decomposed our failures. Of 177 misses at K=5, **94.9% were cases where the true generative posterior also ranked the correct ATM outside its top 5** — the offender simply made a low-probability choice. Only **9 of 618 cases (1.5%)** were genuine model error.

This reframes everything. We didn't stop optimising because we ran out of time or ideas. We stopped because we **measured the ceiling and hit it.** A bigger model cannot buy information the data doesn't contain.

If you take one thing from this post: **compute your ceiling before you spend a week chasing two points of accuracy.** Sometimes the answer is that the remaining error isn't yours.

---

## What actually ships

The product isn't the top-5 list. It's the **search zone** built from the ranked distribution — and there, the model genuinely wins:

| Zone centre | Contains the withdrawal | Median error |
|---|---|---|
| Mule's own location | 75.2% | 6.39 km |
| Nearest-3 ATM centroid | 78.5% | 5.68 km |
| **Model search zone** | **87.4%** | **5.49 km** |

**1,000 ATMs down to a median of 8**, inside a ~10 km radius, at equal search cost against every baseline. Plus a countdown: **11.86 min MAE** against a 14.98 min mean-guess baseline, with a median **42 minutes of lead time** and 85.4% of cases actionable.

End to end, inference runs in **7.5 ms** on one CPU, in about **1 GB of RAM**, with no GPU. The GNN is **10,561 parameters**.

That smallness is deliberate. When a system's output sends police to a physical address, "why did it say that?" isn't a nice-to-have. A conditional logit whose coefficients you can read out loud in a courtroom is worth more than a transformer with two more points of accuracy and no explanation.

---

## The limitation we put on our own slides

A live complaint arrives with **no transaction history attached to it.**

The victim knows they were scammed and how much they lost. They don't know which mule accounts their money passed through — nobody at the intake desk does. That chain has to come from the banking side, via inter-bank tracing infrastructure that RBI and NPCI are still building out.

So our demo synthesises that one input — honestly, on real accounts from the graph, reproducing the transaction patterns our detectors genuinely fire on. Everything downstream of that input is real inference on real features.

We put that on the slide deck, in the challenges section, in plain words. Not because it's flattering, but because someone would have asked, and "we hadn't thought about that" is a much worse answer than "here's exactly where the boundary is."

---

## Three things we'd tell someone starting this

**1. Find out what's already deployed before you build.** Two hours of reading told us the mule-detection problem was solved and where the actual gap was. That single reframing was worth more than any model we trained.

**2. Match the model to the generating process, not to the leaderboard.** A Nobel-winning 1974 linear model beat gradient boosting because the problem was multiplicative and boosting isn't. Fashionable is not the same as correct.

**3. Compute your ceiling.** Baselines tell you if you're winning. Ceilings tell you whether to keep playing. We'd have burned weeks optimising a ranker that was already provably optimal.

---

*MuleShield AI was built for Smart India Hackathon 2026 (problem statement SIH26184, Ministry of Home Affairs / I4C). It runs on synthetic data generated to be deliberately hard — 2% label noise, overlapping class distributions, complaint-level train/test splits — because real NCRP transaction data is legally restricted. Every figure in this post is a held-out measurement, not an estimate. The full implementation carries 291 tests, including the ones that fail if we ever quietly restore a number we retracted.*
