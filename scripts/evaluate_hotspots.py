# -*- coding: utf-8 -*-
"""
Forward hotspot forecast -- held-out evaluation.

    Run: python scripts/evaluate_hotspots.py

WHAT IS BEING MEASURED, AND WHY NOT ACCURACY
--------------------------------------------
The deliverable is "forecast likely cash withdrawal locations in advance", and
the operational question is: if we flag k cells, does the money surface inside
one of them, and how much warning did we have?

Accuracy is meaningless here -- 222 cells and one true cell per cash-out makes
"not this cell" correct 99.5% of the time by saying nothing. The metrics below
are the ones the predictive-policing literature settled on for exactly this
shape of problem:

    hit_rate@k    the true cash-out lands in a flagged cell
    PAI@k         (hits / total) / (area flagged / total area) -- the standard
                  Prediction Accuracy Index. A PAI of 1.0 is a coin flip
                  weighted by area; the number has to clear 1 to mean anything.
    PEI@k         PAI / the best PAI achievable at that same coverage. Answers
                  "how close to the ceiling", which PAI alone cannot.
    rupees_covered@k   hits weighted by money at risk, which is what a recovery
                  rate is actually made of.
    lead_time     minutes between the surface being available and the
                  withdrawal. This IS the phrase "in Advance", quantified.

Area is proxied by ATM SHARE -- the fraction of the 1,000-machine directory
inside the flagged cells -- rather than by km^2. A team is dispatched to
machines, not to empty land, and it is the same denominator the shipped
ranking metric already uses (topk_curve.py reports 1 - K/n_atms). km^2 is
reported alongside for anyone who wants it.

THE BASELINE THAT MATTERS
-------------------------
`baseline_historical_density` ranks cells by the decayed historical prior alone.
That baseline is, functionally, what I4C's Pratibimb already provides. Clearing
it is not a nice-to-have -- it is the entire claim of this component, and it is
asserted by tests/test_metrics_ledger.py::test_forward_forecast_beats_
historical_density. If this script ever reports the forecast losing to it, the
honest response is to say so on the slide, not to retune until it wins.

NO RETRAINING, NO LEAKAGE
-------------------------
  * The shipped checkpoint is loaded and scored. Nothing is fitted here.
  * The split is the frozen GroupShuffleSplit(seed=42, test_size=0.20) BY
    COMPLAINT that train_xgb.py and topk_curve.py both use, so no laundering
    chain straddles train and test.
  * The historical prior is built only from FeatureBuilder.HISTORY_FRACTION --
    the earliest 50% of complaints, which build_ranking_set() already excludes
    from the evaluated rows. The prior therefore cannot see a complaint it is
    later scored on. That guard already existed; this script reuses it rather
    than reinventing a weaker one.
  * The posterior comes from engine.hotspot.posterior_from_scores -- the same
    function the API calls -- so a metrics/serving skew is impossible by
    construction.
  * Forward replay: a cell is scored using only complaints filed at or before
    the moment being forecast from. Nothing filed later is visible.
  * The temporal kernel uses the REGRESSOR'S PREDICTED countdown (median plus
    the q05/q95 quantile models), derived here exactly as
    MuleXGBPredictor.predict() derives it. It does NOT use
    meta["time_to_cashout_min"], which is the observed delay from the ledger --
    that is the answer, and an earlier version of this script fed it into the
    kernel. That was a genuine leak and is recorded here rather than quietly
    corrected, because the whole reason this project re-baselined in
    OVERNIGHT_ML_AUDIT.md is that a leak nobody wrote down survived for weeks.

WHAT GROUND TRUTH IS STILL USED, AND WHY THAT IS CORRECT
--------------------------------------------------------
Three ledger columns are read after the forecast is produced, purely to score it:

    cashout_atm_id       -> the true cell. The label.
    time_to_cashout_min  -> when the withdrawal happened, for lead-time and for
                            deciding which forecast window should have contained
                            it. A target, never an input.
    stolen_amount        -> the rupee weighting in rupees_covered@k. Known at
                            filing time; it is on the 1930 intake form.

The first two are read ONLY inside evaluate(), after rank_fn has returned its
ordering. No ranking function receives them.
"""

import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from feature_builder import FeatureBuilder            # noqa: E402
from xgb_model import MuleXGBPredictor                # noqa: E402
from metrics_io import write_metrics                  # noqa: E402
import hotspot as H                                   # noqa: E402

SEED = 42
TEST_SIZE = 0.20
K_GRID = [1, 2, 3, 5, 10, 20]
OPERATING_K = 5
OPEN_MINUTES = 120
EPOCH_MINUTES = 60


# ── helpers ──────────────────────────────────────────────────────────────────

def _cell_area_km2(radius_km: float) -> float:
    """A cell's nominal coverage. Cells overlap slightly at the seams, so this
    over-states flagged area a little, which makes PAI CONSERVATIVE. Preferred
    to a convex hull that would flatter the number."""
    return 3.141592653589793 * radius_km * radius_km


def _pai(hits: int, total_hits: int, flagged_share: float) -> float:
    if total_hits <= 0 or flagged_share <= 0:
        return 0.0
    return (hits / total_hits) / flagged_share


def main() -> dict:
    print("[1/6] Rebuilding the candidate-ranking set...")
    fb = FeatureBuilder()
    fb.load()
    X, y, groups, meta = fb.build_ranking_set()
    C = fb.RANK_CONTEXT_DIM
    n_atms = len(fb.atm_df)
    print(f"      X={X.shape}  cashouts={len(meta):,}  candidates={fb.CANDIDATE_K}  ATMs={n_atms:,}")
    print(f"      prior-only complaints held back by HISTORY_FRACTION="
          f"{fb.HISTORY_FRACTION}: {len(fb.history_complaints):,}")

    print("\n[2/6] Reproducing the frozen held-out split (seed=42, by complaint)...")
    cid_of_group = dict(zip(meta["group_id"], meta["complaint_id"]))
    row_complaint = np.array([cid_of_group[g] for g in groups])
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=row_complaint))
    overlap = set(row_complaint[train_idx]) & set(row_complaint[test_idx])
    assert not overlap, f"complaint leak across split: {len(overlap)}"
    test_groups = set(groups[test_idx])
    print(f"      test rows={len(test_idx):,}  cashouts={len(test_groups):,}  overlap=0")

    print("\n[3/6] Building cells and the historical prior...")
    atm_rows = fb.atm_df.to_dict("records")
    cells = H.build_cells(atm_rows)
    atm_cell = H.atm_to_cell(cells)
    cell_ids = list(cells.keys())
    print(f"      {len(cells)} cells at {H.CELL_RADIUS_KM} km "
          f"(median {int(np.median([c['atm_count'] for c in cells.values()]))} ATMs)")

    # Complaint timestamps + amounts, for epochs and rupee weighting.
    comp_df = pd.read_csv(ROOT / "data" / "victim_complaints.csv")
    comp_ts = {str(r.ticket_id): H.parse_ts(r.complaint_timestamp)
               for r in comp_df.itertuples()}
    comp_amt = {str(r.ticket_id): float(r.stolen_amount) for r in comp_df.itertuples()}
    comp_state = {str(r.ticket_id): str(r.state) for r in comp_df.itertuples()}
    comp_city = {str(r.ticket_id): str(r.city) for r in comp_df.itertuples()}

    # PRIOR: only from complaints reserved as history. build_ranking_set() skips
    # these, so nothing scored below contributed to the prior it is scored against.
    txn = fb.txn_df
    hist = set(fb.history_complaints)
    prior_rows = txn[(txn["is_terminal"] == 1) & (txn["complaint_id"].isin(hist))]
    as_of_global = max(t for t in comp_ts.values() if t is not None)
    prior_share = H.build_prior(prior_rows.to_dict("records"), atm_cell, as_of_global)
    print(f"      prior from {len(prior_rows):,} history-only cash-outs "
          f"over {len(prior_share)} cells")

    # Static (undecayed) count baseline, same source.
    static_raw = defaultdict(float)
    for r in prior_rows.itertuples():
        cid = atm_cell.get(str(getattr(r, "cashout_atm_id", "") or ""))
        if cid:
            static_raw[cid] += 1.0
    tot = sum(static_raw.values()) or 1.0
    static_share = {k: v / tot for k, v in static_raw.items()}

    print("\n[4/6] Scoring held-out cash-outs with the SHIPPED checkpoint...")
    pred = MuleXGBPredictor.load()
    t0 = time.perf_counter()
    test_rows = np.where(np.isin(groups, list(test_groups)))[0]
    scores_all = pred.classifier.predict(MuleXGBPredictor.log_features(X[test_rows, C:]))
    score_ms = (time.perf_counter() - t0) * 1000

    # Candidate atm_ids per row: candidate_block ordering is stable per group, so
    # recover them from the meta + the builder rather than re-deriving geometry.
    posteriors = {}          # group_id -> {cell_id: P}
    truth_cell = {}          # group_id -> cell_id of the actual cash-out
    countdowns = {}          # group_id -> (median, q05, q95) PREDICTED
    by_group = defaultdict(list)
    for pos, ridx in enumerate(test_rows):
        by_group[int(groups[ridx])].append((pos, ridx))

    meta_by_gid = meta.set_index("group_id")
    unresolved = 0
    for gid, items in by_group.items():
        row = meta_by_gid.loc[gid]
        lat, lon = float(row["lat"]), float(row["lon"])
        acc = str(row["terminal_account"])
        node = fb._node_lookup.get(acc)
        bank = node[4] if node else "UNKNOWN"
        cand_idx, _ = fb.candidate_block(lat, lon, bank, account=acc)
        ids = [str(fb.atm_df.iloc[int(i)]["atm_id"]) for i in cand_idx]
        raw = [float(scores_all[p]) for p, _ in items]
        if len(ids) != len(raw):
            unresolved += 1
            continue
        atm_probs = H.posterior_from_scores(ids, raw)
        posteriors[gid] = H.project_to_cells(atm_probs, atm_cell)

        # PREDICTED countdown, exactly as MuleXGBPredictor.predict() derives it:
        # scale the winning candidate's full row and read the median plus the
        # q05/q95 quantile models. NOT meta["time_to_cashout_min"] -- that is the
        # observed delay from the ledger, i.e. the answer. An earlier version of
        # this script used it here, which both leaked the target into the
        # temporal kernel and measured a pipeline the API does not serve.
        best_local = int(np.argmax(raw))
        best_row = X[items[best_local][1]].reshape(1, -1)
        scaled = pred.scaler.transform(best_row)
        m_hat = float(pred.regressor.predict(scaled)[0])
        lo_hat = hi_hat = None
        if pred.regressor_lo is not None and pred.regressor_hi is not None:
            a_ = float(pred.regressor_lo.predict(scaled)[0])
            b_ = float(pred.regressor_hi.predict(scaled)[0])
            lo_hat, hi_hat = max(1.0, min(a_, b_)), max(1.0, max(a_, b_))
        countdowns[gid] = (max(1.0, m_hat), lo_hat, hi_hat)

        tc = atm_cell.get(str(row["cashout_atm_id"]))
        if tc:
            truth_cell[gid] = tc
    print(f"      {len(posteriors):,} posteriors built "
          f"({score_ms:.0f} ms batch, {unresolved} unresolved)")

    print("\n[5/6] Forward replay by hourly epoch...")
    # An evaluated cash-out is (gid, complaint, truth cell, actual withdrawal time).
    events = []
    for gid, tc in truth_cell.items():
        row = meta_by_gid.loc[gid]
        cid = str(row["complaint_id"])
        ts = comp_ts.get(cid)
        if ts is None:
            continue
        delay = float(row["time_to_cashout_min"])
        events.append({
            "gid": gid, "complaint_id": cid, "truth_cell": tc,
            "filed": ts, "withdrawn": ts + timedelta(minutes=delay),
            "delay": delay, "amount": comp_amt.get(cid, 0.0),
            "state": comp_state.get(cid, ""),
            "city": comp_city.get(cid, ""),
        })
    print(f"      {len(events):,} evaluable held-out cash-outs")

    cell_area = _cell_area_km2(H.CELL_RADIUS_KM)
    total_area = cell_area * len(cells)
    atm_of_cell = {c: cells[c]["atm_count"] for c in cell_ids}

    def evaluate(rank_fn, label: str) -> dict:
        """Score one ranking strategy over every event.

        rank_fn(event, open_events) -> ordered list of cell_ids, best first.
        """
        hits = {k: 0 for k in K_GRID}
        rupees = {k: 0.0 for k in K_GRID}
        atm_share = {k: [] for k in K_GRID}
        leads = []
        n = 0
        total_rupees = 0.0
        for ev in events:
            # Open set: complaints filed in the OPEN_MINUTES before this one's
            # withdrawal, itself included. Nothing filed after is visible.
            t = ev["filed"]
            open_ev = [e for e in events
                       if 0 <= (t - e["filed"]).total_seconds() / 60.0 <= OPEN_MINUTES]
            order = rank_fn(ev, open_ev, t)
            if not order:
                continue
            n += 1
            total_rupees += ev["amount"]
            for k in K_GRID:
                top = order[:k]
                if ev["truth_cell"] in top:
                    hits[k] += 1
                    rupees[k] += ev["amount"]
                    if k == OPERATING_K:
                        leads.append((ev["withdrawn"] - t).total_seconds() / 60.0)
                atm_share[k].append(sum(atm_of_cell.get(c, 0) for c in top) / n_atms)
        out = {}
        for k in K_GRID:
            share = float(np.mean(atm_share[k])) if atm_share[k] else 0.0
            hr = hits[k] / n if n else 0.0
            out[str(k)] = {
                "hit_rate": round(hr, 4),
                "pai": round(_pai(hits[k], n, share), 3),
                "flagged_atm_share": round(share, 5),
                "flagged_area_km2": round(k * cell_area, 1),
                "rupees_covered": round(rupees[k] / total_rupees, 4) if total_rupees else 0.0,
                "hits": hits[k],
            }
        out["_n"] = n
        out["_leads"] = leads
        return out

    # -- the forecast --------------------------------------------------------
    def rank_forecast(ev, open_ev, t):
        agg = defaultdict(float)
        for e in open_ev:
            post = posteriors.get(e["gid"])
            if not post:
                continue
            elapsed = (t - e["filed"]).total_seconds() / 60.0
            cd = countdowns.get(e["gid"])
            if cd is None:
                continue
            m, lo_h, hi_h = cd
            pw = sum(H.window_mass(m, lo_h, hi_h, elapsed, a, b)
                     for a, b in H.WINDOWS)
            if pw <= 0:
                continue
            for c, p in post.items():
                agg[c] += p * pw * e["amount"]
        tot_c = sum(agg.values())
        if tot_c > 0:
            for c in cell_ids:
                agg[c] += H.PRIOR_WEIGHT * prior_share.get(c, 0.0) * tot_c
        return [c for c, _ in sorted(agg.items(), key=lambda kv: -kv[1])]

    # -- baselines -----------------------------------------------------------
    prior_order = [c for c, _ in sorted(prior_share.items(), key=lambda kv: -kv[1])]
    static_order = [c for c, _ in sorted(static_share.items(), key=lambda kv: -kv[1])]

    def rank_prior(ev, open_ev, t):
        return prior_order

    def rank_static(ev, open_ev, t):
        return static_order

    def rank_nearest_cell(ev, open_ev, t):
        """Cells nearest the TRACED TERMINAL ACCOUNT, closest first.

        This is the strong baseline and it is deliberately not hidden. It is
        available only because the graph engine already traced the chain to its
        terminal account -- so it measures the PIPELINE (complaint -> BFS trace
        -> terminal -> geography) without the probabilistic layer on top. If the
        forecast does not clear it, the honest reading is that the tracing is
        doing the work and the ranking is not, which is exactly what
        OVERNIGHT_ML_AUDIT.md already concluded for Top-K ATM ranking.
        """
        row = meta_by_gid.loc[ev["gid"]]
        lat, lon = float(row["lat"]), float(row["lon"])
        return [c for c, _ in sorted(
            ((cid, H.haversine_km(lat, lon, m["lat"], m["lon"]))
             for cid, m in cells.items()), key=lambda kv: kv[1])]

    def rank_victim_city(ev, open_ev, t):
        """Cells nearest the VICTIM's own city -- the folk rule that the money
        comes out where the victim lives. Uses only what the 1930 intake form
        carries, with no tracing at all, so it is the true no-model floor."""
        latlon = city_ll.get((ev["city"], ev["state"]))
        if latlon is None:
            return []
        return [c for c, _ in sorted(
            ((cid, H.haversine_km(latlon[0], latlon[1], m["lat"], m["lon"]))
             for cid, m in cells.items()), key=lambda kv: kv[1])]

    # Victim-city coordinates, averaged from the ATM directory -- the only
    # geocoded city table in the corpus.
    city_ll = {}
    for _, g in fb.atm_df.groupby(["city", "state"]):
        city_ll[(str(g.iloc[0]["city"]), str(g.iloc[0]["state"]))] = (
            float(g["lat"].mean()), float(g["long"].mean()))

    print("      forecast...");            fc = evaluate(rank_forecast, "forecast")
    print("      historical density...");  bp = evaluate(rank_prior, "prior")
    print("      static count...");        bs = evaluate(rank_static, "static")
    print("      nearest cell (traced)..."); bn = evaluate(rank_nearest_cell, "nearest")
    print("      victim city (no trace)..."); bv = evaluate(rank_victim_city, "victim")

    # -- PEI, and why it degenerates for a CONDITIONAL forecast ---------------
    #
    # PEI is PAI divided by the best PAI achievable at the same coverage. It was
    # defined for STATIC hotspot maps, where the ceiling is the empirical
    # concentration of crime and is genuinely below 1.0 hit rate.
    #
    # This forecast is conditional and per-event: every cash-out has exactly ONE
    # true cell, so a perfect forecaster attains hit_rate 1.0 at every k >= 1.
    # The ceiling is therefore PAI* = 1 / flagged_share, and the ratio collapses
    # exactly to the hit rate:
    #
    #     PEI = [(hits/n) / share] / [1 / share] = hits/n = hit_rate
    #
    # An earlier version of this script used a STATIC oracle -- the k cells with
    # the most historical cash-outs -- and reported PEI values above 18, which is
    # arithmetically impossible for a bounded efficiency index and was the
    # symptom of comparing an adaptive method against a fixed map. Reported
    # honestly here as the identity it is, rather than deleted or dressed up.
    for k in K_GRID:
        fc[str(k)]["pei"] = fc[str(k)]["hit_rate"]

    leads = sorted(fc.pop("_leads"))
    n_eval = fc.pop("_n")
    for d in (bp, bs, bn, bv):
        d.pop("_leads", None); d.pop("_n", None)

    lead_median = float(np.median(leads)) if leads else 0.0
    lead_p10 = float(np.percentile(leads, 10)) if leads else 0.0
    actionable = (sum(1 for l in leads if l >= 15) / len(leads)) if leads else 0.0

    print("\n[6/6] Results\n")
    hdr = (f"{'k':>3} | {'hit@k':>7} | {'PAI':>7} | {'PEI':>6} | {'ATM share':>9} | "
           f"{'Rs covered':>10} | {'PAI:prior':>9}")
    print(hdr); print("-" * len(hdr))
    for k in K_GRID:
        f, p = fc[str(k)], bp[str(k)]
        star = "  <-- operating point" if k == OPERATING_K else ""
        print(f"{k:>3} | {f['hit_rate']:>7.4f} | {f['pai']:>7.2f} | {f['pei']:>6.3f} | "
              f"{f['flagged_atm_share']:>9.4f} | {f['rupees_covered']:>10.4f} | "
              f"{p['pai']:>9.2f}{star}")

    at = fc[str(OPERATING_K)]
    pri = bp[str(OPERATING_K)]
    print(f"\n  n = {n_eval:,} held-out cash-outs | {len(cells)} cells | {n_atms:,} ATMs")
    print(f"  Lead time: median {lead_median:.1f} min, p10 {lead_p10:.1f} min, "
          f"{actionable:.1%} with >= 15 min warning")
    verdict = "BEATS" if at["pai"] > pri["pai"] else "DOES NOT BEAT"
    print(f"\n[1] vs HISTORICAL DENSITY (what Pratibimb already provides)")
    print(f"      Forecast {verdict} it at k={OPERATING_K}: "
          f"PAI {at['pai']:.2f} vs {pri['pai']:.2f}  "
          f"({at['pai'] / pri['pai']:.1f}x)" if pri["pai"] > 0 else "")
    print(f"      hit rate {at['hit_rate']:.4f} vs {pri['hit_rate']:.4f}")

    nb = bn[str(OPERATING_K)]
    print(f"\n[2] vs NEAREST CELL TO THE TRACED TERMINAL -- the strong baseline")
    print(f"      hit rate {at['hit_rate']:.4f} vs {nb['hit_rate']:.4f}, "
          f"PAI {at['pai']:.2f} vs {nb['pai']:.2f}")
    if nb["hit_rate"] >= at["hit_rate"]:
        print(f"      The forecast DOES NOT beat plain distance from the traced")
        print(f"      terminal account. This is the same ceiling OVERNIGHT_ML_AUDIT.md")
        print(f"      found for Top-K ATM ranking, and it is reported rather than")
        print(f"      hidden. What the pipeline earns is the TRACE that produces a")
        print(f"      terminal account at all -- plus the time dimension and the")
        print(f"      rupee weighting, which distance cannot supply and which are")
        print(f"      what let many complaints aggregate into one national surface.")

    vb = bv[str(OPERATING_K)]
    print(f"\n[3] vs VICTIM CITY ONLY -- the no-model floor (1930 intake fields alone)")
    print(f"      hit rate {at['hit_rate']:.4f} vs {vb['hit_rate']:.4f}, "
          f"PAI {at['pai']:.2f} vs {vb['pai']:.2f}")

    # precision / coverage, and the false-positive answer
    curve = []
    for k in K_GRID:
        f = fc[str(k)]
        prec = f["hits"] / (k * n_eval) if n_eval else 0.0
        curve.append({
            "k": k,
            "coverage": f["hit_rate"],
            "precision": round(prec, 5),
            "false_cells_per_hit": round((k * n_eval - f["hits"]) / f["hits"], 2)
                                    if f["hits"] else 0.0,
            "pai": f["pai"], "pei": f["pei"],
            "rupees_covered": f["rupees_covered"],
            "flagged_atm_share": f["flagged_atm_share"],
        })
    print(f"\n  At k={OPERATING_K}: {curve[K_GRID.index(OPERATING_K)]['false_cells_per_hit']} "
          f"cells searched per genuine interception.")

    # The cap, MEASURED rather than asserted. Build one real surface at a busy
    # epoch and read back what share of it the historical term actually carries.
    # tests/test_metrics_ledger.py checks this against prior_weight; an algebraic
    # constant would have made that test tautological.
    busiest = max(events, key=lambda e: len(
        [x for x in events
         if 0 <= (e["filed"] - x["filed"]).total_seconds() / 60.0 <= OPEN_MINUTES]))
    t_busy = busiest["filed"]
    open_busy = [e for e in events
                 if 0 <= (t_busy - e["filed"]).total_seconds() / 60.0 <= OPEN_MINUTES]
    agg_c = defaultdict(float)
    for e in open_busy:
        post = posteriors.get(e["gid"]) or {}
        elapsed = (t_busy - e["filed"]).total_seconds() / 60.0
        cd = countdowns.get(e["gid"])
        if cd is None:
            continue
        pw = sum(H.window_mass(cd[0], cd[1], cd[2], elapsed, a, b) for a, b in H.WINDOWS)
        for c, pr in post.items():
            agg_c[c] += pr * pw * e["amount"]
    tot_cond = sum(agg_c.values())
    tot_prior = sum(H.PRIOR_WEIGHT * prior_share.get(c, 0.0) * tot_cond for c in cell_ids)
    measured_prior_share = (tot_prior / (tot_cond + tot_prior)) if (tot_cond + tot_prior) > 0 else 0.0
    print(f"\nMeasured prior share of the surface at the busiest epoch "
          f"({len(open_busy)} open cases): {measured_prior_share:.4f} "
          f"(cap {H.PRIOR_WEIGHT})")

    payload = {
        "n_cells": len(cells),
        "cell_radius_km": H.CELL_RADIUS_KM,
        "n_atms": n_atms,
        "n_test_cashouts": n_eval,
        "n_test_complaints": len(set(e["complaint_id"] for e in events)),
        "windows_min": [list(w) for w in H.WINDOWS],
        "operating_k_cells": OPERATING_K,
        "open_minutes": OPEN_MINUTES,
        "hit_rate_at_k": {k: fc[k]["hit_rate"] for k in map(str, K_GRID)},
        "pai_at_k": {k: fc[k]["pai"] for k in map(str, K_GRID)},
        "pei_at_k": {k: fc[k]["pei"] for k in map(str, K_GRID)},
        "rupees_covered_at_k": {k: fc[k]["rupees_covered"] for k in map(str, K_GRID)},
        "flagged_atm_share_at_k": {k: fc[k]["flagged_atm_share"] for k in map(str, K_GRID)},
        "flagged_area_km2_at_k": {k: fc[k]["flagged_area_km2"] for k in map(str, K_GRID)},
        "baseline_historical_density": {
            "hit_rate_at_5": bp["5"]["hit_rate"], "pai_at_5": bp["5"]["pai"],
            "rupees_covered_at_5": bp["5"]["rupees_covered"],
        },
        "baseline_static_count": {
            "hit_rate_at_5": bs["5"]["hit_rate"], "pai_at_5": bs["5"]["pai"],
            "rupees_covered_at_5": bs["5"]["rupees_covered"],
        },
        "baseline_nearest_cell": {
            "hit_rate_at_5": bn["5"]["hit_rate"], "pai_at_5": bn["5"]["pai"],
            "rupees_covered_at_5": bn["5"]["rupees_covered"],
            "hit_rate_at_k": {k: bn[k]["hit_rate"] for k in map(str, K_GRID)},
            "_note": "Distance from the TRACED TERMINAL ACCOUNT. The strong "
                     "baseline; see the printed verdict [2].",
        },
        "baseline_victim_city": {
            "hit_rate_at_5": bv["5"]["hit_rate"], "pai_at_5": bv["5"]["pai"],
            "rupees_covered_at_5": bv["5"]["rupees_covered"],
            "_note": "Victim city only, no graph trace. The no-model floor.",
        },
        "beats_historical_density": bool(fc["5"]["pai"] > bp["5"]["pai"]),
        "beats_nearest_cell": bool(fc["5"]["hit_rate"] > bn["5"]["hit_rate"]),
        "pei_is_hit_rate": True,
        "lead_time_median_min": round(lead_median, 2),
        "lead_time_p10_min": round(lead_p10, 2),
        "lead_actionable_rate": round(actionable, 4),
        "precision_coverage_curve": curve,
        "prior_weight": H.PRIOR_WEIGHT,
        "prior_share_national": round(measured_prior_share, 4),
        "total_area_km2": round(total_area, 1),
    }
    write_metrics("hotspot", payload, source="python scripts/evaluate_hotspots.py")
    print("\n  wrote data/metrics.json section 'hotspot'")
    return payload


if __name__ == "__main__":
    main()
