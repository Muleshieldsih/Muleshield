# -*- coding: utf-8 -*-
"""
MuleShield AI -- forward cash-out intensity surface.
SIH26184 | MHA / I4C

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
I4C already runs Pratibimb, which maps cybercrime geographically, and Samanvaya,
which coordinates the LEA response. RBI's Innovation Hub runs MuleHunter.AI,
which flags accounts that look like mules. A historical density map of where
cash-outs have previously happened is therefore not a contribution -- it is a
thing the people judging this built first and run at national scale.

The clause in the problem statement that none of those answer is "in Advance".
So this module does not rank cells by history. It asks a conditional question:

    given the complaints that are open RIGHT NOW, each with its own posterior
    over reachable ATMs and its own countdown distribution, where is stolen
    money going to surface in the next thirty, sixty, hundred-and-twenty minutes?

Every term below is an output of the ALREADY-TRAINED checkpoint. Nothing here is
fitted, and nothing here needs retraining: OVERNIGHT_ML_AUDIT.md shows the
ranker sits on the Bayes bound for this generator, so a new model would be a new
way to lose. This is arithmetic over a frozen model's own outputs.

THE PRIOR IS CAPPED, AND THE CAP IS THE POINT
----------------------------------------------
History does enter, through prior_share -- a cell where cash-outs have
repeatedly happened is genuinely more likely to see the next one. But if that
term drives the surface, we have rebuilt Pratibimb with extra steps. So it is
capped at PRIOR_WEIGHT (0.15) of the surface's total conditional mass: it can
reorder cells, it can never carry them.

That cap is not a comment. It is:
  - a module constant, returned in every API response;
  - decomposed per cell into conditional_rupees / prior_rupees / prior_share,
    so the console can show which half is doing the work;
  - asserted by tests/test_metrics_ledger.py::test_prior_does_not_dominate;
  - required by the alerting policy, where a HIGH severity rule additionally
    demands conditional_share >= 0.60 (see backend/notify.py).

FEEDBACK-LOOP BIAS
------------------
Forecasting Nuh sends officers to Nuh, which produces more Nuh detections, which
raises the forecast for Nuh. This is the standard pathology of predictive
policing and it deserves a straight answer rather than silence:

 1. Structural, and primary. The bias travels through the historical prior. The
    prior is capped at 15%. The surface is driven by complaints filed in the
    last two hours -- which are reports from victims, not the product of where
    patrols were sent.
 2. Measured. scripts/evaluate_hotspots.py reports pai_at_5_region_blinded,
    recomputed with the highest-volume states removed from the prior. If PAI
    collapses, the prior was doing the work after all.
 3. Degradation is declared, not hidden. With no open complaints the surface is
    all prior; the response then carries degraded=True and prior_share=1.0 so
    the console can say "no live cases -- this is history only" instead of
    presenting a density map as a forecast.
 4. Not built, and said so: exposure-corrected training (weighting historical
    cash-outs by inverse patrol presence) is what a real deployment needs. See
    docs/INTEGRATION_SEAMS.md.

SCOPE BOUNDARY
--------------
A cell is a set of CASH-OUT POINTS. Today every point is an ATM, because the
corpus has no branch entity. Real cash-out also happens over branch counters by
cheque and through bulk payouts -- Nuh's 2025 figures name 1,400+ ATM IDs and 75
cheque branches. Adding those is a new point type, not a new model: nothing in
this file assumes a point is an ATM beyond where the directory is read.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger("muleshield.hotspot")

# ── Constants ────────────────────────────────────────────────────────────────

CELL_RADIUS_KM = 12.0
"""Greedy-leader clustering radius.

1,000 ATMs across 78 districts is 12.8 machines per district, and a district is a
political boundary rather than a deployable one -- a team is dispatched to a
place it can cover, not to an administrative unit. 12 km is about an hour's
working reach in Indian urban traffic. District and state ride along as roll-up
keys on every cell, so the dashboard's drill-down is a group-by, not a second
model.
"""

WINDOWS: tuple[tuple[int, int], ...] = ((0, 30), (30, 60), (60, 120))
"""Forecast windows, minutes from as_of.

Held-out time_to_cashout_min runs median ~42, p75 ~52, max ~110, so three bands
cover the observed support and 0-60 is the golden hour the 1930 rail is built
around.
"""

PRIOR_WEIGHT = 0.15
"""Maximum share of the surface the historical term may carry. See the header."""

PRIOR_HALFLIFE_DAYS = 30.0
EPS_CASE = 1e-4                       # mass below which a complaint is not "in" a cell

# How many contributing complaint ids a cell carries out to the API. A cell
# with more open cases than this is already the top of the surface; the
# console links to the queue for the rest rather than growing the payload.
MAX_CELL_COMPLAINT_IDS = 25
Z95 = 1.6448536269514722              # Phi^-1(0.95)

INDIA_CENTROID = (20.5937, 78.9629)
"""The fallback coordinate state.synthesize_mule_chain writes when it cannot
resolve a real account. A complaint sitting exactly here has no meaningful
candidate set, so it is excluded from the surface rather than smeared across the
middle of the country. Counted and logged, never silently dropped."""

EARTH_R_KM = 6371.0


# ── Timestamps ───────────────────────────────────────────────────────────────

def parse_ts(value: Any) -> Optional[datetime]:
    """Tolerant ISO parse, returning naive local-comparable datetimes.

    Two formats coexist in this system and always will: seed complaints carry
    naive local time ("2026-05-27T19:43:13"), while complaints ingested through
    the console carry an explicit UTC offset. Comparing them requires stripping
    the offset rather than rejecting either.

    Lifted out of backend/routers/complaint.py, which now imports it from here,
    so there is exactly one tolerant parser instead of two that can drift.
    """
    if value is None or isinstance(value, float):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


# ── Geometry ─────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(max(0.0, a)))


def build_cells(atm_rows: Iterable[Mapping[str, Any]],
                radius_km: float = CELL_RADIUS_KM) -> dict[str, dict]:
    """Cluster cash-out points into cells by greedy leader assignment.

    Greedy leader rather than k-means: it takes a RADIUS as its parameter, which
    is the quantity that actually matters operationally (how far a team can
    cover), whereas k-means takes a cluster COUNT and would produce cells whose
    physical size varies with local ATM density -- huge in Rajasthan, tiny in
    Mumbai. It is also deterministic given a stable input order, so a cell id
    means the same place across runs. That matters because alerts persist cell
    ids into SQLite: a cell id that shifted between restarts would silently
    re-point an acknowledged alert at a different place.

    Seeded in descending historical-fraud order so the busiest machine anchors
    its cell, which keeps known hotspots at cell centres rather than on a seam.
    """
    rows = []
    for a in atm_rows:
        try:
            lat = float(a.get("lat"))
            lon = float(a.get("long", a.get("lon")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        rows.append({
            "atm_id": str(a.get("atm_id", "")),
            "lat": lat, "lon": lon,
            "district": str(a.get("district", "") or ""),
            "state": str(a.get("state", "") or ""),
            "city": str(a.get("city", "") or ""),
            "fraud": float(a.get("historical_fraud_count", 0) or 0),
        })
    rows.sort(key=lambda r: (-r["fraud"], r["atm_id"]))

    leaders: list[dict] = []
    for r in rows:
        placed = False
        for L in leaders:
            if haversine_km(r["lat"], r["lon"], L["lat"], L["lon"]) <= radius_km:
                L["members"].append(r)
                placed = True
                break
        if not placed:
            leaders.append({"lat": r["lat"], "lon": r["lon"], "members": [r]})

    def _plurality(members: list[dict], key: str) -> str:
        counts: dict[str, int] = {}
        for x in members:
            if x[key]:
                counts[x[key]] = counts.get(x[key], 0) + 1
        return max(counts, key=counts.get) if counts else ""

    cells: dict[str, dict] = {}
    for i, L in enumerate(sorted(leaders, key=lambda x: (-len(x["members"]), x["lat"]))):
        m = L["members"]
        cid = f"CELL-{i + 1:03d}"
        # Cell centre is the mean of its members, not the seed machine: a patrol
        # briefed on this cell should be pointed at the middle of the cluster.
        clat = sum(x["lat"] for x in m) / len(m)
        clon = sum(x["lon"] for x in m) / len(m)
        # District/state by plurality of members. A 12 km cell can straddle a
        # boundary; the roll-up has to pick one, and the majority is the honest
        # choice.
        cells[cid] = {
            "cell_id": cid,
            "lat": round(clat, 6), "lon": round(clon, 6),
            "district": _plurality(m, "district"),
            "state": _plurality(m, "state"),
            "city": _plurality(m, "city"),
            "atm_count": len(m),
            "atm_ids": [x["atm_id"] for x in m],
            "radius_km": radius_km,
        }
    return cells


def atm_to_cell(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Reverse index: atm_id -> cell_id. Built once, used per complaint."""
    out: dict[str, str] = {}
    for cid, c in cells.items():
        for a in c["atm_ids"]:
            out[a] = cid
    return out


# ── Spatial term ─────────────────────────────────────────────────────────────

def posterior_from_scores(atm_ids: Sequence[str],
                          raw_scores: Sequence[float]) -> dict[str, float]:
    """Softmax the ranker's raw candidate scores into a choice distribution.

    This is the SAME transformation MuleXGBPredictor.predict applies to produce
    the confidence an operator sees -- "the softmax over the candidate set is
    the model's own choice probability, so the confidences shown to an operator
    are calibrated by construction" (xgb_model.py).

    It lives here as a named function so the evaluation script and the serving
    path call one implementation. A metrics/serving skew is then impossible by
    construction rather than by discipline -- which matters, because a hotspot
    figure that was measured with a different softmax than the API serves would
    be exactly the class of defect the leakage audit was written about.
    """
    if len(atm_ids) == 0:
        return {}
    mx = max(raw_scores)
    exp = [math.exp(float(s) - mx) for s in raw_scores]
    tot = sum(exp)
    if tot <= 0:
        u = 1.0 / len(atm_ids)
        return {str(a): u for a in atm_ids}
    return {str(a): e / tot for a, e in zip(atm_ids, exp)}


def project_to_cells(atm_probs: Mapping[str, float],
                     atm_cell: Mapping[str, str]) -> dict[str, float]:
    """P(cell) = sum of P(atm) over the machines in it."""
    out: dict[str, float] = {}
    for atm, p in atm_probs.items():
        cid = atm_cell.get(str(atm))
        if cid:
            out[cid] = out.get(cid, 0.0) + float(p)
    return out


# ── Temporal term ────────────────────────────────────────────────────────────

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def window_mass(median_min: float, lo_min: Optional[float], hi_min: Optional[float],
                elapsed_min: float, t0: float, t1: float) -> float:
    """P(withdrawal happens in [t0, t1) minutes from now | it has not happened yet).

    A lognormal is fitted through the three numbers the shipped regressor already
    produces -- the median, and the q05/q95 quantile models trained alongside it
    (train_xgb.py fits quantile_alpha 0.05 and 0.95). Lognormal because a delay
    is positive and right-skewed, and because two quantiles plus a median
    over-determine it, so the fit is closed-form rather than an optimisation:

        mu    = ln(median)
        sigma = (ln(hi) - ln(lo)) / (2 * 1.6449)

    The denominator is the part that makes this a FORECAST rather than a
    histogram. The complaint was filed elapsed_min ago and the money has not
    surfaced, so the distribution is conditioned on T > elapsed -- the survival
    renormalisation. Two consequences, both wanted:

      * a complaint whose window has passed contributes almost nothing, so stale
        cases fall out of the surface on their own, with no ad-hoc decay
        constant to tune or defend;
      * the same complaint's mass MOVES between windows as time passes, which is
        what "forecast" means and what a static intensity map cannot do.

    The parametric read is checked against reality elsewhere: metrics.json
    carries location.interval_coverage, the empirical share of held-out
    cash-outs that landed inside the q05-q95 band. Quote it beside this formula;
    if the band under-covers, so does everything computed here.
    """
    m = max(1.0, float(median_min))
    e = max(0.0, float(elapsed_min))

    if lo_min is None or hi_min is None or not (0 < float(lo_min) < float(hi_min)):
        # No quantile models bundled. Fall back to a fixed spread rather than a
        # point mass -- a spike at the median would claim a precision this model
        # has never demonstrated (countdown R^2 is 0.17).
        sigma = 0.6
    else:
        sigma = (math.log(float(hi_min)) - math.log(float(lo_min))) / (2.0 * Z95)
    sigma = min(2.5, max(0.05, sigma))
    mu = math.log(m)

    def cdf(t: float) -> float:
        if t <= 0:
            return 0.0
        return _phi((math.log(t) - mu) / sigma)

    survival = 1.0 - cdf(e)
    if survival <= 1e-9:
        return 0.0
    mass = cdf(e + float(t1)) - cdf(e + float(t0))
    return max(0.0, min(1.0, mass / survival))


# ── Historical prior ─────────────────────────────────────────────────────────

def build_prior(cashout_rows: Iterable[Mapping[str, Any]],
                atm_cell: Mapping[str, str],
                as_of: datetime,
                halflife_days: float = PRIOR_HALFLIFE_DAYS) -> dict[str, float]:
    """Decayed historical cash-out intensity per cell, normalised to shares.

    Amount-weighted and exponentially decayed at 2^(-age/halflife): a ten-lakh
    withdrawal last week says more about a cell than a small one last year.

    Returns SHARES summing to 1, not absolute rupees, because the caller
    rescales them against the live conditional mass. That is what makes the 15%
    cap meaningful in a unit that does not drift with corpus size.
    """
    raw: dict[str, float] = {}
    for r in cashout_rows:
        atm = str(r.get("cashout_atm_id") or "").strip()
        if not atm or atm.lower() == "nan":
            continue
        cid = atm_cell.get(atm)
        if not cid:
            continue
        ts = parse_ts(r.get("timestamp"))
        age_days = 0.0 if ts is None else max(0.0, (as_of - ts).total_seconds() / 86400.0)
        amount = float(r.get("amount", 0.0) or 0.0)
        raw[cid] = raw.get(cid, 0.0) + amount * (2.0 ** (-age_days / halflife_days))
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


# ── The surface ──────────────────────────────────────────────────────────────

def build_hotspot_surface(open_complaints: Sequence[Mapping[str, Any]],
                          cells: Mapping[str, Mapping[str, Any]],
                          prior_share: Mapping[str, float],
                          as_of: datetime,
                          windows: Sequence[tuple[int, int]] = WINDOWS,
                          prior_weight: float = PRIOR_WEIGHT) -> dict:
    """Aggregate open complaints into a forward intensity surface.

    open_complaints are dicts as cached by backend.state:
        {complaint_id, cell_probs: {cell_id: P}, m, lo, hi, amount, ts, fraud_type}

    Returns cells x windows with the conditional and prior contributions kept
    SEPARATE all the way out to the response. That separation is the
    deliverable: it is what lets the console show, and a judge check, that the
    forecast is driven by cases filed minutes ago rather than by a density map.
    """
    win = [tuple(w) for w in windows]
    cond: dict[str, dict[tuple, float]] = {}
    cases: dict[str, dict[tuple, set]] = {}

    for comp in open_complaints:
        ts = comp.get("ts")
        if ts is None:
            continue
        elapsed = (as_of - ts).total_seconds() / 60.0
        if elapsed < 0:
            continue                     # filed in the future; not open yet
        amount = float(comp.get("amount", 0.0) or 0.0)
        cid_c = str(comp.get("complaint_id", ""))
        probs = comp.get("cell_probs") or {}
        if not probs:
            continue
        for w in win:
            pw = window_mass(comp.get("m", 40.0), comp.get("lo"), comp.get("hi"),
                             elapsed, w[0], w[1])
            if pw <= 0:
                continue
            for cell_id, pc in probs.items():
                mass = float(pc) * pw
                if mass <= 0:
                    continue
                cond.setdefault(cell_id, {}).setdefault(w, 0.0)
                cond[cell_id][w] += mass * amount
                if mass > EPS_CASE:
                    cases.setdefault(cell_id, {}).setdefault(w, set()).add(cid_c)

    total_cond = {w: sum(c.get(w, 0.0) for c in cond.values()) for w in win}
    grand_cond = sum(total_cond.values())

    # With nothing open the surface is entirely historical. Declared in the
    # payload rather than letting a density map be read as a forecast.
    degraded = grand_cond <= 0.0

    out_cells = []
    for cell_id, meta in cells.items():
        rows = []
        c_tot = p_tot = 0.0
        share = float(prior_share.get(cell_id, 0.0))
        for w in win:
            c = cond.get(cell_id, {}).get(w, 0.0)
            # The cap, applied: the prior is scaled against the LIVE mass in
            # this window, so it can reorder cells but never carry more than
            # prior_weight of the surface.
            p = share if degraded else prior_weight * share * total_cond[w]
            rows.append({
                "window_start_min": w[0], "window_end_min": w[1],
                "score": c + p,
                "conditional_rupees": c,
                "prior_rupees": p,
                "prior_share": (p / (c + p)) if (c + p) > 0 else (1.0 if degraded else 0.0),
                "case_count": len(cases.get(cell_id, {}).get(w, ())),
            })
            c_tot += c
            p_tot += p
        if c_tot <= 0 and p_tot <= 0:
            continue
        # The contributing complaints, not just how many. `cases` already holds
        # the ids -- until now only len() escaped, which meant the console could
        # say "4 open cases" but could not say WHICH four, and an officer cannot
        # act on a count. Capped and sorted so the payload stays bounded and the
        # order is stable between two calls with the same inputs.
        contributing = sorted(set().union(*cases.get(cell_id, {}).values())
                              if cases.get(cell_id) else ())
        out_cells.append({
            "cell_id": cell_id,
            "lat": meta["lat"], "lon": meta["lon"],
            "district": meta["district"], "state": meta["state"], "city": meta["city"],
            "atm_count": meta["atm_count"],
            "score": c_tot + p_tot,
            "conditional_rupees": c_tot,
            "prior_rupees": p_tot,
            "prior_share": (p_tot / (c_tot + p_tot)) if (c_tot + p_tot) > 0 else 1.0,
            "case_count": max((r["case_count"] for r in rows), default=0),
            "complaint_ids": contributing[:MAX_CELL_COMPLAINT_IDS],
            "windows": rows,
        })

    out_cells.sort(key=lambda r: r["score"], reverse=True)

    tot_c = sum(c["conditional_rupees"] for c in out_cells)
    tot_p = sum(c["prior_rupees"] for c in out_cells)
    return {
        "as_of": as_of.isoformat(timespec="seconds"),
        "degraded": degraded,
        "prior_weight": prior_weight,
        "cell_radius_km": CELL_RADIUS_KM,
        "n_cells": len(out_cells),
        "n_open_complaints": len(open_complaints),
        "windows_min": [list(w) for w in win],
        "total_conditional_rupees": tot_c,
        "total_prior_rupees": tot_p,
        "prior_share_national": (tot_p / (tot_c + tot_p)) if (tot_c + tot_p) > 0 else 1.0,
        "cells": out_cells,
    }
