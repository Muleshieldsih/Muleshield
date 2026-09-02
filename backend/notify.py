# -*- coding: utf-8 -*-
"""
MuleShield AI -- alert rules, dispatch and delivery.
SIH26184 | MHA / I4C

THE CONCURRENCY SEAM
--------------------
Every function in this module is a plain `def` and none of them imports
backend.websocket. That is the rule documented at backend/auth.py:46-59: the app
ships `uvicorn --workers 1`, so a blocking sqlite3 call made on the event loop
freezes every other request and stalls the /ws/feed sockets with it.

A handler that must both write here and broadcast is `async def` and crosses the
seam with starlette.concurrency.run_in_threadpool -- which puts the blocking call
in exactly the threadpool Starlette would have used for a `def` handler, then
returns to the loop to broadcast. One idiom, applied everywhere, including the
background tick in main.py.

backend/tests/test_alerts.py asserts no function here is a coroutine.

WHAT A RULE IS FOR
------------------
The audit's finding 5.2 was that nothing in this system was a trigger. The
WebSocket broadcast four event types, and every one of them was the echo of an
action a user had just taken in the console. An echo is not an alert. A rule is
the conditional that fires when NOBODY is looking -- which is the entire
difference between a dashboard and an early-warning system.

THE ANTI-PRATIBIMB FLOOR
------------------------
R-HIGH-CONVERGE additionally requires conditional_share >= 0.60. This is the same
guarantee engine/hotspot.py enforces on the surface, restated as POLICY: a cell
cannot raise a HIGH-severity alert on historical density alone, no matter how hot
its history is. I4C already runs Pratibimb, which maps where cybercrime has
happened. An alert that fires because a place is historically bad tells them
nothing they do not already have, and it would train officers to ignore us.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from backend import db
from backend.adapters import for_channel

logger = logging.getLogger("muleshield.notify")

# ── Tunables ────────────────────────────────────────────────────────────────
# Module constants rather than literals buried in lambdas, because these are
# operating thresholds an I4C desk would tune against its own alert budget, not
# facts about the world. A force that can action twenty alerts a shift needs
# different cuts from one that can action two hundred.
CRIT_RUPEES = float(os.environ.get("MULESHIELD_CRIT_RUPEES", 5_000_000))
CONVERGE_CASES = int(os.environ.get("MULESHIELD_CONVERGE_CASES", 3))
CONVERGE_EXCESS = float(os.environ.get("MULESHIELD_CONVERGE_EXCESS", 2.5))
"""How far above the CURRENT average a cell must sit to count as convergence.

RECORDED REVERSAL, and the clearest example in this project of a threshold that
was calibrated on the wrong load.

CONVERGE_CASES alone -- "three or more open cases point at this cell" -- is a
sound rule when the surface holds a dozen complaints, and that is the condition
it was tuned against. The problem statement names the real condition twice:
roughly 8,000 complaints a day on NCRP, which puts ~670 complaints in the
120-minute open window. Measured at exactly that load, 664 of 666 cell-windows
held three or more cases: the mean was 10.7 and the maximum 35. A single rule
pass raised 685 alerts, 618 of them HIGH, one of which concerned Rs 2,452.

An alert that fires on the average is not an alert. So the floor is now relative
to the surface it is reading -- a cell must carry CONVERGE_EXCESS times the mean
case count of the live cell-windows, with CONVERGE_CASES as an absolute floor
beneath it. At national load that lifts the cut to roughly 27 cases; on a quiet
surface the mean is near 1 and the rule collapses back to the original "three or
more", which is the behaviour that was right there all along.

2.5 sits above the 95th percentile of the measured distribution and below the
99th, which is the band an alert should live in.
"""
HIGH_MIN_RUPEES = float(os.environ.get("MULESHIELD_HIGH_MIN_RUPEES", 100_000))
"""A HIGH alert has to concern real money. The same stress pass raised a HIGH
over a Rs 2,452 forecast because R-HIGH-CONVERGE had no rupee floor at all --
only R-WATCH-SCORE did. Rs 1 lakh sits an order of magnitude below the CRITICAL
cut and an order of magnitude above the noise."""
CONDITIONAL_FLOOR = 0.60
WATCH_PERCENTILE = float(os.environ.get("MULESHIELD_WATCH_PERCENTILE", 0.98))
"""Top 2% of the live surface, not the top decile.

At national load the top decile is ~67 cell-windows per pass, which is one WATCH
a minute for a national desk -- a feed, not a watchlist. The percentile is
env-tunable precisely because this is an alert-budget decision that belongs to
the force running it, not to us."""
# A watch item still has to be worth an officer reading it. Without this
# floor the top decile of a quiet surface is a list of Rs 40 forecasts.
WATCH_MIN_RUPEES = float(os.environ.get("MULESHIELD_WATCH_MIN_RUPEES", 10_000))
MAX_ATTEMPTS = 4


@dataclass(frozen=True)
class Candidate:
    """One cell-window, flattened into the shape the rules read."""
    cell_id: str
    district: str
    state: str
    window_start_min: int
    window_end_min: int
    score: float
    rupees_at_risk: float
    case_count: int
    complaint_ids: tuple
    prior_share: float
    surface_p90: float
    converge_floor: int = CONVERGE_CASES
    suspect_registry_hits: int = 0

    @property
    def conditional_share(self) -> float:
        return 1.0 - float(self.prior_share)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str
    predicate: Callable[[Candidate], bool]
    why: str


RULES: list[Rule] = [
    Rule("R-CRIT-RUPEES", "CRITICAL",
         lambda c: c.rupees_at_risk >= CRIT_RUPEES and c.window_end_min <= 60,
         "large sum forecast to surface inside the golden hour"),
    Rule("R-HIGH-CONVERGE", "HIGH",
         lambda c: c.case_count >= c.converge_floor
         and c.conditional_share >= CONDITIONAL_FLOOR
         and c.rupees_at_risk >= HIGH_MIN_RUPEES,
         "open cases converging on one cluster far above the current average, "
         "driven by live evidence and worth real money"),
    Rule("R-HIGH-REGISTRY", "HIGH",
         lambda c: c.suspect_registry_hits >= 1,
         "an account on the suspect registry is in the chain"),
    Rule("R-WATCH-SCORE", "WATCH",
         lambda c: (c.surface_p90 > 0 and c.score >= c.surface_p90
                    and c.case_count >= 1
                    and c.conditional_share >= CONDITIONAL_FLOOR
                    and c.rupees_at_risk >= WATCH_MIN_RUPEES),
         "top decile of the live surface, driven by an open case"),
]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _bucket(when: datetime) -> str:
    """Dedupe bucket: one alert per rule per cell per window per HOUR."""
    return when.strftime("%Y-%m-%dT%H")


def _rupees(v: float) -> str:
    v = float(v or 0)
    if v >= 1e7:
        return f"Rs {v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"Rs {v / 1e5:.2f} L"
    return f"Rs {v:,.0f}"


def candidates_from_surface(surface: dict) -> list[Candidate]:
    """Flatten a surface into cell-window candidates.

    surface_p90 is computed over the WHOLE surface, not over whatever subset an
    operator happens to be filtering to. This is the backend analogue of the rule
    at TriageFeed.jsx:504: if the threshold moved with the view, a quiet
    district's local maximum would raise a national WATCH.
    """
    cells = surface.get("cells") or []

    # A cell-window with NO conditional mass is not a forecast -- it is the
    # capped historical prior showing through where nothing live is happening.
    # Alerting on those is alerting on a density map, and measured on the real
    # corpus it was not a theoretical risk: an earlier version of this function
    # admitted every cell-window with score > 0, and the first live rule pass
    # raised 66 alerts of which 48 had zero open cases and prior_share of
    # exactly 1.0. Three quarters of the inbox was Pratibimb.
    #
    # The floor is applied HERE rather than in each rule so no future rule can
    # forget it.
    def _live(w) -> bool:
        return float(w.get("conditional_rupees", 0.0)) > 0.0

    live_windows = [w for c in cells for w in (c.get("windows") or []) if _live(w)]
    live_scores = [float(w.get("score", 0.0)) for w in live_windows]
    p90 = _percentile(live_scores, WATCH_PERCENTILE)

    # The convergence floor is derived from the surface for the same reason
    # surface_p90 is: a cut that means "unusual" has to know what usual is right
    # now. 670 open complaints spread over 222 cells put three cases in almost
    # every cell, so a fixed three stops distinguishing anything. Computed over
    # the WHOLE surface, never the filtered view.
    live_cases = [int(w.get("case_count", 0)) for w in live_windows]
    mean_cases = (sum(live_cases) / len(live_cases)) if live_cases else 0.0
    converge_floor = max(CONVERGE_CASES, int(math.ceil(CONVERGE_EXCESS * mean_cases)))

    out: list[Candidate] = []
    for c in cells:
        for w in (c.get("windows") or []):
            score = float(w.get("score", 0.0))
            if score <= 0 or not _live(w):
                continue
            out.append(Candidate(
                cell_id=str(c.get("cell_id", "")),
                district=str(c.get("district", "")),
                state=str(c.get("state", "")),
                window_start_min=int(w.get("window_start_min", 0)),
                window_end_min=int(w.get("window_end_min", 0)),
                score=score,
                rupees_at_risk=float(w.get("conditional_rupees", 0.0))
                + float(w.get("prior_rupees", 0.0)),
                case_count=int(w.get("case_count", 0)),
                complaint_ids=tuple(c.get("complaint_ids") or []),
                prior_share=float(w.get("prior_share", 0.0)),
                surface_p90=p90,
                converge_floor=converge_floor,
                suspect_registry_hits=int(c.get("suspect_registry_hits", 0)),
            ))
    return out


def evaluate(surface: dict, *, as_of: Optional[datetime] = None,
             dry_run: bool = False) -> list[dict]:
    """Run every rule over the surface and raise what fires.

    A degraded surface raises NOTHING. With no open complaints the surface is
    pure historical prior, and alerting on it would be alerting on a density map
    -- precisely the thing this component exists not to be, and precisely what
    would erode an officer's trust fastest.
    """
    if not surface or surface.get("degraded"):
        logger.info("[NOTIFY] surface is degraded (no live cases); nothing raised.")
        return []

    when = as_of or datetime.now(timezone.utc)
    bucket = _bucket(when)
    raised: list[dict] = []

    for cand in candidates_from_surface(surface):
        for rule in RULES:
            try:
                if not rule.predicate(cand):
                    continue
            except Exception:                      # a bad rule must not stop the pass
                logger.exception("[NOTIFY] rule %s raised", rule.rule_id)
                continue

            headline = (
                f"{_rupees(cand.rupees_at_risk)} forecast in {cand.district or cand.cell_id}"
                f" within {cand.window_start_min}-{cand.window_end_min} min"
                f" ({cand.case_count} open case(s))"
            )
            payload = {
                "id": db.next_alert_id(),
                "rule_id": rule.rule_id,
                "severity": rule.severity,
                "cell_id": cand.cell_id,
                "district": cand.district,
                "state": cand.state,
                "window_start_min": cand.window_start_min,
                "window_end_min": cand.window_end_min,
                "score": cand.score,
                "rupees_at_risk": cand.rupees_at_risk,
                "case_count": cand.case_count,
                "complaint_ids": list(cand.complaint_ids),
                "prior_share": cand.prior_share,
                "headline": headline,
                "dedupe_bucket": bucket,
            }
            if dry_run:
                raised.append({**payload, "why": rule.why})
                continue

            row = db.insert_alert(payload)
            if row is None:
                continue                            # deduped; already raised this hour
            dispatch(row)
            raised.append(db.get_alert(row["id"]) or row)

    if raised:
        logger.info("[NOTIFY] raised %d alert(s)", len(raised))
    return raised


def dispatch(alert: dict) -> list[dict]:
    """Queue and attempt one delivery per responsible recipient.

    The delivery row is written BEFORE the send is attempted. If the process
    dies mid-dispatch the record still says an attempt was owed, which is the
    behaviour an audit needs; the reverse order would lose the fact entirely.
    """
    recipients = db.list_recipients(state=alert.get("state", ""),
                                    district=alert.get("district", ""))
    if not recipients:
        logger.warning("[NOTIFY] %s has no recipient in scope %s/%s",
                       alert.get("id"), alert.get("state"), alert.get("district"))
        return []

    results = []
    subject = f"[{alert.get('severity')}] {alert.get('headline')}"
    body = (f"Cell {alert.get('cell_id')} - {alert.get('district')}, {alert.get('state')}. "
            f"Forecast only; a ranked candidate area, not a confirmed location.")

    for r in recipients:
        did = db.queue_delivery(alert["id"], r["channel"], r["address"])
        results.append(_attempt(did, r["channel"], r["address"], subject, body, alert))
    return results


def _attempt(delivery_id: int, channel: str, recipient: str, subject: str,
             body: str, alert: dict) -> dict:
    adapter = for_channel(channel)
    if adapter is None:
        db.mark_delivery(delivery_id, state="dead", error=f"no adapter for {channel}")
        return {"id": delivery_id, "state": "dead"}

    res = adapter.send(recipient, subject, body, {"alert": alert})
    if res.get("ok"):
        db.mark_delivery(delivery_id, state="sent",
                         provider_ref=res.get("provider_ref", ""))
        return {"id": delivery_id, "state": "sent"}

    # Exponential backoff. A gateway that is down stays down for a while, and
    # hammering it turns one outage into a queue nobody can drain.
    row = next((d for d in db.list_deliveries(alert["id"]) if d["id"] == delivery_id), None)
    attempts = int(row["attempts"]) + 1 if row else 1
    state = "dead" if attempts >= MAX_ATTEMPTS else "failed"
    nxt = (datetime.now(timezone.utc) + timedelta(minutes=2 ** attempts)).isoformat()
    db.mark_delivery(delivery_id, state=state, error=res.get("error", "send failed"),
                     next_retry_at=None if state == "dead" else nxt)
    return {"id": delivery_id, "state": state}


def retry_due(now: Optional[datetime] = None) -> int:
    """Re-attempt failed deliveries whose backoff has elapsed."""
    when = now or datetime.now(timezone.utc)
    due = db.deliveries_due(when.isoformat(), MAX_ATTEMPTS)
    for d in due:
        alert = db.get_alert(d["alert_id"])
        if alert is None:
            continue
        subject = f"[{alert['severity']}] {alert['headline']}"
        body = f"Cell {alert['cell_id']} - {alert['district']}, {alert['state']}."
        _attempt(d["id"], d["channel"], d["recipient"], subject, body, alert)
    if due:
        logger.info("[NOTIFY] retried %d delivery(ies)", len(due))
    return len(due)


def acknowledge(alert_id: str, actor: str, disposition: str) -> Optional[dict]:
    return db.acknowledge_alert(alert_id, actor, disposition)


def summary() -> dict:
    """Counts for the inbox header, including the false-positive rate.

    The false-positive rate is deliberately prominent. It is the number that
    tells an I4C desk whether this system is worth the officers it costs, and a
    tool that hides it is asking to be trusted rather than earning it.
    """
    outcomes = db.alert_outcome_counts()
    actioned = sum(outcomes.values())
    fp = outcomes.get("False positive", 0)
    return {
        "deliveries": db.delivery_state_counts(),
        "dispositions": outcomes,
        "actioned": actioned,
        "false_positive_rate": round(fp / actioned, 4) if actioned else 0.0,
    }
