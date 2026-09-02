# -*- coding: utf-8 -*-
"""
MuleShield AI -- Police intelligence dossier
SIH26184 | MHA / I4C

Deliverable (c) of the problem statement names three things an investigator must
be able to reach: *"alerts, intelligence reports, and evidence documentation"*.
Alerts have an inbox. Evidence has a custody chain and a s.63 certificate. The
middle one had nothing behind it: `COMPLAINT_AUDIT.md` finding 4.4,
`REMEDIATION_AUDIT.md` Sec 7 item 1 and `INDEPENDENT_AUDIT.md` Sec 7.1 all record
the same gap -- an alert carries a headline and a delivery trail, but there is no
case-level document an officer can put in front of a bank nodal officer or a
magistrate. This module is that document.

WHAT MAKES THIS A DOSSIER RATHER THAN A DATA DUMP
-------------------------------------------------
Four properties, and each one is a decision:

  1. **It is generated, never typed.** Every figure is read from the store or
     computed by the shipped model at the moment of production. It cannot
     describe a case the system does not hold, and it cannot quote a forecast
     the model did not make -- which is the same guarantee `evidence.certificate`
     gives, for the same reason.

  2. **It leads with the search zone, not the ranked list.** `SearchZone`'s own
     docstring records that the zone is the deliverable and the five candidates
     are the tactical drill-down. A dossier that opened with "ATM-8D19FE85" would
     invite an officer to read a ranked candidate as a prediction.

  3. **It carries its own caveats, in the body, not a footnote.** The advisory is
     lifted verbatim from `adapters/webhook.py` -- the same sentence that goes
     out on the wire to CFCFRMS. A document that leaves the building must say
     what it is not, and it must say it in the same words everywhere.

  4. **The money trail comes from the ledger, not the graph.** `TransactionRow`'s
     docstring warns that graph edges are de-duplicated by src->dst pair. A
     dossier is what a bank acts on, so it needs every hop with its own IFSC,
     not a de-duplicated visualisation of them.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not re-implement the forecast (`backend/forecast.py` owns that, and both
this and the prediction endpoint call it, so a printed dossier and the console
cannot disagree) and it does not re-implement the evidence certificate
(`backend/evidence.py` owns that, and the dossier nests it whole).

A case with no evidence artefacts still produces a dossier: the nested
certificate reports `artefact_count: 0`, which is the honest statement that
nothing has been collected yet -- not an error, and not a blank section.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from backend import evidence, forecast
import backend.state as state

logger = logging.getLogger("muleshield.dossier")


# ── The caveats block ────────────────────────────────────────────────────────
#
# Lifted verbatim from adapters/webhook.py. One sentence, one wording, whether it
# goes to a printer or to a webhook. If that sentence is ever softened, it must
# be softened in both places, and a test asserts it appears here.
ADVISORY = (
    "Ranked forecast, not a confirmed location. Deploy to observe; do not "
    "treat presence in this cell as grounds for detention."
)

CAVEATS = (
    ADVISORY,
    "The locations below are prioritised search candidates, not a predicted ATM. "
    "The system narrows a 1,000-machine national directory to five; it does not "
    "identify the machine.",
    "The countdown is an estimate with a q05-q95 band, not a deadline. Published "
    "mean absolute error is approximately 12 minutes.",
    "Mule scores are model output on a population where labelling is imperfect. "
    "They are grounds to investigate, never grounds alone to act against a "
    "person or an account.",
)


def _fmt_rupees(v: float) -> str:
    """Indian-notation rupee string. Mirrors notify._rupees so a dossier and an
    alert describe the same amount with the same words."""
    v = float(v or 0)
    if v >= 1e7:
        return f"Rs {v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"Rs {v / 1e5:.2f} L"
    return f"Rs {v:,.0f}"


def _money_trail(complaint_id: str) -> list[dict]:
    """Every hop, in time order, with the bank details a nodal officer acts on.

    From the ledger via state.get_transactions_for -- NOT from graph edges, which
    are de-duplicated by src->dst pair for the visualisation and would silently
    drop a repeated transfer between the same two accounts.
    """
    txns = state.get_transactions_for(complaint_id) or []
    rows = []
    for t in sorted(txns, key=lambda x: str(x.get("timestamp", ""))):
        rows.append({
            "txn_id": str(t.get("txn_id", "") or ""),
            "timestamp": str(t.get("timestamp", "") or ""),
            "src_account": str(t.get("src_account", "") or ""),
            "dst_account": str(t.get("dst_account", "") or ""),
            "amount": float(t.get("amount", 0) or 0),
            "amount_text": _fmt_rupees(t.get("amount", 0)),
            "bank_name": str(t.get("bank_name", "") or ""),
            "ifsc_code": str(t.get("ifsc_code", "") or ""),
            "city": str(t.get("city", "") or ""),
            "state": str(t.get("state", "") or ""),
            "hop_depth": int(t.get("hop_depth", 0) or 0),
            "is_terminal": bool(t.get("is_terminal", 0)),
        })
    return rows


def _detections(complaint_id: str) -> dict:
    """Velocity and fund-splitting detections over this case's sub-graph.

    Calls the same detector the graph endpoint uses, so the dossier and the
    Forensic Graph screen flag the same accounts under the same thresholds. The
    two rule strings are carried through because they are already written for a
    human and belong on a printed page unchanged.
    """
    from backend.routers.graph import _detect_anomalies
    from graph_engine import (SPLIT_AMOUNT_TOLERANCE, SPLIT_DESTINATIONS,
                              VELOCITY_THRESHOLD, VELOCITY_WINDOW_SECONDS)

    txns = state.get_transactions_for(complaint_id) or []
    velocity, splitting, terminals = _detect_anomalies(txns)
    return {
        "velocity_flagged": velocity,
        "fund_split_flagged": splitting,
        "terminal_leaves": terminals,
        "velocity_count": len(velocity),
        "fund_split_count": len(splitting),
        "terminal_count": len(terminals),
        "velocity_rule": (f">{VELOCITY_THRESHOLD} outgoing txns in "
                          f"{VELOCITY_WINDOW_SECONDS // 60}m"),
        "fund_split_rule": (f"1-to-{SPLIT_DESTINATIONS}+ within "
                            f"{int(SPLIT_AMOUNT_TOLERANCE * 100)}% of mean"),
    }


def _forecast_block(complaint_id: str) -> Optional[dict]:
    """The cash-out forecast, or None when the case cannot be traced.

    A case whose bank feed has not landed yet has no chain, so no terminal
    account, so no forecast. That is a real state an officer can be looking at,
    and the dossier prints the rest of the case rather than refusing to render.
    """
    try:
        f = forecast.compute(complaint_id)
    except forecast.NoTransactions:
        logger.info("[DOSSIER] %s has no ledger; dossier omits the forecast.",
                    complaint_id)
        return None

    zone = f.get("search_zone") or None
    return {
        "search_zone": zone,
        "terminal_account": f["terminal_account"],
        "terminal_lat": f["terminal_lat"],
        "terminal_lon": f["terminal_lon"],
        "ranked_candidates": f["ranked_candidates"],
        "time_to_cashout_minutes": f["time_to_cashout_minutes"],
        "time_to_cashout_low": f.get("time_to_cashout_low"),
        "time_to_cashout_high": f.get("time_to_cashout_high"),
        "interception_confidence": f["interception_confidence"],
        "inference_time_ms": f["inference_time_ms"],
    }


def _actions(case_id: str) -> list[dict]:
    """What has been done on this case, from the audit trail.

    Deliberately the audit trail and not state.freeze_log: the freeze log is
    write-only with no reader (state.py:69, 505-506), so the only retrievable
    record that an account was frozen is the audit row written by the freeze
    router. Oldest first here -- a dossier is read as a narrative, while the
    console's activity feed is read newest-first.
    """
    rows = state.get_audit(case_id=case_id, limit=500)
    return [{
        "id": str(r.get("id", "") or ""),
        "timestamp": str(r.get("timestamp", "") or ""),
        "actor": str(r.get("actor", "") or ""),
        "action": str(r.get("action", "") or ""),
        "object": str(r.get("object", "") or ""),
        "result": str(r.get("result", "") or ""),
    } for r in reversed(rows)]


def dossier(case_id: str, *, complaint: dict, produced_by: str = "") -> dict:
    """Assemble the police intelligence dossier for one case.

    Shaped deliberately like `evidence.certificate()`: a document assembled from
    the store at the moment it is asked for, carrying live model output and a
    live evidence verification, so it cannot describe a state the system is not
    actually in.
    """
    trail = _money_trail(case_id)
    fc = _forecast_block(case_id)

    return {
        "case_id": case_id,
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produced_by": produced_by,
        "system": {
            "name": "MuleShield AI",
            "operator": "Indian Cyber Crime Coordination Centre (I4C)",
            "problem_statement": "SIH26184",
        },

        # ── The complaint, as reported on the 1930 rail ──────────────────────
        "complaint": {
            "ticket_id": case_id,
            "victim_name": str(complaint.get("victim_name", "") or ""),
            "victim_bank": str(complaint.get("victim_bank", "") or ""),
            "victim_account": str(complaint.get("victim_account", "") or ""),
            "fraud_type": str(complaint.get("fraud_type", "") or ""),
            "stolen_amount": float(complaint.get("stolen_amount", 0) or 0),
            "stolen_amount_text": _fmt_rupees(complaint.get("stolen_amount", 0)),
            "city": str(complaint.get("city", "") or ""),
            "state": str(complaint.get("state", "") or ""),
            "complaint_timestamp": str(complaint.get("complaint_timestamp", "") or ""),
            "status": state.case_status(case_id),
            "assignee": str(complaint.get("assignee", "") or ""),
        },

        # ── The money, hop by hop ────────────────────────────────────────────
        "money_trail": trail,
        "money_trail_count": len(trail),
        "banks_involved": sorted({r["bank_name"] for r in trail if r["bank_name"]}),

        # ── What the graph engine flagged ────────────────────────────────────
        "detections": _detections(case_id),

        # ── Where and when the cash is expected to surface ───────────────────
        "forecast": fc,

        # ── What has been done, and by whom ──────────────────────────────────
        "actions": _actions(case_id),

        # ── The custody chain, whole and unmodified ──────────────────────────
        "evidence": evidence.certificate(case_id, complaint=complaint,
                                         produced_by=produced_by),

        # ── What this document is not ────────────────────────────────────────
        "caveats": list(CAVEATS),
    }
