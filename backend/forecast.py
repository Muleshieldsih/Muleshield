# -*- coding: utf-8 -*-
"""
MuleShield AI -- the one cash-out forecast, computed once.
SIH26184 | MHA / I4C

WHY THIS MODULE EXISTS
----------------------
The forecast used to live inline in `backend/routers/predict.py`. That was fine
while exactly one caller needed it. The case dossier needs the same forecast, and
copying sixty lines of feature building into a second file is precisely the defect
this codebase has already written comments about twice:

    "it would be a SECOND implementation of one quantity ... the two could drift
     apart without any test noticing -- which is the shape of every leakage defect
     this project has already had to retract a number for."
        -- backend/state.py, on posterior_from_scores

A dossier is a document an officer signs and hands to a bank. If it printed a
different top-ranked ATM than the Interception screen showed five seconds
earlier, the divergence would be discovered in front of the person we were trying
to convince. So both paths call `compute()` and there is nothing to diverge.

WHAT STAYED IN THE ROUTER
-------------------------
The WebSocket broadcast and the Pydantic response construction. This module is
deliberately sync and returns plain dicts: it is called from an `async` route
handler (predict) and from a threadpool `def` handler (dossier), and a coroutine
would be wrong in the second case. See backend/auth.py:65-78 for the rule.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

import backend.state as state

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

logger = logging.getLogger("muleshield.forecast")


class NoSuchCase(LookupError):
    """The complaint id does not exist. Callers map this to 404."""


class NoTransactions(LookupError):
    """The complaint exists but carries no ledger, so nothing can be traced.

    Distinct from NoSuchCase because the two mean different things to an
    officer: one is a typo, the other is a case whose bank feed has not landed.
    Callers map this to 422.
    """


def chain_timing_for(complaint_id: str) -> tuple[float, float, float]:
    """Observed hop timing of this traced chain: median gap, fastest gap, span.

    Available the moment the chain is traced, and uses nothing after the terminal
    transfer -- which is what makes it legitimate at prediction time. Lifted
    verbatim from the prediction router so the ranker sees identical inputs
    whichever caller asked.
    """
    ts = sorted(
        np.datetime64(str(t.get("timestamp"))[:19])
        for t in state.get_transactions_for(complaint_id)
        if t.get("timestamp")
    )
    if len(ts) < 2:
        return (0.0, 0.0, 0.0)
    gaps = np.diff(np.array(ts)).astype("timedelta64[s]").astype(float)
    gaps = gaps[gaps >= 0]
    if not len(gaps):
        return (0.0, 0.0, 0.0)
    return (
        float(np.median(gaps)),
        float(gaps.min()),
        float((ts[-1] - ts[0]) / np.timedelta64(1, "s")),
    )


def compute(complaint_id: str) -> dict:
    """Run the full cash-out forecast for one complaint.

    Returns plain dicts, ATM candidates already enriched from the directory:

        {
          "complaint_id", "search_zone" | None, "ranked_candidates": [...],
          "time_to_cashout_minutes", "time_to_cashout_low", "time_to_cashout_high",
          "interception_confidence", "inference_time_ms", "stolen_amount",
          "terminal_account", "terminal_lat", "terminal_lon",
        }

    Raises NoSuchCase / NoTransactions rather than HTTPException: this module
    knows nothing about HTTP, and the dossier assembler wants to catch the second
    one and print a case without a forecast rather than fail the whole document.
    """
    t0 = time.time()

    complaint = state.get_complaint(complaint_id)
    if not complaint:
        raise NoSuchCase(complaint_id)

    # ── Find the terminal mule account ───────────────────────────────────────
    terminal_txns = state.get_terminal_accounts(complaint_id)
    if not terminal_txns:
        # Fallback: the deepest hop is the closest thing to a terminal we have.
        all_txns = state.get_transactions_for(complaint_id)
        if not all_txns:
            raise NoTransactions(complaint_id)
        terminal_txns = sorted(
            all_txns, key=lambda t: int(t.get("hop_depth", 0)), reverse=True
        )[:1]

    terminal_txn = terminal_txns[0]
    terminal_acc = str(terminal_txn.get("dst_account", ""))
    node_lat = float(terminal_txn.get("lat", 20.5937))
    node_lon = float(terminal_txn.get("long", 78.9629))
    stolen_amount = float(complaint.get("stolen_amount", 100000.0))

    # ── 80-dim hybrid vector (64-dim GNN embedding + 16 spatial/temporal) ────
    fb = state.get_feature_builder()
    feature_vec = fb.build_feature_vector(
        terminal_account=terminal_acc,
        complaint_id=complaint_id,
        stolen_amount=stolen_amount,
    )

    # ── Rank the reachable ATMs, then collapse them into a search zone ───────
    # The ranker needs the account itself: prior cashouts by the account's graph
    # neighbourhood are one of the candidate features.
    predictor = state.get_xgb_predictor()
    node_rec = state.get_node_feature(terminal_acc) or {}
    raw = predictor.predict(
        feature_vec,
        node_lat=node_lat,
        node_lon=node_lon,
        node_bank=str(node_rec.get("bank_name", "UNKNOWN")),
        account=terminal_acc,
        chain_timing=chain_timing_for(complaint_id),
    )

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    # ── Enrich the candidates with ATM directory metadata ────────────────────
    enriched = []
    for pred in raw["ranked_candidates"]:
        meta = state.get_atm(pred["atm_id"]) or {}
        enriched.append({
            "rank": pred["rank"],
            "atm_id": pred["atm_id"],
            "confidence": pred["confidence"],
            "lat": float(meta.get("lat", 20.5937)),
            "lon": float(meta.get("long", 78.9629)),
            "bank": str(meta.get("bank_name", meta.get("bank", "Unknown"))),
            "address": str(meta.get("address", f"ATM {pred['atm_id']}")),
            "historical_fraud_count": int(meta.get("historical_fraud_count", 0)),
            "city": str(meta.get("city", "") or ""),
            "district": str(meta.get("district", "") or ""),
            "state": str(meta.get("state", "") or ""),
            "opening_time": str(meta.get("opening_time", "") or ""),
            "closing_time": str(meta.get("closing_time", "") or ""),
            "cashout_risk_score": float(meta.get("cashout_risk_score", 0) or 0),
        })

    return {
        "complaint_id": complaint_id,
        "search_zone": raw.get("search_zone"),
        "ranked_candidates": enriched,
        "time_to_cashout_minutes": raw["time_to_cashout_minutes"],
        "time_to_cashout_low": raw.get("time_to_cashout_low"),
        "time_to_cashout_high": raw.get("time_to_cashout_high"),
        "interception_confidence": raw["interception_confidence"],
        "inference_time_ms": elapsed_ms,
        "stolen_amount": stolen_amount,
        "terminal_account": terminal_acc,
        "terminal_lat": node_lat,
        "terminal_lon": node_lon,
    }
