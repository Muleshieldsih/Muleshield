# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Prediction Router
SIH26184 | MHA / I4C

GET /api/v1/predict/cashout/{complaint_id}
  Forecasts where and when the cash will be withdrawn -- the primary ask of
  SIH26184:
    1. Find the terminal mule account
    2. Build the 80-dim hybrid feature vector (GNN + spatial tabular)
    3. Rank the reachable ATMs with the conditional-logit choice model
    4. Return the search ZONE (the deliverable), the ranked candidates inside it
       (tactical drill-down), and a countdown with a q05-q95 band
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.models.schemas import ATMPrediction, PredictionResponse, SearchZone
from backend.websocket import manager
import backend.state as state

logger = logging.getLogger("muleshield.predict")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "engine"))

router = APIRouter(prefix="/api/v1/predict", tags=["Prediction"])


@router.get(
    "/cashout/{complaint_id}",
    response_model=PredictionResponse,
    summary="Ranked candidate cash-out locations + countdown",
)
async def predict_cashout(complaint_id: str) -> PredictionResponse:
    """
    Runs the full XGBoost inference pipeline for a given complaint.

    Pipeline:
      1. Locate the terminal mule account from the transaction ledger
      2. Build the 80-dim feature vector (64-dim GNN embedding + 16 spatial)
      3. Rank the 25 reachable ATMs (conditional logit over log-space utilities)
      4. Collapse that distribution into a search zone, and estimate the delay

    Latency target: < 200ms (typical: ~15ms)
    """
    t0 = time.time()

    # ── Validate complaint ───────────────────────────────────────────────────
    complaint = state.get_complaint(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")

    # ── Find terminal mule account ───────────────────────────────────────────
    terminal_txns = state.get_terminal_accounts(complaint_id)
    if not terminal_txns:
        # Fallback: pick deepest hop account
        all_txns = state.get_transactions_for(complaint_id)
        if not all_txns:
            raise HTTPException(
                status_code=422,
                detail=f"No transaction data for complaint '{complaint_id}'."
            )
        terminal_txns = sorted(
            all_txns, key=lambda t: int(t.get("hop_depth", 0)), reverse=True
        )[:1]

    terminal_txn = terminal_txns[0]
    terminal_acc = str(terminal_txn.get("dst_account", ""))
    node_lat = float(terminal_txn.get("lat", 20.5937))
    node_lon = float(terminal_txn.get("long", 78.9629))
    stolen_amount = float(complaint.get("stolen_amount", 100000.0))

    # ── Build 80-dim feature vector (use cached FeatureBuilder) ─────────────
    fb = state.get_feature_builder()

    feature_vec = fb.build_feature_vector(
        terminal_account=terminal_acc,
        complaint_id=complaint_id,
        stolen_amount=stolen_amount,
    )

    # ── Rank the reachable ATMs, then aggregate into a search zone ───────────
    predictor = state.get_xgb_predictor()

    # The ranker scores the ATMs reachable from the terminal account, so it
    # needs the account itself: prior cashouts by the account's graph
    # neighbourhood are one of the candidate features.
    # Observed hop timing of THIS traced chain - median gap, fastest gap, span.
    # Available the moment the chain is traced; uses nothing after the transfer.
    import numpy as _np
    _ts = sorted(
        _np.datetime64(str(t.get("timestamp"))[:19])
        for t in state.get_transactions_for(complaint_id)
        if t.get("timestamp")
    )
    if len(_ts) >= 2:
        _gaps = _np.diff(_np.array(_ts)).astype("timedelta64[s]").astype(float)
        _gaps = _gaps[_gaps >= 0]
        chain_timing = (
            (float(_np.median(_gaps)), float(_gaps.min()),
             float((_ts[-1] - _ts[0]) / _np.timedelta64(1, "s")))
            if len(_gaps) else (0.0, 0.0, 0.0)
        )
    else:
        chain_timing = (0.0, 0.0, 0.0)

    node_rec = state.get_node_feature(terminal_acc) or {}
    raw_result = predictor.predict(
        feature_vec,
        node_lat=node_lat,
        node_lon=node_lon,
        node_bank=str(node_rec.get("bank_name", "UNKNOWN")),
        account=terminal_acc,
        chain_timing=chain_timing,
    )

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    logger.info(
        f"[PREDICT] {complaint_id}: {terminal_acc} → "
        f"Top ATM: {raw_result['ranked_candidates'][0]['atm_id']} "
        f"({raw_result['interception_confidence']:.2%}), "
        f"{elapsed_ms}ms"
    )

    # ── Enrich ATM predictions with directory metadata ────────────────────────
    enriched_atms: list[ATMPrediction] = []
    for pred in raw_result["ranked_candidates"]:
        atm_meta = state.get_atm(pred["atm_id"]) or {}
        enriched_atms.append(ATMPrediction(
            rank=pred["rank"],
            atm_id=pred["atm_id"],
            confidence=pred["confidence"],
            lat=float(atm_meta.get("lat", 20.5937)),
            lon=float(atm_meta.get("long", 78.9629)),
            bank=str(atm_meta.get("bank_name", atm_meta.get("bank", "Unknown"))),
            address=str(atm_meta.get("address", f"ATM {pred['atm_id']}")),
            historical_fraud_count=int(atm_meta.get("historical_fraud_count", 0)),
        ))

    zone_raw = raw_result.get("search_zone")
    response = PredictionResponse(
        complaint_id=complaint_id,
        search_zone=SearchZone(**zone_raw) if zone_raw else None,
        ranked_candidates=enriched_atms,
        time_to_cashout_minutes=raw_result["time_to_cashout_minutes"],
        time_to_cashout_low=raw_result.get("time_to_cashout_low"),
        time_to_cashout_high=raw_result.get("time_to_cashout_high"),
        interception_confidence=raw_result["interception_confidence"],
        inference_time_ms=elapsed_ms,
        stolen_amount=stolen_amount,
        terminal_account=terminal_acc,
        terminal_lat=node_lat,
        terminal_lon=node_lon,
    )

    # Broadcast PREDICTION_READY event to dashboard
    await manager.broadcast({
        "event_type": "PREDICTION_READY",
        "complaint_id": complaint_id,
        "payload": {
            "top_atm": enriched_atms[0].atm_id if enriched_atms else None,
            "confidence": raw_result["interception_confidence"],
            "countdown_minutes": raw_result["time_to_cashout_minutes"],
            "zone_radius_km": zone_raw.get("radius_km") if zone_raw else None,
        },
    })

    return response
