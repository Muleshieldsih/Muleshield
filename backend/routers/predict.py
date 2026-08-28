# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Prediction Router
SIH26184 | MHA / I4C

GET /api/v1/predict/cashout/{complaint_id}
  Runs the full XGBoost inference pipeline for a complaint:
    1. Find the terminal mule account
    2. Build the 80-dim hybrid feature vector (GNN + spatial tabular)
    3. Run MuleXGBPredictor.predict() with Bayesian spatial reranking
    4. Return Top-3 ATMs + countdown + interception confidence
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.models.schemas import ATMPrediction, PredictionResponse
from backend.websocket import manager
import backend.state as state

logger = logging.getLogger("muleshield.predict")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "engine"))

router = APIRouter(prefix="/api/v1/predict", tags=["Prediction"])


@router.get(
    "/cashout/{complaint_id}",
    response_model=PredictionResponse,
    summary="XGBoost Top-3 ATM prediction + cashout countdown",
)
async def predict_cashout(complaint_id: str) -> PredictionResponse:
    """
    Runs the full XGBoost inference pipeline for a given complaint.

    Pipeline:
      1. Locate terminal mule account from transaction ledger
      2. Build 80-dim feature vector (64-dim GNN embedding + 16 spatial tabular features)
      3. Apply Bayesian Gaussian spatial prior reranking
      4. Return Top-3 ATMs with confidence, countdown, and GPS coordinates

    Latency target: < 200ms (typical: ~26ms)
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

    # ── XGBoost inference with Bayesian spatial reranking ────────────────────
    predictor = state.get_xgb_predictor()

    # The ranker scores the ATMs reachable from the terminal account, so it
    # needs the account itself: prior cashouts by the account's graph
    # neighbourhood are one of the candidate features.
    node_rec = state.get_node_feature(terminal_acc) or {}
    raw_result = predictor.predict(
        feature_vec,
        node_lat=node_lat,
        node_lon=node_lon,
        node_bank=str(node_rec.get("bank_name", "UNKNOWN")),
        account=terminal_acc,
    )

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    logger.info(
        f"[PREDICT] {complaint_id}: {terminal_acc} → "
        f"Top ATM: {raw_result['top3_atms'][0]['atm_id']} "
        f"({raw_result['interception_confidence']:.2%}), "
        f"{elapsed_ms}ms"
    )

    # ── Enrich ATM predictions with directory metadata ────────────────────────
    enriched_atms: list[ATMPrediction] = []
    for pred in raw_result["top3_atms"]:
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

    response = PredictionResponse(
        complaint_id=complaint_id,
        top3_atms=enriched_atms,
        time_to_cashout_minutes=raw_result["time_to_cashout_minutes"],
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
        },
    })

    return response
