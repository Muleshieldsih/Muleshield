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

Steps 1-3 moved to `backend/forecast.py` when the case dossier became a second
caller. The alternative was two implementations of one forecast, which would let
a printed dossier and this endpoint disagree about the top-ranked ATM without any
test noticing. This handler is now transport: HTTP mapping, the response model,
and the dashboard broadcast.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend import forecast
from backend.auth import current_user
from backend.models.schemas import ATMPrediction, PredictionResponse, SearchZone
from backend.websocket import manager

logger = logging.getLogger("muleshield.predict")

router = APIRouter(prefix="/api/v1/predict", tags=["Prediction"])


@router.get(
    "/cashout/{complaint_id}",
    response_model=PredictionResponse,
    summary="Ranked candidate cash-out locations + countdown",
)
async def predict_cashout(
    complaint_id: str,
    user: dict = Depends(current_user),
) -> PredictionResponse:
    """
    Runs the full XGBoost inference pipeline for a given complaint.

    Latency target: < 200ms (typical: ~15ms)
    """
    try:
        result = forecast.compute(complaint_id)
    except forecast.NoSuchCase:
        raise HTTPException(status_code=404,
                            detail=f"Complaint '{complaint_id}' not found.")
    except forecast.NoTransactions:
        raise HTTPException(status_code=422,
                            detail=f"No transaction data for complaint '{complaint_id}'.")

    enriched_atms = [ATMPrediction(**c) for c in result["ranked_candidates"]]
    zone_raw = result.get("search_zone")

    logger.info(
        f"[PREDICT] {complaint_id}: {result['terminal_account']} → "
        f"Top ATM: {enriched_atms[0].atm_id if enriched_atms else 'none'} "
        f"({result['interception_confidence']:.2%}), "
        f"{result['inference_time_ms']}ms"
    )

    response = PredictionResponse(
        complaint_id=complaint_id,
        search_zone=SearchZone(**zone_raw) if zone_raw else None,
        ranked_candidates=enriched_atms,
        time_to_cashout_minutes=result["time_to_cashout_minutes"],
        time_to_cashout_low=result.get("time_to_cashout_low"),
        time_to_cashout_high=result.get("time_to_cashout_high"),
        interception_confidence=result["interception_confidence"],
        inference_time_ms=result["inference_time_ms"],
        stolen_amount=result["stolen_amount"],
        terminal_account=result["terminal_account"],
        terminal_lat=result["terminal_lat"],
        terminal_lon=result["terminal_lon"],
    )

    # Broadcast PREDICTION_READY event to dashboard
    await manager.broadcast({
        "event_type": "PREDICTION_READY",
        "complaint_id": complaint_id,
        "payload": {
            "top_atm": enriched_atms[0].atm_id if enriched_atms else None,
            "confidence": result["interception_confidence"],
            "countdown_minutes": result["time_to_cashout_minutes"],
            "zone_radius_km": zone_raw.get("radius_km") if zone_raw else None,
        },
    })

    return response
