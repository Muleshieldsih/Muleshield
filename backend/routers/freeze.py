# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Micro-Freeze Router
SIH26184 | MHA / I4C

POST /api/v1/bank/micro-freeze
  Simulates a one-click emergency card freeze on a mule account.
  In production this would call the bank's NACH/IMPS API.
  For the hackathon demo this returns an instant freeze confirmation.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter

from backend.models.schemas import FreezeRequest, FreezeResponse
from backend.websocket import manager
import backend.state as state

logger = logging.getLogger("muleshield.freeze")

router = APIRouter(prefix="/api/v1/bank", tags=["Freeze"])


@router.post(
    "/micro-freeze",
    response_model=FreezeResponse,
    summary="Emergency micro-freeze on a mule account",
)
async def micro_freeze(request: FreezeRequest) -> FreezeResponse:
    """
    Simulates an emergency card freeze on a flagged mule account.

    In production: triggers NPCI NACH debit block API via secure
    bank channel. For the hackathon this returns an instant freeze
    confirmation with a unique reference number.

    Response:
        { "status": "FROZEN", "account": "...", "timestamp": "..." }
    """
    freeze_ref = f"FRZ-{str(uuid.uuid4())[:8].upper()}"
    ts = datetime.now(timezone.utc).isoformat()

    # The console dispatches a freeze against an account; the issuing bank is
    # looked up from the account record so the caller never has to supply it.
    bank = request.bank
    if not bank or bank == "UNKNOWN":
        node = state.get_node_feature(request.account_id) or {}
        bank = str(node.get("bank_name", "UNKNOWN"))

    record = FreezeResponse(
        status="FROZEN",
        account=request.account_id,
        bank=bank,
        complaint_id=request.complaint_id,
        timestamp=ts,
        officer_id=request.officer_id,
        freeze_reference=freeze_ref,
    )

    state.log_freeze(record.model_dump())
    logger.info(
        f"[FREEZE] {freeze_ref}: Account {request.account_id} "
        f"({bank}) FROZEN by {request.officer_id}"
    )

    # Broadcast freeze event to all connected dashboards
    await manager.broadcast({
        "event_type": "FREEZE_EXECUTED",
        "complaint_id": request.complaint_id,
        "payload": {
            "freeze_reference": freeze_ref,
            "account": request.account_id,
            "bank": bank,
            "timestamp": ts,
            "officer_id": request.officer_id,
        },
    })

    return record
