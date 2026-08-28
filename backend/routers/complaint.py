# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Complaint Router
SIH26184 | MHA / I4C

POST /api/v1/complaint/ingest  — Ingest a new 1930 complaint
GET  /api/v1/complaint/list    — List all active complaints (newest first)
GET  /api/v1/complaint/{id}    — Get a single complaint record
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.models.schemas import ComplaintIngestRequest, ComplaintResponse
from backend.websocket import manager
import backend.state as state

logger = logging.getLogger("muleshield.complaint")

router = APIRouter(prefix="/api/v1/complaint", tags=["Complaints"])


@router.post(
    "/ingest",
    response_model=ComplaintResponse,
    summary="Ingest a new 1930 cybercrime complaint",
)
async def ingest_complaint(request: ComplaintIngestRequest) -> ComplaintResponse:
    """
    Accepts a new 1930 complaint, stores it in the in-memory state,
    and immediately broadcasts a `NEW_COMPLAINT` event to all
    connected WebSocket dashboard clients.
    """
    data = request.model_dump()
    if not data.get("complaint_timestamp"):
        data["complaint_timestamp"] = datetime.now(timezone.utc).isoformat()

    record = state.add_complaint(data)
    logger.info(f"[COMPLAINT] Ingested: {record['ticket_id']} | ₹{record['stolen_amount']:,.0f} | {record['city']}")

    # Broadcast to all connected dashboards
    await manager.broadcast({
        "event_type": "NEW_COMPLAINT",
        "complaint_id": record["ticket_id"],
        "payload": record,
    })

    return ComplaintResponse(**record)


@router.get(
    "/list",
    response_model=list[ComplaintResponse],
    summary="List all active complaints (newest first)",
)
async def list_complaints() -> list[ComplaintResponse]:
    """Returns all complaints sorted by timestamp descending."""
    return [ComplaintResponse(**c) for c in state.get_all_complaints()]


@router.get(
    "/{complaint_id}",
    response_model=ComplaintResponse,
    summary="Get a specific complaint by ticket ID",
)
async def get_complaint(complaint_id: str) -> ComplaintResponse:
    """Returns a single complaint record."""
    record = state.get_complaint(complaint_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")
    return ComplaintResponse(**record)
