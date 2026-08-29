# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Complaint Router
SIH26184 | MHA / I4C

POST /api/v1/complaint/ingest  — Ingest a new 1930 complaint
GET  /api/v1/complaint/list    — List all active complaints (newest first)
GET  /api/v1/complaint/{id}    — Get a single complaint record
PATCH /api/v1/complaint/{id}   — Move a case through the workflow
POST /api/v1/complaint/{id}/note          — Attach an investigation note
GET  /api/v1/complaint/{id}/notes         — Read the notes on a case
GET  /api/v1/complaint/{id}/transactions  — The money trail, every row
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import (
    ComplaintIngestRequest, ComplaintResponse, CaseUpdateRequest,
    NoteRequest, NoteResponse, TransactionRow,
)
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

    state.record_audit("SYSTEM", "Case created", record["ticket_id"],
                       case_id=record["ticket_id"])
    return ComplaintResponse(**_decorate(record))


@router.get(
    "/list",
    response_model=list[ComplaintResponse],
    summary="List active complaints (newest first)",
)
async def list_complaints(
    limit: int = Query(default=50, ge=1, le=2500, description="Max complaints to return"),
    offset: int = Query(default=0, ge=0, description="Number of complaints to skip"),
) -> list[ComplaintResponse]:
    """
    Returns complaints sorted by timestamp descending.

    Paginated: the national dataset holds thousands of tickets, and the triage
    console only ever renders the live head of the queue. Callers that genuinely
    need everything can page through with `offset`.
    """
    window = state.get_all_complaints()[offset: offset + limit]
    return [ComplaintResponse(**_decorate(c)) for c in window]


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
    return ComplaintResponse(**_decorate(record))


def _decorate(record: dict) -> dict:
    """
    Present a stored complaint as the API describes it.

    Seed records predate the workflow fields and carry the legacy status
    "ACTIVE", which is not one of the six workflow states. Normalising here
    rather than rewriting 2,500 rows on load keeps the CSV as the source of
    truth for what was reported, and the workflow as a layer over it.
    """
    out = dict(record)
    out["status"] = state.case_status(record["ticket_id"])
    out["note_count"] = len(state.case_notes.get(record["ticket_id"], []))
    return out


@router.patch(
    "/{complaint_id}",
    response_model=ComplaintResponse,
    summary="Update a case's status or assignment",
)
async def update_case(complaint_id: str, payload: CaseUpdateRequest) -> ComplaintResponse:
    """
    Move a case through the workflow. Every change is written to the audit trail
    with the actor who made it.
    """
    if not state.get_complaint(complaint_id):
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")
    try:
        record = state.update_case(
            complaint_id,
            status=payload.status,
            assignee=payload.assignee,
            actor=payload.actor,
        )
    except ValueError as e:
        # A bad status is the caller's mistake, and naming the valid set is more
        # use than "invalid input".
        raise HTTPException(
            status_code=422,
            detail=f"{e}. Valid statuses: {', '.join(state.CASE_STATUSES)}",
        ) from e

    await manager.broadcast({
        "event_type": "CASE_UPDATED",
        "complaint_id": complaint_id,
        "payload": {"status": record.get("status"), "assignee": record.get("assignee")},
    })
    return ComplaintResponse(**_decorate(record))


@router.post(
    "/{complaint_id}/note",
    response_model=NoteResponse,
    summary="Attach an investigation note to a case",
)
async def create_note(complaint_id: str, payload: NoteRequest) -> NoteResponse:
    note = state.add_note(complaint_id, payload.text.strip(), payload.author)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")
    return NoteResponse(**note)


@router.get(
    "/{complaint_id}/notes",
    response_model=list[NoteResponse],
    summary="Notes on a case, newest first",
)
async def list_notes(complaint_id: str) -> list[NoteResponse]:
    if not state.get_complaint(complaint_id):
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")
    return [NoteResponse(**n) for n in state.get_notes(complaint_id)]


@router.get(
    "/{complaint_id}/transactions",
    response_model=list[TransactionRow],
    summary="The money trail for a case, in order",
)
async def list_transactions(complaint_id: str) -> list[TransactionRow]:
    """
    Every transfer on the case, oldest first.

    The graph endpoint also returns edges, but it collapses them by
    source-destination pair -- two transfers between the same accounts become
    one and the second is lost. An investigator reading a ledger cannot have
    rows quietly disappear, so this returns them all.
    """
    if not state.get_complaint(complaint_id):
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")

    rows = state.get_transactions_for(complaint_id)
    if not rows:
        return []

    def ts(r):
        return str(r.get("timestamp", ""))

    ordered = sorted(rows, key=ts)
    first = _parse(ts(ordered[0])) if ordered else None

    out: list[TransactionRow] = []
    for r in ordered:
        when = _parse(ts(r))
        delta = 0.0
        if first is not None and when is not None:
            delta = round((when - first).total_seconds() / 60.0, 1)

        # Seed rows come from pandas, so absent values arrive as float('nan')
        # rather than None and are not JSON-serialisable.
        atm = r.get("cashout_atm_id")
        if atm is None or (isinstance(atm, float) and atm != atm) or str(atm) == "nan":
            atm = None

        out.append(TransactionRow(
            txn_id=str(r.get("txn_id", "")),
            timestamp=ts(r),
            src_account=str(r.get("src_account", "")),
            dst_account=str(r.get("dst_account", "")),
            amount=float(r.get("amount", 0) or 0),
            bank_name=str(r.get("bank_name", "Unknown")),
            ifsc_code=str(r.get("ifsc_code", "") or ""),
            city=str(r.get("city", "") or ""),
            state=str(r.get("state", "") or ""),
            channel=str(r.get("txn_purpose", "") or "TRANSFER"),
            hop_depth=int(r.get("hop_depth", 0) or 0),
            is_terminal=bool(int(r.get("is_terminal", 0) or 0)),
            cashout_atm_id=str(atm) if atm else None,
            minutes_from_first=delta,
        ))
    return out


def _parse(value: str):
    """Tolerant ISO parse — seed timestamps are naive, live ones carry an offset."""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None
