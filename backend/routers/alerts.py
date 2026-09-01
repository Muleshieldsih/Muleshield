# -*- coding: utf-8 -*-
"""
MuleShield AI -- Alert inbox and dispatch router
SIH26184 | MHA / I4C

GET  /api/v1/alerts               -- the inbox, severity then recency
GET  /api/v1/alerts/summary       -- delivery + disposition counts
GET  /api/v1/alerts/recipients    -- who is in scope for what
GET  /api/v1/alerts/{id}          -- one alert with its delivery attempts
POST /api/v1/alerts/{id}/ack      -- close the loop, with a required disposition
POST /api/v1/alerts/evaluate      -- force a rule pass (administrator)

TIER A throughout. An alert names where officers are about to be sent and which
victims' cases are behind it.

THE SEAM, APPLIED
-----------------
Read-only handlers are plain `def`: Starlette runs them in a threadpool
automatically, so their sqlite calls never touch the event loop. The two
handlers that write AND broadcast are `async def` and cross with
run_in_threadpool. See backend/auth.py:46-59 and backend/notify.py.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from backend import db, notify
from backend.auth import actor_for, admin_user, current_user
from backend.models.schemas import (
    AlertAckRequest, AlertDetail, AlertOut, AlertSummary, RecipientOut,
    RuleRunSummary,
)
import backend.state as state
from backend.websocket import manager

logger = logging.getLogger("muleshield.alerts")

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])

VALID_DISPOSITIONS = ("Dispatched", "Monitoring", "False positive", "Duplicate")


@router.get("", response_model=list[AlertOut], summary="The alert inbox")
def list_alerts(
    status: str = Query(default="", description="open | acknowledged | dismissed"),
    severity: str = Query(default="", description="CRITICAL | HIGH | WATCH"),
    state_filter: str = Query(default="", alias="state"),
    limit: int = Query(default=100, ge=1, le=500),
    user: dict = Depends(current_user),
) -> list[AlertOut]:
    return [AlertOut(**a) for a in db.list_alerts(
        status=status.strip(), severity=severity.strip(),
        state=state_filter.strip(), limit=limit)]


@router.get("/summary", response_model=AlertSummary,
            summary="Delivery and disposition counts")
def alert_summary(user: dict = Depends(current_user)) -> AlertSummary:
    return AlertSummary(**notify.summary())


@router.get("/recipients", response_model=list[RecipientOut],
            summary="Who receives an alert, and for what ground")
def recipients(
    state_filter: str = Query(default="", alias="state"),
    district: str = Query(default=""),
    user: dict = Depends(current_user),
) -> list[RecipientOut]:
    return [RecipientOut(**r) for r in db.list_recipients(
        state=state_filter.strip(), district=district.strip())]


@router.get("/{alert_id}", response_model=AlertDetail,
            summary="One alert with every delivery attempt")
def get_alert(alert_id: str, user: dict = Depends(current_user)) -> AlertDetail:
    row = db.get_alert(alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return AlertDetail(**row, deliveries=db.list_deliveries(alert_id))


@router.post("/{alert_id}/ack", response_model=AlertOut,
             summary="Acknowledge an alert and record what was decided")
async def acknowledge(alert_id: str, request: AlertAckRequest,
                      user: dict = Depends(current_user)) -> AlertOut:
    """Close the loop.

    The disposition is REQUIRED, and "False positive" is a first-class value
    rather than a hidden one. Nothing in this system previously recorded whether
    an intervention was worth making (COMPLIANCE_AUDIT.md finding 2.5), which
    means it could never have improved and I4C could never have measured it.

    SYNC ACROSS A SEAM: notify.acknowledge writes sqlite. Running it inline on
    the event loop would stall /ws/feed for every client -- including the one
    about to receive the broadcast two lines below.
    """
    if request.disposition not in VALID_DISPOSITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"disposition must be one of {list(VALID_DISPOSITIONS)}")

    actor = actor_for(user, "", "OFFICER")
    row = await run_in_threadpool(notify.acknowledge, alert_id, actor,
                                  request.disposition)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")

    state.record_audit(actor, "Acknowledged alert",
                       f"{alert_id} ({request.disposition})",
                       case_id=(row.get("complaint_ids") or [""])[0])
    await manager.broadcast({
        "event_type": "ALERT_ACKNOWLEDGED",
        "complaint_id": "",
        "payload": {"alert_id": alert_id, "disposition": request.disposition,
                    "by": actor},
    })
    return AlertOut(**row)


@router.post("/evaluate", response_model=RuleRunSummary,
             summary="Force a rule pass over the current surface")
async def evaluate_now(
    as_of: str = Query(default="", description="ISO timestamp to evaluate at."),
    dry_run: bool = Query(default=False,
                          description="Report what would fire without persisting."),
    admin: dict = Depends(admin_user),
) -> RuleRunSummary:
    """Run the rules on demand.

    Administrator-only even though it is idempotent under the dedupe index: a
    rule pass dispatches to real recipients, and "who can cause an SMS to go to
    a district unit" is an authorisation question, not a convenience one.
    """
    from hotspot import parse_ts
    when = parse_ts(as_of) if as_of else None

    surface = await run_in_threadpool(state.hotspot_surface, as_of=when)
    raised = await run_in_threadpool(notify.evaluate, surface,
                                     as_of=when or datetime.now(timezone.utc),
                                     dry_run=dry_run)
    if not dry_run:
        for a in raised:
            await manager.broadcast({"event_type": "ALERT_RAISED",
                                     "complaint_id": "", "payload": a})
        if raised:
            state.record_audit(actor_for(admin, "", "ADMIN"), "Raised alert",
                               f"{len(raised)} alert(s) from a manual rule pass")

    return RuleRunSummary(
        as_of=surface.get("as_of", ""),
        degraded=bool(surface.get("degraded")),
        cells_considered=int(surface.get("n_cells", 0)),
        open_complaints=int(surface.get("n_open_complaints", 0)),
        raised=len(raised),
        dry_run=dry_run,
        alerts=[AlertOut(**a) for a in raised] if not dry_run else [],
    )
