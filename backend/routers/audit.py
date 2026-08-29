# -*- coding: utf-8 -*-
"""
MuleShield AI -- Audit trail
SIH26184 | MHA / I4C

GET /api/v1/audit        — recent activity across all cases
GET /api/v1/audit/{id}   — activity on one case

Every action that changes a case is recorded: status transitions, assignment,
notes, and account freezes. Before this the only trace a freeze left was a log
line and a write-only dict that no endpoint ever read back, which meant the
system could take an irreversible action against a person's account and then be
unable to say who had ordered it.

The trail is in-memory and does not survive a restart. That is a real
limitation, stated plainly rather than papered over: what it gives is a truthful
record of the running session, not a compliance-grade archive.
"""

import logging

from fastapi import APIRouter, Query

from backend.models.schemas import AuditEntry
import backend.state as state

logger = logging.getLogger("muleshield.audit")

router = APIRouter(prefix="/api/v1/audit", tags=["Audit"])


@router.get("", response_model=list[AuditEntry], summary="Recent activity, newest first")
async def list_audit(
    limit: int = Query(default=100, ge=1, le=1000),
    case_id: str = Query(default="", description="Restrict to one case"),
) -> list[AuditEntry]:
    return [AuditEntry(**e) for e in state.get_audit(case_id=case_id, limit=limit)]


@router.get("/{complaint_id}", response_model=list[AuditEntry],
            summary="Activity on one case, newest first")
async def case_audit(
    complaint_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[AuditEntry]:
    return [AuditEntry(**e) for e in state.get_audit(case_id=complaint_id, limit=limit)]
