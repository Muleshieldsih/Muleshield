# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 4: Cross-Case Intelligence Router
SIH26184 | MHA / I4C

GET /api/v1/intel/atms  -- ATMs ranked by how many distinct complaints touch them

Read-only, and deliberately open: it exposes no account numbers, no victim
details and no case identifiers, only aggregate counts over machines. It is also
the one endpoint in the system that answers a question about the network rather
than about a single case.
"""

import logging

from fastapi import APIRouter, Query

from backend.models.schemas import ATMIntelRow
import backend.state as state

logger = logging.getLogger("muleshield.intel")

router = APIRouter(prefix="/api/v1/intel", tags=["Intelligence"])


@router.get("/atms", response_model=list[ATMIntelRow],
            summary="Recurring cash-out ATMs, ranked across every case")
async def recurring_atms(
    limit: int = Query(default=25, ge=1, le=200,
                       description="How many machines to return."),
) -> list[ATMIntelRow]:
    return [ATMIntelRow(**row) for row in state.atm_intelligence(limit=limit)]
