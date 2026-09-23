# -*- coding: utf-8 -*-
"""
MuleShield AI -- Forward hotspot surface router
SIH26184 | MHA / I4C

GET /api/v1/hotspots/cells  -- forward cash-out intensity per cell and window
"""


import logging
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from backend.auth import current_user
from backend.models.schemas import HotspotSurfaceResponse
import backend.state as state

_ENGINE = str(Path(__file__).resolve().parents[2] / "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from hotspot import parse_ts  # noqa: E402

logger = logging.getLogger("muleshield.hotspot")

router = APIRouter(prefix="/api/v1/hotspots", tags=["Hotspots"])


@router.get("/cells", response_model=HotspotSurfaceResponse,
            summary="Forward cash-out intensity by cell and time window")
async def hotspot_cells(
    window_start_min: int = Query(default=0, ge=0, le=120,
                                  description="Earliest forecast window bound."),
    window_end_min: int = Query(default=120, ge=30, le=120,
                                description="Latest forecast window bound."),
    fraud_type: str = Query(default="", description="Restrict to these crime categories, comma-separated."),
    state_filter: str = Query(default="", alias="state",
                              description="Restrict to complaints from one state."),
    as_of: str = Query(default="",
                       description="ISO timestamp to forecast from. Defaults to now. "
                                   "The response echoes the value actually used."),
    open_minutes: int = Query(default=120, ge=15, le=1440,
                              description="How far back a complaint still counts as open."),
    user: dict = Depends(current_user),
) -> HotspotSurfaceResponse:
    """Aggregate every open complaint's own forecast into a national surface.

    Nothing here is a new model. Each complaint already carries a posterior over
    its reachable ATMs and a countdown with a q05-q95 band, both produced by the
    frozen checkpoint; this projects them onto cells and sums them, weighted by
    the rupees at risk. The historical prior is one capped term, and its
    contribution is reported separately on every cell so a reader can see which
    half is carrying the forecast.

    `as_of` is honest rather than cosmetic: the evaluation script replays
    historical epochs through this same code path, and a demo against a corpus
    generated days ago has to point the surface at a time when complaints were
    actually arriving. The response always echoes the timestamp it used.
    """
    when = parse_ts(as_of) if as_of else None
    surface = state.hotspot_surface(
        window_start=window_start_min,
        window_end=window_end_min,
        fraud_type=fraud_type.strip(),
        state_filter=state_filter.strip(),
        as_of=when,
        open_minutes=open_minutes,
    )
    return HotspotSurfaceResponse(**surface)
