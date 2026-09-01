# -*- coding: utf-8 -*-
"""
MuleShield AI -- Police intelligence dossier router
SIH26184 | MHA / I4C

GET /api/v1/dossier/{case_id}  -- the printable case dossier

TIER A. A dossier is the most concentrated PII object this system produces: the
victim's name and account number, every mule account in the chain with its IFSC,
and the forward forecast of where a patrol is about to be sent. It is exactly
what `backend/auth.py` means by "forward operational intelligence closes".

WHY `def` AND NOT `async def`
-----------------------------
It reaches sqlite through the nested evidence certificate, so it must run in the
threadpool rather than on the event loop. Same rule as every other read-only
handler here; stated at backend/auth.py:65-78 and enforced by
backend/tests/test_evidence.py::TestConcurrencySeam.

It is also not cheap -- it runs the full forecast -- which is the second reason
it must not sit on the loop.
"""

import logging

from fastapi import APIRouter, Depends

from backend import dossier as dossier_mod
from backend.auth import current_user
from backend.models.schemas import CaseDossier
from backend.routers.evidence import _officer, _require_case
import backend.state as state

logger = logging.getLogger("muleshield.dossier")

router = APIRouter(prefix="/api/v1/dossier", tags=["Dossier"])


@router.get("/{case_id}", response_model=CaseDossier,
            summary="Printable police intelligence dossier for a case")
def case_dossier(case_id: str,
                 user: dict = Depends(current_user)) -> CaseDossier:
    """Assemble the dossier for one case.

    `_require_case` and `_officer` are imported from the evidence router rather
    than copied. They encode two decisions -- that a document must belong to a
    case that exists, and that a document names the login and not just the
    display name -- and a second copy would let those two decisions drift apart
    between the certificate and the dossier that embeds it.

    Production is audited. A document that names a victim and forecasts a patrol
    deployment should leave a record of who asked for it.
    """
    comp = _require_case(case_id)
    actor = _officer(user)
    state.record_audit(actor, "Produced case dossier", case_id, case_id=case_id)
    logger.info("[DOSSIER] %s produced by %s", case_id, actor)
    return CaseDossier(
        **dossier_mod.dossier(case_id, complaint=comp, produced_by=actor))
