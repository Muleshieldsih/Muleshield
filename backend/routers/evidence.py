# -*- coding: utf-8 -*-
"""
MuleShield AI -- Evidence documentation router
SIH26184 | MHA / I4C

POST /api/v1/evidence/{case_id}              -- collect an artefact (multipart)
GET  /api/v1/evidence/case/{case_id}         -- the case's chain of custody
GET  /api/v1/evidence/case/{case_id}/verify  -- re-hash every artefact and the chain
GET  /api/v1/evidence/case/{case_id}/certificate -- BSA 2023 s.63 certificate
GET  /api/v1/evidence/{id}                   -- one artefact's record
GET  /api/v1/evidence/{id}/download          -- the bytes, hash-checked first
POST /api/v1/evidence/{id}/withdraw          -- withdraw, with a reason. No delete.

TIER A throughout. Evidence is the most sensitive thing in this system: it is
victim material, it is bank material, and it is what a prosecution rests on.

THE SEAM, APPLIED
-----------------
Read-only handlers are plain `def` -- Starlette runs them in a threadpool, so
their sqlite calls never touch the event loop. The upload handler must be
`async def` because reading an UploadFile is awaitable, so it crosses to
sqlite and to the disk write with run_in_threadpool. Same rule as
backend/routers/alerts.py; see backend/auth.py:46-59.

TWO THINGS THAT LOOK LIKE PARANOIA AND ARE NOT
----------------------------------------------
**Downloads are always `application/octet-stream` with an attachment
disposition**, never the content type the uploader declared. Serving a file
back under a caller-supplied type on the same origin as the console is how an
uploaded .html or .svg becomes stored XSS against the next officer who opens
the case. The declared type is kept in the record because it is part of the
custody description; it is not used to render anything.

**A hash mismatch is a 409, not a download.** If the bytes on disk no longer
match what was recorded at collection, the honest answer is that the artefact
is no longer evidence -- handing it over anyway would let it be produced in
court as though nothing had happened.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend import db, evidence
from backend.auth import current_user
from backend.models.schemas import (
    EvidenceCertificate, EvidenceItem, EvidenceVerification,
    EvidenceWithdrawRequest,
)
import backend.state as state

logger = logging.getLogger("muleshield.evidence")

router = APIRouter(prefix="/api/v1/evidence", tags=["Evidence"])


def _require_case(case_id: str) -> dict:
    """An artefact must belong to a case that exists.

    Without this, a typo in a ticket id silently creates a custody chain nobody
    will ever find, and the artefact is effectively lost at the moment it is
    collected.
    """
    comp = state.get_complaint(case_id)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    return comp


def _officer(user: dict) -> str:
    """Who to name on a custody record.

    Deliberately NOT auth.actor_for(), which returns the display name and falls
    back to a claimed string. A custody record needs the account as well as the
    label: "Duty Officer" identifies nobody on a roster of duty officers, and a
    certificate that has to be adopted and signed needs to say which login
    collected the artefact.
    """
    display = str(user.get("display_name") or "").strip()
    username = str(user.get("username") or "").strip()
    if display and username and display.lower() != username.lower():
        return f"{display} ({username})"
    return display or username or "UNKNOWN"


def _out(item: dict) -> EvidenceItem:
    return EvidenceItem(**{k: v for k, v in item.items()
                           if k in EvidenceItem.model_fields})


@router.post("/{case_id}", response_model=EvidenceItem, status_code=201,
             summary="Collect an artefact against a case")
async def collect(
    case_id: str,
    file: UploadFile = File(..., description="The artefact itself"),
    kind: str = Form(default="other"),
    description: str = Form(default=""),
    source: str = Form(default=""),
    user: dict = Depends(current_user),
) -> EvidenceItem:
    """Take custody of one artefact.

    The collecting officer is taken from the bearer token, never from the body.
    Who collected a piece of evidence is the one field a custody record cannot
    let the caller choose.
    """
    comp = _require_case(case_id)
    if kind not in evidence.KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"kind must be one of {', '.join(evidence.KINDS)}")

    actor = _officer(user)
    name = evidence.display_name(file.filename or "")

    # Read the upload here, on the loop, because UploadFile is awaitable; hash
    # and write on a worker, because both block.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(evidence.CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > evidence.MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"artefact exceeds {evidence.MAX_BYTES // (1024 * 1024)} MB")
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(status_code=422, detail="artefact is empty")

    def _persist() -> dict:
        digest, size, staged = evidence.stage(case_id, chunks)

        existing = db.find_evidence_by_digest(case_id, digest)
        if existing is not None:
            # Same bytes, same case. One object, one custody record -- minting a
            # second would make the chain describe a history that did not happen.
            evidence.discard(staged)
            return existing

        row = db.insert_evidence({
            "case_id": case_id, "filename": name,
            "content_type": str(file.content_type or ""), "size_bytes": size,
            "sha256": digest, "kind": kind, "description": description[:1000],
            "source": source[:200], "collected_by": actor,
        })
        try:
            evidence.commit(staged, case_id, row["id"])
        except OSError:
            # The row landed and the bytes did not. Unwind it, or the chain
            # claims custody of something the store does not hold.
            db.drop_evidence_row(row["id"])
            evidence.discard(staged)
            raise
        return row

    item = await run_in_threadpool(_persist)

    state.record_audit(actor, "Collected evidence",
                       f"{item['id']} ({name})", case_id=case_id)
    logger.info("[EVIDENCE] %s %s on %s by %s (%d bytes, sha256 %s)",
                item["id"], name, case_id, actor, item["size_bytes"],
                item["sha256"][:16])
    return _out(item)


@router.get("/case/{case_id}", response_model=list[EvidenceItem],
            summary="Chain of custody for one case")
def list_for_case(
    case_id: str,
    include_withdrawn: bool = Query(default=True),
    user: dict = Depends(current_user),
) -> list[EvidenceItem]:
    _require_case(case_id)
    return [_out(i) for i in db.list_evidence(case_id,
                                              include_withdrawn=include_withdrawn)]


@router.get("/case/{case_id}/verify", response_model=EvidenceVerification,
            summary="Re-hash every artefact and check the chain")
def verify(case_id: str, user: dict = Depends(current_user)) -> EvidenceVerification:
    _require_case(case_id)
    return EvidenceVerification(**evidence.verify_case(case_id))


@router.get("/case/{case_id}/certificate", response_model=EvidenceCertificate,
            summary="BSA 2023 s.63 certificate for the case's electronic records")
def certificate(case_id: str,
                user: dict = Depends(current_user)) -> EvidenceCertificate:
    """The document a court asks for when electronic records are produced.

    Generated from the store, carrying a live re-verification, so it cannot
    describe artefacts the store does not hold or claim an integrity it does not
    have. The custodian is the officer requesting it -- they are the one who
    would sign it.
    """
    comp = _require_case(case_id)
    actor = _officer(user)
    state.record_audit(actor, "Produced evidence certificate", case_id,
                       case_id=case_id)
    return EvidenceCertificate(
        **evidence.certificate(case_id, complaint=comp, produced_by=actor))


@router.get("/{evidence_id}", response_model=EvidenceItem,
            summary="One artefact's custody record")
def get_one(evidence_id: str, user: dict = Depends(current_user)) -> EvidenceItem:
    item = db.get_evidence(evidence_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"No evidence {evidence_id}")
    return _out(item)


@router.get("/{evidence_id}/download", summary="The bytes, hash-checked first")
def download(evidence_id: str, user: dict = Depends(current_user)) -> FileResponse:
    item = db.get_evidence(evidence_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"No evidence {evidence_id}")

    check = evidence.verify_item(item)
    if not check["ok"]:
        # 409, not 500: the request is well formed and the system is working.
        # What is wrong is the artefact, and saying so is the point.
        raise HTTPException(
            status_code=409,
            detail=f"{evidence_id} failed integrity check: {check['reason']}. "
                   f"It is not served, because bytes that no longer match their "
                   f"collection hash are not evidence.")

    state.record_audit(_officer(user), "Downloaded evidence",
                       f"{item['id']} ({item['filename']})",
                       case_id=item["case_id"])

    return FileResponse(
        path=str(evidence.stored_path(item)),
        # Never the declared type. See the module docstring.
        media_type="application/octet-stream",
        filename=item["filename"],
        headers={"X-Content-Type-Options": "nosniff",
                 "X-Evidence-SHA256": item["sha256"]},
    )


@router.post("/{evidence_id}/withdraw", response_model=EvidenceItem,
             summary="Withdraw an artefact, with a reason. There is no delete.")
async def withdraw(evidence_id: str, body: EvidenceWithdrawRequest,
                   user: dict = Depends(current_user)) -> EvidenceItem:
    item = db.get_evidence(evidence_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"No evidence {evidence_id}")
    if item["withdrawn_at"]:
        raise HTTPException(status_code=409,
                            detail=f"{evidence_id} was already withdrawn")

    actor = _officer(user)
    updated = await run_in_threadpool(
        db.withdraw_evidence, evidence_id, actor, body.reason.strip())
    if updated is None:
        raise HTTPException(status_code=409, detail=f"{evidence_id} was already withdrawn")

    state.record_audit(actor, "Withdrew evidence",
                       f"{evidence_id}: {body.reason.strip()}",
                       case_id=item["case_id"])
    logger.info("[EVIDENCE] %s withdrawn by %s: %s", evidence_id, actor, body.reason)
    return _out(updated)
