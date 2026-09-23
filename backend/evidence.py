# -*- coding: utf-8 -*-
"""
MuleShield AI -- Electronic evidence documentation & BSA 2023 s.63 certification
SIH26184 | MHA / I4C
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from backend import db

logger = logging.getLogger("muleshield.evidence")

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = Path(os.environ.get("MULESHIELD_EVIDENCE_DIR", str(ROOT / "data" / "evidence")))

MAX_BYTES = int(os.environ.get("MULESHIELD_EVIDENCE_MAX_BYTES", 25 * 1024 * 1024))
"""25 MB. A bank statement PDF or a phone screenshot fits comfortably; a video
does not, and a system that accepts one has to think about storage it has not
thought about. Env-tunable so a deployment can raise it deliberately."""

CHUNK = 1024 * 1024

KINDS = ("bank_statement", "screenshot", "fir_copy", "transaction_export",
         "device_image", "correspondence", "other")
"""What an artefact is. Free text would give a dossier nine spellings of
"statement"; this is the list an investigator actually picks from."""

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


def display_name(filename: str) -> str:
    """A filename safe to render and to log.

    Never used as a path -- the stored name is derived from the artefact id --
    but it is echoed back to a browser, so directory traversal, control
    characters and anything that could break out of a table cell come off here.
    """
    base = os.path.basename(str(filename or "")).strip()
    base = _SAFE_NAME.sub("_", base)
    base = base.lstrip(".") or "artefact"
    return base[:180]


def case_dir(case_id: str) -> Path:
    """One directory per case, named by a digest of the ticket id.

    The ticket id is a UUID and would be safe as a path component, but deriving
    the directory rather than trusting the argument means no future caller can
    turn a case id into a traversal.
    """
    key = hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:24]
    return EVIDENCE_DIR / key


def stage(case_id: str, chunks: Iterable[bytes]) -> tuple[str, int, Path]:
    """Stream to a staging file, hashing as we go. Returns (sha256, size, path).

    Hashing during the write rather than reading the file back afterwards: the
    digest must describe the bytes that arrived, and a second read could pick up
    something that changed in between.

    Staged rather than final because the artefact id is minted by the database,
    under the same lock that fixes the artefact's position in the chain -- so
    the bytes have to land somewhere before there is an id to name them by.
    commit() moves them once there is.
    """
    target_dir = case_dir(case_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f".staging-{uuid.uuid4().hex}"

    sha = hashlib.sha256()
    size = 0
    try:
        with open(path, "wb") as fh:
            for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError(
                        f"artefact exceeds {MAX_BYTES // (1024 * 1024)} MB")
                sha.update(chunk)
                fh.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return sha.hexdigest(), size, path


def commit(staged: Path, case_id: str, item_id: str) -> Path:
    """Move staged bytes into place under the artefact id."""
    final = case_dir(case_id) / item_id
    staged.replace(final)
    return final


def discard(staged: Path) -> None:
    """Drop staged bytes whose row never landed."""
    try:
        staged.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - a locked temp file is not worth failing on
        logger.warning("could not remove staged artefact %s", staged)


def stored_path(item: dict) -> Path:
    return case_dir(item["case_id"]) / item["stored_name"]


def verify_item(item: dict) -> dict:
    """Re-hash one artefact's bytes and compare against the collection record."""
    path = stored_path(item)
    if not path.exists():
        return {"id": item["id"], "ok": False, "reason": "bytes missing from the store"}
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != item["sha256"]:
        return {"id": item["id"], "ok": False, "reason": "content hash mismatch",
                "recorded": item["sha256"], "actual": actual}
    return {"id": item["id"], "ok": True, "reason": ""}


def verify_case(case_id: str) -> dict:
    """Verify every artefact on a case, and the chain that links them.

    Two independent checks, reported separately because they fail for different
    reasons and mean different things:

      * **content** -- the bytes on disk still hash to what was recorded. Fails
        if a file was edited or replaced.
      * **chain** -- each row's entry_hash is what the previous row's hash and
        this row's collection facts produce. Fails if a row was inserted,
        removed, reordered or rewritten.
    """
    items = db.list_evidence(case_id)
    content = [verify_item(i) for i in items]

    chain: list[dict] = []
    prev = ""
    for i, item in enumerate(items, start=1):
        expected = db.chain_hash(prev, item["id"], item["case_id"], item["sha256"],
                                 item["collected_by"], item["collected_at"])
        ok = (item["prev_hash"] == prev and item["entry_hash"] == expected
              and int(item["seq"]) == i)
        chain.append({
            "id": item["id"], "seq": item["seq"], "ok": ok,
            "reason": "" if ok else "link does not reproduce from the previous entry",
        })
        # Carry the RECOMPUTED hash forward, never the stored one.
        #
        # An earlier revision propagated item["entry_hash"], which made the walk
        # forgiving in exactly the wrong way: rewriting one row flagged that row
        # and then resynchronised, so every artefact after it verified clean. A
        # chain whose damage does not propagate is not a chain -- it is a
        # per-row checksum with extra steps, and it would let somebody alter one
        # artefact and hand over a report showing a single isolated problem.
        # Recomputing means one broken link invalidates everything downstream of
        # it, which is the property the whole construction exists for.
        # backend/tests/test_evidence.py::test_a_rewritten_row_breaks_the_chain
        # is what caught this.
        prev = expected

    return {
        "case_id": case_id,
        "items": len(items),
        "content_ok": all(c["ok"] for c in content),
        "chain_ok": all(c["ok"] for c in chain),
        "intact": all(c["ok"] for c in content) and all(c["ok"] for c in chain),
        "content": content,
        "chain": chain,
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ── The certificate ─────────────────────────────────────────────────────────

STATUTE = "Bharatiya Sakshya Adhiniyam, 2023 - section 63"
STATUTE_NOTE = (
    "BSA 2023 s.63 replaced Information Technology Act s.65B for electronic "
    "records produced in evidence, in force from 1 July 2024."
)

STATEMENTS = (
    "The electronic records listed below were produced by a computer system used "
    "regularly to store and process information for the activities of the "
    "Indian Cyber Crime Coordination Centre (I4C).",
    "Each record was collected by the named officer through the authenticated "
    "interface of that system, and the SHA-256 value shown for each was computed "
    "from the bytes at the moment of collection.",
    "Each record has been re-hashed at the time this certificate was produced "
    "and the result is stated per artefact; a mismatch is reported rather than "
    "suppressed.",
    "Each record carries the hash of the preceding record for the same case, so "
    "that the order and completeness of the set can be checked independently of "
    "any single artefact.",
)

LIMITATIONS = (
    "The transports that notify recipients of alerts are simulated in this "
    "build; nothing in this certificate depends on them.",
    "The chain establishes that this set of records is internally consistent. It "
    "is not anchored outside the operator's own store, so it does not by itself "
    "establish authenticity against an external reference.",
    "This certificate is generated from the evidence store and is not a legal "
    "opinion. It must be adopted and signed by the officer named as custodian "
    "before it is produced.",
)


def certificate(case_id: str, *, complaint: Optional[dict] = None,
                produced_by: str = "") -> dict:
    """Assemble the s.63 certificate for one case, from the store.

    Generated rather than typed, and it carries the LIVE verification result --
    including a failure. A certificate that could only ever say "intact" would
    be decoration; this one reports what it finds, and the console renders a
    failed verification in red rather than printing anyway.
    """
    items = db.list_evidence(case_id)
    check = verify_case(case_id)
    by_item = {c["id"]: c for c in check["content"]}

    artefacts = []
    for item in items:
        v = by_item.get(item["id"], {})
        artefacts.append({
            "id": item["id"],
            "seq": item["seq"],
            "filename": item["filename"],
            "kind": item["kind"],
            "description": item["description"],
            "source": item["source"],
            "content_type": item["content_type"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
            "collected_by": item["collected_by"],
            "collected_at": item["collected_at"],
            "entry_hash": item["entry_hash"],
            "prev_hash": item["prev_hash"],
            "withdrawn": bool(item["withdrawn_at"]),
            "withdrawn_reason": item["withdrawn_reason"] or "",
            "reverified_ok": bool(v.get("ok")),
            "reverification_note": v.get("reason", ""),
        })

    live = [a for a in artefacts if not a["withdrawn"]]
    return {
        "statute": STATUTE,
        "statute_note": STATUTE_NOTE,
        "case_id": case_id,
        "complaint": {
            "victim_name": str((complaint or {}).get("victim_name", "")),
            "victim_bank": str((complaint or {}).get("victim_bank", "")),
            "fraud_type": str((complaint or {}).get("fraud_type", "")),
            "stolen_amount": float((complaint or {}).get("stolen_amount", 0) or 0),
            "city": str((complaint or {}).get("city", "")),
            "state": str((complaint or {}).get("state", "")),
            "complaint_timestamp": str((complaint or {}).get("complaint_timestamp", "")),
        } if complaint else None,
        "system": {
            "name": "MuleShield AI",
            "operator": "Indian Cyber Crime Coordination Centre (I4C)",
            "problem_statement": "SIH26184",
        },
        "custodian": produced_by,
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artefact_count": len(live),
        "withdrawn_count": len(artefacts) - len(live),
        "total_bytes": sum(a["size_bytes"] for a in live),
        "statements": list(STATEMENTS),
        "limitations": list(LIMITATIONS),
        "verification": {
            "content_ok": check["content_ok"],
            "chain_ok": check["chain_ok"],
            "intact": check["intact"],
            "verified_at": check["verified_at"],
        },
        "artefacts": artefacts,
    }
