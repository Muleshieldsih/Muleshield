# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Embeddings Router
SIH26184 | MHA / I4C

GET /api/v1/embeddings/{complaint_id}
  Returns the top-N highest-risk mule accounts for a complaint,
  ranked by their GNN embedding risk score (L2 norm of 64-dim vector).
"""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import EmbeddingResponse, MuleNodeRisk
import backend.state as state

logger = logging.getLogger("muleshield.embeddings")

router = APIRouter(prefix="/api/v1/embeddings", tags=["Embeddings"])


@router.get(
    "/{complaint_id}",
    response_model=EmbeddingResponse,
    summary="Get top-N highest-risk mule nodes with GNN risk scores",
)
async def get_embeddings(
    complaint_id: str,
    top_n: int = Query(default=5, ge=1, le=20, description="Number of top mule nodes to return"),
) -> EmbeddingResponse:
    """
    Returns the top-N mule accounts with the highest GNN risk scores
    (measured as L2 norm of the 64-dim GraphSAGE embedding).

    Higher norm → higher fraud-ring centrality.
    """
    complaint = state.get_complaint(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")

    txns = state.get_transactions_for(complaint_id)
    if not txns:
        raise HTTPException(
            status_code=404,
            detail=f"No transaction data found for complaint '{complaint_id}'."
        )

    # Collect all unique mule accounts (exclude victim)
    victim_acc = complaint["victim_account"]
    mule_accounts: set[str] = set()
    for txn in txns:
        for key in ("src_account", "dst_account"):
            acc = str(txn.get(key, ""))
            if acc and acc != victim_acc:
                mule_accounts.add(acc)

    # Score each mule by embedding L2 norm
    scored: list[tuple[float, str]] = []
    for acc in mule_accounts:
        emb = state.embeddings.get(acc)
        if emb is not None:
            norm = float(np.linalg.norm(emb))
            scored.append((norm, acc))

    # Sort descending by norm
    scored.sort(key=lambda x: x[0], reverse=True)
    top_accounts = scored[:top_n]

    result_nodes: list[MuleNodeRisk] = []
    for norm, acc in top_accounts:
        emb = state.embeddings[acc]
        features = state.get_node_feature(acc) or {}

        # Find the transaction for this account to get bank + hop info
        acc_txn = next(
            (t for t in txns if t.get("dst_account") == acc or t.get("src_account") == acc),
            {}
        )
        hop = int(acc_txn.get("hop_depth", 1))
        bank = str(acc_txn.get("bank_name", features.get("bank", "Unknown")))

        result_nodes.append(MuleNodeRisk(
            account_id=acc,
            bank=bank,
            hop_depth=hop,
            risk_score=round(norm, 4),
            embedding_norm=round(norm, 4),
            embedding_preview=[round(float(v), 4) for v in emb[:8]],
        ))

    logger.info(
        f"[EMBEDDINGS] {complaint_id}: returned top-{len(result_nodes)} mule nodes "
        f"out of {len(mule_accounts)} total."
    )

    return EmbeddingResponse(
        complaint_id=complaint_id,
        top_mules=result_nodes,
        total_nodes=len(mule_accounts),
    )
