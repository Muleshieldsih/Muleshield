# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Graph Router
SIH26184 | MHA / I4C

GET /api/v1/graph/{complaint_id}
  Returns node-link data in React Flow format for the money-flow
  graph of a given complaint. Builds the graph from the transaction
  ledger stored in state.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from backend.models.schemas import GraphNode, GraphEdge, GraphResponse
import backend.state as state

logger = logging.getLogger("muleshield.graph")

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])

# Node type colour/class mapping for the frontend
_NODE_TYPE = {
    0: "victim",
    1: "mule",
    2: "mule",
    3: "mule",
    4: "terminal",
}


def _get_node_type(hop_depth: int) -> str:
    if hop_depth == 0:
        return "victim"
    if hop_depth >= 4:
        return "terminal"
    return "mule"


@router.get(
    "/{complaint_id}",
    response_model=GraphResponse,
    summary="Get React Flow money-flow graph for a complaint",
)
async def get_graph(complaint_id: str) -> GraphResponse:
    """
    Builds the directed money-flow graph for a complaint from the
    transaction ledger. Returns nodes and edges in React Flow format.

    Node types:
      victim   — the original defrauded account
      mule     — intermediate layering accounts (hop 1–3)
      terminal — final cashout account (hop 4+, is_terminal=1)
    """
    t0 = time.time()

    complaint = state.get_complaint(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")

    txns = state.get_transactions_for(complaint_id)
    if not txns:
        raise HTTPException(
            status_code=404,
            detail=f"No transaction data found for complaint '{complaint_id}'."
        )

    # ── Build nodes ──────────────────────────────────────────────────────────
    seen_accounts: dict[str, GraphNode] = {}

    # Add victim node (hop 0)
    victim_acc = complaint["victim_account"]
    victim_features = state.get_node_feature(victim_acc) or {}
    seen_accounts[victim_acc] = GraphNode(
        id=victim_acc,
        label=f"VICTIM\n{complaint['victim_name']}",
        node_type="victim",
        hop_depth=0,
        bank=complaint["victim_bank"],
        amount=complaint["stolen_amount"],
        lat=float(victim_features.get("lat", 20.5937)),
        lon=float(victim_features.get("long", 78.9629)),
        risk_score=0.0,
    )

    for txn in txns:
        for acc_key, hop_key in [("src_account", "hop_depth"), ("dst_account", "hop_depth")]:
            acc = str(txn.get(acc_key, ""))
            if not acc or acc == victim_acc:
                continue
            if acc in seen_accounts:
                continue

            hop = int(txn.get("hop_depth", 1))
            features = state.get_node_feature(acc) or {}
            emb = state.embeddings.get(acc)
            risk = float(emb.mean()) if emb is not None else 0.0
            is_terminal = int(txn.get("is_terminal", 0)) == 1

            seen_accounts[acc] = GraphNode(
                id=acc,
                label=f"{'TERMINAL' if is_terminal else f'HOP-{hop}'}\n{txn.get('bank_name', '?')}",
                node_type="terminal" if is_terminal else _get_node_type(hop),
                hop_depth=hop,
                bank=str(txn.get("bank_name", "Unknown")),
                amount=float(txn.get("amount", 0)),
                lat=float(txn.get("lat", features.get("lat", 20.5937))),
                lon=float(txn.get("long", features.get("long", 78.9629))),
                risk_score=round(risk, 4),
            )

    # ── Build edges ──────────────────────────────────────────────────────────
    edges: list[GraphEdge] = []
    seen_edges: set[str] = set()

    for txn in txns:
        src = str(txn.get("src_account", ""))
        dst = str(txn.get("dst_account", ""))
        if not src or not dst:
            continue

        edge_key = f"{src}->{dst}"
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        amount = float(txn.get("amount", 0))
        ts = str(txn.get("timestamp", ""))
        txn_id = str(txn.get("txn_id", edge_key))

        edges.append(GraphEdge(
            id=f"e-{txn_id}",
            source=src,
            target=dst,
            amount=amount,
            timestamp=ts,
            label=f"₹{amount:,.0f}",
        ))

    nodes = list(seen_accounts.values())
    build_ms = round((time.time() - t0) * 1000, 2)

    logger.info(
        f"[GRAPH] {complaint_id}: {len(nodes)} nodes, {len(edges)} edges, built in {build_ms}ms"
    )

    return GraphResponse(
        complaint_id=complaint_id,
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
        build_time_ms=build_ms,
    )
