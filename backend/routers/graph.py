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
import sys
import time
from pathlib import Path

import networkx as nx
from fastapi import APIRouter, HTTPException

from backend.models.schemas import GraphAnomalies, GraphNode, GraphEdge, GraphResponse
import backend.state as state

ROOT = Path(__file__).parent.parent.parent
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

# Same thresholds the offline engine uses — the dashboard must not invent its own.
from graph_engine import (  # noqa: E402
    SPLIT_AMOUNT_TOLERANCE,
    SPLIT_DESTINATIONS,
    VELOCITY_THRESHOLD,
    VELOCITY_WINDOW_SECONDS,
)

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


def _detect_anomalies(txns: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """
    Run the engine's velocity / fund-splitting / terminal-leaf detectors over a
    single complaint's sub-graph.

    Mirrors engine/graph_engine.py exactly (same thresholds, same sliding-window
    and split-variance logic) so the console reports what the offline engine
    would report, just scoped to one complaint.

    Returns:
        (velocity_flagged, split_flagged, terminal_leaves)
    """
    import pandas as pd

    g = nx.MultiDiGraph()
    for t in txns:
        src, dst = str(t.get("src_account", "")), str(t.get("dst_account", ""))
        if not src or not dst:
            continue
        try:
            ts = pd.Timestamp(t.get("timestamp"))
        except Exception:
            ts = None
        g.add_edge(src, dst, amount=float(t.get("amount", 0.0)), timestamp=ts)

    velocity, splitting = [], []
    for node in g.nodes():
        out_edges = list(g.out_edges(node, data=True))

        # ── Velocity: > VELOCITY_THRESHOLD outgoing txns inside the window ────
        if len(out_edges) > VELOCITY_THRESHOLD:
            stamps = sorted(
                d["timestamp"] for _, _, d in out_edges if d.get("timestamp") is not None
            )
            for i in range(len(stamps)):
                in_window = sum(
                    1 for j in range(i + 1, len(stamps))
                    if (stamps[j] - stamps[i]).total_seconds() <= VELOCITY_WINDOW_SECONDS
                )
                if in_window >= VELOCITY_THRESHOLD:
                    velocity.append(node)
                    break

        # ── Fund splitting: 1 → N near-equal dispersal ────────────────────────
        if len(out_edges) >= SPLIT_DESTINATIONS:
            amounts = [d.get("amount", 0.0) for _, _, d in out_edges]
            mean_amt = sum(amounts) / len(amounts) if amounts else 0.0
            if mean_amt and all(
                abs(a - mean_amt) / mean_amt <= SPLIT_AMOUNT_TOLERANCE for a in amounts
            ):
                splitting.append(node)

    terminals = [n for n in g.nodes() if g.out_degree(n) == 0]
    return sorted(set(velocity)), sorted(set(splitting)), sorted(set(terminals))


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

    # A transaction's hop_depth describes its DESTINATION. The source therefore
    # sits one hop upstream — deriving both from the same field mislabels every
    # source node (and can brand a mid-chain account "TERMINAL").
    terminal_accounts = {
        str(t.get("dst_account", ""))
        for t in txns
        if int(t.get("is_terminal", 0)) == 1
    }

    for txn in txns:
        txn_hop = int(txn.get("hop_depth", 1))
        for acc_key, hop in (("src_account", txn_hop - 1), ("dst_account", txn_hop)):
            acc = str(txn.get(acc_key, ""))
            if not acc or acc == victim_acc or acc in seen_accounts:
                continue

            hop = max(hop, 0)
            features = state.get_node_feature(acc) or {}
            # True GraphSAGE mule probability from the trained classification
            # head — not a summary statistic of the embedding.
            risk = state.gnn_risk_score(acc)
            is_terminal = acc in terminal_accounts

            # Geo/bank come from the account's own node record; the transaction
            # row only describes the destination side of that transfer.
            is_dst = acc_key == "dst_account"
            bank = str(
                features.get("bank_name")
                or (txn.get("bank_name") if is_dst else None)
                or "Unknown"
            )
            lat = float(features.get("lat", txn.get("lat", 20.5937) if is_dst else 20.5937))
            lon = float(features.get("long", txn.get("long", 78.9629) if is_dst else 78.9629))

            seen_accounts[acc] = GraphNode(
                id=acc,
                label=f"{'TERMINAL' if is_terminal else f'HOP-{hop}'}\n{bank}",
                node_type="terminal" if is_terminal else _get_node_type(hop),
                hop_depth=hop,
                bank=bank,
                amount=float(txn.get("amount", 0)) if is_dst else 0.0,
                lat=lat,
                lon=lon,
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

    # ── Run the real anomaly detectors over this complaint's sub-graph ────────
    velocity_flagged, split_flagged, terminal_leaves = _detect_anomalies(txns)
    anomalies = GraphAnomalies(
        velocity_flagged=velocity_flagged,
        fund_split_flagged=split_flagged,
        terminal_leaves=terminal_leaves,
        velocity_count=len(velocity_flagged),
        fund_split_count=len(split_flagged),
        terminal_count=len(terminal_leaves),
        velocity_rule=f">{VELOCITY_THRESHOLD} outgoing txns in {VELOCITY_WINDOW_SECONDS // 60}m",
        fund_split_rule=f"1-to-{SPLIT_DESTINATIONS}+ within {int(SPLIT_AMOUNT_TOLERANCE * 100)}% of mean",
    )

    build_ms = round((time.time() - t0) * 1000, 2)

    logger.info(
        f"[GRAPH] {complaint_id}: {len(nodes)} nodes, {len(edges)} edges, "
        f"{anomalies.velocity_count} velocity / {anomalies.fund_split_count} split "
        f"anomalies, built in {build_ms}ms"
    )

    return GraphResponse(
        complaint_id=complaint_id,
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
        build_time_ms=build_ms,
        anomalies=anomalies,
    )
