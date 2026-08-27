# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2a: Graph Intelligence Engine
SIH26184 | MHA / I4C

Responsibilities:
  - Build directed NetworkX graph from CSV data
  - BFS traversal to enumerate all mule nodes reachable from a victim
  - Velocity anomaly detection (>2 outgoing txns within 5 minutes)
  - Fund-splitting detection (1 src -> 3+ destinations of similar amounts)
  - Terminal node identification (cashout candidates)
  - Export JSON node-link format for React Flow / API
"""

import json
import time
from collections import deque
from pathlib import Path

import networkx as nx
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# ── Constants ─────────────────────────────────────────────────────────────────
VELOCITY_WINDOW_SECONDS = 300   # 5-minute window for velocity check
VELOCITY_THRESHOLD = 2          # >2 outgoing txns in window = anomaly
SPLIT_DESTINATIONS = 3          # 1 src -> 3+ dst = fund splitting
SPLIT_AMOUNT_TOLERANCE = 0.30   # amounts within 30% of mean = even split


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class MuleGraph:
    """
    Directed transaction graph built from the MuleShield synthetic dataset.

    Nodes: bank account IDs
    Edges: transactions (directed: src -> dst), with attributes:
        amount, timestamp, hop_depth, is_terminal, complaint_id
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self._loaded = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_from_csv(
        self,
        transactions_path: Path = DATA_DIR / "transactions.csv",
        node_features_path: Path = DATA_DIR / "node_features.csv",
    ) -> "MuleGraph":
        """Load graph from Phase 1 CSV files (fast batch insertion)."""
        txn_df = pd.read_csv(transactions_path)
        node_df = pd.read_csv(node_features_path)
        return self.load_from_dataframes(txn_df, node_df)

    def load_from_dataframes(
        self,
        transactions_df: pd.DataFrame,
        node_features_df: pd.DataFrame,
    ) -> "MuleGraph":
        """Load graph directly from DataFrames using fast batch insertion."""
        txn_df = transactions_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(txn_df["timestamp"]):
            txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])

        # Batch Add Nodes
        node_nodes = [
            (
                acc,
                {
                    "bank_name": bank,
                    "city": city,
                    "lat": float(lat),
                    "long": float(lon),
                    "total_received": float(rec),
                    "total_sent": float(sent),
                    "txn_count_24h": int(cnt),
                    "avg_txn_amount": float(avg),
                    "is_mule_label": int(lbl),
                    "hop_depth": int(hop),
                }
            )
            for acc, bank, city, lat, lon, rec, sent, cnt, avg, lbl, hop in zip(
                node_features_df["account_id"],
                node_features_df.get("bank_name", [""] * len(node_features_df)),
                node_features_df.get("city", [""] * len(node_features_df)),
                node_features_df.get("lat", [0.0] * len(node_features_df)),
                node_features_df.get("long", [0.0] * len(node_features_df)),
                node_features_df.get("total_received", [0.0] * len(node_features_df)),
                node_features_df.get("total_sent", [0.0] * len(node_features_df)),
                node_features_df.get("txn_count_24h", [0] * len(node_features_df)),
                node_features_df.get("avg_txn_amount", [0.0] * len(node_features_df)),
                node_features_df.get("is_mule_label", [0] * len(node_features_df)),
                node_features_df.get("hop_depth", [0] * len(node_features_df)),
            )
        ]
        self.G.add_nodes_from(node_nodes)

        # Batch Add Edges
        edge_edges = [
            (
                src,
                dst,
                {
                    "txn_id": tid,
                    "complaint_id": cid,
                    "amount": float(amt),
                    "timestamp": ts,
                    "hop_depth": int(hop),
                    "is_terminal": int(term),
                    "bank_name": bank,
                    "city": city,
                    "lat": float(lat),
                    "long": float(lon),
                }
            )
            for src, dst, tid, cid, amt, ts, hop, term, bank, city, lat, lon in zip(
                txn_df["src_account"],
                txn_df["dst_account"],
                txn_df["txn_id"],
                txn_df["complaint_id"],
                txn_df["amount"],
                txn_df["timestamp"],
                txn_df["hop_depth"],
                txn_df["is_terminal"],
                txn_df.get("bank_name", [""] * len(txn_df)),
                txn_df.get("city", [""] * len(txn_df)),
                txn_df.get("lat", [0.0] * len(txn_df)),
                txn_df.get("long", [0.0] * len(txn_df)),
            )
        ]
        self.G.add_edges_from(edge_edges)

        self._loaded = True
        return self

    # ── BFS Traversal ─────────────────────────────────────────────────────────

    def bfs_from_victim(
        self,
        victim_account: str,
        max_depth: int = 5,
    ) -> dict:
        """
        BFS traversal from a victim account to enumerate all reachable mule nodes.

        Args:
            victim_account: Starting node (victim's account ID)
            max_depth:      Maximum hop depth to traverse

        Returns:
            dict with keys:
                nodes: list of (account_id, node_attrs, depth)
                edges: list of edge dicts
                terminal_nodes: list of account_ids that are terminal (cashout)
                depth_map: {account_id: depth}
        """
        if victim_account not in self.G:
            return {"nodes": [], "edges": [], "terminal_nodes": [], "depth_map": {}}

        visited = {victim_account: 0}
        queue = deque([(victim_account, 0)])
        result_nodes = []
        result_edges = []
        terminal_nodes = []

        while queue:
            current, depth = queue.popleft()
            node_attrs = dict(self.G.nodes.get(current, {}))
            node_attrs["account_id"] = current
            node_attrs["depth"] = depth
            result_nodes.append((current, node_attrs, depth))

            if depth >= max_depth:
                continue

            for _, neighbor, edge_data in self.G.out_edges(current, data=True):
                result_edges.append({
                    "src": current,
                    "dst": neighbor,
                    **{k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                       for k, v in edge_data.items()},
                })
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

                if edge_data.get("is_terminal", 0) == 1:
                    if neighbor not in terminal_nodes:
                        terminal_nodes.append(neighbor)

        return {
            "nodes": result_nodes,
            "edges": result_edges,
            "terminal_nodes": terminal_nodes,
            "depth_map": visited,
        }

    def bfs_for_complaint(
        self,
        complaint_id: str,
        max_depth: int = 5,
    ) -> dict:
        """
        BFS traversal for all accounts linked to a specific complaint_id.
        Useful when victim account ID is not known, only complaint ticket.
        """
        # Find all src accounts for this complaint (L1 origins)
        origin_nodes = set()
        for u, v, data in self.G.edges(data=True):
            if data.get("complaint_id") == complaint_id and data.get("hop_depth") == 1:
                origin_nodes.add(u)

        all_nodes, all_edges, all_terminals, depth_map = [], [], [], {}
        for origin in origin_nodes:
            result = self.bfs_from_victim(origin, max_depth)
            all_nodes.extend(result["nodes"])
            all_edges.extend(result["edges"])
            all_terminals.extend(result["terminal_nodes"])
            depth_map.update(result["depth_map"])

        return {
            "nodes": all_nodes,
            "edges": all_edges,
            "terminal_nodes": list(set(all_terminals)),
            "depth_map": depth_map,
        }

    # ── Anomaly Detection ─────────────────────────────────────────────────────

    def detect_velocity_anomalies(self) -> list[str]:
        """
        Flag accounts with >VELOCITY_THRESHOLD outgoing transactions
        within a VELOCITY_WINDOW_SECONDS window.

        Returns:
            List of account_ids with velocity anomaly
        """
        flagged = []
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node, data=True))
            if len(out_edges) <= VELOCITY_THRESHOLD:
                continue

            timestamps = []
            for _, _, data in out_edges:
                ts = data.get("timestamp")
                if isinstance(ts, pd.Timestamp):
                    timestamps.append(ts)
                elif isinstance(ts, str):
                    try:
                        timestamps.append(pd.Timestamp(ts))
                    except Exception:
                        pass

            if not timestamps:
                continue

            timestamps.sort()
            # Sliding window check
            for i in range(len(timestamps)):
                window_count = sum(
                    1 for j in range(i + 1, len(timestamps))
                    if (timestamps[j] - timestamps[i]).total_seconds() <= VELOCITY_WINDOW_SECONDS
                )
                if window_count >= VELOCITY_THRESHOLD:
                    flagged.append(node)
                    break

        return list(set(flagged))

    def detect_fund_splitting(self) -> list[str]:
        """
        Flag accounts where a single source splits funds to 3+ destinations
        of approximately equal amounts (within SPLIT_AMOUNT_TOLERANCE of mean).

        Returns:
            List of source account_ids involved in fund-splitting
        """
        flagged = []
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node, data=True))
            if len(out_edges) < SPLIT_DESTINATIONS:
                continue

            amounts = [data.get("amount", 0.0) for _, _, data in out_edges]
            if not amounts:
                continue

            mean_amt = sum(amounts) / len(amounts)
            if mean_amt == 0:
                continue

            # Check if all amounts are within tolerance of mean
            all_similar = all(
                abs(a - mean_amt) / mean_amt <= SPLIT_AMOUNT_TOLERANCE
                for a in amounts
            )
            if all_similar:
                flagged.append(node)

        return list(set(flagged))

    def get_terminal_nodes(self) -> list[str]:
        """
        Return all terminal nodes (leaf nodes with no outgoing edges),
        which are cashout candidates.
        """
        return [n for n in self.G.nodes() if self.G.out_degree(n) == 0]

    # ── JSON Export (for React Flow / API) ────────────────────────────────────

    def to_node_link_json(
        self,
        complaint_id: str = None,
        victim_account: str = None,
        include_anomalies: bool = True,
    ) -> dict:
        """
        Export the graph (or a subgraph) as JSON node-link format for React Flow.

        Args:
            complaint_id:      Filter to a specific complaint (optional)
            victim_account:    Start BFS from this account (optional)
            include_anomalies: Add velocity/split flags to nodes

        Returns:
            dict with 'nodes' and 'edges' lists (React Flow compatible)
        """
        velocity_set = set()
        split_set = set()
        if include_anomalies:
            velocity_set = set(self.detect_velocity_anomalies())
            split_set = set(self.detect_fund_splitting())
        terminal_set = set(self.get_terminal_nodes())

        if victim_account:
            result = self.bfs_from_victim(victim_account)
            node_ids = {n[0] for n in result["nodes"]}
            subgraph = self.G.subgraph(node_ids)
        elif complaint_id:
            result = self.bfs_for_complaint(complaint_id)
            node_ids = {n[0] for n in result["nodes"]}
            subgraph = self.G.subgraph(node_ids)
        else:
            subgraph = self.G

        nodes_out = []
        for nid, attrs in subgraph.nodes(data=True):
            nodes_out.append({
                "id": nid,
                "data": {
                    **{k: v for k, v in attrs.items()
                       if isinstance(v, (str, int, float, bool))},
                    "is_terminal": nid in terminal_set,
                    "velocity_anomaly": nid in velocity_set,
                    "fund_split": nid in split_set,
                },
                "position": {"x": 0, "y": 0},  # Frontend positions nodes via layout
            })

        edges_out = []
        for u, v, data in subgraph.edges(data=True):
            edges_out.append({
                "id": data.get("txn_id", f"{u}->{v}"),
                "source": u,
                "target": v,
                "data": {
                    k: str(v_val) if isinstance(v_val, pd.Timestamp) else v_val
                    for k, v_val in data.items()
                    if isinstance(v_val, (str, int, float, bool, pd.Timestamp))
                },
            })

        return {"nodes": nodes_out, "edges": edges_out}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return basic graph statistics."""
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "terminal_nodes": len(self.get_terminal_nodes()),
            "velocity_anomalies": len(self.detect_velocity_anomalies()),
            "fund_splitting_nodes": len(self.detect_fund_splitting()),
            "is_directed": self.G.is_directed(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO (run directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building MuleShield graph from CSV...")
    t0 = time.time()
    mg = MuleGraph().load_from_csv()
    elapsed = (time.time() - t0) * 1000

    stats = mg.stats()
    print(f"\nGraph Stats (built in {elapsed:.1f}ms):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # BFS demo
    complaints_df = pd.read_csv(DATA_DIR / "victim_complaints.csv")
    sample_complaint = complaints_df["ticket_id"].iloc[0]
    print(f"\nBFS for complaint: {sample_complaint}")
    result = mg.bfs_for_complaint(sample_complaint)
    print(f"  Reachable nodes : {len(result['nodes'])}")
    print(f"  Terminal nodes  : {len(result['terminal_nodes'])}")

    print("\n[OK] graph_engine.py verified.")
