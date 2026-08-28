# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2a: GraphSAGE Model Definition
SIH26184 | MHA / I4C

Architecture: 2-layer GraphSAGE binary node classifier
  in_channels   = 15  (behavioural features measured from the ledger)
  hidden_channels = 64
  out_channels  = 64  (embedding dimension)
  num_layers    = 2
  aggr          = 'mean'

The model is trained as a binary node classifier (mule=1 / clean=0).
The penultimate layer output (64-dim) is used as the risk embedding for XGBoost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ── Feature columns used for GNN input (must match node_features.csv) ────────
# Behavioural features, every one of them measurable by a bank from its own
# transaction log.
#
# `hop_depth` is deliberately absent. It records an account's position in a
# traced fraud chain, so it is only ever non-zero for accounts already known to
# be part of one — using it to decide whether an account is a mule assumes the
# answer. It stays in node_features.csv for the money-flow visualisation.
#
# `total_received` is retained: legitimate accounts now receive money too, so it
# no longer separates the classes on its own (best single-threshold F1 ≈ 0.42).
# Chosen from measured signal, not habit. `scripts/feature_analysis.py` reports
# each feature's standalone AUC and the feature-feature correlation heatmap; the
# selection below follows directly from it.
#
# Dropped as exact or near-duplicates (zero information lost):
#   in_degree         r = 1.000 with txn_count_24h  (the same column twice)
#   distinct_senders  r = 0.991 with txn_count_24h
#   out_degree        r = 0.996 with distinct_receivers
#
# Dropped as pure noise:
#   lat, long         AUC 0.511 / 0.509 alone. Mules are spread across all 79
#                     cities, so an account's coordinates say nothing about
#                     whether it is one. They stay in node_features.csv for the
#                     map, but feeding them here only adds variance.
#
# `hop_depth` remains excluded: it is non-zero only for accounts already known to
# sit in a traced fraud chain, so using it assumes the answer.
BASE_FEATURE_COLS = [
    "total_received",
    "total_sent",
    "txn_count_24h",
    "avg_txn_amount",
    "distinct_receivers",
    "median_dwell_seconds",     # credit -> next debit; mules forward fast
    "passthrough_ratio",        # share of inflow forwarded on
    "account_age_days",         # rented mule accounts are young  (AUC 0.772)
    "night_txn_ratio",
    "burst_out_5min",           # peak outgoing count in 5 min    (AUC 0.760)
]

# Derived on the fly from the columns above. Each is a ratio or a log the raw
# features cannot express to a model that only splits on axis-aligned thresholds,
# and all are computable by a bank from its own ledger.
DERIVED_FEATURE_COLS = [
    "dwell_log",                # heavy right tail; the raw seconds span 5 orders
    "activity_per_day",         # a young account moving constantly is the signal
    "in_out_amount_ratio",      # sweep behaviour, amount-based not count-based
    "avg_amount_per_credit",
    "counterparty_concentration",
]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS


def derive_features(df):
    """
    Add DERIVED_FEATURE_COLS to a node-features frame.

    Applied identically at training and inference. Kept here rather than baked
    into node_features.csv so the derivation lives next to the feature list it
    belongs to, and cannot drift away from it.
    """
    import numpy as np

    out = df.copy()
    recv = out["total_received"].astype(float)
    sent = out["total_sent"].astype(float)
    ind = out["txn_count_24h"].astype(float)
    outd = out["distinct_receivers"].astype(float)
    age = out["account_age_days"].astype(float)

    out["dwell_log"] = np.log1p(out["median_dwell_seconds"].astype(float))
    # Per ACTIVE day, not per day since the account opened. Dividing by age
    # made this 1/age for any account older than the 120-day observation
    # window (corr 0.92 with 1/age), so it was an age proxy rather than an
    # activity measure.
    observed_days = np.minimum(age, OBSERVATION_WINDOW_DAYS)
    out["activity_per_day"] = (ind + outd) / (observed_days + 1.0)
    out["in_out_amount_ratio"] = sent / (recv + 1.0)
    out["avg_amount_per_credit"] = recv / (ind + 1.0)
    out["counterparty_concentration"] = ind / (outd + 1.0)
    return out


OBSERVATION_WINDOW_DAYS = 120.0   # ledger span the features are measured over

IN_CHANNELS = len(FEATURE_COLS)   # 10 base + 5 derived
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 64                  # embedding dimension
NUM_LAYERS = 2                     # documentation only - depth is fixed by
                                   # conv1/conv2 below, not driven by this


class GraphSAGEMule(nn.Module):
    """
    2-layer GraphSAGE model for mule node detection.

    Forward pass produces:
      - logits (shape: [N, 1]) for BCEWithLogitsLoss training
      - embeddings (shape: [N, 64]) via get_embeddings()

    Usage:
        model = GraphSAGEMule()
        logits = model(x, edge_index)              # training
        embeddings = model.get_embeddings(x, edge_index)  # inference
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = HIDDEN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: in_channels -> hidden_channels
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        # Layer 2: hidden_channels -> out_channels (64-dim embedding)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")

        # Classification head: 64 -> 1 (binary: mule vs clean)
        self.classifier = nn.Linear(out_channels, 1)

        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Run the 2 GraphSAGE layers and return 64-dim node embeddings.

        Args:
            x:          Node feature matrix [N, in_channels]
            edge_index: Edge connectivity [2, E]

        Returns:
            Tensor [N, out_channels] — the 64-dim embedding per node
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        return x  # shape: [N, 64]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode + classify.

        Returns:
            logits [N, 1] for use with BCEWithLogitsLoss
        """
        embeddings = self.encode(x, edge_index)
        logits = self.classifier(embeddings)  # [N, 1]
        return logits

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get 64-dim risk embeddings without classification head.
        Call after model.eval().

        Returns:
            Tensor [N, 64]
        """
        self.eval()
        with torch.no_grad():
            return self.encode(x, edge_index)


def build_model(
    in_channels: int = IN_CHANNELS,
    hidden_channels: int = HIDDEN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
    dropout: float = 0.3,
) -> GraphSAGEMule:
    """Factory function — returns a new untrained GraphSAGEMule model."""
    return GraphSAGEMule(in_channels, hidden_channels, out_channels, dropout)


if __name__ == "__main__":
    # Sanity check: verify output shapes without real data
    import torch

    model = build_model()
    model.eval()

    N = 100   # dummy nodes
    E = 200   # dummy edges
    x = torch.randn(N, IN_CHANNELS)
    edge_index = torch.randint(0, N, (2, E))

    logits = model(x, edge_index)
    embeddings = model.get_embeddings(x, edge_index)

    assert logits.shape == (N, 1), f"Expected ({N}, 1), got {logits.shape}"
    assert embeddings.shape == (N, OUT_CHANNELS), \
        f"Expected ({N}, {OUT_CHANNELS}), got {embeddings.shape}"

    print(f"[OK] GraphSAGEMule forward pass: logits={logits.shape}, embeddings={embeddings.shape}")
    print(f"[OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
