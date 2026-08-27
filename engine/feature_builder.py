# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b: Hybrid Feature Builder
SIH26184 | MHA / I4C

Combines GraphSAGE 64-dim embeddings with 8 tabular features
into a 72-dim vector for XGBoost training and inference.

Feature vector layout:
  [0:64]   GraphSAGE embedding (risk vector from GNN)
  [64]     stolen_amount
  [65]     hop_depth
  [66]     transaction_velocity  (txns/min from terminal account)
  [67]     branch_distance_to_atm (km, Haversine)
  [68]     hour_of_day
  [69]     historical_hotspot_density (ATM fraud count within 5km)
  [70]     day_of_week (0=Mon, 6=Sun)
  [71]     amount_after_split
"""

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "node_embeddings.pkl"

EMBEDDING_DIM = 64
TABULAR_DIM = 8
TOTAL_FEATURE_DIM = EMBEDDING_DIM + TABULAR_DIM  # 72

TABULAR_FEATURE_NAMES = [
    "stolen_amount",
    "hop_depth",
    "transaction_velocity",
    "branch_distance_to_atm",
    "hour_of_day",
    "historical_hotspot_density",
    "day_of_week",
    "amount_after_split",
]

FEATURE_NAMES = [f"emb_{i}" for i in range(EMBEDDING_DIM)] + TABULAR_FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# HAVERSINE DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _haversine_vectorized(
    lat1: float,
    lon1: float,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """Vectorized Haversine: one point vs many points. Returns km array."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = np.radians(lats2)
    dphi = phi2 - phi1
    dlam = np.radians(lons2 - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def nearest_atm_info(
    node_lat: float,
    node_lon: float,
    atm_df: pd.DataFrame,
) -> tuple[str, float, float]:
    """
    Find the nearest ATM to a given (lat, lon) point (vectorized).

    Returns:
        (atm_id, distance_km, historical_fraud_count)
    """
    dists = _haversine_vectorized(
        node_lat, node_lon,
        atm_df["lat"].to_numpy(),
        atm_df["long"].to_numpy(),
    )
    idx = int(np.argmin(dists))
    row = atm_df.iloc[idx]
    return str(row["atm_id"]), float(dists[idx]), float(row["historical_fraud_count"])


def atms_within_radius(
    node_lat: float,
    node_lon: float,
    atm_df: pd.DataFrame,
    radius_km: float = 5.0,
) -> int:
    """Count ATMs within `radius_km` of a given point (vectorized)."""
    dists = _haversine_vectorized(
        node_lat, node_lon,
        atm_df["lat"].to_numpy(),
        atm_df["long"].to_numpy(),
    )
    return int((dists <= radius_km).sum())


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION VELOCITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_transaction_velocity(
    account_id: str,
    transactions_df: pd.DataFrame,
    window_minutes: float = 60.0,
) -> float:
    """
    Compute transactions-per-minute rate for `account_id` within the
    latest `window_minutes` window of activity.

    Returns:
        float — txns/min (0.0 if no transactions found)
    """
    txn = transactions_df[
        (transactions_df["src_account"] == account_id) |
        (transactions_df["dst_account"] == account_id)
    ].copy()

    if txn.empty:
        return 0.0

    txn["ts"] = pd.to_datetime(txn["timestamp"])
    txn = txn.sort_values("ts")
    latest = txn["ts"].max()
    cutoff = latest - pd.Timedelta(minutes=window_minutes)
    window_txns = txn[txn["ts"] >= cutoff]

    if len(window_txns) <= 1:
        return 0.0

    span_minutes = (window_txns["ts"].max() - window_txns["ts"].min()).total_seconds() / 60.0
    if span_minutes == 0:
        return float(len(window_txns))

    return len(window_txns) / span_minutes


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Builds the 72-dim hybrid feature vector for each terminal mule node.

    Each sample = one terminal account (cashout candidate) derived from
    one complaint chain.

    Usage:
        fb = FeatureBuilder()
        fb.load()
        X, y_atm, y_time, meta = fb.build_training_set()
    """

    def __init__(
        self,
        transactions_path: Path = DATA_DIR / "transactions.csv",
        node_features_path: Path = DATA_DIR / "node_features.csv",
        atm_path: Path = DATA_DIR / "atm_directory.csv",
        complaints_path: Path = DATA_DIR / "victim_complaints.csv",
        embeddings_path: Path = EMBEDDINGS_PATH,
    ):
        self.transactions_path = transactions_path
        self.node_features_path = node_features_path
        self.atm_path = atm_path
        self.complaints_path = complaints_path
        self.embeddings_path = embeddings_path

        self.txn_df: Optional[pd.DataFrame] = None
        self.node_df: Optional[pd.DataFrame] = None
        self.atm_df: Optional[pd.DataFrame] = None
        self.complaints_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[dict] = None

    def load(self) -> "FeatureBuilder":
        """Load all data sources."""
        self.txn_df = pd.read_csv(self.transactions_path)
        self.node_df = pd.read_csv(self.node_features_path)
        self.atm_df = pd.read_csv(self.atm_path)
        self.complaints_df = pd.read_csv(self.complaints_path)
        with open(self.embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)
        return self

    def _get_embedding(self, account_id: str) -> np.ndarray:
        """Get 64-dim embedding for account, or zero vector if not found."""
        if account_id in self.embeddings:
            return self.embeddings[account_id].astype(np.float32)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def build_feature_vector(
        self,
        terminal_account: str,
        complaint_id: str,
        stolen_amount: float,
    ) -> np.ndarray:
        """
        Build a single 72-dim feature vector for one terminal account.

        Args:
            terminal_account: The cashout (terminal) mule account ID
            complaint_id:     The associated complaint ticket ID
            stolen_amount:    Original amount stolen from victim

        Returns:
            np.ndarray of shape (72,)
        """
        # ── Part A: GNN Embedding (64 dims) ──────────────────────────────────
        embedding = self._get_embedding(terminal_account)

        # ── Part B: Tabular features (8 dims) ────────────────────────────────
        # Get node metadata
        node_row = self.node_df[self.node_df["account_id"] == terminal_account]
        if not node_row.empty:
            node_lat = float(node_row["lat"].iloc[0])
            node_lon = float(node_row["long"].iloc[0])
            hop_depth = int(node_row["hop_depth"].iloc[0])
            amount_at_terminal = float(node_row["total_received"].iloc[0])
        else:
            node_lat, node_lon = 20.5937, 78.9629  # India centroid fallback
            hop_depth = 3
            amount_at_terminal = stolen_amount * 0.7

        # Terminal transaction for this complaint
        terminal_txn = self.txn_df[
            (self.txn_df["complaint_id"] == complaint_id) &
            (self.txn_df["dst_account"] == terminal_account) &
            (self.txn_df["is_terminal"] == 1)
        ]
        if not terminal_txn.empty:
            ts = pd.to_datetime(terminal_txn["timestamp"].iloc[0])
            hour_of_day = ts.hour
            day_of_week = ts.dayofweek
            amount_after_split = float(terminal_txn["amount"].iloc[0])
        else:
            hour_of_day = 14   # 2pm default
            day_of_week = 2    # Wednesday
            amount_after_split = amount_at_terminal

        # Velocity: txns/min from terminal account
        velocity = compute_transaction_velocity(terminal_account, self.txn_df)

        # Nearest ATM distance + hotspot density
        _, dist_km, _ = nearest_atm_info(node_lat, node_lon, self.atm_df)
        hotspot_density = atms_within_radius(node_lat, node_lon, self.atm_df, radius_km=5.0)

        tabular = np.array([
            stolen_amount,
            float(hop_depth),
            velocity,
            dist_km,
            float(hour_of_day),
            float(hotspot_density),
            float(day_of_week),
            amount_after_split,
        ], dtype=np.float32)

        return np.concatenate([embedding, tabular])  # shape (72,)

    def build_training_set(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Build training set by iterating over all terminal nodes across complaints.

        Returns:
            X         : np.ndarray [N, 72] feature matrix
            y_atm     : np.ndarray [N]     nearest ATM label (integer index)
            y_time    : np.ndarray [N]     synthetic time-to-cashout (minutes)
            meta_df   : pd.DataFrame       metadata (account_id, complaint_id, atm_id, etc.)
        """
        # Get terminal transactions
        terminal_txns = self.txn_df[self.txn_df["is_terminal"] == 1].copy()

        # Merge complaint amounts
        complaint_amounts = self.complaints_df.set_index("ticket_id")["stolen_amount"].to_dict()

        # Build ATM ID -> index mapping for classification
        atm_ids = self.atm_df["atm_id"].tolist()
        atm_to_idx = {atm: i for i, atm in enumerate(atm_ids)}

        X_rows, y_atm_rows, y_time_rows, meta_rows = [], [], [], []

        for _, row in terminal_txns.iterrows():
            terminal_acc = row["dst_account"]
            complaint_id = row["complaint_id"]
            stolen_amount = complaint_amounts.get(complaint_id, 100000.0)

            # Feature vector
            feat = self.build_feature_vector(terminal_acc, complaint_id, stolen_amount)

            # Target 1: Nearest ATM (label = ATM index)
            node_row = self.node_df[self.node_df["account_id"] == terminal_acc]
            if not node_row.empty:
                lat, lon = float(node_row["lat"].iloc[0]), float(node_row["long"].iloc[0])
            else:
                lat, lon = 20.5937, 78.9629

            nearest_atm_id, dist_km, fraud_count = nearest_atm_info(lat, lon, self.atm_df)
            atm_label = atm_to_idx[nearest_atm_id]

            # Target 2: Synthetic time-to-cashout (minutes)
            # Model: cashout happens faster when closer to ATM and higher velocity
            velocity = feat[66]
            base_time = 45.0  # base 45 min
            dist_factor = dist_km * 2.0  # farther = more time
            velocity_factor = -min(velocity * 5.0, 30.0)  # faster = less time
            time_to_cashout = max(5.0, base_time + dist_factor + velocity_factor)

            X_rows.append(feat)
            y_atm_rows.append(atm_label)
            y_time_rows.append(time_to_cashout)
            meta_rows.append({
                "terminal_account": terminal_acc,
                "complaint_id": complaint_id,
                "nearest_atm_id": nearest_atm_id,
                "dist_to_atm_km": dist_km,
                "time_to_cashout_min": time_to_cashout,
                "lat": lat,
                "lon": lon,
            })

        X = np.array(X_rows, dtype=np.float32)
        y_atm = np.array(y_atm_rows, dtype=np.int32)
        y_time = np.array(y_time_rows, dtype=np.float32)
        meta_df = pd.DataFrame(meta_rows)

        # Store ATM metadata for inverse mapping at inference time
        self.atm_ids = atm_ids
        self.atm_to_idx = atm_to_idx

        return X, y_atm, y_time, meta_df


if __name__ == "__main__":
    print("Building hybrid feature matrix...")
    fb = FeatureBuilder()
    fb.load()
    X, y_atm, y_time, meta = fb.build_training_set()
    print(f"  X shape       : {X.shape}  (should be [N, 72])")
    print(f"  y_atm shape   : {y_atm.shape}")
    print(f"  y_time shape  : {y_time.shape}")
    print(f"  ATM classes   : {len(set(y_atm))}")
    print(f"  Time range    : {y_time.min():.1f} -- {y_time.max():.1f} min")
    assert X.shape[1] == TOTAL_FEATURE_DIM, f"Wrong feature dim: {X.shape[1]}"
    print("[OK] feature_builder.py verified.")
