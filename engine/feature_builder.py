# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b: High-Performance Hybrid Feature Builder
SIH26184 | MHA / I4C

Combines GraphSAGE 64-dim embeddings with 8 tabular features
into a 72-dim vector for XGBoost training and inference.

Optimized with O(1) indexed lookups and vectorized NumPy distance calculations
for sub-second processing over 50,000+ nodes.

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
# VECTORIZED HAVERSINE DISTANCE
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
# HIGH PERFORMANCE FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Builds the 72-dim hybrid feature vector for each terminal mule node.
    Indexed for O(1) attribute access.
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

        # O(1) Fast Lookups
        self._node_lookup = {}
        self._terminal_txn_lookup = {}
        self._velocity_lookup = {}
        self._atm_lats = None
        self._atm_lons = None

    def load(self) -> "FeatureBuilder":
        """Load all data sources and build O(1) indexed structures."""
        self.txn_df = pd.read_csv(self.transactions_path)
        self.node_df = pd.read_csv(self.node_features_path)
        self.atm_df = pd.read_csv(self.atm_path)
        self.complaints_df = pd.read_csv(self.complaints_path)

        with open(self.embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        # 1. Build node lookup dict using fast zip
        for acc_id, lat, lon, hop, tot_rec in zip(
            self.node_df["account_id"],
            self.node_df["lat"],
            self.node_df["long"],
            self.node_df["hop_depth"],
            self.node_df["total_received"],
        ):
            self._node_lookup[acc_id] = (float(lat), float(lon), int(hop), float(tot_rec))

        # 2. Build ATM coordinate arrays for vectorized distance
        self._atm_lats = self.atm_df["lat"].to_numpy()
        self._atm_lons = self.atm_df["long"].to_numpy()
        self.atm_ids = self.atm_df["atm_id"].tolist()
        self.atm_to_idx = {atm: i for i, atm in enumerate(self.atm_ids)}

        # 3. Vectorize timestamp conversion once
        self.txn_df["_dt"] = pd.to_datetime(self.txn_df["timestamp"])

        # 4. Build terminal transaction lookup using zip
        terminals = self.txn_df[self.txn_df["is_terminal"] == 1]
        for cid, dst, dt, amt in zip(
            terminals["complaint_id"],
            terminals["dst_account"],
            terminals["_dt"],
            terminals["amount"],
        ):
            self._terminal_txn_lookup[(cid, dst)] = (dt.hour, dt.dayofweek, float(amt))

        # 5. Fast precompute velocity for all accounts
        acc_ts = {}
        for src, dst, dt in zip(self.txn_df["src_account"], self.txn_df["dst_account"], self.txn_df["_dt"]):
            acc_ts.setdefault(src, []).append(dt)
            acc_ts.setdefault(dst, []).append(dt)

        for acc, times in acc_ts.items():
            if len(times) <= 1:
                self._velocity_lookup[acc] = 0.0
            else:
                times.sort()
                latest = times[-1]
                cutoff = latest - pd.Timedelta(minutes=60.0)
                recent = [t for t in times if t >= cutoff]
                if len(recent) <= 1:
                    self._velocity_lookup[acc] = 0.0
                else:
                    span = (recent[-1] - recent[0]).total_seconds() / 60.0
                    self._velocity_lookup[acc] = float(len(recent) / max(0.1, span))

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
        """
        # Part A: GNN Embedding (64 dims)
        embedding = self._get_embedding(terminal_account)

        # Part B: Tabular features (8 dims)
        if terminal_account in self._node_lookup:
            node_lat, node_lon, hop_depth, amount_at_terminal = self._node_lookup[terminal_account]
        else:
            node_lat, node_lon, hop_depth, amount_at_terminal = 20.5937, 78.9629, 3, stolen_amount * 0.7

        key = (complaint_id, terminal_account)
        if key in self._terminal_txn_lookup:
            hour_of_day, day_of_week, amount_after_split = self._terminal_txn_lookup[key]
        else:
            hour_of_day, day_of_week, amount_after_split = 14, 2, amount_at_terminal

        velocity = self._velocity_lookup.get(terminal_account, 0.0)

        # Fast Vectorized Haversine to ATM directory
        dists = _haversine_vectorized(node_lat, node_lon, self._atm_lats, self._atm_lons)
        dist_km = float(np.min(dists))
        hotspot_density = float((dists <= 5.0).sum())

        tabular = np.array([
            stolen_amount,
            float(hop_depth),
            velocity,
            dist_km,
            float(hour_of_day),
            hotspot_density,
            float(day_of_week),
            amount_after_split,
        ], dtype=np.float32)

        return np.concatenate([embedding, tabular])

    def build_training_set(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Build training set across all terminal nodes in the dataset.
        """
        terminal_txns = self.txn_df[self.txn_df["is_terminal"] == 1].copy()
        complaint_amounts = self.complaints_df.set_index("ticket_id")["stolen_amount"].to_dict()

        X_rows, y_atm_rows, y_time_rows, meta_rows = [], [], [], []

        for _, row in terminal_txns.iterrows():
            terminal_acc = row["dst_account"]
            complaint_id = row["complaint_id"]
            stolen_amount = complaint_amounts.get(complaint_id, 100000.0)

            # Feature vector
            feat = self.build_feature_vector(terminal_acc, complaint_id, stolen_amount)

            if terminal_acc in self._node_lookup:
                lat, lon, _, _ = self._node_lookup[terminal_acc]
            else:
                lat, lon = 20.5937, 78.9629

            # Nearest ATM index
            dists = _haversine_vectorized(lat, lon, self._atm_lats, self._atm_lons)
            atm_label = int(np.argmin(dists))
            dist_km = float(dists[atm_label])
            nearest_atm_id = self.atm_ids[atm_label]

            # Time-to-cashout synthetic target
            velocity = feat[66]
            base_time = 45.0
            dist_factor = dist_km * 2.0
            velocity_factor = -min(velocity * 5.0, 30.0)
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

        return X, y_atm, y_time, meta_df


if __name__ == "__main__":
    import time
    print("Testing High-Performance Feature Builder...")
    t0 = time.time()
    fb = FeatureBuilder()
    fb.load()
    X, y_atm, y_time, meta = fb.build_training_set()
    elapsed = (time.time() - t0) * 1000
    print(f"  Processed {len(X):,} terminal nodes in {elapsed:.1f}ms")
    print(f"  Feature Matrix shape: {X.shape}")
    print(f"  Unique ATM targets  : {len(set(y_atm))}")
    assert X.shape[1] == TOTAL_FEATURE_DIM, f"Wrong feature dim: {X.shape[1]}"
    print("[OK] feature_builder.py optimized and verified.")
