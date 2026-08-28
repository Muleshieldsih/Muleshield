# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b: High-Performance Hybrid Feature Builder (v2)
SIH26184 | MHA / I4C

Combines GraphSAGE 64-dim embeddings with 16 tabular features
into an 80-dim vector for XGBoost training and inference.

Accuracy Upgrades (v2):
  1. 3D Cartesian coordinates (X, Y, Z) — removes spherical lat/long bias in trees.
  2. Directional bearing angle to nearest ATM — distinguishes cardinal directions.
  3. Distances to 1st, 2nd, and 3rd nearest ATMs — richer spatial context.
  4. Bank affiliation features — same-bank ATM preference modelling.

Feature vector layout:
  [0:64]  GraphSAGE embedding (64-dim topological risk vector)
  [64]    stolen_amount
  [65]    hop_depth
  [66]    transaction_velocity        (txns/min from terminal account)
  [67]    dist_to_atm_1_km            (nearest ATM, Haversine)
  [68]    hour_of_day
  [69]    historical_hotspot_density  (ATM fraud count within 5km)
  [70]    day_of_week                 (0=Mon, 6=Sun)
  [71]    amount_after_split
  [72]    node_x                      (3D Cartesian X — tree-friendly coordinate)
  [73]    node_y                      (3D Cartesian Y)
  [74]    node_z                      (3D Cartesian Z)
  [75]    bearing_to_atm_1_deg        (bearing angle 0–360° to nearest ATM)
  [76]    dist_to_atm_2_km            (2nd nearest ATM distance)
  [77]    dist_to_atm_3_km            (3rd nearest ATM distance)
  [78]    is_nearest_same_bank        (binary: 1 if nearest ATM is same bank)
  [79]    nearest_same_bank_atm_dist  (km to closest same-bank ATM, or 999 if none)
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
TABULAR_DIM = 16          # upgraded from 8 → 16
TOTAL_FEATURE_DIM = EMBEDDING_DIM + TABULAR_DIM  # 80

TABULAR_FEATURE_NAMES = [
    "stolen_amount",
    "hop_depth",
    "transaction_velocity",
    "dist_to_atm_1_km",
    "hour_of_day",
    "historical_hotspot_density",
    "day_of_week",
    "amount_after_split",
    # ── v2 spatial accuracy upgrades ──────────────────────────────
    "node_x",
    "node_y",
    "node_z",
    "bearing_to_atm_1_deg",
    "dist_to_atm_2_km",
    "dist_to_atm_3_km",
    "is_nearest_same_bank",
    "nearest_same_bank_atm_dist",
]

FEATURE_NAMES = [f"emb_{i}" for i in range(EMBEDDING_DIM)] + TABULAR_FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# GEODESY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_EARTH_R = 6371.0  # km


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two GPS points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return _EARTH_R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _haversine_vectorized(
    lat1: float,
    lon1: float,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """Vectorized Haversine: one point vs many points. Returns km array."""
    phi1 = math.radians(lat1)
    phi2 = np.radians(lats2)
    dphi = phi2 - phi1
    dlam = np.radians(lons2 - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return _EARTH_R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def latlon_to_cartesian(lat: float, lon: float) -> tuple[float, float, float]:
    """
    Convert (lat, lon) in degrees to 3D Cartesian (X, Y, Z) on Earth's surface.
    Eliminates angular discontinuities that confuse decision tree splits.
    """
    phi = math.radians(lat)
    lam = math.radians(lon)
    x = _EARTH_R * math.cos(phi) * math.cos(lam)
    y = _EARTH_R * math.cos(phi) * math.sin(lam)
    z = _EARTH_R * math.sin(phi)
    return x, y, z


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute initial bearing (0–360°, clockwise from North) from point 1 to point 2.
    Allows XGBoost to distinguish ATMs in different compass quadrants.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


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
# HIGH PERFORMANCE FEATURE BUILDER (v2 — 80-dim)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Builds the 80-dim hybrid feature vector for each terminal mule node.

    v2 accuracy upgrades:
      • 3D Cartesian coordinates (node_x, node_y, node_z)
      • Bearing angle to nearest ATM
      • Distances to top-3 nearest ATMs
      • Bank affiliation match features
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
        self._node_lookup: dict = {}       # account_id → (lat, lon, hop, total_received, bank)
        self._terminal_txn_lookup: dict = {}
        self._velocity_lookup: dict = {}
        self._atm_lats: Optional[np.ndarray] = None
        self._atm_lons: Optional[np.ndarray] = None
        # Bank-indexed ATM arrays for affinity lookup
        self._atm_bank_arr: Optional[np.ndarray] = None   # str array of bank names

    def load(self) -> "FeatureBuilder":
        """Load all data sources and build O(1) indexed structures."""
        self.txn_df = pd.read_csv(self.transactions_path)
        self.node_df = pd.read_csv(self.node_features_path)
        self.atm_df = pd.read_csv(self.atm_path)
        self.complaints_df = pd.read_csv(self.complaints_path)

        with open(self.embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        # ── 1. Node lookup: lat, lon, hop, total_received, bank ────────────────
        # Derive bank from node_df if the column exists (generated by generate_data.py)
        has_bank_col = "bank" in self.node_df.columns
        for row_t in self.node_df.itertuples(index=False):
            acc_id = row_t.account_id
            bank = getattr(row_t, "bank", "UNKNOWN") if has_bank_col else "UNKNOWN"
            self._node_lookup[acc_id] = (
                float(row_t.lat),
                float(row_t.long),
                int(row_t.hop_depth),
                float(row_t.total_received),
                str(bank),
            )

        # ── 2. ATM coordinate arrays for vectorized distance ──────────────────
        self._atm_lats = self.atm_df["lat"].to_numpy()
        self._atm_lons = self.atm_df["long"].to_numpy()
        self.atm_ids = self.atm_df["atm_id"].tolist()
        self.atm_to_idx = {atm: i for i, atm in enumerate(self.atm_ids)}

        # Build bank array for affinity checks
        if "bank" in self.atm_df.columns:
            self._atm_bank_arr = self.atm_df["bank"].to_numpy(dtype=str)
        else:
            self._atm_bank_arr = np.array(["UNKNOWN"] * len(self.atm_ids), dtype=str)

        # ── 3. Vectorize timestamp conversion once ────────────────────────────
        self.txn_df["_dt"] = pd.to_datetime(self.txn_df["timestamp"])

        # ── 4. Terminal transaction lookup ────────────────────────────────────
        terminals = self.txn_df[self.txn_df["is_terminal"] == 1]
        for cid, dst, dt, amt in zip(
            terminals["complaint_id"],
            terminals["dst_account"],
            terminals["_dt"],
            terminals["amount"],
        ):
            self._terminal_txn_lookup[(cid, dst)] = (dt.hour, dt.dayofweek, float(amt))

        # ── 5. Precompute velocity for all accounts ───────────────────────────
        acc_ts: dict = {}
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

    def _spatial_features(
        self,
        node_lat: float,
        node_lon: float,
        node_bank: str,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Compute all 10 spatial/affinity tabular features in a single vectorized pass.

        Returns:
            dist1, hotspot_density, node_x, node_y, node_z,
            bearing1, dist2, dist3, is_same_bank, same_bank_dist
        """
        dists = _haversine_vectorized(node_lat, node_lon, self._atm_lats, self._atm_lons)

        # Sort to find top-3 nearest
        top3_idx = np.argpartition(dists, min(2, len(dists) - 1))[:3]
        top3_sorted = top3_idx[np.argsort(dists[top3_idx])]

        idx1 = int(top3_sorted[0])
        dist1 = float(dists[idx1])
        dist2 = float(dists[top3_sorted[1]]) if len(top3_sorted) > 1 else dist1 + 1.0
        dist3 = float(dists[top3_sorted[2]]) if len(top3_sorted) > 2 else dist2 + 1.0

        hotspot_density = float((dists <= 5.0).sum())

        # Cartesian coordinates
        node_x, node_y, node_z = latlon_to_cartesian(node_lat, node_lon)

        # Bearing to nearest ATM
        atm1_lat = float(self._atm_lats[idx1])
        atm1_lon = float(self._atm_lons[idx1])
        bear1 = bearing_degrees(node_lat, node_lon, atm1_lat, atm1_lon)

        # Bank affinity: nearest same-bank ATM distance
        same_bank_mask = self._atm_bank_arr == node_bank
        if same_bank_mask.any():
            same_bank_dists = dists[same_bank_mask]
            same_bank_dist = float(same_bank_dists.min())
            nearest_same_bank_idx = int(np.where(same_bank_mask)[0][np.argmin(same_bank_dists)])
            is_same_bank = 1.0 if nearest_same_bank_idx == idx1 else 0.0
        else:
            same_bank_dist = 999.0
            is_same_bank = 0.0

        return (
            dist1, hotspot_density,
            node_x, node_y, node_z,
            bear1, dist2, dist3,
            is_same_bank, same_bank_dist,
        )

    def build_feature_vector(
        self,
        terminal_account: str,
        complaint_id: str,
        stolen_amount: float,
        node_bank: str = "UNKNOWN",
    ) -> np.ndarray:
        """
        Build a single 80-dim feature vector for one terminal account.
        """
        # Part A: GNN Embedding (64 dims)
        embedding = self._get_embedding(terminal_account)

        # Part B: Node attributes
        if terminal_account in self._node_lookup:
            node_lat, node_lon, hop_depth, amount_at_terminal, node_bank = self._node_lookup[terminal_account]
        else:
            node_lat, node_lon, hop_depth, amount_at_terminal = 20.5937, 78.9629, 3, stolen_amount * 0.7

        key = (complaint_id, terminal_account)
        if key in self._terminal_txn_lookup:
            hour_of_day, day_of_week, amount_after_split = self._terminal_txn_lookup[key]
        else:
            hour_of_day, day_of_week, amount_after_split = 14, 2, amount_at_terminal

        velocity = self._velocity_lookup.get(terminal_account, 0.0)

        # Part C: All spatial/affinity features (single vectorized pass)
        (
            dist1, hotspot_density,
            node_x, node_y, node_z,
            bear1, dist2, dist3,
            is_same_bank, same_bank_dist,
        ) = self._spatial_features(node_lat, node_lon, node_bank)

        tabular = np.array([
            stolen_amount,
            float(hop_depth),
            velocity,
            dist1,                  # replaces old branch_distance_to_atm
            float(hour_of_day),
            hotspot_density,
            float(day_of_week),
            amount_after_split,
            node_x,
            node_y,
            node_z,
            bear1,
            dist2,
            dist3,
            is_same_bank,
            same_bank_dist,
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
            terminal_acc = str(row["dst_account"])
            complaint_id = str(row["complaint_id"])
            stolen_amount = complaint_amounts.get(complaint_id, 100000.0)

            feat = self.build_feature_vector(terminal_acc, complaint_id, stolen_amount)

            if terminal_acc in self._node_lookup:
                lat, lon, _, _, _ = self._node_lookup[terminal_acc]
            else:
                lat, lon = 20.5937, 78.9629

            # Nearest ATM label (ground truth for classification)
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
    print("Testing High-Performance Feature Builder v2 (80-dim)...")
    t0 = time.time()
    fb = FeatureBuilder()
    fb.load()
    X, y_atm, y_time, meta = fb.build_training_set()
    elapsed = (time.time() - t0) * 1000
    print(f"  Processed {len(X):,} terminal nodes in {elapsed:.1f}ms")
    print(f"  Feature Matrix shape: {X.shape}")
    print(f"  Unique ATM targets  : {len(set(y_atm))}")
    assert X.shape[1] == TOTAL_FEATURE_DIM, f"Wrong feature dim: {X.shape[1]}"
    print("[OK] feature_builder v2 (80-dim) verified.")
