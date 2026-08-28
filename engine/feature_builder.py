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

# Single source of truth for the crew-neighbourhood depth, shared with the
# serving path so training and inference cannot drift apart.
from xgb_model import CREW_HOPS

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

# Account-behaviour columns handed to the countdown regressor. Constant within a
# candidate group, so they are inert for the conditional-logit ranker (anything
# constant within a group cancels in a softmax) but directly informative for
# "how long until the withdrawal".
BEHAVIOUR_COLS = [
    "median_dwell_seconds",
    "account_age_days",
    "passthrough_ratio",
    "burst_out_5min",
    "night_txn_ratio",
]


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
        self._atm_risk: Optional[np.ndarray] = None       # cashout_risk_score per ATM
        self._atm_fraud_ct: Optional[np.ndarray] = None   # historical incidents per ATM
        self._atm_prior_count: Optional[np.ndarray] = None  # past cashouts per ATM
        self._acct_atm_hist: dict = {}                    # account -> {atm_idx: count}
        self._graph = None                                # undirected money graph
        self._gnn_head = None                             # (weight, bias) of the GNN head
        self._behaviour_lookup: dict = {}                 # account -> BEHAVIOUR_COLS values
        self.history_complaints: set = set()              # prior-only complaints
        self._chain_timing: dict = {}                     # complaint -> observed hop timing

    def load(self) -> "FeatureBuilder":
        """Load all data sources and build O(1) indexed structures."""
        self.txn_df = pd.read_csv(self.transactions_path)
        self.node_df = pd.read_csv(self.node_features_path)
        self.atm_df = pd.read_csv(self.atm_path)
        self.complaints_df = pd.read_csv(self.complaints_path)

        with open(self.embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        # ── 1. Node lookup: lat, lon, hop, total_received, bank ────────────────
        # generate_data.py writes the bank as "bank_name"; older dumps used "bank".
        bank_col = next(
            (c for c in ("bank_name", "bank") if c in self.node_df.columns), None
        )
        for row_t in self.node_df.itertuples(index=False):
            acc_id = row_t.account_id
            bank = getattr(row_t, bank_col, "UNKNOWN") if bank_col else "UNKNOWN"
            self._node_lookup[acc_id] = (
                float(row_t.lat),
                float(row_t.long),
                int(row_t.hop_depth),
                float(row_t.total_received),
                str(bank),
            )

        # ── 1b. Behavioural columns, for the countdown regressor ──────────────
        # The generator's delay law is
        #   22 + 1.9*travel_km + night_penalty(hour) + min(dwell_hours*2.2, 25) + noise
        # so an account's dwell behaviour drives up to 25 of the ~45-minute mean.
        # These columns are measured from the ledger and already feed the GNN;
        # they simply never reached the regressor.
        for col in BEHAVIOUR_COLS:
            if col not in self.node_df.columns:
                raise ValueError(
                    f"node_features.csv is missing '{col}'. "
                    "Regenerate: python scripts/generate_data.py"
                )
        beh = self.node_df.set_index("account_id")[BEHAVIOUR_COLS]
        self._behaviour_lookup = dict(
            zip(beh.index.astype(str), beh.to_numpy(dtype=np.float32))
        )

        # ── 2. ATM coordinate arrays for vectorized distance ──────────────────
        self._atm_lats = self.atm_df["lat"].to_numpy()
        self._atm_lons = self.atm_df["long"].to_numpy()
        self.atm_ids = self.atm_df["atm_id"].tolist()
        self.atm_to_idx = {atm: i for i, atm in enumerate(self.atm_ids)}

        # Build bank array for affinity checks.
        # atm_directory.csv stores the operator as "bank_name".
        atm_bank_col = next(
            (c for c in ("bank_name", "bank") if c in self.atm_df.columns), None
        )
        if atm_bank_col:
            self._atm_bank_arr = self.atm_df[atm_bank_col].to_numpy(dtype=str)
        else:
            self._atm_bank_arr = np.array(["UNKNOWN"] * len(self.atm_ids), dtype=str)

        # Per-ATM priors used by the candidate-ranking features.
        self._atm_risk = (
            self.atm_df["cashout_risk_score"].to_numpy(dtype=np.float32)
            if "cashout_risk_score" in self.atm_df.columns else None
        )
        self._atm_fraud_ct = (
            self.atm_df["historical_fraud_count"].to_numpy(dtype=np.float32)
            if "historical_fraud_count" in self.atm_df.columns else None
        )

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

    # ── Candidate-ranking construction ────────────────────────────────────────

    CANDIDATE_K = 25          # ATMs considered per cashout
    CANDIDATE_FEATURES = 12   # per-candidate feature block appended to the base
    HISTORY_FRACTION = 0.50   # earliest share of complaints used only as priors

    def build_chain_timing(self) -> None:
        """
        Measure how fast each traced laundering chain is actually moving.

        The countdown target is driven by the crew's operating speed. The only
        proxy the regressor had for that was `median_dwell_seconds`, measured
        over ALL of an account's traffic - and since a mule now carries mostly
        ordinary banking activity, that statistic is dominated by civilian
        behaviour and barely reflects the crew at all. R2 sat at 0.16 against an
        achievable 0.89.

        The chain's own hop-to-hop timing is the direct measurement, and it is
        fully available at prediction time: when a 1930 complaint is traced, the
        timestamps of every hop are on the statement in front of you. Nothing
        here uses the cashout, or anything after the terminal transfer.
        """
        df = self.txn_df[self.txn_df["is_terminal"].notna()].copy()
        df = df[df["complaint_id"].astype(str).str.len() > 0]
        df["_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["_dt"]).sort_values(["complaint_id", "_dt"])

        self._chain_timing = {}
        for cid, grp in df.groupby("complaint_id", sort=False):
            ts = grp["_dt"].to_numpy()
            if len(ts) < 2:
                self._chain_timing[cid] = (0.0, 0.0, 0.0)
                continue
            gaps = np.diff(ts).astype("timedelta64[s]").astype(float)
            gaps = gaps[gaps >= 0]
            if not len(gaps):
                self._chain_timing[cid] = (0.0, 0.0, 0.0)
                continue
            span = float((ts[-1] - ts[0]) / np.timedelta64(1, "s"))
            self._chain_timing[cid] = (
                float(np.median(gaps)),   # typical hop-to-hop delay
                float(gaps.min()),        # fastest hop = upper bound on crew speed
                span,                     # total time the chain has been running
            )

    def build_cashout_priors(self) -> None:
        """
        Build ATM cashout priors from the EARLIEST slice of complaints only.

        Syndicates reuse cashout points they have already tested, so where a
        crew withdrew before is the strongest non-distance signal available.
        Counting those priors over the whole dataset would leak the test labels,
        so the complaint timeline is cut: the earliest `HISTORY_FRACTION` acts as
        case history and is excluded from training and evaluation entirely —
        which is also how this would run in deployment, where you only ever have
        yesterday's cases.
        """
        import networkx as nx

        comp = self.complaints_df.copy()
        comp["_ts"] = pd.to_datetime(comp["complaint_timestamp"])
        comp = comp.sort_values("_ts")
        cut = int(len(comp) * self.HISTORY_FRACTION)
        self.history_complaints = set(comp["ticket_id"].iloc[:cut])

        hist = self.txn_df[
            (self.txn_df["is_terminal"] == 1)
            & (self.txn_df["complaint_id"].isin(self.history_complaints))
        ]

        # Global prior: how often each ATM has served a cashout before.
        self._atm_prior_count = np.zeros(len(self.atm_ids), dtype=np.float32)
        # Per-account prior: which ATMs this specific account has used.
        acct_atm: dict[str, dict[int, int]] = {}
        for acc, atm in zip(hist["dst_account"], hist["cashout_atm_id"]):
            if not isinstance(atm, str) or atm not in self.atm_to_idx:
                continue
            i = self.atm_to_idx[atm]
            self._atm_prior_count[i] += 1.0
            acct_atm.setdefault(str(acc), {})
            acct_atm[str(acc)][i] = acct_atm[str(acc)].get(i, 0) + 1

        # Undirected view of the money graph, used to reach an account's crew.
        g = nx.Graph()
        g.add_edges_from(zip(self.txn_df["src_account"], self.txn_df["dst_account"]))
        self._graph = g
        self._acct_atm_hist = acct_atm

    def neighbourhood_prior(self, account: str, cand_idx: np.ndarray) -> np.ndarray:
        """
        Count historical cashouts at each candidate ATM by the account's crew.

        The crew is approximated by the account's CREW_HOPS-hop neighbourhood in
        the money graph — the accounts it actually transacts with. This is the
        feature the GNN embedding complements: the embedding says *who* this
        account moves money with, and this says *where those people cash out*.

        CREW_HOPS is imported from xgb_model so the serving path expands exactly
        the same neighbourhood; the two drifted apart once already (3 hops here,
        2 at inference), which silently weakened the feature in production.
        """
        counts = np.zeros(len(cand_idx), dtype=np.float32)
        g = getattr(self, "_graph", None)
        if g is None or account not in g:
            return counts

        # A terminal's crew-mates sit behind the accounts that fed it, so two
        # hops frequently stops just short of the rest of the ring.
        neigh = {account}
        frontier = {account}
        for _ in range(CREW_HOPS):
            nxt = set()
            for n in frontier:
                nxt.update(g.neighbors(n))
            nxt -= neigh
            if not nxt:
                break
            neigh |= nxt
            frontier = nxt

        pos = {int(a): j for j, a in enumerate(cand_idx)}
        for acc in neigh:
            for atm_i, c in self._acct_atm_hist.get(acc, {}).items():
                j = pos.get(atm_i)
                if j is not None:
                    counts[j] += c
        return counts

    def candidate_block(
        self,
        node_lat: float,
        node_lon: float,
        node_bank: str = "UNKNOWN",
        k: int | None = None,
        account: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the K nearest ATM candidates for one terminal account.

        Returns:
            (candidate_indices, candidate_feature_matrix[K, CANDIDATE_FEATURES])

        Predicting the cashout ATM as a 953-way classification over the whole
        national directory cannot work with a few thousand training rows — the
        model saw about five examples per class and scored below a plain
        "walk to the nearest ATM" rule. Framing it as ranking a local candidate
        set is both learnable and closer to how the choice is actually made: an
        operator only cares about the handful of ATMs the suspect could reach.
        """
        k = k or self.CANDIDATE_K
        dists = _haversine_vectorized(node_lat, node_lon, self._atm_lats, self._atm_lons)
        k = min(k, len(dists))
        order = np.argpartition(dists, k - 1)[:k]
        order = order[np.argsort(dists[order])]

        d = dists[order]
        nearest = max(float(d[0]), 1e-6)

        risk = (self._atm_risk[order] if self._atm_risk is not None
                else np.zeros(k, dtype=np.float32))
        fraud_ct = (self._atm_fraud_ct[order] if self._atm_fraud_ct is not None
                    else np.zeros(k, dtype=np.float32))
        same_bank = (self._atm_bank_arr[order] == node_bank).astype(np.float32)

        # Local ATM density around each candidate — a busy cluster offers cover.
        density = np.array([
            float((_haversine_vectorized(
                float(self._atm_lats[i]), float(self._atm_lons[i]),
                self._atm_lats, self._atm_lons) <= 3.0).sum())
            for i in order
        ], dtype=np.float32)

        # Prior cashout activity — global, and specific to this account's crew.
        atm_prior = (self._atm_prior_count[order]
                     if getattr(self, "_atm_prior_count", None) is not None
                     else np.zeros(k, dtype=np.float32))
        crew_prior = (self.neighbourhood_prior(account, order)
                      if account is not None else np.zeros(k, dtype=np.float32))

        block = np.column_stack([
            d.astype(np.float32),                    # absolute distance (km)
            np.arange(k, dtype=np.float32),          # distance rank
            (d / nearest).astype(np.float32),        # distance relative to nearest
            risk.astype(np.float32),                 # cashout_risk_score
            np.log1p(fraud_ct).astype(np.float32),   # historical incidents
            same_bank,                               # own-bank ATM
            density,                                 # ATMs within 3 km
            np.exp(-d / 5.0).astype(np.float32),     # distance-decay prior
            np.log1p(atm_prior).astype(np.float32),  # past cashouts at this ATM
            np.log1p(crew_prior).astype(np.float32), # past cashouts here by the crew
            (crew_prior > 0).astype(np.float32),     # crew has used this ATM at all
            np.full(k, float((crew_prior > 0).any()), dtype=np.float32),  # any signal?
        ]).astype(np.float32)

        return order, block

    RANK_CONTEXT_DIM = 7 + len(BEHAVIOUR_COLS) + 3   # 15 (+3 chain timing)

    def gnn_mule_probability(self, account: str) -> float:
        """
        Trained GraphSAGE mule probability for an account, sigmoid(Wh + b).

        The ranker takes this single scalar rather than the raw 64-dim
        embedding. Every embedding dimension is identical across the 25
        candidates of one cashout, so it carries no information about which
        candidate to rank first — it only bloats the feature space and gives
        the trees 64 useless columns to overfit on. The graph signal that does
        vary per candidate is `crew_prior`, and that stays.
        """
        if self._gnn_head is None:
            self._load_gnn_head()
        emb = self.embeddings.get(account) if self.embeddings else None
        if emb is None or self._gnn_head is None:
            return 0.0
        w, b = self._gnn_head
        z = float(np.dot(w, np.asarray(emb, dtype=np.float64)) + b)
        return float(1.0 / (1.0 + np.exp(-z))) if z >= 0 else float(np.exp(z) / (1.0 + np.exp(z)))

    def _load_gnn_head(self) -> None:
        """Lift the classification head out of the GraphSAGE checkpoint."""
        try:
            import torch
            ckpt = torch.load(ROOT / "models" / "graphsage_mule.pt",
                              map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            w = sd["classifier.weight"].detach().cpu().numpy().reshape(-1)
            b = float(sd["classifier.bias"].detach().cpu().numpy().reshape(-1)[0])
            self._gnn_head = (w, b)
        except Exception:
            self._gnn_head = None

    def rank_context(self, base_vec: np.ndarray, account: str,
                     complaint_id: str | None = None) -> np.ndarray:
        """
        Per-cashout context handed to the ranker alongside the candidate block.

        Holds the case-level quantities that modulate where a cashout happens —
        amount, depth, velocity, time of day — plus the account's GNN mule
        probability, in place of the full embedding.
        """
        beh = self._behaviour_lookup.get(
            account, np.zeros(len(BEHAVIOUR_COLS), dtype=np.float32)
        )
        return np.concatenate([
            np.array([
                base_vec[64],    # stolen_amount
                base_vec[65],    # hop_depth
                base_vec[66],    # transaction_velocity
                base_vec[68],    # hour_of_day
                base_vec[70],    # day_of_week
                base_vec[71],    # amount_after_split
                self.gnn_mule_probability(account),
            ], dtype=np.float32),
            beh.astype(np.float32),
            # Observed speed of THIS chain: median hop gap, fastest hop, elapsed
            # span. Log-scaled because hop gaps span seconds to days.
            np.log1p(np.array(
                self._chain_timing.get(complaint_id, (0.0, 0.0, 0.0)),
                dtype=np.float32,
            )),
        ])

    def build_ranking_set(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Build the pairwise ranking training set.

        Each terminal cashout expands into K rows — one per candidate ATM —
        labelled 1 for the ATM the withdrawal actually happened at and 0 for the
        rest. A single binary model then scores any candidate, which also means
        it generalises to ATMs never seen during training.

        Returns:
            (X[N*K, 17], y[N*K], group_ids[N*K], meta_df[N rows])
        """
        self.build_cashout_priors()
        self.build_chain_timing()

        terminal_txns = self.txn_df[self.txn_df["is_terminal"] == 1].copy()
        complaint_amounts = self.complaints_df.set_index("ticket_id")["stolen_amount"].to_dict()

        X_rows, y_rows, grp_rows, meta_rows = [], [], [], []
        skipped = 0

        for gid, (_, row) in enumerate(terminal_txns.iterrows()):
            terminal_acc = str(row["dst_account"])
            complaint_id = str(row["complaint_id"])

            # Complaints in the history window built the priors, so training or
            # scoring on them would be scoring on their own labels.
            if complaint_id in self.history_complaints:
                continue

            atm_id = row.get("cashout_atm_id")
            if not isinstance(atm_id, str) or atm_id not in self.atm_to_idx:
                skipped += 1
                continue

            stolen_amount = complaint_amounts.get(complaint_id, 100000.0)
            base = self.build_feature_vector(terminal_acc, complaint_id, stolen_amount)

            if terminal_acc in self._node_lookup:
                lat, lon, _, _, bank = self._node_lookup[terminal_acc]
            else:
                lat, lon, bank = 20.5937, 78.9629, "UNKNOWN"

            cand_idx, block = self.candidate_block(lat, lon, bank, account=terminal_acc)
            truth = int(self.atm_to_idx[atm_id])

            # A cashout outside the candidate radius cannot be ranked; recording
            # it as unreachable keeps the accuracy denominator honest.
            reachable = truth in set(int(i) for i in cand_idx)

            ctx = self.rank_context(base, terminal_acc, complaint_id)
            for j, atm_i in enumerate(cand_idx):
                X_rows.append(np.concatenate([ctx, block[j]]))
                y_rows.append(1 if int(atm_i) == truth else 0)
                grp_rows.append(gid)

            meta_rows.append({
                "group_id": gid,
                "terminal_account": terminal_acc,
                "complaint_id": complaint_id,
                "cashout_atm_id": atm_id,
                "time_to_cashout_min": float(row["time_to_cashout_min"]),
                "truth_in_candidates": reachable,
                "lat": lat,
                "lon": lon,
            })

        if skipped:
            print(f"      [warn] skipped {skipped} terminals with no cashout label")

        return (
            np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int8),
            np.array(grp_rows, dtype=np.int32),
            pd.DataFrame(meta_rows),
        )

    def build_training_set(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Build training set across all terminal nodes in the dataset.
        """
        terminal_txns = self.txn_df[self.txn_df["is_terminal"] == 1].copy()
        complaint_amounts = self.complaints_df.set_index("ticket_id")["stolen_amount"].to_dict()

        X_rows, y_atm_rows, y_time_rows, meta_rows = [], [], [], []

        # Ground truth comes from the ledger, not from the features.
        #
        # This previously set `atm_label = argmin(distance)` and derived the
        # countdown from a closed-form line in two of its own inputs. Since
        # `dist_to_atm_1_km` is feature #67, the classifier was being asked to
        # find the nearest ATM while holding the distance to it — a tautology
        # that inflated Top-3 accuracy to 98%. The cashout ATM and delay are now
        # sampled by a behavioural choice model in scripts/generate_data.py and
        # read from `cashout_atm_id` / `time_to_cashout_min`.
        has_ground_truth = "cashout_atm_id" in terminal_txns.columns
        if not has_ground_truth:
            raise ValueError(
                "transactions.csv has no 'cashout_atm_id' column. "
                "Regenerate the dataset: python scripts/generate_data.py"
            )

        skipped = 0
        for _, row in terminal_txns.iterrows():
            terminal_acc = str(row["dst_account"])
            complaint_id = str(row["complaint_id"])
            stolen_amount = complaint_amounts.get(complaint_id, 100000.0)

            atm_id = row.get("cashout_atm_id")
            if not isinstance(atm_id, str) or atm_id not in self.atm_to_idx:
                skipped += 1
                continue

            feat = self.build_feature_vector(terminal_acc, complaint_id, stolen_amount)

            if terminal_acc in self._node_lookup:
                lat, lon, _, _, _ = self._node_lookup[terminal_acc]
            else:
                lat, lon = 20.5937, 78.9629

            atm_label = int(self.atm_to_idx[atm_id])
            dist_km = float(haversine_km(
                lat, lon,
                float(self._atm_lats[atm_label]), float(self._atm_lons[atm_label]),
            ))
            time_to_cashout = float(row["time_to_cashout_min"])

            X_rows.append(feat)
            y_atm_rows.append(atm_label)
            y_time_rows.append(time_to_cashout)
            meta_rows.append({
                "terminal_account": terminal_acc,
                "complaint_id": complaint_id,
                "cashout_atm_id": atm_id,
                "dist_to_atm_km": dist_km,
                "time_to_cashout_min": time_to_cashout,
                "lat": lat,
                "lon": lon,
            })

        if skipped:
            print(f"      [warn] skipped {skipped} terminals with no cashout label")

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
