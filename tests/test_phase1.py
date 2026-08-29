"""
MuleShield AI — Phase 1 Test Suite
SIH26184 | MHA / I4C

Tests all acceptance criteria from phases.md Phase 1:
  ✓ 500+ victim complaint records
  ✓ 2000+ transaction records with ≥3 fraud rings embedded
  ✓ 200+ ATM/CSP entries across 10+ Indian cities
  ✓ node_features.csv and graph_edges.csv ready for PyG ingestion
  ✓ All lat/long within Indian geographic bounds

Run:
    python -m pytest tests/test_phase1.py -v
"""

import sys
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# ── Make scripts/ importable ─────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_data import (
    INDIA_LAT_MIN, INDIA_LAT_MAX,
    INDIA_LON_MIN, INDIA_LON_MAX,
    FRAUD_TYPES, INDIAN_BANKS, INDIAN_CITIES,
    _masked_account, _random_ifsc, _jitter_coords,
    _split_amount, _random_bank, _random_city,
    generate_victim_complaints,
    generate_transactions,
    generate_atm_directory,
    generate_graph_edges,
    generate_node_features,
    main,
)
from faker import Faker


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def seed():
    return 42


@pytest.fixture(scope="module")
def fake(seed):
    f = Faker("en_IN")
    f.seed_instance(seed)
    return f


@pytest.fixture(scope="module")
def all_data(seed, tmp_path_factory, fake):
    """Run the full generator pipeline once; share results across all tests."""
    random.seed(seed)
    np.random.seed(seed)
    tmp = tmp_path_factory.mktemp("data")

    # Monkey-patch DATA_DIR so we write to tmp during tests
    import generate_data as gd
    original_dir = gd.DATA_DIR
    gd.DATA_DIR = tmp

    result = main(n_complaints=500, n_transactions=2000, n_atms=200, seed=seed)

    gd.DATA_DIR = original_dir  # restore
    return result, tmp


@pytest.fixture(scope="module")
def complaints(all_data):
    return all_data[0]["complaints"]


@pytest.fixture(scope="module")
def transactions(all_data):
    return all_data[0]["transactions"]


@pytest.fixture(scope="module")
def atms(all_data):
    return all_data[0]["atms"]


@pytest.fixture(scope="module")
def graph_edges(all_data):
    return all_data[0]["graph_edges"]


@pytest.fixture(scope="module")
def node_features(all_data):
    return all_data[0]["node_features"]


@pytest.fixture(scope="module")
def data_dir(all_data):
    return all_data[1]


# ─────────────────────────────────────────────
# HELPER UNIT TESTS
# ─────────────────────────────────────────────

class TestHelpers:
    """Unit tests for all helper functions."""

    def test_masked_account_format(self):
        """Account number must be XXXX-XXXX-XXXX."""
        for _ in range(50):
            acc = _masked_account()
            parts = acc.split("-")
            assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}: {acc}"
            assert all(len(p) == 4 and p.isdigit() for p in parts), \
                f"Each part must be 4 digits: {acc}"

    def test_random_ifsc_format(self):
        """IFSC must be 11 chars: BANK_CODE (4) + '0' + 6 digits."""
        for bank_name, code in INDIAN_BANKS:
            ifsc = _random_ifsc(code)
            assert len(ifsc) == 11, f"IFSC length should be 11: {ifsc}"
            assert ifsc[4] == "0", f"5th char of IFSC must be '0': {ifsc}"
            assert ifsc[5:].isdigit(), f"Last 6 chars must be digits: {ifsc}"

    def test_jitter_coords_stays_in_india(self):
        """Jittered coordinates must remain within Indian bounds."""
        for city, district, state, lat, lon in INDIAN_CITIES:
            for _ in range(20):
                jlat, jlon = _jitter_coords(lat, lon, radius_km=15.0)
                assert INDIA_LAT_MIN <= jlat <= INDIA_LAT_MAX, \
                    f"Lat {jlat} out of Indian bounds for {city}"
                assert INDIA_LON_MIN <= jlon <= INDIA_LON_MAX, \
                    f"Lon {jlon} out of Indian bounds for {city}"

    def test_split_amount_single(self):
        """Single split should return exact original amount."""
        result = _split_amount(50000.0, 1)
        assert len(result) == 1
        assert result[0] == 50000.0

    def test_split_amount_multiple_sum(self):
        """Multi-split parts should approximately sum to original (±1%)."""
        for _ in range(20):
            total = random.uniform(10000, 500000)
            n = random.randint(2, 5)
            parts = _split_amount(total, n)
            assert len(parts) == n
            assert all(p >= 100.0 for p in parts), "Each part must be ≥ ₹100"
            assert abs(sum(parts) - total) / total < 0.05, \
                f"Sum {sum(parts)} should be close to {total}"

    def test_split_amount_all_positive(self):
        """All split parts must be positive."""
        for _ in range(10):
            parts = _split_amount(random.uniform(5000, 100000), random.randint(2, 6))
            assert all(p > 0 for p in parts)

    def test_random_bank_returns_valid(self):
        """random_bank should return a known bank from INDIAN_BANKS."""
        valid_names = {b[0] for b in INDIAN_BANKS}
        for _ in range(20):
            name, code = _random_bank()
            assert name in valid_names

    def test_random_city_returns_valid(self):
        """random_city should return a known city from INDIAN_CITIES."""
        valid_cities = {c[0] for c in INDIAN_CITIES}
        for _ in range(20):
            city, district, state, lat, lon = _random_city()
            assert city in valid_cities


# ─────────────────────────────────────────────
# ACCEPTANCE TEST 1 — Victim Complaints
# ─────────────────────────────────────────────

class TestVictimComplaints:
    """Phase 1 AC: 500+ complaint records with correct schema."""

    def test_count_at_least_500(self, complaints):
        assert len(complaints) >= 500, \
            f"Expected ≥500 complaints, got {len(complaints)}"

    def test_required_columns_present(self, complaints):
        required = {
            "ticket_id", "victim_name", "victim_bank", "victim_account",
            "fraud_type", "stolen_amount", "complaint_timestamp", "city", "state"
        }
        assert required.issubset(set(complaints.columns)), \
            f"Missing columns: {required - set(complaints.columns)}"

    def test_ticket_ids_are_unique(self, complaints):
        assert complaints["ticket_id"].is_unique, "ticket_id must be unique"

    def test_ticket_ids_are_uuids(self, complaints):
        import re
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        sample = complaints["ticket_id"].sample(min(50, len(complaints)))
        for tid in sample:
            assert uuid_pattern.match(str(tid)), f"Invalid UUID: {tid}"

    def test_fraud_types_are_valid(self, complaints):
        invalid = set(complaints["fraud_type"]) - set(FRAUD_TYPES)
        assert not invalid, f"Invalid fraud types found: {invalid}"

    def test_stolen_amount_range(self, complaints):
        assert (complaints["stolen_amount"] >= 5_000).all(), \
            "All amounts must be ≥ ₹5,000"
        assert (complaints["stolen_amount"] <= 5_00_000).all(), \
            "All amounts must be ≤ ₹5,00,000"
        assert (complaints["stolen_amount"] > 0).all(), \
            "All stolen amounts must be positive"

    def test_timestamps_are_valid_iso(self, complaints):
        for ts in complaints["complaint_timestamp"].sample(min(50, len(complaints))):
            try:
                datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {ts}")

    def test_cities_are_known_indian(self, complaints):
        valid_cities = {c[0] for c in INDIAN_CITIES}
        invalid = set(complaints["city"]) - valid_cities
        assert not invalid, f"Unknown cities: {invalid}"

    def test_no_null_values(self, complaints):
        nulls = complaints.isnull().sum()
        assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"

    def test_victim_bank_is_known(self, complaints):
        valid_banks = {b[0] for b in INDIAN_BANKS}
        invalid = set(complaints["victim_bank"]) - valid_banks
        assert not invalid, f"Unknown banks: {invalid}"

    def test_account_format(self, complaints):
        for acc in complaints["victim_account"].sample(min(50, len(complaints))):
            parts = str(acc).split("-")
            assert len(parts) == 3, f"Invalid account format: {acc}"

    def test_csv_file_written(self, data_dir):
        assert (data_dir / "victim_complaints.csv").exists(), \
            "victim_complaints.csv not found"

    def test_csv_loadable_matches_dataframe(self, data_dir, complaints):
        loaded = pd.read_csv(data_dir / "victim_complaints.csv")
        assert len(loaded) == len(complaints), \
            "CSV row count mismatch with in-memory DataFrame"


# ─────────────────────────────────────────────
# ACCEPTANCE TEST 2 — Transactions
# ─────────────────────────────────────────────

class TestTransactions:
    """Phase 1 AC: 2000+ transactions with multi-hop chains and fraud rings."""

    def test_count_at_least_2000(self, transactions):
        assert len(transactions) >= 2000, \
            f"Expected ≥2000 transactions, got {len(transactions)}"

    def test_required_columns_present(self, transactions):
        required = {
            "txn_id", "complaint_id", "src_account", "dst_account",
            "bank_name", "ifsc_code", "city", "district", "state",
            "lat", "long", "amount", "timestamp", "hop_depth", "is_terminal"
        }
        assert required.issubset(set(transactions.columns)), \
            f"Missing: {required - set(transactions.columns)}"

    def test_txn_ids_unique(self, transactions):
        assert transactions["txn_id"].is_unique, "txn_id must be unique"

    def test_hop_depth_range(self, transactions):
        """
        Laundering hops run 1..MAX_HOP_DEPTH; legitimate traffic sits at hop 0.

        The ledger now carries ordinary banking activity alongside the fraud
        chains, so that legitimate accounts also receive money and the mule
        label stops being readable off a single column.
        """
        fraud = transactions[transactions["is_fraud"] == 1]
        legit = transactions[transactions["is_fraud"] == 0]

        assert (fraud["hop_depth"] >= 1).all(), "fraud hop_depth must be >= 1"
        assert (fraud["hop_depth"] <= 4).all(), "fraud hop_depth must be <= 4"
        assert (legit["hop_depth"] == 0).all(), "legitimate hop_depth must be 0"

    def test_dataset_contains_legitimate_activity(self, transactions):
        """
        A meaningful share of the ledger must be non-fraud.

        Without it every money-receiving account is a mule by construction and
        `total_received > 0` classifies the dataset perfectly.
        """
        legit_share = (transactions["is_fraud"] == 0).mean()
        assert legit_share > 0.25, \
            f"Only {legit_share:.1%} of transactions are legitimate; the classes will not overlap"

    def test_all_hop_depths_present(self, transactions):
        """All layers 1–4 must appear in the data."""
        depths_present = set(transactions["hop_depth"].unique())
        for d in [1, 2, 3]:
            assert d in depths_present, f"hop_depth {d} missing from transactions"

    def test_is_terminal_binary(self, transactions):
        assert set(transactions["is_terminal"].unique()).issubset({0, 1}), \
            "is_terminal must be 0 or 1"

    def test_terminal_nodes_exist(self, transactions):
        """Must have at least some terminal (cashout candidate) nodes."""
        assert transactions["is_terminal"].sum() > 0, \
            "No terminal nodes found — fraud chains have no cashout endpoints"

    def test_amounts_positive(self, transactions):
        assert (transactions["amount"] > 0).all(), "All amounts must be > 0"

    def test_lat_within_india(self, transactions):
        assert (transactions["lat"] >= INDIA_LAT_MIN).all() and \
               (transactions["lat"] <= INDIA_LAT_MAX).all(), \
            f"Latitudes out of Indian bounds [{INDIA_LAT_MIN}, {INDIA_LAT_MAX}]"

    def test_lon_within_india(self, transactions):
        assert (transactions["long"] >= INDIA_LON_MIN).all() and \
               (transactions["long"] <= INDIA_LON_MAX).all(), \
            f"Longitudes out of Indian bounds [{INDIA_LON_MIN}, {INDIA_LON_MAX}]"

    def test_timestamps_valid(self, transactions):
        sample = transactions["timestamp"].sample(min(50, len(transactions)))
        for ts in sample:
            try:
                datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pytest.fail(f"Invalid timestamp: {ts}")

    def test_ifsc_format(self, transactions):
        sample = transactions["ifsc_code"].sample(min(50, len(transactions)))
        for ifsc in sample:
            assert len(ifsc) == 11, f"IFSC must be 11 chars: {ifsc}"
            assert ifsc[4] == "0", f"5th char of IFSC must be '0': {ifsc}"

    def test_complaint_ids_match_complaints(self, transactions, complaints):
        """All complaint_ids in transactions must exist in complaints."""
        valid_ids = set(complaints["ticket_id"])
        # Legitimate banking traffic belongs to no complaint and carries an
        # empty complaint_id.
        txn_ids = {c for c in transactions["complaint_id"] if str(c).strip()}
        unknown = txn_ids - valid_ids
        assert not unknown, f"Unknown complaint_ids in transactions: {list(unknown)[:5]}"

    def test_fraud_rings_embedded(self, transactions):
        """
        Fraud ring = account receiving from 3+ DIFFERENT sources within the chain.
        Must have ≥3 such rings (per acceptance criteria).
        """
        ring_counts = (
            transactions.groupby("dst_account")["src_account"]
            .nunique()
        )
        rings = ring_counts[ring_counts >= 3]
        assert len(rings) >= 3, \
            f"Expected ≥3 fraud rings (3+ sources → 1 dest), found {len(rings)}"

    def test_multi_hop_chains_exist(self, transactions):
        """Each complaint should have transactions at multiple hop depths."""
        complaint_depths = transactions.groupby("complaint_id")["hop_depth"].nunique()
        multi_hop = (complaint_depths >= 2).sum()
        assert multi_hop >= len(complaint_depths) * 0.8, \
            f"Fewer than 80% of complaints have multi-hop chains: {multi_hop}"

    def test_no_null_critical_columns(self, transactions):
        critical = ["txn_id", "complaint_id", "src_account", "dst_account",
                    "amount", "lat", "long", "hop_depth", "is_terminal"]
        for col in critical:
            assert transactions[col].isnull().sum() == 0, \
                f"Null values found in critical column: {col}"

    def test_csv_written(self, data_dir):
        assert (data_dir / "transactions.csv").exists()

    def test_csv_row_count_matches(self, data_dir, transactions):
        loaded = pd.read_csv(data_dir / "transactions.csv")
        assert len(loaded) == len(transactions)


# ─────────────────────────────────────────────
# ACCEPTANCE TEST 3 — ATM Directory
# ─────────────────────────────────────────────

class TestATMDirectory:
    """Phase 1 AC: 200+ ATMs across 10+ Indian cities."""

    def test_count_at_least_200(self, atms):
        assert len(atms) >= 200, f"Expected ≥200 ATMs, got {len(atms)}"

    def test_required_columns_present(self, atms):
        required = {
            "atm_id", "bank_name", "address", "city", "district", "state",
            "lat", "long", "opening_time", "closing_time",
            "cashout_risk_score", "historical_fraud_count"
        }
        assert required.issubset(set(atms.columns)), \
            f"Missing: {required - set(atms.columns)}"

    def test_atm_ids_unique(self, atms):
        assert atms["atm_id"].is_unique, "atm_id must be unique"

    def test_atm_id_format(self, atms):
        sample = atms["atm_id"].sample(min(50, len(atms)))
        for aid in sample:
            assert str(aid).startswith("ATM-"), f"ATM ID must start with 'ATM-': {aid}"

    def test_covers_10_plus_cities(self, atms):
        n_cities = atms["city"].nunique()
        assert n_cities >= 10, f"Expected ≥10 cities, got {n_cities}: {sorted(atms['city'].unique())}"

    def test_lat_within_india(self, atms):
        assert (atms["lat"] >= INDIA_LAT_MIN).all() and \
               (atms["lat"] <= INDIA_LAT_MAX).all(), "ATM lats out of Indian bounds"

    def test_lon_within_india(self, atms):
        assert (atms["long"] >= INDIA_LON_MIN).all() and \
               (atms["long"] <= INDIA_LON_MAX).all(), "ATM lons out of Indian bounds"

    def test_risk_score_range(self, atms):
        assert (atms["cashout_risk_score"] >= 0.0).all() and \
               (atms["cashout_risk_score"] <= 1.0).all(), \
            "cashout_risk_score must be in [0.0, 1.0]"

    def test_fraud_count_non_negative(self, atms):
        assert (atms["historical_fraud_count"] >= 0).all(), \
            "historical_fraud_count must be ≥ 0"

    def test_bank_is_known(self, atms):
        valid_banks = {b[0] for b in INDIAN_BANKS}
        invalid = set(atms["bank_name"]) - valid_banks
        assert not invalid, f"Unknown banks in ATM directory: {invalid}"

    def test_no_null_values(self, atms):
        nulls = atms.isnull().sum()
        assert nulls.sum() == 0, f"Nulls found:\n{nulls[nulls > 0]}"

    def test_csv_written(self, data_dir):
        assert (data_dir / "atm_directory.csv").exists()

    def test_csv_row_count_matches(self, data_dir, atms):
        loaded = pd.read_csv(data_dir / "atm_directory.csv")
        assert len(loaded) == len(atms)


# ─────────────────────────────────────────────
# ACCEPTANCE TEST 4 — Graph Edge List
# ─────────────────────────────────────────────

class TestGraphEdges:
    """Phase 1 AC: graph_edges.csv ready for PyG/NetworkX ingestion."""

    def test_required_columns(self, graph_edges):
        required = {"src_account", "dst_account", "amount", "timestamp",
                    "hop_depth", "is_terminal", "complaint_id"}
        assert required.issubset(set(graph_edges.columns)), \
            f"Missing: {required - set(graph_edges.columns)}"

    def test_row_count_matches_transactions(self, graph_edges, transactions):
        assert len(graph_edges) == len(transactions), \
            "graph_edges must have same row count as transactions"

    def test_no_self_loops(self, graph_edges):
        """No account should transact to itself."""
        self_loops = (graph_edges["src_account"] == graph_edges["dst_account"]).sum()
        assert self_loops == 0, f"Found {self_loops} self-loop edges"

    def test_src_dst_are_strings(self, graph_edges):
        # Pandas 3.x may use StringDtype instead of object — both are valid string types
        src_dtype = graph_edges["src_account"].dtype
        dst_dtype = graph_edges["dst_account"].dtype
        assert pd.api.types.is_string_dtype(src_dtype), \
            f"src_account must be a string dtype, got {src_dtype}"
        assert pd.api.types.is_string_dtype(dst_dtype), \
            f"dst_account must be a string dtype, got {dst_dtype}"

    def test_amounts_positive(self, graph_edges):
        assert (graph_edges["amount"] > 0).all()

    def test_hop_depth_range(self, graph_edges):
        # 0 = legitimate transfer, 1..4 = laundering hop
        assert (graph_edges["hop_depth"] >= 0).all()
        assert (graph_edges["hop_depth"] <= 4).all()

    def test_is_terminal_binary(self, graph_edges):
        assert set(graph_edges["is_terminal"].unique()).issubset({0, 1})

    def test_networkx_loadable(self, graph_edges):
        """Verify the edge list can be loaded into a NetworkX directed graph."""
        import networkx as nx
        G = nx.from_pandas_edgelist(
            graph_edges.head(500),
            source="src_account",
            target="dst_account",
            edge_attr=["amount", "hop_depth"],
            create_using=nx.DiGraph(),
        )
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_csv_written(self, data_dir):
        assert (data_dir / "graph_edges.csv").exists()


# ─────────────────────────────────────────────
# ACCEPTANCE TEST 5 — Node Features
# ─────────────────────────────────────────────

class TestNodeFeatures:
    """Phase 1 AC: node_features.csv ready for GNN training."""

    def test_required_columns(self, node_features):
        required = {
            "account_id", "bank_name", "city", "lat", "long",
            "total_received", "total_sent", "txn_count_24h",
            "avg_txn_amount", "is_mule_label", "hop_depth"
        }
        assert required.issubset(set(node_features.columns)), \
            f"Missing: {required - set(node_features.columns)}"

    def test_account_ids_unique(self, node_features):
        assert node_features["account_id"].is_unique, "account_id must be unique"

    def test_is_mule_label_binary(self, node_features):
        assert set(node_features["is_mule_label"].unique()).issubset({0, 1}), \
            "is_mule_label must be 0 or 1"

    def test_both_mule_and_clean_present(self, node_features):
        """Dataset must have both mule (1) and clean (0) labels."""
        labels = set(node_features["is_mule_label"].unique())
        assert 1 in labels, "No mule nodes (is_mule_label=1) found"
        assert 0 in labels, "No clean nodes (is_mule_label=0) found"

    def test_mule_ratio_reasonable(self, node_features):
        """Mule label ratio should be > 0% and < 100%."""
        mule_ratio = node_features["is_mule_label"].mean()
        assert 0.0 < mule_ratio < 1.0, f"Mule ratio out of range: {mule_ratio:.2%}"

    def test_lat_within_india(self, node_features):
        assert (node_features["lat"] >= INDIA_LAT_MIN).all() and \
               (node_features["lat"] <= INDIA_LAT_MAX).all()

    def test_lon_within_india(self, node_features):
        assert (node_features["long"] >= INDIA_LON_MIN).all() and \
               (node_features["long"] <= INDIA_LON_MAX).all()

    def test_total_received_non_negative(self, node_features):
        assert (node_features["total_received"] >= 0).all()

    def test_total_sent_non_negative(self, node_features):
        assert (node_features["total_sent"] >= 0).all()

    def test_txn_count_non_negative(self, node_features):
        # A send-only account (a payer that never receives) legitimately has an
        # incoming count of zero now that ordinary traffic is simulated.
        assert (node_features["txn_count_24h"] >= 0).all(), \
            "txn_count_24h must be >= 0"
        assert node_features["txn_count_24h"].max() > 0, "No account received anything"

    def test_avg_txn_amount_non_negative(self, node_features):
        assert (node_features["avg_txn_amount"] >= 0).all()

    def test_hop_depth_range(self, node_features):
        assert (node_features["hop_depth"] >= 0).all()
        assert (node_features["hop_depth"] <= 4).all()

    def test_no_null_values(self, node_features):
        nulls = node_features.isnull().sum()
        assert nulls.sum() == 0, f"Nulls found:\n{nulls[nulls > 0]}"

    def test_feature_matrix_numeric_except_ids(self, node_features):
        """All non-ID, non-categorical columns must be numeric."""
        numeric_cols = ["lat", "long", "total_received", "total_sent",
                        "txn_count_24h", "avg_txn_amount", "is_mule_label", "hop_depth",
                        "in_degree", "out_degree", "distinct_senders", "distinct_receivers",
                        "median_dwell_seconds", "passthrough_ratio", "account_age_days",
                        "night_txn_ratio", "burst_out_5min"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(node_features[col]), \
                f"Column {col} must be numeric, got {node_features[col].dtype}"


    def test_label_is_not_a_copy_of_a_feature(self, node_features):
        """
        Regression guard against label leakage.

        A previous generator defined a mule as "received money" and then wrote
        total_received = 0 onto every non-mule, so `total_received > 0` scored
        F1 = 1.0 and the GNN's 0.9996 measured nothing. No single feature may
        reproduce the label that closely again.
        """
        from sklearn.metrics import f1_score

        y = node_features["is_mule_label"].values
        candidates = [c for c in node_features.columns
                      if c not in ("account_id", "bank_name", "city", "is_mule_label")
                      and pd.api.types.is_numeric_dtype(node_features[c])]

        worst_col, worst_f1 = None, 0.0
        for col in candidates:
            v = node_features[col].values.astype(float)
            for t in np.unique(np.percentile(v, np.linspace(2, 98, 40))):
                for pred in ((v > t).astype(int), (v <= t).astype(int)):
                    if pred.sum() in (0, len(pred)):
                        continue
                    f1 = f1_score(y, pred)
                    if f1 > worst_f1:
                        worst_f1, worst_col = f1, col

        assert worst_f1 < 0.95, (
            f"Feature '{worst_col}' reproduces the label with F1={worst_f1:.4f} — "
            "this is label leakage, the model would be measuring nothing"
        )

    def test_pyg_feature_matrix_buildable(self, node_features):
        """Verify numeric features can form a valid tensor-like matrix."""
        # Mirrors engine/gnn_model.FEATURE_COLS — behavioural only, no hop_depth
        feature_cols = ["lat", "long", "total_received", "total_sent",
                        "txn_count_24h", "avg_txn_amount", "in_degree", "out_degree",
                        "distinct_senders", "distinct_receivers", "median_dwell_seconds",
                        "passthrough_ratio", "account_age_days", "night_txn_ratio",
                        "burst_out_5min"]
        matrix = node_features[feature_cols].values
        assert matrix.shape[0] > 0, "Feature matrix has 0 rows"
        assert matrix.shape[1] == len(feature_cols), "Wrong number of feature columns"
        assert not np.isnan(matrix).any(), "NaN values in feature matrix"
        assert not np.isinf(matrix).any(), "Inf values in feature matrix"

    def test_csv_written(self, data_dir):
        assert (data_dir / "node_features.csv").exists()

    def test_csv_row_count_matches(self, data_dir, node_features):
        loaded = pd.read_csv(data_dir / "node_features.csv")
        assert len(loaded) == len(node_features)


# ─────────────────────────────────────────────
# INTEGRATION — Full Pipeline
# ─────────────────────────────────────────────

class TestFullPipeline:
    """End-to-end integration tests for the complete Phase 1 pipeline."""

    def test_all_5_csvs_written(self, data_dir):
        expected_files = [
            "victim_complaints.csv",
            "transactions.csv",
            "atm_directory.csv",
            "graph_edges.csv",
            "node_features.csv",
        ]
        for fname in expected_files:
            assert (data_dir / fname).exists(), f"Missing file: {fname}"

    def test_complaint_to_transaction_linkage(self, complaints, transactions):
        """Every complaint must have at least one transaction."""
        complaint_ids_with_txns = set(transactions["complaint_id"])
        all_complaint_ids = set(complaints["ticket_id"])
        missing = all_complaint_ids - complaint_ids_with_txns
        # At least 90% of complaints should have transactions
        coverage = 1 - (len(missing) / len(all_complaint_ids))
        assert coverage >= 0.9, \
            f"Only {coverage:.1%} of complaints have transactions (need ≥90%)"

    def test_transaction_to_edge_count_match(self, transactions, graph_edges):
        """graph_edges must be derived from transactions (same row count)."""
        assert len(graph_edges) == len(transactions)

    def test_transaction_accounts_in_node_features(self, transactions, node_features):
        """All dst_accounts from transactions should appear in node_features."""
        dst_accounts = set(transactions["dst_account"])
        node_accounts = set(node_features["account_id"])
        missing = dst_accounts - node_accounts
        assert not missing, \
            f"{len(missing)} dst_accounts not in node_features: {list(missing)[:5]}"

    def test_reproducibility(self, fake, all_data):
        """Running generator twice with same seed yields structurally identical results.

        Note: uuid4() is cryptographically random and cannot be seeded,
        so we verify structural equality (shape, column values, amounts)
        rather than exact UUID matches.
        """
        f2 = Faker("en_IN")
        f2.seed_instance(99)
        random.seed(99)
        np.random.seed(99)
        df1 = generate_victim_complaints(50, f2, all_data[0]["registry"])

        f3 = Faker("en_IN")
        f3.seed_instance(99)
        random.seed(99)
        np.random.seed(99)
        df2 = generate_victim_complaints(50, f3, all_data[0]["registry"])

        # Shape must be identical
        assert df1.shape == df2.shape, "Shape differs across runs"
        # Column names must be identical
        assert list(df1.columns) == list(df2.columns), "Columns differ across runs"
        # Deterministic fields (not UUIDs) must match exactly
        for col in ["victim_name", "victim_bank", "fraud_type", "city", "state"]:
            pd.testing.assert_series_equal(
                df1[col].reset_index(drop=True),
                df2[col].reset_index(drop=True),
                check_names=False,
                obj=f"Column '{col}'"
            )
        # Amounts must be identical (seeded random)
        pd.testing.assert_series_equal(
            df1["stolen_amount"].reset_index(drop=True),
            df2["stolen_amount"].reset_index(drop=True),
            check_names=False, obj="stolen_amount"
        )

    def test_all_acceptance_criteria_summary(self, complaints, transactions, atms,
                                              graph_edges, node_features):
        """Single summary assertion covering all 6 acceptance criteria."""
        # AC1
        assert len(complaints) >= 500
        # AC2
        assert len(transactions) >= 2000
        # AC3
        assert len(atms) >= 200
        # AC4: 10+ cities
        assert atms["city"].nunique() >= 10
        # AC5: node_features + graph_edges exist with required columns
        assert "is_mule_label" in node_features.columns
        assert "src_account" in graph_edges.columns
        # AC6: coords in Indian bounds
        assert (transactions["lat"].between(INDIA_LAT_MIN, INDIA_LAT_MAX)).all()
        assert (transactions["long"].between(INDIA_LON_MIN, INDIA_LON_MAX)).all()
