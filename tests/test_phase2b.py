# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b Test Suite (v2)
SIH26184 | MHA / I4C

Tests all acceptance criteria from phases.md Phase 2b:
  AC1: feature_builder.py correctly concatenates GNN + tabular (80 dims v2)
  AC2: XGBoost trained and saved as models/xgb_cashout.pkl
  AC3: Top-3 ATM prediction returned with confidence scores
  AC4: Countdown prediction within +-5 minutes of synthetic ground truth
  AC5: End-to-end inference (embed -> XGBoost predict) in <200ms

Run:
    python -m pytest tests/test_phase2b.py -v
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from feature_builder import (
    FeatureBuilder,
    TOTAL_FEATURE_DIM,
    EMBEDDING_DIM,
    TABULAR_DIM,
    FEATURE_NAMES,
    TABULAR_FEATURE_NAMES,
    haversine_km,
    nearest_atm_info,
    atms_within_radius,
    compute_transaction_velocity,
)
from xgb_model import MuleXGBPredictor

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EMBEDDINGS_DIR = ROOT / "embeddings"
XGB_MODEL_PATH = MODELS_DIR / "xgb_cashout.pkl"


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def transactions_df():
    return pd.read_csv(DATA_DIR / "transactions.csv")


@pytest.fixture(scope="module")
def node_features_df():
    return pd.read_csv(DATA_DIR / "node_features.csv")


@pytest.fixture(scope="module")
def atm_df():
    return pd.read_csv(DATA_DIR / "atm_directory.csv")


@pytest.fixture(scope="module")
def feature_builder():
    fb = FeatureBuilder()
    fb.load()
    return fb


@pytest.fixture(scope="module")
def training_set(feature_builder):
    X, y_atm, y_time, meta = feature_builder.build_training_set()
    return X, y_atm, y_time, meta


@pytest.fixture(scope="module")
def predictor():
    if not XGB_MODEL_PATH.exists():
        pytest.skip("XGBoost model not trained yet. Run: python engine/train_xgb.py")
    return MuleXGBPredictor.load(XGB_MODEL_PATH)


# ─────────────────────────────────────────────
# HELPER FUNCTION UNIT TESTS
# ─────────────────────────────────────────────

class TestHelpers:

    def test_haversine_same_point_is_zero(self):
        """Same point should give 0km distance."""
        dist = haversine_km(28.6, 77.2, 28.6, 77.2)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        """Delhi to Mumbai is roughly 1150km."""
        delhi_lat, delhi_lon = 28.6139, 77.2090
        mumbai_lat, mumbai_lon = 19.0760, 72.8777
        dist = haversine_km(delhi_lat, delhi_lon, mumbai_lat, mumbai_lon)
        assert 1100 < dist < 1200, f"Delhi-Mumbai distance: {dist:.1f}km (expected ~1150km)"

    def test_haversine_symmetric(self):
        """Distance A->B must equal B->A."""
        d1 = haversine_km(19.07, 72.87, 13.08, 80.27)
        d2 = haversine_km(13.08, 80.27, 19.07, 72.87)
        assert d1 == pytest.approx(d2, rel=1e-6)

    def test_haversine_returns_float(self):
        dist = haversine_km(12.0, 77.0, 12.1, 77.1)
        assert isinstance(dist, float)

    def test_nearest_atm_returns_valid_id(self, atm_df):
        atm_id, dist, fraud_cnt = nearest_atm_info(28.6, 77.2, atm_df)
        assert atm_id in atm_df["atm_id"].values
        assert dist >= 0.0
        assert fraud_cnt >= 0.0

    def test_nearest_atm_is_actually_nearest(self, atm_df):
        """The returned ATM must be the closest one."""
        lat, lon = 19.07, 72.87
        atm_id, dist, _ = nearest_atm_info(lat, lon, atm_df)
        all_dists = atm_df.apply(
            lambda r: haversine_km(lat, lon, r["lat"], r["long"]), axis=1
        )
        assert dist == pytest.approx(all_dists.min(), rel=1e-3)

    def test_atms_within_radius_non_negative(self, atm_df):
        count = atms_within_radius(20.0, 77.0, atm_df, radius_km=50.0)
        assert count >= 0

    def test_atms_within_radius_large_includes_all(self, atm_df):
        """Radius of 10,000km should include all ATMs in India."""
        count = atms_within_radius(20.0, 77.0, atm_df, radius_km=10000.0)
        assert count == len(atm_df)

    def test_atms_within_radius_zero_includes_none(self, atm_df):
        """Radius of 0km should include 0 ATMs (no exact coincidence)."""
        count = atms_within_radius(0.0, 0.0, atm_df, radius_km=0.0)
        assert count == 0

    def test_velocity_empty_account_returns_zero(self, transactions_df):
        v = compute_transaction_velocity("NONEXISTENT-ACC", transactions_df)
        assert v == 0.0

    def test_velocity_known_account_non_negative(self, transactions_df):
        acc = transactions_df["src_account"].iloc[0]
        v = compute_transaction_velocity(acc, transactions_df)
        assert v >= 0.0


# ─────────────────────────────────────────────
# AC1: Feature Builder — 72-dim vector
# ─────────────────────────────────────────────

class TestFeatureBuilder:

    def test_feature_dim_constants_correct(self):
        """TOTAL_FEATURE_DIM must be 80 = 64 + 16."""
        assert EMBEDDING_DIM == 64
        assert TABULAR_DIM == 16
        assert TOTAL_FEATURE_DIM == 80

    def test_feature_names_length(self):
        """FEATURE_NAMES must have exactly 80 entries."""
        assert len(FEATURE_NAMES) == TOTAL_FEATURE_DIM

    def test_tabular_feature_names_match_spec(self):
        """Core 8 tabular features must be present (v2 may have extras)."""
        core_features = [
            "stolen_amount", "hop_depth", "transaction_velocity",
            "dist_to_atm_1_km", "hour_of_day",
            "historical_hotspot_density", "day_of_week", "amount_after_split",
        ]
        for feat in core_features:
            assert feat in TABULAR_FEATURE_NAMES, \
                f"Missing expected feature: '{feat}'"

    def test_X_shape_is_72(self, training_set):
        """AC1: Feature matrix must have exactly 80 columns (v2 80-dim)."""
        X, y_atm, y_time, meta = training_set
        assert X.shape[1] == TOTAL_FEATURE_DIM, \
            f"Expected {TOTAL_FEATURE_DIM} features, got {X.shape[1]}"

    def test_X_has_samples(self, training_set):
        """Training set must have at least 100 samples."""
        X, _, _, _ = training_set
        assert X.shape[0] >= 100, f"Only {X.shape[0]} training samples"

    def test_X_is_float32(self, training_set):
        """Feature matrix must be float32 for XGBoost efficiency."""
        X, _, _, _ = training_set
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"

    def test_no_nan_in_X(self, training_set):
        """Feature matrix must not contain NaN."""
        X, _, _, _ = training_set
        assert not np.isnan(X).any(), "NaN values found in feature matrix"

    def test_no_inf_in_X(self, training_set):
        """Feature matrix must not contain Inf."""
        X, _, _, _ = training_set
        assert not np.isinf(X).any(), "Inf values found in feature matrix"

    def test_embedding_slice_not_all_zeros(self, training_set):
        """First 64 features (GNN embeddings) must not be all zeros."""
        X, _, _, _ = training_set
        emb_slice = X[:, :EMBEDDING_DIM]
        non_zero_rows = np.any(emb_slice != 0, axis=1).sum()
        assert non_zero_rows > 0, "All GNN embedding columns are zero"

    def test_stolen_amount_positive(self, training_set):
        """Tabular feature stolen_amount (col 64) must be > 0."""
        X, _, _, _ = training_set
        assert (X[:, EMBEDDING_DIM] > 0).all(), \
            "stolen_amount feature has non-positive values"

    def test_hop_depth_range(self, training_set):
        """hop_depth (col 65) must be in [1, 4]."""
        X, _, _, _ = training_set
        hop = X[:, EMBEDDING_DIM + 1]
        assert (hop >= 1).all() and (hop <= 4).all(), \
            f"hop_depth out of [1,4]: min={hop.min()}, max={hop.max()}"

    def test_hour_of_day_range(self, training_set):
        """hour_of_day (col 68) must be in [0, 23]."""
        X, _, _, _ = training_set
        hour = X[:, EMBEDDING_DIM + 4]
        assert (hour >= 0).all() and (hour <= 23).all(), \
            f"hour_of_day out of [0,23]: min={hour.min()}, max={hour.max()}"

    def test_day_of_week_range(self, training_set):
        """day_of_week (col 70) must be in [0, 6]."""
        X, _, _, _ = training_set
        dow = X[:, EMBEDDING_DIM + 6]
        assert (dow >= 0).all() and (dow <= 6).all(), \
            f"day_of_week out of [0,6]: min={dow.min()}, max={dow.max()}"

    def test_y_atm_valid_labels(self, training_set, feature_builder):
        """ATM labels must be valid indices into the ATM list."""
        _, y_atm, _, _ = training_set
        assert (y_atm >= 0).all()
        assert (y_atm < len(feature_builder.atm_ids)).all(), \
            f"ATM label out of range: max={y_atm.max()}, n_atms={len(feature_builder.atm_ids)}"

    def test_y_time_positive(self, training_set):
        """Time-to-cashout labels must be > 0."""
        _, _, y_time, _ = training_set
        assert (y_time > 0).all(), "Time-to-cashout has non-positive values"

    def test_meta_df_has_required_columns(self, training_set):
        """Metadata DataFrame must contain expected columns."""
        _, _, _, meta = training_set
        required = {"terminal_account", "complaint_id", "nearest_atm_id",
                    "dist_to_atm_km", "time_to_cashout_min"}
        assert required.issubset(set(meta.columns))

    def test_single_feature_vector_shape(self, feature_builder, transactions_df):
        """Single build_feature_vector call must return (TOTAL_FEATURE_DIM,) == (80,)."""
        terminal_txns = transactions_df[transactions_df["is_terminal"] == 1]
        row = terminal_txns.iloc[0]
        feat = feature_builder.build_feature_vector(
            row["dst_account"],
            row["complaint_id"],
            100000.0,
        )
        assert feat.shape == (TOTAL_FEATURE_DIM,), \
            f"Expected ({TOTAL_FEATURE_DIM},), got {feat.shape}"


# ─────────────────────────────────────────────
# AC2: Model File
# ─────────────────────────────────────────────

class TestModelFile:

    def test_model_file_exists(self):
        """AC2: models/xgb_cashout.pkl must exist."""
        assert XGB_MODEL_PATH.exists(), \
            "xgb_cashout.pkl not found. Run: python engine/train_xgb.py"

    def test_model_loadable(self, predictor):
        """AC2: Model must load without errors."""
        assert predictor is not None

    def test_predictor_has_classifier(self, predictor):
        assert hasattr(predictor, "classifier")
        assert predictor.classifier is not None

    def test_predictor_has_regressor(self, predictor):
        assert hasattr(predictor, "regressor")
        assert predictor.regressor is not None

    def test_predictor_has_atm_ids(self, predictor):
        assert hasattr(predictor, "atm_ids")
        assert len(predictor.atm_ids) > 0

    def test_predictor_has_scaler(self, predictor):
        assert hasattr(predictor, "scaler")
        assert predictor.scaler is not None

    def test_atm_ids_count_matches_directory(self, predictor, atm_df):
        """ATM ID list must match count in atm_directory.csv."""
        assert len(predictor.atm_ids) == len(atm_df), \
            f"ATM ID mismatch: model={len(predictor.atm_ids)}, csv={len(atm_df)}"


# ─────────────────────────────────────────────
# AC3: Top-3 ATM Prediction
# ─────────────────────────────────────────────

class TestTop3Prediction:

    def test_predict_returns_dict(self, predictor, training_set):
        """AC3: predict() on a single vector must return a dict."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        assert isinstance(result, dict)

    def test_predict_has_top3_atms(self, predictor, training_set):
        """AC3: Result must have 'top3_atms' key."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        assert "top3_atms" in result
        assert isinstance(result["top3_atms"], list)

    def test_top3_has_exactly_3_entries(self, predictor, training_set):
        """AC3: top3_atms must have exactly 3 entries."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        assert len(result["top3_atms"]) == 3

    def test_top3_entries_have_required_fields(self, predictor, training_set):
        """AC3: Each ATM entry must have atm_id, confidence, rank."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        for entry in result["top3_atms"]:
            assert "atm_id" in entry
            assert "confidence" in entry
            assert "rank" in entry

    def test_top3_ranks_are_1_2_3(self, predictor, training_set):
        """Ranks must be 1, 2, 3 in order."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        ranks = [e["rank"] for e in result["top3_atms"]]
        assert ranks == [1, 2, 3]

    def test_confidence_scores_sum_to_le_1(self, predictor, training_set):
        """Top-3 confidence scores must sum to <= 1.0."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        total_conf = sum(e["confidence"] for e in result["top3_atms"])
        assert total_conf <= 1.0 + 1e-4, f"Confidence sum > 1: {total_conf}"

    def test_confidence_scores_are_positive(self, predictor, training_set):
        """All confidence scores must be > 0."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        for e in result["top3_atms"]:
            assert e["confidence"] >= 0.0

    def test_top3_atm_ids_are_valid(self, predictor, training_set, atm_df):
        """All predicted ATM IDs must exist in atm_directory.csv."""
        X, _, _, _ = training_set
        valid_ids = set(atm_df["atm_id"])
        for i in range(min(20, len(X))):
            result = predictor.predict(X[i])
            for e in result["top3_atms"]:
                assert e["atm_id"] in valid_ids, \
                    f"Unknown ATM ID: {e['atm_id']}"

    def test_interception_confidence_range(self, predictor, training_set):
        """interception_confidence must be in [0, 1]."""
        X, _, _, _ = training_set
        for i in range(min(20, len(X))):
            result = predictor.predict(X[i])
            conf = result["interception_confidence"]
            assert 0.0 <= conf <= 1.0, f"Confidence out of [0,1]: {conf}"

    def test_batch_prediction(self, predictor, training_set):
        """Batch predict on 10 vectors must return a list of 10 dicts."""
        X, _, _, _ = training_set
        results = predictor.predict(X[:10])
        assert isinstance(results, list)
        assert len(results) == 10
        for r in results:
            assert "top3_atms" in r


# ─────────────────────────────────────────────
# AC4: Time-to-Cashout Accuracy
# ─────────────────────────────────────────────

class TestTimeTocashout:

    def test_time_prediction_positive(self, predictor, training_set):
        """AC4: time_to_cashout_minutes must be > 0."""
        X, _, _, _ = training_set
        result = predictor.predict(X[0])
        assert result["time_to_cashout_minutes"] > 0

    def test_time_prediction_reasonable_range(self, predictor, training_set):
        """Time predictions must be within a realistic range (1–500 min)."""
        X, _, _, _ = training_set
        for i in range(min(20, len(X))):
            t = predictor.predict(X[i])["time_to_cashout_minutes"]
            assert 1.0 <= t <= 500.0, f"Unrealistic time: {t} min"

    def test_time_mae_within_5_minutes(self, predictor, training_set):
        """AC4: MAE of time predictions vs synthetic ground truth must be <= 5 min."""
        X, _, y_time, _ = training_set
        X_scaled = predictor.scaler.transform(X)
        y_pred = predictor.regressor.predict(X_scaled)
        mae = float(np.mean(np.abs(y_pred - y_time)))
        assert mae <= 5.0, \
            f"Time MAE {mae:.2f} min exceeds 5-minute AC"


# ─────────────────────────────────────────────
# AC5: End-to-End Inference Speed <200ms
# ─────────────────────────────────────────────

class TestInferenceSpeed:

    def test_single_inference_under_200ms(self, predictor, training_set):
        """AC5: Single inference must complete in <200ms."""
        X, _, _, _ = training_set
        sample = X[0]

        # Warm up
        predictor.predict(sample)

        # Measure
        times = []
        for _ in range(30):
            t0 = time.time()
            predictor.predict(sample)
            times.append((time.time() - t0) * 1000)

        mean_ms = np.mean(times)
        assert mean_ms < 200, \
            f"Mean inference {mean_ms:.2f}ms exceeds 200ms AC"

    def test_end_to_end_with_embedding_under_200ms(
        self, feature_builder, predictor, transactions_df
    ):
        """
        AC5: Full pipeline (GNN embed lookup + feature build + XGBoost predict)
        must complete in <200ms for a single terminal account.
        """
        terminal_txns = transactions_df[transactions_df["is_terminal"] == 1]
        row = terminal_txns.iloc[0]

        t0 = time.time()
        # Step 1: Build feature vector (includes embedding lookup)
        feat = feature_builder.build_feature_vector(
            row["dst_account"],
            row["complaint_id"],
            100000.0,
        )
        # Step 2: XGBoost predict
        result = predictor.predict(feat)
        elapsed_ms = (time.time() - t0) * 1000

        assert elapsed_ms < 200, \
            f"End-to-end took {elapsed_ms:.2f}ms — exceeds 200ms AC"
        assert "top3_atms" in result


# ─────────────────────────────────────────────
# INTEGRATION
# ─────────────────────────────────────────────

class TestIntegration:

    def test_all_phase2b_files_exist(self):
        """All required Phase 2b output files must exist."""
        files = {
            "engine/feature_builder.py": ROOT / "engine" / "feature_builder.py",
            "engine/xgb_model.py":       ROOT / "engine" / "xgb_model.py",
            "engine/train_xgb.py":       ROOT / "engine" / "train_xgb.py",
            "models/xgb_cashout.pkl":    ROOT / "models" / "xgb_cashout.pkl",
        }
        for name, path in files.items():
            assert path.exists(), f"Missing file: {name}"

    def test_full_complaint_pipeline(self, feature_builder, predictor, transactions_df):
        """
        Full pipeline: pick a complaint's terminal accounts,
        build features, run prediction, verify output structure.
        """
        terminal_txns = transactions_df[transactions_df["is_terminal"] == 1]
        # Pick a complaint with terminal transactions
        complaint_id = terminal_txns["complaint_id"].iloc[0]
        complaint_terminals = terminal_txns[terminal_txns["complaint_id"] == complaint_id]

        for _, row in complaint_terminals.head(3).iterrows():
            feat = feature_builder.build_feature_vector(
                row["dst_account"],
                complaint_id,
                200000.0,
            )
            result = predictor.predict(feat)

            assert len(result["top3_atms"]) == 3
            assert result["time_to_cashout_minutes"] > 0
            assert 0.0 <= result["interception_confidence"] <= 1.0

    def test_predictor_serialization_roundtrip(self, predictor, training_set, tmp_path):
        """Save and reload predictor — predictions must be identical."""
        X, _, _, _ = training_set
        sample = X[0]

        # Get original prediction
        orig = predictor.predict(sample)

        # Save to tmp, reload, predict again
        tmp_path_pkl = tmp_path / "xgb_test.pkl"
        predictor.save(tmp_path_pkl)
        loaded = MuleXGBPredictor.load(tmp_path_pkl)
        reloaded = loaded.predict(sample)

        assert orig["top3_atms"][0]["atm_id"] == reloaded["top3_atms"][0]["atm_id"]
        assert orig["time_to_cashout_minutes"] == pytest.approx(
            reloaded["time_to_cashout_minutes"], abs=0.01
        )
