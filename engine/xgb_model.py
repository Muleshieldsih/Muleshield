# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b: XGBoost ATM Predictor (v2 — Bayesian Spatial Reranking)
SIH26184 | MHA / I4C

Two models trained on the 80-dim hybrid feature vector:
  1. ATM Classifier   -- XGBClassifier  -> Top-3 ATM IDs + confidence scores
  2. Time Regressor   -- XGBRegressor   -> time_to_cashout_minutes

v2 Accuracy Upgrade — Bayesian Spatial Prior Reranking:
  After XGBoost raw softmax probabilities are computed, scores are multiplied
  by a Gaussian distance decay kernel and an ATM risk weight:

      FinalScore(ATM_i) = P(ATM_i | x)
                          × exp(−dist(node, ATM_i)² / 2σ²)
                          × (1 + 0.35 × risk_score_i)

  This guarantees ATMs thousands of km away (impossible candidates) get near-zero
  score, while nearby high-risk ATMs rise to the top of the rankings.

Inference output per complaint:
    {
        "top3_atms": [
            {"atm_id": "ATM-001", "confidence": 0.91, "rank": 1},
            {"atm_id": "ATM-042", "confidence": 0.07, "rank": 2},
            {"atm_id": "ATM-017", "confidence": 0.02, "rank": 3},
        ],
        "time_to_cashout_minutes": 23.4,
        "interception_confidence": 0.91,
    }
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
XGB_MODEL_PATH = MODELS_DIR / "xgb_cashout.pkl"

# Bayesian spatial prior sigma (km): within this radius ATMs get full score.
# Set to 15km — covers dense urban ATM clusters without over-restricting rural.
_SPATIAL_SIGMA_KM = 15.0
_SPATIAL_SIGMA_SQ = _SPATIAL_SIGMA_KM ** 2


def _gaussian_decay(dist_km: float) -> float:
    """Gaussian decay kernel: exp(−d² / 2σ²). Returns 1.0 at d=0."""
    return float(np.exp(-(dist_km ** 2) / (2 * _SPATIAL_SIGMA_SQ)))


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class MuleXGBPredictor:
    """
    Wraps XGBClassifier (ATM) + XGBRegressor (time) for unified inference.

    v2: supports optional Bayesian spatial reranking for higher accuracy.

    Usage:
        predictor = MuleXGBPredictor.load()
        result = predictor.predict(feature_vector_80d)
        # with spatial reranking:
        result = predictor.predict(feature_vector_80d,
                                   node_lat=28.6, node_lon=77.2,
                                   atm_lats=atm_lats_arr,
                                   atm_lons=atm_lons_arr,
                                   atm_risk_scores=risk_arr)
    """

    def __init__(
        self,
        classifier,         # XGBClassifier
        regressor,          # XGBRegressor
        atm_ids: list[str],
        scaler,             # sklearn StandardScaler
        label_encoder=None, # sklearn LabelEncoder (encoded -> atm_ids index)
        atm_lats: Optional[np.ndarray] = None,   # for spatial reranking
        atm_lons: Optional[np.ndarray] = None,
        atm_risk_scores: Optional[np.ndarray] = None,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.atm_ids = atm_ids
        self.scaler = scaler
        self.label_encoder = label_encoder  # maps 0..N-1 -> original ATM index
        self.atm_lats = atm_lats
        self.atm_lons = atm_lons
        self.atm_risk_scores = atm_risk_scores

    # ── Serialization ──────────────────────────────────────────────────────────

    def save(self, path: Path = XGB_MODEL_PATH):
        """Save the full predictor bundle to pickle."""
        bundle = {
            "classifier": self.classifier,
            "regressor": self.regressor,
            "atm_ids": self.atm_ids,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "atm_lats": self.atm_lats,
            "atm_lons": self.atm_lons,
            "atm_risk_scores": self.atm_risk_scores,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path = XGB_MODEL_PATH) -> "MuleXGBPredictor":
        """Load the full predictor bundle from pickle."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(
            classifier=bundle["classifier"],
            regressor=bundle["regressor"],
            atm_ids=bundle["atm_ids"],
            scaler=bundle["scaler"],
            label_encoder=bundle.get("label_encoder"),
            atm_lats=bundle.get("atm_lats"),
            atm_lons=bundle.get("atm_lons"),
            atm_risk_scores=bundle.get("atm_risk_scores"),
        )

    # ── Bayesian Spatial Reranking ─────────────────────────────────────────────

    def _apply_spatial_prior(
        self,
        probs: np.ndarray,      # shape (num_classes,) — XGBoost softmax output
        node_lat: float,
        node_lon: float,
    ) -> np.ndarray:
        """
        Rerank XGBoost probabilities using Bayesian Gaussian spatial decay.

        For each class i (mapped to a physical ATM), compute:
            score_i = P(i | x) × exp(−dist(node, ATM_i)² / 2σ²) × (1 + 0.35 × risk_i)

        Returns normalised scores (sum to 1).
        """
        if self.atm_lats is None or self.label_encoder is None:
            return probs  # graceful fallback — no spatial data available

        scores = probs.copy()
        for enc_idx, p in enumerate(probs):
            orig_idx = int(self.label_encoder.classes_[enc_idx])
            if orig_idx < len(self.atm_lats):
                atm_lat = float(self.atm_lats[orig_idx])
                atm_lon = float(self.atm_lons[orig_idx])
                # Haversine distance (inline for speed)
                import math
                phi1, phi2 = math.radians(node_lat), math.radians(atm_lat)
                dphi = math.radians(atm_lat - node_lat)
                dlam = math.radians(atm_lon - node_lon)
                a = (math.sin(dphi / 2) ** 2
                     + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
                dist_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                decay = _gaussian_decay(dist_km)
                risk_boost = 1.0
                if self.atm_risk_scores is not None and orig_idx < len(self.atm_risk_scores):
                    risk_boost = 1.0 + 0.35 * float(self.atm_risk_scores[orig_idx])
                scores[enc_idx] = p * decay * risk_boost

        total = scores.sum()
        if total > 0:
            scores /= total
        return scores

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        top_k: int = 3,
        node_lat: Optional[float] = None,
        node_lon: Optional[float] = None,
    ) -> dict:
        """
        Run full inference on a single feature vector or a batch.

        Args:
            X:         np.ndarray of shape (80,) or (N, 80)
            top_k:     Number of top ATM predictions to return
            node_lat:  Optional — terminal node latitude for spatial reranking
            node_lon:  Optional — terminal node longitude for spatial reranking

        Returns:
            Single prediction dict (if X is 1-D) or list of dicts (if 2-D)
        """
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        # ATM class probabilities [N, num_classes]
        atm_probs = self.classifier.predict_proba(X_scaled)

        # Time-to-cashout regression [N]
        time_preds = self.regressor.predict(X_scaled)

        results = []
        for i in range(len(X)):
            probs = atm_probs[i]

            # Apply Bayesian spatial prior if coordinates provided
            if node_lat is not None and node_lon is not None:
                probs = self._apply_spatial_prior(probs, node_lat, node_lon)
            elif self.atm_lats is not None:
                # Try to read lat/lon from the feature vector (dims 72/73 → node_x/y/z)
                # Fall back gracefully if feature dim doesn't support this
                pass

            top_k_enc = np.argsort(probs)[::-1][:top_k]

            # Decode encoded class -> original ATM index -> atm_ids
            top3 = []
            for rank, enc_idx in enumerate(top_k_enc):
                if self.label_encoder is not None:
                    orig_idx = int(self.label_encoder.classes_[enc_idx])
                else:
                    orig_idx = int(enc_idx)
                atm_id = self.atm_ids[orig_idx] if orig_idx < len(self.atm_ids) else f"ATM-{orig_idx:04d}"
                top3.append({
                    "atm_id": atm_id,
                    "confidence": round(float(probs[enc_idx]), 4),
                    "rank": rank + 1,
                })

            results.append({
                "top3_atms": top3,
                "time_to_cashout_minutes": round(float(max(1.0, time_preds[i])), 2),
                "interception_confidence": top3[0]["confidence"],
            })

        return results[0] if single else results

    def predict_batch(self, X: np.ndarray, top_k: int = 3) -> list[dict]:
        """Predict for a batch of feature vectors."""
        return self.predict(X, top_k=top_k) if X.ndim == 2 else [self.predict(X, top_k)]
