# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2b: XGBoost ATM Predictor
SIH26184 | MHA / I4C

Two models trained on the 72-dim hybrid feature vector:
  1. ATM Classifier   -- XGBClassifier  -> Top-3 ATM IDs + confidence scores
  2. Time Regressor   -- XGBRegressor   -> time_to_cashout_minutes

Inference output per complaint:
    {
        "top3_atms": [
            {"atm_id": "ATM-001", "confidence": 0.87, "rank": 1},
            {"atm_id": "ATM-042", "confidence": 0.09, "rank": 2},
            {"atm_id": "ATM-017", "confidence": 0.04, "rank": 3},
        ],
        "time_to_cashout_minutes": 23.4,
        "interception_confidence": 0.87,
    }
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
XGB_MODEL_PATH = MODELS_DIR / "xgb_cashout.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class MuleXGBPredictor:
    """
    Wraps XGBClassifier (ATM) + XGBRegressor (time) for unified inference.

    Usage:
        predictor = MuleXGBPredictor.load()
        result = predictor.predict(feature_vector_72d)
    """

    def __init__(
        self,
        classifier,     # XGBClassifier
        regressor,      # XGBRegressor
        atm_ids: list[str],
        scaler,         # sklearn StandardScaler
        label_encoder=None,  # sklearn LabelEncoder (encoded -> atm_ids index)
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.atm_ids = atm_ids
        self.scaler = scaler
        self.label_encoder = label_encoder  # maps 0..N-1 -> original ATM index

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: Path = XGB_MODEL_PATH):
        """Save the full predictor bundle to pickle."""
        bundle = {
            "classifier": self.classifier,
            "regressor": self.regressor,
            "atm_ids": self.atm_ids,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
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
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, top_k: int = 3) -> dict:
        """
        Run full inference on a single feature vector or a batch.

        Args:
            X:     np.ndarray of shape (72,) or (N, 72)
            top_k: Number of top ATM predictions to return

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

