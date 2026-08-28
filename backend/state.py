# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: In-Memory State Store
SIH26184 | MHA / I4C

Single source of truth for all runtime data.
No database needed — loaded from CSV at startup, updated in memory.

Boot sequence:
  1. Load all victim_complaints.csv → complaints dict
  2. Load all transactions.csv → transaction lookup per complaint
  3. Load node_features.csv → node attribute lookup
  4. Load atm_directory.csv → ATM metadata lookup
  5. Load GNN embeddings → embeddings dict (account → 64-dim vector)
"""

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("muleshield.state")

ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "embeddings"
MODELS_DIR = ROOT / "models"

COMPLAINTS_CSV = DATA_DIR / "victim_complaints.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
NODE_FEATURES_CSV = DATA_DIR / "node_features.csv"
ATM_CSV = DATA_DIR / "atm_directory.csv"
EMBEDDINGS_PKL = EMBEDDINGS_DIR / "node_embeddings.pkl"
XGB_MODEL_PKL = MODELS_DIR / "xgb_cashout.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

# Complaints: complaint_id → dict (raw record)
complaints: dict[str, dict] = {}

# Transactions: complaint_id → list of transaction dicts
transactions_by_complaint: dict[str, list[dict]] = {}

# Node features: account_id → dict of feature values
node_features: dict[str, dict] = {}

# ATM directory: atm_id → dict
atm_directory: dict[str, dict] = {}
atm_df: Optional[pd.DataFrame] = None

# GNN Embeddings: account_id → np.ndarray (64-dim)
embeddings: dict[str, np.ndarray] = {}

# Freeze log: freeze_ref → dict
freeze_log: dict[str, dict] = {}

# Cached FeatureBuilder singleton (lazy-loaded on first predict call)
_feature_builder = None

# Loaded model singleton (lazy-loaded on first predict call)
_xgb_predictor = None


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def load_all() -> None:
    """
    Load all data from CSVs and embeddings pickle into global state.
    Called once at FastAPI startup.
    """
    global atm_df
    logger.info("[STATE] Loading MuleShield AI state from disk...")

    # 1. Complaints
    if COMPLAINTS_CSV.exists():
        df = pd.read_csv(COMPLAINTS_CSV)
        for _, row in df.iterrows():
            cid = str(row["ticket_id"])
            complaints[cid] = {
                "ticket_id": cid,
                "victim_name": str(row.get("victim_name", "Unknown")),
                "victim_bank": str(row.get("victim_bank", "Unknown")),
                "victim_account": str(row.get("victim_account", "ACC-00000000")),
                "fraud_type": str(row.get("fraud_type", "UPI Fraud")),
                "stolen_amount": float(row.get("stolen_amount", 0)),
                "city": str(row.get("city", "Unknown")),
                "state": str(row.get("state", "Unknown")),
                "complaint_timestamp": str(row.get("complaint_timestamp", datetime.now(timezone.utc).isoformat())),
                "status": "ACTIVE",
            }
        logger.info(f"[STATE] Loaded {len(complaints)} complaints.")
    else:
        logger.warning(f"[STATE] {COMPLAINTS_CSV} not found — starting with empty complaint store.")

    # 2. Transactions — index by complaint_id
    if TRANSACTIONS_CSV.exists():
        txn_df = pd.read_csv(TRANSACTIONS_CSV)
        for _, row in txn_df.iterrows():
            cid = str(row["complaint_id"])
            txn = row.to_dict()
            transactions_by_complaint.setdefault(cid, []).append(txn)
        total_txns = sum(len(v) for v in transactions_by_complaint.values())
        logger.info(f"[STATE] Loaded {total_txns} transactions across {len(transactions_by_complaint)} complaints.")
    else:
        logger.warning(f"[STATE] {TRANSACTIONS_CSV} not found.")

    # 3. Node features — index by account_id
    if NODE_FEATURES_CSV.exists():
        nf_df = pd.read_csv(NODE_FEATURES_CSV)
        for _, row in nf_df.iterrows():
            aid = str(row["account_id"])
            node_features[aid] = row.to_dict()
        logger.info(f"[STATE] Loaded {len(node_features)} node feature records.")
    else:
        logger.warning(f"[STATE] {NODE_FEATURES_CSV} not found.")

    # 4. ATM directory
    if ATM_CSV.exists():
        atm_df = pd.read_csv(ATM_CSV)
        for _, row in atm_df.iterrows():
            aid = str(row["atm_id"])
            atm_directory[aid] = row.to_dict()
        logger.info(f"[STATE] Loaded {len(atm_directory)} ATM records.")
    else:
        logger.warning(f"[STATE] {ATM_CSV} not found.")

    # 5. GNN Embeddings
    if EMBEDDINGS_PKL.exists():
        import pickle
        with open(EMBEDDINGS_PKL, "rb") as f:
            embeddings.update(pickle.load(f))
        logger.info(f"[STATE] Loaded {len(embeddings)} GNN embeddings.")
    else:
        logger.warning(f"[STATE] {EMBEDDINGS_PKL} not found — embeddings unavailable.")

    logger.info("[STATE] ✅ MuleShield AI state fully loaded and ready.")

    # Pre-load FeatureBuilder now so first predict call is instant
    try:
        global _feature_builder
        from feature_builder import FeatureBuilder
        _feature_builder = FeatureBuilder()
        _feature_builder.load()
        logger.info("[STATE] FeatureBuilder pre-loaded and cached.")
    except Exception as e:
        logger.warning(f"[STATE] FeatureBuilder pre-load failed: {e}")


def get_xgb_predictor():
    """Lazy-load the XGBoost predictor model (cached after first call)."""
    global _xgb_predictor
    if _xgb_predictor is None:
        if not XGB_MODEL_PKL.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {XGB_MODEL_PKL}. "
                "Run: python engine/train_xgb.py"
            )
        sys.path.insert(0, str(ROOT / "engine"))
        from xgb_model import MuleXGBPredictor
        _xgb_predictor = MuleXGBPredictor.load(XGB_MODEL_PKL)
        logger.info(f"[STATE] XGBoost predictor loaded from {XGB_MODEL_PKL}")
    return _xgb_predictor


def get_feature_builder():
    """Return the cached FeatureBuilder instance (pre-loaded at startup)."""
    global _feature_builder
    if _feature_builder is None:
        from feature_builder import FeatureBuilder
        _feature_builder = FeatureBuilder()
        _feature_builder.load()
        logger.info("[STATE] FeatureBuilder lazy-loaded and cached.")
    return _feature_builder


# ─────────────────────────────────────────────────────────────────────────────
# COMPLAINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_complaint(data: dict) -> dict:
    """Add a new complaint to the in-memory store and return the record."""
    ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"
    record = {
        **data,
        "ticket_id": ticket_id,
        "complaint_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ACTIVE",
    }
    complaints[ticket_id] = record
    return record


def get_complaint(complaint_id: str) -> Optional[dict]:
    return complaints.get(complaint_id)


def get_all_complaints() -> list[dict]:
    """Return all complaints sorted newest-first."""
    return sorted(
        complaints.values(),
        key=lambda c: c.get("complaint_timestamp", ""),
        reverse=True,
    )


def get_transactions_for(complaint_id: str) -> list[dict]:
    return transactions_by_complaint.get(complaint_id, [])


def get_terminal_accounts(complaint_id: str) -> list[dict]:
    """Return all terminal mule accounts (is_terminal == 1) for a complaint."""
    txns = get_transactions_for(complaint_id)
    return [t for t in txns if int(t.get("is_terminal", 0)) == 1]


def get_node_feature(account_id: str) -> Optional[dict]:
    return node_features.get(account_id)


def get_atm(atm_id: str) -> Optional[dict]:
    return atm_directory.get(atm_id)


def get_all_atms() -> list[dict]:
    return list(atm_directory.values())


def log_freeze(freeze_record: dict) -> None:
    freeze_log[freeze_record["freeze_reference"]] = freeze_record
