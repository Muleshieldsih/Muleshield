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
import random
from datetime import datetime, timedelta, timezone
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

# Accounts indexed by lowercase city → list of account_ids that carry a GNN
# embedding. Used to ground live-ingested complaints in real graph nodes.
accounts_by_city: dict[str, list[str]] = {}

# Mule-labelled accounts (is_mule_label == 1) by lowercase city.
mules_by_city: dict[str, list[str]] = {}

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

    # Idempotent: a second call (tests, --reload) must not duplicate rows.
    complaints.clear()
    transactions_by_complaint.clear()
    node_features.clear()
    atm_directory.clear()
    accounts_by_city.clear()
    mules_by_city.clear()

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
            rec = row.to_dict()
            node_features[aid] = rec
            city_key = str(rec.get("city", "")).strip().lower()
            if city_key:
                accounts_by_city.setdefault(city_key, []).append(aid)
                if int(rec.get("is_mule_label", 0)) == 1:
                    mules_by_city.setdefault(city_key, []).append(aid)
        logger.info(
            f"[STATE] Loaded {len(node_features)} node feature records "
            f"across {len(accounts_by_city)} cities."
        )
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
        if str(ROOT / "engine") not in sys.path:
            sys.path.insert(0, str(ROOT / "engine"))
        from feature_builder import FeatureBuilder
        _feature_builder = FeatureBuilder()
        _feature_builder.load()
        logger.info("[STATE] FeatureBuilder pre-loaded and cached.")
    except Exception as e:
        logger.warning(f"[STATE] FeatureBuilder pre-load failed: {e}")

    # Warm the GNN head and the XGBoost bundle too. Without this the first
    # prediction of a demo pays a ~1.5s cold-load cost and the console reports
    # a latency an order of magnitude worse than the benchmark.
    try:
        _load_gnn_head()
    except Exception as e:
        logger.warning(f"[STATE] GNN head warm-up failed: {e}")

    try:
        get_xgb_predictor()
        logger.info("[STATE] XGBoost predictor warmed.")
    except Exception as e:
        logger.warning(f"[STATE] XGBoost warm-up failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GNN RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

# Trained GraphSAGE classification head, lifted straight out of the checkpoint.
# The cached embeddings ARE the classifier's input (encode() output), so applying
# this head reproduces the exact mule probability the GNN was trained to emit —
# no proxy, no heuristic.
_gnn_head_w: Optional[np.ndarray] = None
_gnn_head_b: float = 0.0


def _load_gnn_head() -> bool:
    """Load classifier.weight / classifier.bias from the GraphSAGE checkpoint."""
    global _gnn_head_w, _gnn_head_b
    if _gnn_head_w is not None:
        return True
    ckpt_path = MODELS_DIR / "graphsage_mule.pt"
    if not ckpt_path.exists():
        return False
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        _gnn_head_w = sd["classifier.weight"].detach().cpu().numpy().reshape(-1)
        _gnn_head_b = float(sd["classifier.bias"].detach().cpu().numpy().reshape(-1)[0])
        logger.info("[STATE] GraphSAGE classification head loaded for risk scoring.")
        return True
    except Exception as e:  # pragma: no cover — defensive
        logger.warning(f"[STATE] Could not load GNN classification head: {e}")
        return False


def gnn_risk_score(account_id: str) -> float:
    """
    True GraphSAGE mule probability for an account, in [0, 1].

    Computed as sigmoid(W·h + b) where h is the account's cached 64-dim
    embedding and (W, b) is the trained binary classification head.
    Returns 0.0 for accounts with no embedding (e.g. a fresh victim account).
    """
    emb = embeddings.get(account_id)
    if emb is None or not _load_gnn_head():
        return 0.0
    logit = float(np.dot(_gnn_head_w, np.asarray(emb, dtype=np.float64)) + _gnn_head_b)
    # Numerically stable sigmoid
    if logit >= 0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    z = np.exp(logit)
    return float(z / (1.0 + z))


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
        if str(ROOT / "engine") not in sys.path:
            sys.path.insert(0, str(ROOT / "engine"))
        from feature_builder import FeatureBuilder
        _feature_builder = FeatureBuilder()
        _feature_builder.load()
        logger.info("[STATE] FeatureBuilder lazy-loaded and cached.")
    return _feature_builder


# ─────────────────────────────────────────────────────────────────────────────
# COMPLAINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pick_chain_accounts(city: str, count: int) -> list[str]:
    """
    Pick `count` real account IDs to act as the mule chain for a live complaint.

    Preference order:
      1. Mule-labelled accounts in the complaint's own city
      2. Any account in that city
      3. Mule-labelled accounts anywhere (national syndicate fallback)

    Real account IDs matter: they carry pre-computed GraphSAGE embeddings and
    genuine GPS coordinates, so a live complaint runs through exactly the same
    64-dim GNN + XGBoost path as a dataset complaint.
    """
    key = str(city or "").strip().lower()
    pool = list(mules_by_city.get(key, []))
    if len(pool) < count:
        pool += [a for a in accounts_by_city.get(key, []) if a not in pool]
    if len(pool) < count:
        for city_mules in mules_by_city.values():
            pool += [a for a in city_mules if a not in pool]
            if len(pool) >= count * 4:
                break
    if not pool:
        pool = list(node_features.keys())
    if not pool:
        return []

    rng = random.Random(f"{city}:{count}:{len(complaints)}")
    rng.shuffle(pool)
    return pool[:count]


def synthesize_mule_chain(record: dict) -> list[dict]:
    """
    Build a realistic multi-hop laundering chain for a newly ingested complaint.

    A live 1930 complaint arrives with no transaction history attached — in
    production the bank/NPCI feed supplies it. For the demo we synthesise that
    feed, but we anchor every hop on a REAL account node from the graph so the
    GNN embeddings, ATM directory and XGBoost inference are all genuine.

    Topology (mirrors observed syndicate behaviour):

        victim ──► L1 mule ──┬──► L2 mule A ──► terminal A   (is_terminal=1)
                             ├──► L2 mule B ──► terminal B   (is_terminal=1)
                             └──► L2 mule C ──► terminal C   (is_terminal=1)

    The 1-to-3 near-equal split at hop 1, inside a 5-minute window, is what the
    velocity and fund-splitting detectors in engine/graph_engine.py look for —
    so a live complaint produces genuine detections, not decorative ones.
    """
    cid = record["ticket_id"]
    city = record.get("city", "")
    victim_acc = record.get("victim_account", "ACC-00000000")
    total = float(record.get("stolen_amount", 100000.0))

    picks = _pick_chain_accounts(city, 7)
    if len(picks) < 7:
        logger.warning(f"[STATE] Not enough graph accounts to build a chain for {cid}.")
        return []

    l1, l2a, l2b, l2c, ta, tb, tc = picks
    base_ts = datetime.now(timezone.utc)
    txns: list[dict] = []

    def _emit(src: str, dst: str, amount: float, hop: int, minutes: float, terminal: bool):
        feat = node_features.get(dst, {})
        txns.append({
            "txn_id": f"TXN-{cid}-{len(txns) + 1:02d}",
            "complaint_id": cid,
            "src_account": src,
            "dst_account": dst,
            "bank_name": str(feat.get("bank_name", record.get("victim_bank", "Unknown"))),
            "ifsc_code": f"{str(feat.get('bank_name', 'BANK'))[:4].upper()}0{random.Random(dst).randint(100000, 999999)}",
            "city": str(feat.get("city", city)),
            "district": str(feat.get("city", city)),
            "state": record.get("state", ""),
            "lat": float(feat.get("lat", 20.5937)),
            "long": float(feat.get("long", 78.9629)),
            "amount": round(amount, 2),
            "timestamp": (base_ts + timedelta(minutes=minutes)).isoformat(),
            "hop_depth": hop,
            "is_terminal": 1 if terminal else 0,
        })

    # Hop 1 — full amount lands on the first layering account
    _emit(victim_acc, l1, total, 1, 0.0, False)

    # Hop 2 — near-equal 1-to-3 dispersal inside 5 minutes. Three destinations
    # within 30% of the mean is the fund-splitting rule; three outgoing
    # transfers in the window is the velocity rule.
    share = total / 3.0
    _emit(l1, l2a, share * 0.99, 2, 1.4, False)
    _emit(l1, l2b, share * 0.97, 2, 2.2, False)
    _emit(l1, l2c, share * 0.95, 2, 3.1, False)

    # Hop 3 — terminal cashout accounts, minus the mule's commission
    _emit(l2a, ta, share * 0.99 * 0.93, 3, 6.1, True)
    _emit(l2b, tb, share * 0.97 * 0.93, 3, 7.4, True)
    _emit(l2c, tc, share * 0.95 * 0.93, 3, 8.9, True)

    transactions_by_complaint[cid] = txns
    logger.info(
        f"[STATE] Synthesised {len(txns)}-hop mule chain for {cid} "
        f"(terminals: {ta}, {tb}, {tc})"
    )
    return txns


def add_complaint(data: dict) -> dict:
    """
    Add a new complaint to the in-memory store, attach a mule chain grounded in
    real graph accounts, and return the record.
    """
    ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"
    record = {
        **data,
        "ticket_id": ticket_id,
        "complaint_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ACTIVE",
        "is_live": True,
    }
    complaints[ticket_id] = record
    synthesize_mule_chain(record)
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
