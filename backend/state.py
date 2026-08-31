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
    #
    # Only rows that belong to a complaint are kept. The corpus is 622,188 rows
    # but 599,987 of them are the legitimate banking activity the generator
    # creates so that mule behaviour is not trivially separable — they carry no
    # complaint_id and so can never be returned, because every read here is
    # get_transactions_for(complaint_id), a keyed lookup.
    #
    # Loading them anyway cost 380 MB of process memory to hold 6 MB of
    # reachable data. They still matter on disk: the generator writes them and
    # the models train on them. They just have no business being resident in an
    # API process that can only ever serve the other 4%.
    if TRANSACTIONS_CSV.exists():
        txn_df = pd.read_csv(TRANSACTIONS_CSV)
        skipped = 0
        for _, row in txn_df.iterrows():
            cid = row["complaint_id"]
            # pandas gives NaN for a blank cell, and str(nan) is the truthy
            # "nan" — so this has to test the value, not its string form.
            if cid is None or (isinstance(cid, float) and cid != cid) or not str(cid).strip():
                skipped += 1
                continue
            transactions_by_complaint.setdefault(str(cid), []).append(row.to_dict())
        total_txns = sum(len(v) for v in transactions_by_complaint.values())
        logger.info(
            f"[STATE] Loaded {total_txns:,} transactions across "
            f"{len(transactions_by_complaint):,} complaints "
            f"({skipped:,} unattached rows left on disk)."
        )
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
# Isotonic calibration curve (x, y) fitted on the GNN validation split.
_gnn_calib: Optional[tuple] = None


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
        global _gnn_calib
        cal = ckpt.get("calibration")
        if cal:
            _gnn_calib = (np.asarray(cal["x"], dtype=float),
                          np.asarray(cal["y"], dtype=float))
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
        raw = float(1.0 / (1.0 + np.exp(-logit)))
    else:
        z = np.exp(logit)
        raw = float(z / (1.0 + z))

    # Map to a calibrated probability. The raw sigmoid saturates, so an uncalibrated
    # score reads 100% for every account in a traced chain - true of the ordering,
    # misleading as a number on screen.
    if _gnn_calib is not None:
        return float(np.interp(raw, _gnn_calib[0], _gnn_calib[1]))
    return raw


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


# ─────────────────────────────────────────────────────────────────────────────
# CASE WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
#
# A complaint arrives and is worked. Until now `status` was written once as
# "ACTIVE" and never read, so the console could show a queue but not a caseload:
# nothing recorded that an analyst had looked at a case, formed a view, acted on
# it, or closed it.
#
# State lives in the same in-memory dicts as everything else. It does not
# survive a restart, which is honest about what this build is -- but every
# transition is recorded while the process lives, and that is what makes the
# audit trail below mean anything.

CASE_STATUSES = [
    "New",
    "Under Review",
    "Investigating",
    "Intervention Required",
    "Resolved",
    "Closed",
]

# Terminal states. A case can be reopened out of them, but nothing auto-advances.
CLOSED_STATUSES = {"Resolved", "Closed"}

# Audit entries, newest last. A list rather than a dict because order is the
# point: an audit trail that cannot be read in sequence is not a trail.
audit_log: list[dict] = []

# Per-case notes: complaint_id -> list of note dicts.
case_notes: dict[str, list[dict]] = {}

_AUDIT_LIMIT = 5000


def record_audit(actor: str, action: str, obj: str, result: str = "ok",
                 case_id: str = "") -> dict:
    """
    Append one entry to the audit trail.

    Deliberately free of any judgement about what is worth recording -- callers
    decide. The trail is capped so a long-running process cannot exhaust memory;
    the oldest entries fall off, which is the wrong trade-off for a real system
    and the right one for a demo that must not fall over.
    """
    entry = {
        "id": f"AUD-{len(audit_log) + 1:06d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor or "UNKNOWN",
        "action": action,
        "object": obj,
        "result": result,
        "case_id": case_id,
    }
    audit_log.append(entry)
    if len(audit_log) > _AUDIT_LIMIT:
        del audit_log[: len(audit_log) - _AUDIT_LIMIT]
    return entry


def get_audit(case_id: str = "", limit: int = 100) -> list[dict]:
    """Newest first. Filtered to one case when `case_id` is given."""
    rows = audit_log if not case_id else [e for e in audit_log if e["case_id"] == case_id]
    return list(reversed(rows))[:limit]


def case_status(complaint_id: str) -> str:
    """
    The working status of a case.

    Seed complaints load with the legacy "ACTIVE", which is not one of the six
    workflow states. Rather than rewrite 2,500 records on load, that value is
    read as "New" -- an untouched case is exactly what it means.
    """
    rec = complaints.get(complaint_id) or {}
    status = rec.get("status", "New")
    return "New" if status == "ACTIVE" else status


def update_case(complaint_id: str, *, status: Optional[str] = None,
                assignee: Optional[str] = None, actor: str = "SYSTEM") -> Optional[dict]:
    """Move a case through the workflow. Returns the updated record, or None."""
    rec = complaints.get(complaint_id)
    if rec is None:
        return None

    if status is not None:
        if status not in CASE_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        before = case_status(complaint_id)
        rec["status"] = status
        record_audit(actor, "Changed status", f"{before} -> {status}",
                     case_id=complaint_id)

    if assignee is not None:
        rec["assignee"] = assignee
        record_audit(actor, "Assigned case",
                     assignee or "unassigned", case_id=complaint_id)

    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    return rec


def add_note(complaint_id: str, text: str, author: str) -> Optional[dict]:
    """Attach an investigation note to a case."""
    if complaint_id not in complaints:
        return None
    note = {
        "id": f"NOTE-{complaint_id}-{len(case_notes.get(complaint_id, [])) + 1:03d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "author": author,
        "text": text,
    }
    case_notes.setdefault(complaint_id, []).append(note)
    record_audit(author, "Added note", note["id"], case_id=complaint_id)
    return note


def get_notes(complaint_id: str) -> list[dict]:
    return list(reversed(case_notes.get(complaint_id, [])))


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CASE INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def atm_intelligence(limit: int = 25) -> list[dict]:
    """
    Rank ATMs by how many distinct complaints they appear in.

    Every other screen in this console answers a question about ONE case. This
    answers the one I4C asks across a district: which machines keep coming back.

    The models already consume this history -- `atm_prior_count` and
    `historical_hotspot_density` are ranking features -- but nothing surfaced it
    to a person, so a claim about spotting the network rather than the incident
    had nothing behind it on screen.

    Ranked by DISTINCT COMPLAINTS rather than raw cash-out count. One chain that
    splits four ways and converges on a single terminal produces four cash-outs
    at one machine; that is one case, not four, and counting it as four would
    make an ordinary pooling pattern look like a hotspot.
    """
    by_atm: dict[str, dict] = {}

    for cid, rows in transactions_by_complaint.items():
        for t in rows:
            if int(t.get("is_terminal", 0) or 0) != 1:
                continue
            atm_id = str(t.get("cashout_atm_id") or "").strip()
            if not atm_id or atm_id.lower() == "nan":
                continue

            rec = by_atm.get(atm_id)
            if rec is None:
                rec = by_atm[atm_id] = {
                    "atm_id": atm_id,
                    "cashouts": 0,
                    "_complaints": set(),
                    "total_amount": 0.0,
                    "last_seen": "",
                }
            rec["cashouts"] += 1
            rec["_complaints"].add(cid)
            rec["total_amount"] += float(t.get("amount", 0.0) or 0.0)
            ts = str(t.get("timestamp") or "")
            if ts > rec["last_seen"]:
                rec["last_seen"] = ts

    out: list[dict] = []
    for rec in by_atm.values():
        atm = atm_directory.get(rec["atm_id"], {})
        out.append({
            "atm_id": rec["atm_id"],
            "cashouts": rec["cashouts"],
            "distinct_complaints": len(rec["_complaints"]),
            "total_amount": round(rec["total_amount"], 2),
            "city": str(atm.get("city", "")),
            "state": str(atm.get("state", "")),
            "lat": float(atm.get("lat", 0.0) or 0.0),
            "lon": float(atm.get("long", atm.get("lon", 0.0)) or 0.0),
            "cashout_risk_score": float(atm.get("cashout_risk_score", 0.0) or 0.0),
            "last_seen": rec["last_seen"],
        })

    out.sort(key=lambda r: (r["distinct_complaints"], r["cashouts"]), reverse=True)
    return out[:limit]
