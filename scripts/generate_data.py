# -*- coding: utf-8 -*-
"""
MuleShield AI — Phase 1: Realistic Indian Banking & Cybercrime Dataset Generator
SIH26184 | Ministry of Home Affairs / I4C

Generates:
  - data/victim_complaints.csv   → 1930-style complaint tickets
  - data/transactions.csv        → Multi-hop IMPS/UPI ledger with fraud rings
  - data/atm_directory.csv       → Geo-coded ATM/CSP database
  - data/graph_edges.csv         → Edge list for PyG/NetworkX ingestion
  - data/node_features.csv       → Per-account feature matrix for GNN training

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --complaints 500 --transactions 2000 --atms 200 --seed 42
"""

import argparse
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
INDIA_LAT_MIN, INDIA_LAT_MAX = 8.0, 37.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 97.5
MAX_HOP_DEPTH = 4
VELOCITY_WINDOW_SECONDS = 300
MIN_SPLIT_DESTINATIONS = 3
DATA_DIR = Path(__file__).parent.parent / "data"

FRAUD_TYPES = [
    "Digital Arrest",
    "Job Scam",
    "UPI Fraud",
    "Investment Scam",
    "Romance Scam",
]

INDIAN_BANKS = [
    ("State Bank of India", "SBIN"),
    ("HDFC Bank", "HDFC"),
    ("ICICI Bank", "ICIC"),
    ("Axis Bank", "UTIB"),
    ("Punjab National Bank", "PUNB"),
    ("Bank of Baroda", "BARB"),
    ("Canara Bank", "CNRB"),
    ("Kotak Mahindra Bank", "KKBK"),
    ("Union Bank of India", "UBIN"),
    ("UCO Bank", "UCBA"),
    ("IndusInd Bank", "INDB"),
    ("IDBI Bank", "IBKL"),
    ("YES Bank", "YESB"),
    ("Federal Bank", "FDRL"),
    ("South Indian Bank", "SIBL"),
]

INDIAN_CITIES = [
    ("Delhi", "Central Delhi", "Delhi", 28.6139, 77.2090),
    ("Noida", "Gautam Buddh Nagar", "Uttar Pradesh", 28.5355, 77.3910),
    ("Gurgaon", "Gurugram", "Haryana", 28.4595, 77.0266),
    ("Mumbai", "Mumbai City", "Maharashtra", 19.0760, 72.8777),
    ("Pune", "Pune", "Maharashtra", 18.5204, 73.8567),
    ("Bengaluru", "Bengaluru Urban", "Karnataka", 12.9716, 77.5946),
    ("Hyderabad", "Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Chennai", "Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Kolkata", "Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Ahmedabad", "Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Jaipur", "Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Lucknow", "Lucknow", "Uttar Pradesh", 26.8467, 80.9462),
    ("Patna", "Patna", "Bihar", 25.5941, 85.1376),
    ("Bhopal", "Bhopal", "Madhya Pradesh", 23.2599, 77.4126),
    ("Chandigarh", "Chandigarh", "Chandigarh", 30.7333, 76.7794),
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _random_ifsc(bank_code: str) -> str:
    """Generate a realistic IFSC code: BANKCODE + '0' + 6-digit branch."""
    return f"{bank_code}0{random.randint(100000, 999999)}"


def _masked_account() -> str:
    """Return a masked account number: XXXX-XXXX-XXXX."""
    return f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"


def _jitter_coords(lat: float, lon: float, radius_km: float = 5.0):
    """Jitter lat/long by ±radius_km, clamped to Indian bounds."""
    delta = radius_km / 111.0
    new_lat = max(INDIA_LAT_MIN, min(INDIA_LAT_MAX, lat + random.uniform(-delta, delta)))
    new_lon = max(INDIA_LON_MIN, min(INDIA_LON_MAX, lon + random.uniform(-delta, delta)))
    return round(new_lat, 6), round(new_lon, 6)


def _random_bank():
    return random.choice(INDIAN_BANKS)


def _random_city():
    return random.choice(INDIAN_CITIES)


def _split_amount(total: float, n: int):
    """Split total into n parts with Dirichlet noise. Each part >= 100."""
    if n == 1:
        return [total]
    parts = np.random.dirichlet(np.ones(n)) * total
    return [round(max(p, 100.0), 2) for p in parts]


# ─────────────────────────────────────────────
# GENERATOR 1 — Victim Complaints
# ─────────────────────────────────────────────

def generate_victim_complaints(n: int, fake: Faker) -> pd.DataFrame:
    """Generate n 1930-style victim complaint records."""
    records = []
    base_ts = datetime.now() - timedelta(days=90)
    for _ in range(n):
        bank_name, _ = _random_bank()
        city, district, state, lat, lon = _random_city()
        records.append({
            "ticket_id": str(uuid.uuid4()),
            "victim_name": fake.name(),
            "victim_bank": bank_name,
            "victim_account": _masked_account(),
            "fraud_type": random.choice(FRAUD_TYPES),
            "stolen_amount": round(random.uniform(5_000, 5_00_000), 2),
            "complaint_timestamp": (
                base_ts + timedelta(seconds=random.randint(0, 90 * 24 * 3600))
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "city": city,
            "state": state,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# GENERATOR 2 — Multi-Hop Transactions
# ─────────────────────────────────────────────

def _build_chain(complaint_id: str, complaint_ts: datetime, stolen_amount: float, fake: Faker):
    """Build a single multi-hop mule chain for one complaint.

    Explicitly embeds a fraud RING: multiple L2 accounts converge to a single
    shared terminal node (many-to-one), satisfying the fraud ring AC.
    """
    txns = []
    current_time = complaint_ts + timedelta(seconds=random.randint(30, 180))
    victim_account = _masked_account()

    # L0 → L1
    n_l1 = random.randint(1, 3)
    l1_accounts = [_masked_account() for _ in range(n_l1)]
    l1_amounts = _split_amount(stolen_amount, n_l1)

    for l1_acc, l1_amt in zip(l1_accounts, l1_amounts):
        current_time += timedelta(seconds=random.randint(10, 60))
        bank_name, bank_code = _random_bank()
        city, dist, state, lat, lon = _random_city()
        txns.append({
            "txn_id": str(uuid.uuid4()), "complaint_id": complaint_id,
            "src_account": victim_account, "dst_account": l1_acc,
            "bank_name": bank_name, "ifsc_code": _random_ifsc(bank_code),
            "city": city, "district": dist, "state": state,
            "lat": round(lat + random.uniform(-0.05, 0.05), 6),
            "long": round(lon + random.uniform(-0.05, 0.05), 6),
            "amount": round(l1_amt, 2),
            "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hop_depth": 1, "is_terminal": 0,
        })

    # L1 → L2
    l2_groups = []
    for l1_acc, l1_amt in zip(l1_accounts, l1_amounts):
        n_l2 = random.randint(1, 3)
        l2_accs = [_masked_account() for _ in range(n_l2)]
        l2_amounts = _split_amount(l1_amt, n_l2)
        l2_groups.append(list(zip(l2_accs, l2_amounts)))
        for l2_acc, l2_amt in zip(l2_accs, l2_amounts):
            current_time += timedelta(seconds=random.randint(30, 90))
            bank_name, bank_code = _random_bank()
            city, dist, state, lat, lon = _random_city()
            txns.append({
                "txn_id": str(uuid.uuid4()), "complaint_id": complaint_id,
                "src_account": l1_acc, "dst_account": l2_acc,
                "bank_name": bank_name, "ifsc_code": _random_ifsc(bank_code),
                "city": city, "district": dist, "state": state,
                "lat": round(lat + random.uniform(-0.05, 0.05), 6),
                "long": round(lon + random.uniform(-0.05, 0.05), 6),
                "amount": round(l2_amt, 2),
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "hop_depth": 2, "is_terminal": 0,
            })

    # FRAUD RING: Collect 3+ L2 accounts and funnel them into ONE shared terminal
    all_l2 = [acc for group in l2_groups for acc, _ in group]
    all_l2_amts = [amt for group in l2_groups for _, amt in group]
    if len(all_l2) >= MIN_SPLIT_DESTINATIONS:
        # Pick 3+ L2 accounts as ring members converging to one terminal
        ring_size = min(len(all_l2), random.randint(3, min(5, len(all_l2))))
        ring_members = random.sample(list(zip(all_l2, all_l2_amts)), k=ring_size)
        shared_terminal = _masked_account()
        bank_name, bank_code = _random_bank()
        city, dist, state, lat, lon = _random_city()
        for ring_src, ring_amt in ring_members:
            current_time += timedelta(seconds=random.randint(60, 180))
            txns.append({
                "txn_id": str(uuid.uuid4()), "complaint_id": complaint_id,
                "src_account": ring_src, "dst_account": shared_terminal,
                "bank_name": bank_name, "ifsc_code": _random_ifsc(bank_code),
                "city": city, "district": dist, "state": state,
                "lat": round(lat + random.uniform(-0.02, 0.02), 6),
                "long": round(lon + random.uniform(-0.02, 0.02), 6),
                "amount": round(ring_amt * 0.9, 2),
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "hop_depth": 3, "is_terminal": 1,
            })
    else:
        # Fallback: deep chain L2 -> L3 -> L4 (terminal)
        for l2_group in l2_groups:
            deepened = random.sample(l2_group, k=min(1, len(l2_group)))
            for l2_acc, l2_amt in deepened:
                depth = random.choice([3, 4])
                src = l2_acc
                remaining = l2_amt
                for d in range(3, depth + 1):
                    terminal = int(d == depth)
                    dst_acc = _masked_account()
                    current_time += timedelta(seconds=random.randint(60, 300))
                    bank_name, bank_code = _random_bank()
                    city, dist, state, lat, lon = _random_city()
                    txns.append({
                        "txn_id": str(uuid.uuid4()), "complaint_id": complaint_id,
                        "src_account": src, "dst_account": dst_acc,
                        "bank_name": bank_name, "ifsc_code": _random_ifsc(bank_code),
                        "city": city, "district": dist, "state": state,
                        "lat": round(lat + random.uniform(-0.05, 0.05), 6),
                        "long": round(lon + random.uniform(-0.05, 0.05), 6),
                        "amount": round(remaining, 2),
                        "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "hop_depth": d, "is_terminal": terminal,
                    })
                    src = dst_acc

    return txns


def generate_transactions(complaints_df: pd.DataFrame, target: int, fake: Faker) -> pd.DataFrame:
    """Generate multi-hop transaction chains.

    Guarantees:
      - Every complaint gets at least ONE chain (100% complaint coverage).
      - Keeps adding chains until total transaction count >= target.
    """
    all_txns = []
    cids = complaints_df["ticket_id"].tolist()
    ts_map = dict(zip(complaints_df["ticket_id"],
                      pd.to_datetime(complaints_df["complaint_timestamp"])))
    amt_map = dict(zip(complaints_df["ticket_id"], complaints_df["stolen_amount"]))

    # Pass 1: every complaint gets exactly one chain
    for cid in cids:
        all_txns.extend(_build_chain(cid, ts_map[cid], amt_map[cid], fake))

    # Pass 2: keep adding extra chains (cycling) until target is met
    idx = 0
    while len(all_txns) < target:
        cid = cids[idx % len(cids)]
        all_txns.extend(_build_chain(cid, ts_map[cid], amt_map[cid], fake))
        idx += 1

    df = pd.DataFrame(all_txns)
    df["lat"] = df["lat"].clip(INDIA_LAT_MIN, INDIA_LAT_MAX)
    df["long"] = df["long"].clip(INDIA_LON_MIN, INDIA_LON_MAX)
    return df


# ─────────────────────────────────────────────
# GENERATOR 3 — ATM Directory
# ─────────────────────────────────────────────

def generate_atm_directory(n: int, fake: Faker) -> pd.DataFrame:
    """Generate n geo-coded ATM/CSP entries."""
    hotspot_cities = {"Delhi", "Mumbai", "Noida", "Gurgaon", "Patna"}
    records = []
    for _ in range(n):
        bank_name, _ = _random_bank()
        city, district, state, lat, lon = _random_city()
        jlat, jlon = _jitter_coords(lat, lon, radius_km=15.0)
        base_risk = 0.6 if city in hotspot_cities else 0.3
        risk_score = round(min(1.0, base_risk + random.uniform(-0.2, 0.4)), 3)
        records.append({
            "atm_id": f"ATM-{str(uuid.uuid4())[:8].upper()}",
            "bank_name": bank_name,
            "address": fake.address().replace("\n", ", "),
            "city": city, "district": district, "state": state,
            "lat": jlat, "long": jlon,
            "opening_time": random.choice(["00:00", "06:00", "07:00", "08:00"]),
            "closing_time": random.choice(["23:59", "22:00", "21:00"]),
            "cashout_risk_score": risk_score,
            "historical_fraud_count": random.randint(0, 120),
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# GENERATOR 4 — Graph Edge List
# ─────────────────────────────────────────────

def generate_graph_edges(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Derive clean edge list from transactions for PyG/NetworkX."""
    cols = ["src_account", "dst_account", "amount", "timestamp",
            "hop_depth", "is_terminal", "complaint_id"]
    return transactions_df[cols].copy().reset_index(drop=True)


# ─────────────────────────────────────────────
# GENERATOR 5 — Node Features
# ─────────────────────────────────────────────

def generate_node_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-account node feature matrix for GNN training.

    Includes:
      - Mule nodes (hop_depth >= 1):  all accounts appearing as dst in chains
      - Clean nodes (hop_depth == 0): source-only accounts (victims / feeders)
        that appear as src but NEVER as dst — labelled is_mule_label=0
    """
    # ── Mule nodes: appeared as transaction destination ──────────────────────
    received = (
        transactions_df.groupby("dst_account")
        .agg(total_received=("amount", "sum"),
             bank_name=("bank_name", "first"),
             city=("city", "first"),
             lat=("lat", "first"),
             long=("long", "first"))
        .reset_index().rename(columns={"dst_account": "account_id"})
    )
    sent_to_dst = (
        transactions_df.groupby("src_account")
        .agg(total_sent=("amount", "sum"))
        .reset_index().rename(columns={"src_account": "account_id"})
    )
    txn_count = (
        transactions_df.groupby("dst_account").size()
        .reset_index(name="txn_count_24h")
        .rename(columns={"dst_account": "account_id"})
    )
    avg_amt = (
        transactions_df.groupby("dst_account")
        .agg(avg_txn_amount=("amount", "mean"))
        .reset_index().rename(columns={"dst_account": "account_id"})
    )
    hop = (
        transactions_df.groupby("dst_account")
        .agg(hop_depth=("hop_depth", "max"))
        .reset_index().rename(columns={"dst_account": "account_id"})
    )

    COL_ORDER = ["account_id", "bank_name", "city", "lat", "long",
                 "total_received", "total_sent", "txn_count_24h",
                 "avg_txn_amount", "is_mule_label", "hop_depth"]

    # Merge all mule aggregates
    mule_df = (received
               .merge(sent_to_dst, on="account_id", how="left")
               .merge(txn_count, on="account_id", how="left")
               .merge(avg_amt, on="account_id", how="left")
               .merge(hop, on="account_id", how="left"))
    mule_df["total_sent"] = mule_df["total_sent"].fillna(0.0)
    mule_df["txn_count_24h"] = mule_df["txn_count_24h"].fillna(1).astype(int)
    mule_df["avg_txn_amount"] = mule_df["avg_txn_amount"].round(2)
    mule_df["hop_depth"] = mule_df["hop_depth"].fillna(1).astype(int)
    mule_df["is_mule_label"] = 1  # all dst accounts are mules

    # Clean nodes: src-only accounts that NEVER appear as dst
    all_dst = set(transactions_df["dst_account"])
    clean_src = transactions_df[~transactions_df["src_account"].isin(all_dst)]
    if not clean_src.empty:
        clean_df = (
            clean_src.groupby("src_account")
            .agg(total_sent=("amount", "sum"),
                 bank_name=("bank_name", "first"),
                 city=("city", "first"),
                 lat=("lat", "first"),
                 long=("long", "first"),
                 avg_txn_amount=("amount", "mean"),
                 txn_count_24h=("amount", "count"))
            .reset_index().rename(columns={"src_account": "account_id"})
        )
        clean_df["total_received"] = 0.0
        clean_df["hop_depth"] = 0
        clean_df["is_mule_label"] = 0
        clean_df["avg_txn_amount"] = clean_df["avg_txn_amount"].round(2)
        node_df = pd.concat([mule_df[COL_ORDER], clean_df[COL_ORDER]], ignore_index=True)
    else:
        node_df = mule_df[COL_ORDER].copy()

    node_df = node_df.drop_duplicates(subset=["account_id"]).reset_index(drop=True)
    node_df["lat"] = node_df["lat"].clip(INDIA_LAT_MIN, INDIA_LAT_MAX).round(6)
    node_df["long"] = node_df["long"].clip(INDIA_LON_MIN, INDIA_LON_MAX).round(6)
    return node_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(n_complaints=500, n_transactions=2000, n_atms=200, seed=42):
    """Run the full Phase 1 data generation pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker("en_IN")
    fake.seed_instance(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MuleShield AI - Phase 1 Data Generator  |  SIH26184")
    print("=" * 60)

    print(f"\n[1/5] Generating {n_complaints} victim complaints...")
    complaints_df = generate_victim_complaints(n_complaints, fake)
    complaints_df.to_csv(DATA_DIR / "victim_complaints.csv", index=False)
    print(f"      [OK] {len(complaints_df)} rows -> data/victim_complaints.csv")

    print(f"\n[2/5] Generating >={n_transactions} transaction records...")
    transactions_df = generate_transactions(complaints_df, n_transactions, fake)
    transactions_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    print(f"      [OK] {len(transactions_df)} rows -> data/transactions.csv")
    print(f"      [OK] Terminal nodes: {transactions_df['is_terminal'].sum()}")

    print(f"\n[3/5] Generating {n_atms} ATM/CSP entries...")
    atm_df = generate_atm_directory(n_atms, fake)
    atm_df.to_csv(DATA_DIR / "atm_directory.csv", index=False)
    print(f"      [OK] {len(atm_df)} rows -> data/atm_directory.csv")

    print("\n[4/5] Building graph edge list...")
    edges_df = generate_graph_edges(transactions_df)
    edges_df.to_csv(DATA_DIR / "graph_edges.csv", index=False)
    print(f"      [OK] {len(edges_df)} edges -> data/graph_edges.csv")

    print("\n[5/5] Building node feature matrix...")
    node_df = generate_node_features(transactions_df)
    node_df.to_csv(DATA_DIR / "node_features.csv", index=False)
    mule_count = node_df["is_mule_label"].sum()
    print(f"      [OK] {len(node_df)} nodes -> data/node_features.csv")
    print(f"      [OK] Mule nodes: {mule_count} | Clean nodes: {len(node_df) - mule_count}")

    print("\n" + "=" * 60)
    print(f"  Phase 1 Complete! Files saved to: {DATA_DIR.resolve()}")
    print("=" * 60)

    return {
        "complaints": complaints_df,
        "transactions": transactions_df,
        "atms": atm_df,
        "graph_edges": edges_df,
        "node_features": node_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuleShield AI — Phase 1 Data Generator")
    parser.add_argument("--complaints", type=int, default=500)
    parser.add_argument("--transactions", type=int, default=2000)
    parser.add_argument("--atms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.complaints, args.transactions, args.atms, args.seed)
