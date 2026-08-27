# -*- coding: utf-8 -*-
"""
MuleShield AI — Phase 1: Realistic Pan-India Banking & Cybercrime Dataset Generator
SIH26184 | Ministry of Home Affairs / I4C

Scales dataset to Pan-India national coverage:
  - 65+ Cities & Notorious Cybercrime Hotspots (Jamtara, Mewat/Nuh, Bharatpur, Deoghar, etc.)
  - 16 Major Public & Private Indian Banks with Authentic IFSC Prefixes
  - 9 Realistic 1930 Cybercrime Modalities (Digital Arrest, Task Scams, APK Loan Extortion, etc.)
  - 2,500+ Victim Complaints, 20,000+ Multi-Hop Transactions, 1,000+ Geo-coded ATMs
  - Fraud Rings (1-to-many splitting, many-to-one pooling, deep 4-hop layering)

Generates:
  - data/victim_complaints.csv   -> 1930-style complaint tickets
  - data/transactions.csv        -> Multi-hop IMPS/UPI ledger with fraud rings
  - data/atm_directory.csv       -> Geo-coded ATM/CSP database across 65+ cities
  - data/graph_edges.csv         -> Edge list for PyG/NetworkX ingestion
  - data/node_features.csv       -> Per-account feature matrix for GNN training

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --complaints 2500 --transactions 20000 --atms 1000 --seed 42
"""

import argparse
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ─────────────────────────────────────────────
# CONSTANTS & BOUNDS
# ─────────────────────────────────────────────
INDIA_LAT_MIN, INDIA_LAT_MAX = 8.0, 37.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 97.5
MAX_HOP_DEPTH = 4
VELOCITY_WINDOW_SECONDS = 300
MIN_SPLIT_DESTINATIONS = 3
DATA_DIR = Path(__file__).parent.parent / "data"

# Realistic Cybercrime Modalities (NCRP / 1930 Helpline taxonomies)
FRAUD_TYPES = [
    "Digital Arrest",
    "Investment Scam",
    "Job Scam",
    "UPI Fraud",
    "APK Loan Scam",
    "Romance Scam",
    "SIM Swap / KYC",
    "Electricity Bill Scam",
    "Bank Impersonation",
]

# 16 Major Indian Banks with accurate 4-character IFSC prefixes
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
    ("IndusInd Bank", "INDB"),
    ("Federal Bank", "FDRL"),
    ("IDBI Bank", "IBKL"),
    ("YES Bank", "YESB"),
    ("Indian Bank", "IDIB"),
    ("Central Bank of India", "CBIN"),
    ("Bank of India", "BKID"),
]

# 65+ Pan-India Cities & Notorious Cybercrime Hotspots
# (City Name, District, State, Latitude, Longitude)
INDIAN_CITIES = [
    # ── Notorious Cybercrime Hotspots ─────────────────────────────────────────
    ("Jamtara", "Jamtara", "Jharkhand", 23.9624, 86.8016),
    ("Nuh", "Mewat", "Haryana", 28.1188, 77.0090),
    ("Bharatpur", "Bharatpur", "Rajasthan", 27.2152, 77.5030),
    ("Alwar", "Alwar", "Rajasthan", 27.5530, 76.6346),
    ("Deoghar", "Deoghar", "Jharkhand", 24.4826, 86.7000),
    ("Giridih", "Giridih", "Jharkhand", 24.1866, 86.3059),
    ("Mathura", "Mathura", "Uttar Pradesh", 27.4924, 77.6737),
    ("Karmatar", "Jamtara", "Jharkhand", 24.0921, 86.8521),
    ("Surat", "Surat", "Gujarat", 21.1702, 72.8311),
    ("Bidhannagar", "North 24 Parganas", "West Bengal", 22.5804, 88.4194),
    ("Cyberabad", "Ranga Reddy", "Telangana", 17.4435, 78.3772),

    # ── North Zone ────────────────────────────────────────────────────────────
    ("Delhi", "Central Delhi", "Delhi", 28.6139, 77.2090),
    ("Noida", "Gautam Buddh Nagar", "Uttar Pradesh", 28.5355, 77.3910),
    ("Gurgaon", "Gurugram", "Haryana", 28.4595, 77.0266),
    ("Ghaziabad", "Ghaziabad", "Uttar Pradesh", 28.6692, 77.4538),
    ("Faridabad", "Faridabad", "Haryana", 28.4089, 77.3178),
    ("Jaipur", "Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Lucknow", "Lucknow", "Uttar Pradesh", 26.8467, 80.9462),
    ("Kanpur", "Kanpur Nagar", "Uttar Pradesh", 26.4499, 80.3319),
    ("Varanasi", "Varanasi", "Uttar Pradesh", 25.3176, 82.9739),
    ("Agra", "Agra", "Uttar Pradesh", 27.1767, 78.0081),
    ("Prayagraj", "Prayagraj", "Uttar Pradesh", 25.4358, 81.8463),
    ("Chandigarh", "Chandigarh", "Chandigarh", 30.7333, 76.7794),
    ("Ludhiana", "Ludhiana", "Punjab", 30.9010, 75.8573),
    ("Amritsar", "Amritsar", "Punjab", 31.6340, 74.8723),
    ("Dehradun", "Dehradun", "Uttarakhand", 30.3165, 78.0322),
    ("Haridwar", "Haridwar", "Uttarakhand", 29.9457, 78.1642),
    ("Jammu", "Jammu", "Jammu and Kashmir", 32.7266, 74.8570),
    ("Srinagar", "Srinagar", "Jammu and Kashmir", 34.0837, 74.7973),
    ("Shimla", "Shimla", "Himachal Pradesh", 31.1048, 77.1734),

    # ── South Zone ────────────────────────────────────────────────────────────
    ("Bengaluru", "Bengaluru Urban", "Karnataka", 12.9716, 77.5946),
    ("Hyderabad", "Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Chennai", "Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Kochi", "Ernakulam", "Kerala", 9.9312, 76.2673),
    ("Thiruvananthapuram", "Thiruvananthapuram", "Kerala", 8.5241, 76.9366),
    ("Kozhikode", "Kozhikode", "Kerala", 11.2588, 75.7804),
    ("Visakhapatnam", "Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185),
    ("Vijayawada", "NTR", "Andhra Pradesh", 16.5062, 80.6480),
    ("Guntur", "Guntur", "Andhra Pradesh", 16.3067, 80.4365),
    ("Coimbatore", "Coimbatore", "Tamil Nadu", 11.0168, 76.9558),
    ("Madurai", "Madurai", "Tamil Nadu", 9.9252, 78.1198),
    ("Mysuru", "Mysuru", "Karnataka", 12.2958, 76.6394),
    ("Mangaluru", "Dakshina Kannada", "Karnataka", 12.9141, 74.8560),
    ("Hubballi", "Dharwad", "Karnataka", 15.3647, 75.1240),

    # ── West Zone ─────────────────────────────────────────────────────────────
    ("Mumbai", "Mumbai City", "Maharashtra", 19.0760, 72.8777),
    ("Pune", "Pune", "Maharashtra", 18.5204, 73.8567),
    ("Nagpur", "Nagpur", "Maharashtra", 21.1458, 79.0882),
    ("Nashik", "Nashik", "Maharashtra", 19.9975, 73.7898),
    ("Thane", "Thane", "Maharashtra", 19.2183, 72.9781),
    ("Navi Mumbai", "Thane", "Maharashtra", 19.0330, 73.0297),
    ("Ahmedabad", "Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Vadodara", "Vadodara", "Gujarat", 22.3072, 73.1812),
    ("Rajkot", "Rajkot", "Gujarat", 22.3039, 70.8022),
    ("Bhopal", "Bhopal", "Madhya Pradesh", 23.2599, 77.4126),
    ("Indore", "Indore", "Madhya Pradesh", 22.7196, 75.8577),
    ("Gwalior", "Gwalior", "Madhya Pradesh", 26.2183, 78.1828),
    ("Jabalpur", "Jabalpur", "Madhya Pradesh", 23.1815, 79.9864),
    ("Goa (Panaji)", "North Goa", "Goa", 15.4909, 73.8278),

    # ── East & Central Zone ───────────────────────────────────────────────────
    ("Kolkata", "Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Howrah", "Howrah", "West Bengal", 22.5958, 88.2636),
    ("Patna", "Patna", "Bihar", 25.5941, 85.1376),
    ("Gaya", "Gaya", "Bihar", 24.7914, 85.0002),
    ("Muzaffarpur", "Muzaffarpur", "Bihar", 26.1209, 85.3647),
    ("Bhubaneswar", "Khurda", "Odisha", 20.2961, 85.8245),
    ("Cuttack", "Cuttack", "Odisha", 20.4625, 85.8828),
    ("Ranchi", "Ranchi", "Jharkhand", 23.3441, 85.3096),
    ("Jamshedpur", "East Singhbhum", "Jharkhand", 22.8046, 86.2029),
    ("Dhanbad", "Dhanbad", "Jharkhand", 23.7957, 86.4304),
    ("Raipur", "Raipur", "Chhattisgarh", 21.2514, 81.6296),
    ("Bilaspur", "Bilaspur", "Chhattisgarh", 22.0797, 82.1409),

    # ── Northeast Zone ────────────────────────────────────────────────────────
    ("Guwahati", "Kamrup Metropolitan", "Assam", 26.1445, 91.7362),
    ("Silchar", "Cachar", "Assam", 24.8333, 92.7789),
    ("Shillong", "East Khasi Hills", "Meghalaya", 25.5788, 91.8933),
    ("Agartala", "West Tripura", "Tripura", 23.8315, 91.2868),
    ("Imphal", "Imphal West", "Manipur", 24.8170, 93.9368),
    ("Dimapur", "Dimapur", "Nagaland", 25.9068, 93.7273),
    ("Aizawl", "Aizawl", "Mizoram", 23.7271, 92.7176),
    ("Gangtok", "East Sikkim", "Sikkim", 27.3389, 88.6065),
]


# ─────────────────────────────────────────────
# HELPER GENERATION UTILITIES
# ─────────────────────────────────────────────

def _random_ifsc(bank_code: str) -> str:
    """Generate authentic IFSC format: BANKCODE + '0' + 6-digit branch code."""
    return f"{bank_code}0{random.randint(100000, 999999)}"


def _masked_account() -> str:
    """Return a masked 12-digit Indian bank account number."""
    return f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"


def _jitter_coords(lat: float, lon: float, radius_km: float = 6.0):
    """Jitter geographical coordinates within realistic urban radius."""
    delta = radius_km / 111.0
    new_lat = max(INDIA_LAT_MIN, min(INDIA_LAT_MAX, lat + random.uniform(-delta, delta)))
    new_lon = max(INDIA_LON_MIN, min(INDIA_LON_MAX, lon + random.uniform(-delta, delta)))
    return round(new_lat, 6), round(new_lon, 6)


def _random_bank():
    return random.choice(INDIAN_BANKS)


def _random_city():
    return random.choice(INDIAN_CITIES)


def _split_amount(total: float, n: int):
    """Split total stolen amount into n parts using Dirichlet distribution."""
    if n == 1:
        return [round(total, 2)]
    parts = np.random.dirichlet(np.ones(n)) * total
    return [round(max(p, 250.0), 2) for p in parts]


# ─────────────────────────────────────────────
# GENERATOR 1 — Victim Complaints (1930 Feed)
# ─────────────────────────────────────────────

def generate_victim_complaints(n: int, fake: Faker) -> pd.DataFrame:
    """Generate n national 1930-style victim complaints."""
    records = []
    base_ts = datetime.now() - timedelta(days=120)

    for _ in range(n):
        bank_name, _ = _random_bank()
        city, district, state, lat, lon = _random_city()
        fraud_type = random.choice(FRAUD_TYPES)

        # Scale amounts based on fraud complexity (capped at 5,00,000)
        if fraud_type == "Digital Arrest":
            amount = round(random.uniform(50_000, 5_00_000), 2)
        elif fraud_type == "Investment Scam":
            amount = round(random.uniform(25_000, 5_00_000), 2)
        elif fraud_type in ["APK Loan Scam", "Job Scam"]:
            amount = round(random.uniform(10_000, 3_50_000), 2)
        else:
            amount = round(random.uniform(5_000, 2_00_000), 2)

        records.append({
            "ticket_id": str(uuid.uuid4()),
            "victim_name": fake.name(),
            "victim_bank": bank_name,
            "victim_account": _masked_account(),
            "fraud_type": fraud_type,
            "stolen_amount": amount,
            "complaint_timestamp": (
                base_ts + timedelta(seconds=random.randint(0, 120 * 24 * 3600))
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "city": city,
            "state": state,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# GENERATOR 2 — Multi-Hop Transactions & Fraud Rings
# ─────────────────────────────────────────────

def _build_chain(complaint_id: str, complaint_ts: datetime, stolen_amount: float, fake: Faker):
    """Build multi-hop transaction chain embedding complex fraud topologies."""
    txns = []
    current_time = complaint_ts + timedelta(seconds=random.randint(30, 180))
    victim_account = _masked_account()

    # Hop 1: L0 -> L1 (1 to 3 mule accounts)
    n_l1 = random.randint(1, 3)
    l1_accounts = [_masked_account() for _ in range(n_l1)]
    l1_amounts = _split_amount(stolen_amount, n_l1)

    for l1_acc, l1_amt in zip(l1_accounts, l1_amounts):
        current_time += timedelta(seconds=random.randint(15, 60))
        bank_name, bank_code = _random_bank()
        city, dist, state, lat, lon = _random_city()
        txns.append({
            "txn_id": str(uuid.uuid4()), "complaint_id": complaint_id,
            "src_account": victim_account, "dst_account": l1_acc,
            "bank_name": bank_name, "ifsc_code": _random_ifsc(bank_code),
            "city": city, "district": dist, "state": state,
            "lat": round(lat + random.uniform(-0.04, 0.04), 6),
            "long": round(lon + random.uniform(-0.04, 0.04), 6),
            "amount": round(l1_amt, 2),
            "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hop_depth": 1, "is_terminal": 0,
        })

    # Hop 2: L1 -> L2
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
                "lat": round(lat + random.uniform(-0.04, 0.04), 6),
                "long": round(lon + random.uniform(-0.04, 0.04), 6),
                "amount": round(l2_amt, 2),
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "hop_depth": 2, "is_terminal": 0,
            })

    # FRAUD RING: Multiple L2 accounts converging into shared terminal nodes
    all_l2 = [acc for group in l2_groups for acc, _ in group]
    all_l2_amts = [amt for group in l2_groups for _, amt in group]

    if len(all_l2) >= MIN_SPLIT_DESTINATIONS:
        ring_size = min(len(all_l2), random.randint(3, min(5, len(all_l2))))
        ring_members = random.sample(list(zip(all_l2, all_l2_amts)), k=ring_size)
        shared_terminal = _masked_account()
        bank_name, bank_code = _random_bank()
        city, dist, state, lat, lon = _random_city()

        for ring_src, ring_amt in ring_members:
            current_time += timedelta(seconds=random.randint(45, 180))
            txns.append({
                "txn_id": str(uuid.uuid4()), "complaint_id": complaint_id,
                "src_account": ring_src, "dst_account": shared_terminal,
                "bank_name": bank_name, "ifsc_code": _random_ifsc(bank_code),
                "city": city, "district": dist, "state": state,
                "lat": round(lat + random.uniform(-0.03, 0.03), 6),
                "long": round(lon + random.uniform(-0.03, 0.03), 6),
                "amount": round(ring_amt * 0.92, 2),
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "hop_depth": 3, "is_terminal": 1,
            })
    else:
        # Deep chain L2 -> L3 -> L4 (terminal cashout)
        for l2_group in l2_groups:
            deepened = random.sample(l2_group, k=min(1, len(l2_group)))
            for l2_acc, l2_amt in deepened:
                depth = random.choice([3, 4])
                src = l2_acc
                remaining = l2_amt
                for d in range(3, depth + 1):
                    terminal = int(d == depth)
                    dst_acc = _masked_account()
                    current_time += timedelta(seconds=random.randint(60, 240))
                    bank_name, bank_code = _random_bank()
                    city, dist, state, lat, lon = _random_city()
                    txns.append({
                        "txn_id": str(uuid.uuid4()), "complaint_id": complaint_id,
                        "src_account": src, "dst_account": dst_acc,
                        "bank_name": bank_name, "ifsc_code": _random_ifsc(bank_code),
                        "city": city, "district": dist, "state": state,
                        "lat": round(lat + random.uniform(-0.04, 0.04), 6),
                        "long": round(lon + random.uniform(-0.04, 0.04), 6),
                        "amount": round(remaining, 2),
                        "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "hop_depth": d, "is_terminal": terminal,
                    })
                    src = dst_acc

    return txns


def generate_transactions(complaints_df: pd.DataFrame, target: int, fake: Faker) -> pd.DataFrame:
    """Generate Pan-India multi-hop transaction dataset ensuring target volume."""
    all_txns = []
    cids = complaints_df["ticket_id"].tolist()
    ts_map = dict(zip(complaints_df["ticket_id"],
                      pd.to_datetime(complaints_df["complaint_timestamp"])))
    amt_map = dict(zip(complaints_df["ticket_id"], complaints_df["stolen_amount"]))

    # Pass 1: Every complaint gets a complete multi-hop chain
    for cid in cids:
        all_txns.extend(_build_chain(cid, ts_map[cid], amt_map[cid], fake))

    # Pass 2: Expand until volume threshold is fulfilled
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
# GENERATOR 3 — Pan-India Geo-coded ATM Directory
# ─────────────────────────────────────────────

def generate_atm_directory(n: int, fake: Faker) -> pd.DataFrame:
    """Generate n geo-coded ATM/CSP units distributed across all 65+ cities."""
    hotspot_cities = {
        "Jamtara", "Nuh", "Bharatpur", "Alwar", "Deoghar", "Mathura",
        "Delhi", "Mumbai", "Noida", "Gurgaon", "Patna", "Surat", "Bidhannagar", "Cyberabad",
    }
    records = []
    for _ in range(n):
        bank_name, _ = _random_bank()
        city, district, state, lat, lon = _random_city()
        jlat, jlon = _jitter_coords(lat, lon, radius_km=12.0)
        base_risk = 0.7 if city in hotspot_cities else 0.35
        risk_score = round(min(1.0, max(0.05, base_risk + random.uniform(-0.25, 0.35))), 3)

        records.append({
            "atm_id": f"ATM-{str(uuid.uuid4())[:8].upper()}",
            "bank_name": bank_name,
            "address": fake.address().replace("\n", ", "),
            "city": city, "district": district, "state": state,
            "lat": jlat, "long": jlon,
            "opening_time": random.choice(["00:00", "06:00", "07:00", "08:00"]),
            "closing_time": random.choice(["23:59", "22:00", "21:00"]),
            "cashout_risk_score": risk_score,
            "historical_fraud_count": random.randint(5, 180) if city in hotspot_cities else random.randint(0, 45),
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
# GENERATOR 5 — Node Feature Matrix
# ─────────────────────────────────────────────

def generate_node_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-account node feature matrix for GNN training."""
    # Mule nodes: accounts receiving funds in transaction chains
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

    mule_df = (received
               .merge(sent_to_dst, on="account_id", how="left")
               .merge(txn_count, on="account_id", how="left")
               .merge(avg_amt, on="account_id", how="left")
               .merge(hop, on="account_id", how="left"))
    mule_df["total_sent"] = mule_df["total_sent"].fillna(0.0)
    mule_df["txn_count_24h"] = mule_df["txn_count_24h"].fillna(1).astype(int)
    mule_df["avg_txn_amount"] = mule_df["avg_txn_amount"].round(2)
    mule_df["hop_depth"] = mule_df["hop_depth"].fillna(1).astype(int)
    mule_df["is_mule_label"] = 1

    # Clean nodes: source-only accounts (victims / legitimate feeders)
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
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main(n_complaints=2500, n_transactions=20000, n_atms=1000, seed=42):
    """Run full Pan-India data generation pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker("en_IN")
    fake.seed_instance(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  MuleShield AI — Pan-India National Dataset Generator (SIH26184)")
    print("=" * 65)

    print(f"\n[1/5] Generating {n_complaints:,} victim complaints across 65+ cities...")
    complaints_df = generate_victim_complaints(n_complaints, fake)
    complaints_df.to_csv(DATA_DIR / "victim_complaints.csv", index=False)
    print(f"      [OK] {len(complaints_df):,} rows -> data/victim_complaints.csv")

    print(f"\n[2/5] Generating >={n_transactions:,} multi-hop transaction records...")
    transactions_df = generate_transactions(complaints_df, n_transactions, fake)
    transactions_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    terminals = transactions_df["is_terminal"].sum()
    print(f"      [OK] {len(transactions_df):,} rows -> data/transactions.csv")
    print(f"      [OK] Cashout Terminal Nodes: {terminals:,}")

    print(f"\n[3/5] Generating {n_atms:,} geo-coded ATM/CSP entries across 65+ cities...")
    atm_df = generate_atm_directory(n_atms, fake)
    atm_df.to_csv(DATA_DIR / "atm_directory.csv", index=False)
    print(f"      [OK] {len(atm_df):,} rows -> data/atm_directory.csv")

    print("\n[4/5] Building graph edge list...")
    edges_df = generate_graph_edges(transactions_df)
    edges_df.to_csv(DATA_DIR / "graph_edges.csv", index=False)
    print(f"      [OK] {len(edges_df):,} edges -> data/graph_edges.csv")

    print("\n[5/5] Building node feature matrix...")
    node_df = generate_node_features(transactions_df)
    node_df.to_csv(DATA_DIR / "node_features.csv", index=False)
    mule_count = node_df["is_mule_label"].sum()
    clean_count = len(node_df) - mule_count
    print(f"      [OK] {len(node_df):,} nodes -> data/node_features.csv")
    print(f"      [OK] Mule nodes: {mule_count:,} | Clean nodes: {clean_count:,}")

    print("\n" + "=" * 65)
    print(f"  Dataset successfully generated in: {DATA_DIR.resolve()}")
    print("=" * 65)

    return {
        "complaints": complaints_df,
        "transactions": transactions_df,
        "atms": atm_df,
        "graph_edges": edges_df,
        "node_features": node_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuleShield AI — Pan-India Dataset Generator")
    parser.add_argument("--complaints", type=int, default=2500)
    parser.add_argument("--transactions", type=int, default=20000)
    parser.add_argument("--atms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.complaints, args.transactions, args.atms, args.seed)
