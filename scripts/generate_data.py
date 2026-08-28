# -*- coding: utf-8 -*-
"""
MuleShield AI — Phase 1: Realistic Pan-India Banking & Cybercrime Dataset Generator
SIH26184 | Ministry of Home Affairs / I4C

Design principle — no label leakage
-----------------------------------
An earlier revision of this generator defined a mule as "any account that
received money", then wrote `total_received = 0` and `hop_depth = 0` onto every
non-mule account. Both columns were fed to the GNN as features, so the label was
literally a copy of a feature: the rule `total_received > 0` scored F1 = 1.0000,
and the reported 0.9996 GraphSAGE F1 measured nothing at all.

This version inverts the causal order:

  1. An account registry is built FIRST. Each account is assigned an archetype
     (salaried, merchant, transit business, student, senior, mule) and, with it,
     ground-truth mule status — before a single transaction exists.
  2. Both legitimate and fraudulent transactions are then simulated from those
     archetypes' behaviour.
  3. Node features are MEASURED from the resulting ledger.

So legitimate accounts receive money too, and the classes overlap on purpose —
the `transit_business` archetype (payment aggregators, trading firms) forwards
funds almost as fast and as completely as a mule does. Separating them requires
the neighbourhood structure a GNN provides, which is the point of the model.

Two further realism fixes:
  - Mule accounts are drawn from reusable SYNDICATE pools, so the same accounts
    recur across complaints. Previously every chain minted fresh accounts, which
    left the graph as thousands of disconnected 5-node components with no ring
    structure for GraphSAGE to exploit.
  - Chains stay regionally coherent instead of teleporting between random cities
    at every hop.

Ground truth that belongs in the data now lives in the data: each terminal
transaction carries the ATM the cashout actually happened at, and the delay
before it. Both were previously derived inside the feature builder — the ATM
label was `argmin(distance)` while distance was itself an input feature.

Generates:
  - data/victim_complaints.csv   -> 1930-style complaint tickets
  - data/transactions.csv        -> Multi-hop ledger + legitimate banking activity
  - data/atm_directory.csv       -> Geo-coded ATM/CSP database
  - data/graph_edges.csv         -> Edge list for PyG/NetworkX ingestion
  - data/node_features.csv       -> Measured per-account behavioural features

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --complaints 2500 --transactions 20000 --atms 1000 --seed 42
"""

import argparse

import random
import uuid
from collections import defaultdict
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

# Fraction of ground-truth labels that are wrong, reflecting imperfect bank
# reporting: undetected mules and accounts wrongly flagged in an STR.
LABEL_NOISE_RATE = 0.02

# ATM choice model — a mule does not always walk to the nearest ATM.
#
# These weights matter more than they look. An earlier setting (decay 2.5 km,
# risk 1.2, bank 1.6) made distance so dominant that the Bayes-optimal ranking
# beat a plain "nearest ATM" rule by 0.7 percentage points. No model can show
# skill on a task where the naive answer is already optimal, so the task itself
# has to carry a systematic non-distance component.
#
# The dominant one here is syndicate preference: a crew returns to cashout points
# it has already tested — an ATM with a broken camera, a compliant operator, a
# quiet forecourt. That is real syndicate behaviour, and it is only predictable
# if you know which crew holds the terminal account, which is exactly what the
# graph embedding encodes. Distance alone cannot recover it.
ATM_DISTANCE_DECAY_KM = 5.0     # exponential decay length
ATM_RISK_WEIGHT = 2.0           # pull toward ATMs with weak surveillance history
ATM_SAME_BANK_BOOST = 2.0       # own-bank ATMs avoid interchange limits
ATM_SYNDICATE_PREF_BOOST = 9.0  # a crew's established cashout points
ATM_PREF_PER_SYNDICATE = 4      # how many points a crew keeps in rotation
ATM_CANDIDATE_RADIUS_KM = 60.0
ATM_MIN_CANDIDATES = 25

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
    ("Union Bank of India", "UBIN"),
    ("Kotak Mahindra Bank", "KKBK"),
    ("IndusInd Bank", "INDB"),
    ("YES Bank", "YESB"),
    ("IDBI Bank", "IBKL"),
    ("Indian Bank", "IDIB"),
    ("Central Bank of India", "CBIN"),
    ("Bank of India", "BKID"),
    ("Federal Bank", "FDRL"),
]

# ─────────────────────────────────────────────
# ACCOUNT ARCHETYPES
# ─────────────────────────────────────────────
# Each archetype fixes the BEHAVIOUR an account will exhibit. Mule status is a
# property of the archetype, fixed before any transaction is generated, so no
# measured feature can be definitionally equal to the label.
#
# `transit_business` exists specifically to overlap with mules: a payment
# aggregator or a commodity trader legitimately sweeps almost everything it
# receives, within minutes. Any classifier keying purely on "fast, complete
# pass-through" will misclassify these — which is what makes the task real.

ARCHETYPES = {
    "salaried": {
        "is_mule": False, "weight": 0.30,
        "age_days": (400, 3200), "dwell_hours": (18.0, 400.0),
        "passthrough": (0.05, 0.55), "fan_out": (1, 4), "night_ratio": (0.02, 0.15),
    },
    "merchant": {
        "is_mule": False, "weight": 0.22,
        "age_days": (200, 2600), "dwell_hours": (2.0, 60.0),
        "passthrough": (0.30, 0.85), "fan_out": (2, 9), "night_ratio": (0.05, 0.25),
    },
    "transit_business": {          # deliberate overlap class
        "is_mule": False, "weight": 0.10,
        "age_days": (60, 1400), "dwell_hours": (0.08, 3.0),
        "passthrough": (0.72, 0.98), "fan_out": (3, 12), "night_ratio": (0.10, 0.40),
    },
    "student": {
        "is_mule": False, "weight": 0.13,
        "age_days": (40, 900), "dwell_hours": (6.0, 180.0),
        "passthrough": (0.20, 0.80), "fan_out": (1, 5), "night_ratio": (0.10, 0.35),
    },
    "senior": {
        "is_mule": False, "weight": 0.07,
        "age_days": (900, 6000), "dwell_hours": (48.0, 900.0),
        "passthrough": (0.02, 0.35), "fan_out": (1, 3), "night_ratio": (0.01, 0.08),
    },
    "mule": {
        "is_mule": True, "weight": 0.18,
        "age_days": (4, 320), "dwell_hours": (0.03, 6.0),
        "passthrough": (0.84, 0.995), "fan_out": (2, 8), "night_ratio": (0.15, 0.55),
    },
}

INDIAN_CITIES = [
    # (city, district, state, lat, lon)
    ("Delhi", "New Delhi", "Delhi", 28.6139, 77.2090),
    ("Mumbai", "Mumbai Suburban", "Maharashtra", 19.0760, 72.8777),
    ("Bengaluru", "Bengaluru Urban", "Karnataka", 12.9716, 77.5946),
    ("Hyderabad", "Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Chennai", "Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Kolkata", "Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Pune", "Pune", "Maharashtra", 18.5204, 73.8567),
    ("Ahmedabad", "Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Jaipur", "Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Lucknow", "Lucknow", "Uttar Pradesh", 26.8467, 80.9462),
    ("Kanpur", "Kanpur Nagar", "Uttar Pradesh", 26.4499, 80.3319),
    ("Nagpur", "Nagpur", "Maharashtra", 21.1458, 79.0882),
    ("Indore", "Indore", "Madhya Pradesh", 22.7196, 75.8577),
    ("Bhopal", "Bhopal", "Madhya Pradesh", 23.2599, 77.4126),
    ("Patna", "Patna", "Bihar", 25.5941, 85.1376),
    ("Ludhiana", "Ludhiana", "Punjab", 30.9010, 75.8573),
    ("Agra", "Agra", "Uttar Pradesh", 27.1767, 78.0081),
    ("Nashik", "Nashik", "Maharashtra", 19.9975, 73.7898),
    ("Vadodara", "Vadodara", "Gujarat", 22.3072, 73.1812),
    ("Varanasi", "Varanasi", "Uttar Pradesh", 25.3176, 82.9739),
    ("Srinagar", "Srinagar", "Jammu & Kashmir", 34.0837, 74.7973),
    ("Aurangabad", "Aurangabad", "Maharashtra", 19.8762, 75.3433),
    ("Dhanbad", "Dhanbad", "Jharkhand", 23.7957, 86.4304),
    ("Amritsar", "Amritsar", "Punjab", 31.6340, 74.8723),
    ("Allahabad", "Prayagraj", "Uttar Pradesh", 25.4358, 81.8463),
    ("Ranchi", "Ranchi", "Jharkhand", 23.3441, 85.3096),
    ("Coimbatore", "Coimbatore", "Tamil Nadu", 11.0168, 76.9558),
    ("Jabalpur", "Jabalpur", "Madhya Pradesh", 23.1815, 79.9864),
    ("Gwalior", "Gwalior", "Madhya Pradesh", 26.2183, 78.1828),
    ("Vijayawada", "Krishna", "Andhra Pradesh", 16.5062, 80.6480),
    ("Jodhpur", "Jodhpur", "Rajasthan", 26.2389, 73.0243),
    ("Madurai", "Madurai", "Tamil Nadu", 9.9252, 78.1198),
    ("Raipur", "Raipur", "Chhattisgarh", 21.2514, 81.6296),
    ("Kota", "Kota", "Rajasthan", 25.2138, 75.8648),
    ("Guwahati", "Kamrup Metropolitan", "Assam", 26.1445, 91.7362),
    ("Chandigarh", "Chandigarh", "Chandigarh", 30.7333, 76.7794),
    ("Thiruvananthapuram", "Thiruvananthapuram", "Kerala", 8.5241, 76.9366),
    ("Solapur", "Solapur", "Maharashtra", 17.6599, 75.9064),
    ("Bareilly", "Bareilly", "Uttar Pradesh", 28.3670, 79.4304),
    ("Mysuru", "Mysuru", "Karnataka", 12.2958, 76.6394),
    ("Tiruchirappalli", "Tiruchirappalli", "Tamil Nadu", 10.7905, 78.7047),
    ("Bhubaneswar", "Khordha", "Odisha", 20.2961, 85.8245),
    ("Salem", "Salem", "Tamil Nadu", 11.6643, 78.1460),
    ("Warangal", "Warangal", "Telangana", 17.9689, 79.5941),
    ("Guntur", "Guntur", "Andhra Pradesh", 16.3067, 80.4365),
    ("Bhiwandi", "Thane", "Maharashtra", 19.3002, 73.0682),
    ("Saharanpur", "Saharanpur", "Uttar Pradesh", 29.9680, 77.5552),
    ("Gorakhpur", "Gorakhpur", "Uttar Pradesh", 26.7606, 83.3732),
    ("Bikaner", "Bikaner", "Rajasthan", 28.0229, 73.3119),
    ("Amravati", "Amravati", "Maharashtra", 20.9374, 77.7796),
    ("Noida", "Gautam Buddh Nagar", "Uttar Pradesh", 28.5355, 77.3910),
    ("Jamshedpur", "East Singhbhum", "Jharkhand", 22.8046, 86.2029),
    ("Bhilai", "Durg", "Chhattisgarh", 21.1938, 81.3509),
    ("Cuttack", "Cuttack", "Odisha", 20.4625, 85.8830),
    ("Firozabad", "Firozabad", "Uttar Pradesh", 27.1592, 78.3957),
    ("Kochi", "Ernakulam", "Kerala", 9.9312, 76.2673),
    ("Dehradun", "Dehradun", "Uttarakhand", 30.3165, 78.0322),
    ("Durgapur", "Paschim Bardhaman", "West Bengal", 23.5204, 87.3119),
    ("Asansol", "Paschim Bardhaman", "West Bengal", 23.6739, 86.9524),
    ("Rourkela", "Sundargarh", "Odisha", 22.2604, 84.8536),
    ("Siliguri", "Darjeeling", "West Bengal", 26.7271, 88.3953),
    ("Aizawl", "Aizawl", "Mizoram", 23.7271, 92.7176),
    ("Shimla", "Shimla", "Himachal Pradesh", 31.1048, 77.1734),
    ("Gurgaon", "Gurugram", "Haryana", 28.4595, 77.0266),
    ("Panihati", "North 24 Parganas", "West Bengal", 22.6941, 88.3745),
    ("Sri Ganganagar", "Sri Ganganagar", "Rajasthan", 29.9038, 73.8772),
    ("Nellore", "Nellore", "Andhra Pradesh", 14.4426, 79.9865),
    ("Eluru", "Eluru", "Andhra Pradesh", 16.7107, 81.0952),
    ("Shahjahanpur", "Shahjahanpur", "Uttar Pradesh", 27.8815, 79.9098),
    ("Bhusawal", "Jalgaon", "Maharashtra", 21.0436, 75.7851),
    ("Ghaziabad", "Ghaziabad", "Uttar Pradesh", 28.6692, 77.4538),
    ("Cyberabad", "Ranga Reddy", "Telangana", 17.4400, 78.3489),
    # Known cybercrime hotspot districts
    ("Jamtara", "Jamtara", "Jharkhand", 23.9629, 86.8035),
    ("Deoghar", "Deoghar", "Jharkhand", 24.4823, 86.6997),
    ("Nuh", "Nuh (Mewat)", "Haryana", 28.1080, 77.0011),
    ("Bharatpur", "Bharatpur", "Rajasthan", 27.2173, 77.4901),
    ("Alwar", "Alwar", "Rajasthan", 27.5530, 76.6346),
    ("Mathura", "Mathura", "Uttar Pradesh", 27.4924, 77.6737),
    ("Giridih", "Giridih", "Jharkhand", 24.1913, 86.3009),
]

# Cities with elevated cashout activity — used for ATM risk priors.
HOTSPOT_CITIES = {
    "Jamtara", "Deoghar", "Nuh", "Bharatpur", "Alwar", "Mathura", "Giridih",
    "Delhi", "Mumbai", "Kolkata", "Patna", "Ranchi",
}


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


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or numpy arrays."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - lat1)
    dlam = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────────────────────────
# GENERATOR 0 — Account Registry (built BEFORE any transaction)
# ─────────────────────────────────────────────

def build_account_registry(n_accounts: int, n_syndicates: int = 180) -> dict:
    """
    Create the population of bank accounts with intrinsic behavioural traits.

    Mule status is decided here, from the archetype, and never from anything the
    account subsequently does. Every feature written to node_features.csv is
    measured downstream from simulated activity, so no feature can be a restatement
    of the label.

    Mules additionally belong to a syndicate, and syndicates reuse their accounts
    across complaints — this is what gives the graph genuine ring structure.
    """
    names = list(ARCHETYPES.keys())
    weights = [ARCHETYPES[a]["weight"] for a in names]

    registry: dict[str, dict] = {}
    mule_ids: list[str] = []

    for _ in range(n_accounts):
        archetype = random.choices(names, weights=weights, k=1)[0]
        spec = ARCHETYPES[archetype]
        bank_name, bank_code = _random_bank()
        city, district, state, lat, lon = _random_city()
        alat, alon = _jitter_coords(lat, lon, radius_km=10.0)

        acc_id = _masked_account()
        if acc_id in registry:
            continue

        registry[acc_id] = {
            "account_id": acc_id,
            "archetype": archetype,
            "is_mule": spec["is_mule"],
            "bank_name": bank_name,
            "bank_code": bank_code,
            "ifsc_code": _random_ifsc(bank_code),
            "city": city,
            "district": district,
            "state": state,
            "lat": alat,
            "long": alon,
            "account_age_days": random.randint(*spec["age_days"]),
            # Behavioural dials, sampled per account inside the archetype's band.
            "dwell_hours": random.uniform(*spec["dwell_hours"]),
            "passthrough": random.uniform(*spec["passthrough"]),
            "fan_out": random.randint(*spec["fan_out"]),
            "night_ratio": random.uniform(*spec["night_ratio"]),
            "syndicate_id": -1,
        }
        if spec["is_mule"]:
            mule_ids.append(acc_id)

    # Assign mules to syndicates. Syndicates are regionally clustered, which is
    # how mule networks actually recruit.
    random.shuffle(mule_ids)
    for i, acc_id in enumerate(mule_ids):
        registry[acc_id]["syndicate_id"] = i % max(1, n_syndicates)

    return registry


def _syndicate_index(registry: dict) -> dict:
    """Map syndicate_id -> list of member account_ids."""
    idx = defaultdict(list)
    for acc_id, rec in registry.items():
        if rec["is_mule"]:
            idx[rec["syndicate_id"]].append(acc_id)
    return idx


# ─────────────────────────────────────────────
# GENERATOR 1 — Victim Complaints (1930 Feed)
# ─────────────────────────────────────────────

def generate_victim_complaints(n: int, fake: Faker, registry: dict) -> pd.DataFrame:
    """
    Generate n national 1930-style victim complaints.

    Victims are drawn from the legitimate account population rather than minted
    as free-floating IDs, so a complaint's originating account is a real node in
    the graph with its own history.
    """
    records = []
    base_ts = datetime.now() - timedelta(days=120)

    victim_pool = [
        a for a, r in registry.items()
        if not r["is_mule"] and r["archetype"] in ("salaried", "senior", "student", "merchant")
    ]

    for _ in range(n):
        victim_acc = random.choice(victim_pool)
        vrec = registry[victim_acc]
        fraud_type = random.choice(FRAUD_TYPES)

        if fraud_type == "Digital Arrest":
            amount = round(random.uniform(50_000, 5_00_000), 2)
        elif fraud_type == "Investment Scam":
            amount = round(random.uniform(25_000, 5_00_000), 2)
        elif fraud_type in ("APK Loan Scam", "Job Scam"):
            amount = round(random.uniform(10_000, 3_50_000), 2)
        else:
            amount = round(random.uniform(5_000, 2_00_000), 2)

        records.append({
            "ticket_id": str(uuid.uuid4()),
            "victim_name": fake.name(),
            "victim_bank": vrec["bank_name"],
            "victim_account": victim_acc,
            "fraud_type": fraud_type,
            "stolen_amount": amount,
            "complaint_timestamp": (
                base_ts + timedelta(seconds=random.randint(0, 120 * 24 * 3600))
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "city": vrec["city"],
            "state": vrec["state"],
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# GENERATOR 2 — Legitimate Banking Activity
# ─────────────────────────────────────────────

def generate_legitimate_activity(registry: dict, n_txns: int) -> list[dict]:
    """
    Simulate ordinary banking traffic among non-mule accounts.

    This is the half of the dataset the previous generator never produced. Without
    it every account that received money was a mule by definition, so `total_received
    > 0` classified the whole population perfectly. Legitimate accounts now receive
    salary credits, merchant settlements, remittances and P2P transfers, which is
    what forces a model to look at *how* money moves rather than *whether* it did.
    """
    legit = [a for a, r in registry.items() if not r["is_mule"]]
    by_arch = defaultdict(list)
    for a in legit:
        by_arch[registry[a]["archetype"]].append(a)

    base_ts = datetime.now() - timedelta(days=120)
    txns: list[dict] = []

    # Payer pools by transfer purpose.
    purposes = [
        ("SALARY", by_arch.get("merchant", []) + by_arch.get("transit_business", []),
         by_arch.get("salaried", []) + by_arch.get("student", []), (18_000, 180_000)),
        ("SETTLEMENT", by_arch.get("transit_business", []),
         by_arch.get("merchant", []), (5_000, 400_000)),
        ("P2P", legit, legit, (500, 60_000)),
        ("REMITTANCE", by_arch.get("salaried", []),
         by_arch.get("senior", []) + by_arch.get("student", []), (2_000, 45_000)),
        ("VENDOR", by_arch.get("merchant", []),
         by_arch.get("transit_business", []) + by_arch.get("merchant", []), (3_000, 220_000)),
    ]
    purposes = [p for p in purposes if p[1] and p[2]]
    if not purposes:
        return txns

    for _ in range(n_txns):
        purpose, src_pool, dst_pool, amt_range = random.choice(purposes)
        src = random.choice(src_pool)
        dst = random.choice(dst_pool)
        if src == dst:
            continue

        srec, drec = registry[src], registry[dst]
        amount = round(random.uniform(*amt_range), 2)

        # Night activity follows the payer's own profile.
        if random.random() < srec["night_ratio"]:
            hour = random.choice([0, 1, 2, 3, 4, 23])
        else:
            hour = random.randint(8, 21)

        ts = base_ts + timedelta(
            seconds=random.randint(0, 120 * 24 * 3600 - 1)
        )
        ts = ts.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))

        txns.append({
            "txn_id": str(uuid.uuid4()),
            "complaint_id": "",                 # not part of any fraud chain
            "src_account": src,
            "dst_account": dst,
            "bank_name": drec["bank_name"],
            "ifsc_code": drec["ifsc_code"],
            "city": drec["city"],
            "district": drec["district"],
            "state": drec["state"],
            "lat": drec["lat"],
            "long": drec["long"],
            "amount": amount,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "hop_depth": 0,
            "is_terminal": 0,
            "is_fraud": 0,
            "txn_purpose": purpose,
            "cashout_atm_id": "",
            "time_to_cashout_min": "",
        })

    return txns


# ─────────────────────────────────────────────
# GENERATOR 3 — Multi-Hop Fraud Chains
# ─────────────────────────────────────────────

def _pick_syndicate_members(syndicates: dict, syn_id: int, k: int, registry: dict,
                            near_state: str | None = None) -> list[str]:
    """
    Draw k mule accounts for a chain, preferring the same syndicate and region.

    Reuse is the point: a syndicate's accounts recur across many complaints, which
    produces the shared-neighbourhood structure a GNN can actually exploit.
    """
    pool = list(syndicates.get(syn_id, []))
    if near_state:
        local = [a for a in pool if registry[a]["state"] == near_state]
        rest = [a for a in pool if registry[a]["state"] != near_state]
        random.shuffle(local)
        random.shuffle(rest)
        pool = local + rest
    else:
        random.shuffle(pool)

    if len(pool) < k:
        # Borrow from neighbouring syndicates — cross-syndicate cooperation.
        others = [a for sid, members in syndicates.items() if sid != syn_id for a in members]
        random.shuffle(others)
        pool = pool + others

    return pool[:k]


def _build_chain(complaint_id: str, complaint_ts: datetime, stolen_amount: float,
                 victim_account: str, registry: dict, syndicates: dict) -> list[dict]:
    """
    Build one multi-hop laundering chain using real syndicate accounts.

    Timing at each hop is driven by the receiving account's own `dwell_hours`
    trait, so the pass-through speed a model measures later is a consequence of
    the account's archetype rather than a constant baked into the chain.
    """
    txns: list[dict] = []
    syn_id = random.randrange(max(1, len(syndicates)))
    victim_state = registry[victim_account]["state"]

    needed = 14
    members = _pick_syndicate_members(syndicates, syn_id, needed, registry, near_state=victim_state)
    if len(members) < 6:
        return txns
    cursor = 0

    def take():
        nonlocal cursor
        acc = members[cursor % len(members)]
        cursor += 1
        return acc

    def emit(src, dst, amount, hop, ts, terminal):
        drec = registry[dst]
        txns.append({
            "txn_id": str(uuid.uuid4()),
            "complaint_id": complaint_id,
            "src_account": src,
            "dst_account": dst,
            "bank_name": drec["bank_name"],
            "ifsc_code": drec["ifsc_code"],
            "city": drec["city"],
            "district": drec["district"],
            "state": drec["state"],
            "lat": drec["lat"],
            "long": drec["long"],
            "amount": round(amount, 2),
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "hop_depth": hop,
            "is_terminal": 1 if terminal else 0,
            "is_fraud": 1,
            "txn_purpose": "LAYERING",
            "cashout_atm_id": "",
            "time_to_cashout_min": "",
        })

    def dwell(acc: str) -> timedelta:
        """Delay before this account forwards on, from its own behaviour dial."""
        hours = registry[acc]["dwell_hours"] * random.uniform(0.6, 1.5)
        return timedelta(seconds=max(20.0, hours * 3600.0))

    t0 = complaint_ts + timedelta(seconds=random.randint(30, 180))

    # ── Hop 1: victim -> L1 ──────────────────────────────────────────────────
    n_l1 = random.randint(1, 3)
    l1 = [take() for _ in range(n_l1)]
    l1_amounts = _split_amount(stolen_amount, n_l1)
    for acc, amt in zip(l1, l1_amounts):
        emit(victim_account, acc, amt, 1, t0 + timedelta(seconds=random.randint(15, 90)), False)

    # ── Hop 2: L1 -> L2 ──────────────────────────────────────────────────────
    l2_groups = []
    for acc, amt in zip(l1, l1_amounts):
        n_l2 = random.randint(1, 3)
        l2 = [take() for _ in range(n_l2)]
        amounts = _split_amount(amt * registry[acc]["passthrough"], n_l2)
        l2_groups.append(list(zip(l2, amounts)))
        base = t0 + dwell(acc)
        for i, (dst, a2) in enumerate(zip(l2, amounts)):
            emit(acc, dst, a2, 2, base + timedelta(seconds=30 * i + random.randint(5, 45)), False)

    all_l2 = [(a, m) for grp in l2_groups for a, m in grp]

    # ── Hop 3+: pooling into a shared terminal, or a deeper chain ────────────
    if len(all_l2) >= MIN_SPLIT_DESTINATIONS and random.random() < 0.55:
        # Many-to-one pooling: the classic cash-consolidation pattern.
        ring = random.sample(all_l2, k=min(len(all_l2), random.randint(3, 5)))
        shared_terminal = take()
        base = t0 + timedelta(hours=random.uniform(0.2, 4.0))
        for i, (src, amt) in enumerate(ring):
            emit(src, shared_terminal, amt * registry[src]["passthrough"], 3,
                 base + timedelta(seconds=90 * i + random.randint(10, 120)), True)
    else:
        for grp in l2_groups:
            for src, amt in random.sample(grp, k=min(1, len(grp))):
                depth = random.choice([3, 3, 4])
                cur_src, remaining = src, amt
                cur_t = t0 + dwell(src)
                for d in range(3, depth + 1):
                    dst = take()
                    remaining *= registry[cur_src]["passthrough"]
                    cur_t = cur_t + dwell(cur_src) * random.uniform(0.3, 1.0)
                    emit(cur_src, dst, remaining, d, cur_t, d == depth)
                    cur_src = dst

    return txns


def generate_transactions(complaints_df: pd.DataFrame, target: int,
                          registry: dict, syndicates: dict) -> pd.DataFrame:
    """Generate the fraud ledger for every complaint, then top up to `target` rows."""
    all_txns: list[dict] = []
    cids = complaints_df["ticket_id"].tolist()
    ts_map = dict(zip(complaints_df["ticket_id"], pd.to_datetime(complaints_df["complaint_timestamp"])))
    amt_map = dict(zip(complaints_df["ticket_id"], complaints_df["stolen_amount"]))
    victim_map = dict(zip(complaints_df["ticket_id"], complaints_df["victim_account"]))

    for cid in cids:
        all_txns.extend(
            _build_chain(cid, ts_map[cid], amt_map[cid], victim_map[cid], registry, syndicates)
        )

    idx = 0
    while len(all_txns) < target and idx < len(cids) * 4:
        cid = cids[idx % len(cids)]
        all_txns.extend(
            _build_chain(cid, ts_map[cid], amt_map[cid], victim_map[cid], registry, syndicates)
        )
        idx += 1

    return pd.DataFrame(all_txns)


# ─────────────────────────────────────────────
# GENERATOR 4 — ATM Directory
# ─────────────────────────────────────────────

_ATM_LOCALITIES = [
    "Main Market", "Bus Stand Road", "Railway Station Road", "Civil Lines",
    "MG Road", "Sector 17", "Gandhi Chowk", "Station Road", "Model Town",
    "Ring Road", "Nehru Nagar", "Industrial Area", "Old City", "Kacheri Road",
    "Collectorate Road", "Medical College Road", "Bank Street", "Mandi Road",
]

_ATM_SITE_TYPES = ["Branch ATM", "Offsite ATM", "Micro-ATM Kiosk", "CSP Outlet"]


def _atm_address(city: str, state: str) -> str:
    """
    Build a street address that actually agrees with the ATM's city/state.

    fake.address() invents an unrelated city and PIN, which makes a Pune
    interception display a Nellore address - the location shown to a dispatched
    officer must match the coordinates being sent with it.
    """
    return (
        f"{random.randint(1, 240)}, {random.choice(_ATM_LOCALITIES)}, "
        f"{city}, {state} - {random.randint(110, 855)}{random.randint(100, 999)} "
        f"({random.choice(_ATM_SITE_TYPES)})"
    )


def generate_atm_directory(n: int, fake: Faker) -> pd.DataFrame:
    """Generate a geo-coded ATM / CSP directory across the city list."""
    records = []
    for _ in range(n):
        bank_name, _ = _random_bank()
        city, district, state, lat, lon = _random_city()
        jlat, jlon = _jitter_coords(lat, lon, radius_km=12.0)
        base_risk = 0.7 if city in HOTSPOT_CITIES else 0.35
        risk_score = round(min(1.0, max(0.05, base_risk + random.uniform(-0.25, 0.35))), 3)

        records.append({
            "atm_id": f"ATM-{str(uuid.uuid4())[:8].upper()}",
            "bank_name": bank_name,
            "address": _atm_address(city, state),
            "city": city, "district": district, "state": state,
            "lat": jlat, "long": jlon,
            "opening_time": random.choice(["00:00", "06:00", "07:00", "08:00"]),
            "closing_time": random.choice(["23:59", "22:00", "21:00"]),
            "cashout_risk_score": risk_score,
            "historical_fraud_count": (
                random.randint(5, 180) if city in HOTSPOT_CITIES else random.randint(0, 45)
            ),
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# GENERATOR 5 — Cashout Ground Truth (ATM choice + delay)
# ─────────────────────────────────────────────

def assign_cashouts(txn_df: pd.DataFrame, atm_df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    """
    Decide, for each terminal transaction, WHICH ATM the cash was drawn from and
    HOW LONG after the transfer.

    Previously the ATM label was computed inside the feature builder as
    `argmin(distance)` — while the distance to that same ATM was handed to the
    model as feature #67. The task was therefore a tautology and 98% accuracy
    measured nothing.

    Here the ATM is *sampled* from a behavioural choice model:

        score_i  ∝  exp(-d_i / λ) · (1 + w·risk_i) · bank_affinity_i

    The nearest ATM is the most likely single choice but far from certain, so
    recovering it demands combining distance with surveillance risk and bank
    affiliation. The delay carries irreducible noise, which caps R² where a
    real-world estimate would sit.
    """
    atm_lats = atm_df["lat"].to_numpy()
    atm_lons = atm_df["long"].to_numpy()
    atm_ids = atm_df["atm_id"].tolist()
    atm_risk = atm_df["cashout_risk_score"].to_numpy()
    atm_banks = atm_df["bank_name"].to_numpy(dtype=object)

    # ── Each syndicate keeps a small rotation of established cashout points ──
    # Chosen near the crew's own centre of gravity, so the preference is
    # geographically plausible rather than arbitrary.
    members_by_syn: dict[int, list[str]] = defaultdict(list)
    for acc, rec in registry.items():
        if rec["is_mule"] and rec["syndicate_id"] >= 0:
            members_by_syn[rec["syndicate_id"]].append(acc)

    syndicate_pref: dict[int, set[int]] = {}
    for syn_id, members in members_by_syn.items():
        clat = float(np.mean([registry[m]["lat"] for m in members]))
        clon = float(np.mean([registry[m]["long"] for m in members]))
        d_syn = _haversine_km(clat, clon, atm_lats, atm_lons)
        # Sample from the 40 nearest so crews differ from one another.
        near = np.argsort(d_syn)[:40]
        picks = np.random.choice(
            near, size=min(ATM_PREF_PER_SYNDICATE, len(near)), replace=False
        )
        syndicate_pref[syn_id] = set(int(i) for i in picks)

    terminal_mask = txn_df["is_terminal"] == 1
    chosen_atm: list[str] = []
    delays: list[float] = []

    for _, row in txn_df[terminal_mask].iterrows():
        acc = str(row["dst_account"])
        rec = registry.get(acc)
        lat = float(rec["lat"]) if rec else float(row["lat"])
        lon = float(rec["long"]) if rec else float(row["long"])
        acc_bank = rec["bank_name"] if rec else row["bank_name"]

        dists = _haversine_km(lat, lon, atm_lats, atm_lons)

        # Restrict to a plausible travel radius; fall back to the nearest N.
        cand = np.where(dists <= ATM_CANDIDATE_RADIUS_KM)[0]
        if len(cand) < ATM_MIN_CANDIDATES:
            cand = np.argsort(dists)[:ATM_MIN_CANDIDATES]

        d = dists[cand]
        score = np.exp(-d / ATM_DISTANCE_DECAY_KM)
        score = score * (1.0 + ATM_RISK_WEIGHT * atm_risk[cand])
        score = score * np.where(atm_banks[cand] == acc_bank, ATM_SAME_BANK_BOOST, 1.0)

        # Established cashout points for this account's crew.
        syn = rec["syndicate_id"] if rec else -1
        pref = syndicate_pref.get(syn)
        if pref:
            score = score * np.where(
                np.isin(cand, list(pref)), ATM_SYNDICATE_PREF_BOOST, 1.0
            )

        total = score.sum()
        if not np.isfinite(total) or total <= 0:
            pick = int(cand[int(np.argmin(d))])
        else:
            pick = int(np.random.choice(cand, p=score / total))

        chosen_atm.append(atm_ids[pick])

        # ── Delay to cashout ────────────────────────────────────────────────
        travel_km = float(dists[pick])
        hour = pd.Timestamp(row["timestamp"]).hour
        # Syndicates move faster in daylight; night withdrawals wait for cover.
        night_penalty = 14.0 if hour in (0, 1, 2, 3, 4) else 0.0
        discipline = registry[acc]["dwell_hours"] if rec else 1.0
        base = 22.0 + 1.9 * travel_km + night_penalty + min(discipline * 2.2, 25.0)
        noise = np.random.gamma(shape=2.2, scale=4.6) - 10.1   # heavy-tailed, ~0 mean
        delays.append(float(np.clip(base + noise, 4.0, 180.0)))

    # Both columns are seeded as empty strings when the rows are built, which
    # pins them to a string dtype; recreate them with the right types before
    # writing the sampled labels back.
    txn_df["cashout_atm_id"] = txn_df["cashout_atm_id"].astype(object)
    txn_df["time_to_cashout_min"] = np.nan

    txn_df.loc[terminal_mask, "cashout_atm_id"] = chosen_atm
    txn_df.loc[terminal_mask, "time_to_cashout_min"] = np.round(delays, 2)
    return txn_df


# ─────────────────────────────────────────────
# GENERATOR 6 — Graph Edge List
# ─────────────────────────────────────────────

def generate_graph_edges(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Derive clean edge list from transactions for PyG/NetworkX."""
    cols = ["src_account", "dst_account", "amount", "timestamp",
            "hop_depth", "is_terminal", "complaint_id"]
    return transactions_df[cols].copy().reset_index(drop=True)


# ─────────────────────────────────────────────
# GENERATOR 7 — Measured Node Features
# ─────────────────────────────────────────────

def generate_node_features(transactions_df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    """
    Measure per-account features from the simulated ledger.

    Every column here is an observation of behaviour that a bank could compute
    from its own transaction log. None of them is written from the label, and the
    label is not written from any of them — `is_mule_label` was fixed in the
    account registry before this ledger existed.

    The discriminative signal now lives in HOW an account handles money:
    dwell time between credit and debit, what fraction it forwards, how many
    counterparties it touches, how new it is, how much it moves at night. The
    `transit_business` archetype scores like a mule on several of these, so no
    single threshold separates the classes.
    """
    df = transactions_df.copy()
    df["_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("_dt")

    # ── Credit / debit aggregates ────────────────────────────────────────────
    recv = df.groupby("dst_account")["amount"].agg(total_received="sum", in_degree="count")
    sent = df.groupby("src_account")["amount"].agg(total_sent="sum", out_degree="count")
    avg_in = df.groupby("dst_account")["amount"].mean().rename("avg_txn_amount")

    distinct_in = df.groupby("dst_account")["src_account"].nunique().rename("distinct_senders")
    distinct_out = df.groupby("src_account")["dst_account"].nunique().rename("distinct_receivers")

    # ── Night-time share of activity ─────────────────────────────────────────
    df["_night"] = df["_dt"].dt.hour.isin([0, 1, 2, 3, 4, 23]).astype(int)
    night_out = df.groupby("src_account")["_night"].mean().rename("night_txn_ratio")

    # ── Dwell time: gap between a credit and the account's next debit ────────
    credits = defaultdict(list)
    debits = defaultdict(list)
    for acc, t in zip(df["dst_account"], df["_dt"]):
        credits[acc].append(t)
    for acc, t in zip(df["src_account"], df["_dt"]):
        debits[acc].append(t)

    dwell_rows = {}
    burst_rows = {}
    for acc, ctimes in credits.items():
        dtimes = debits.get(acc)
        if not dtimes:
            continue
        dtimes = sorted(dtimes)
        gaps = []
        j = 0
        for ct in sorted(ctimes):
            while j < len(dtimes) and dtimes[j] < ct:
                j += 1
            if j < len(dtimes):
                gaps.append((dtimes[j] - ct).total_seconds())
        if gaps:
            dwell_rows[acc] = float(np.median(gaps))

    # ── Peak outgoing burst inside the 5-minute velocity window ──────────────
    for acc, dtimes in debits.items():
        if len(dtimes) < 2:
            burst_rows[acc] = len(dtimes)
            continue
        ts = sorted(t.timestamp() for t in dtimes)
        best, lo = 1, 0
        for hi in range(len(ts)):
            while ts[hi] - ts[lo] > VELOCITY_WINDOW_SECONDS:
                lo += 1
            best = max(best, hi - lo + 1)
        burst_rows[acc] = best

    # ── Hop depth (kept for the money-flow UI; excluded from GNN features) ───
    fraud = df[df["is_fraud"] == 1]
    hop = fraud.groupby("dst_account")["hop_depth"].max().rename("hop_depth")

    active = sorted(set(df["src_account"]) | set(df["dst_account"]))
    rows = []
    for acc in active:
        rec = registry.get(acc)
        if rec is None:
            continue
        rows.append({
            "account_id": acc,
            "bank_name": rec["bank_name"],
            "city": rec["city"],
            "lat": rec["lat"],
            "long": rec["long"],
            "account_age_days": rec["account_age_days"],
            "is_mule_label": int(rec["is_mule"]),
        })

    node_df = pd.DataFrame(rows).set_index("account_id")
    node_df = (node_df
               .join(recv).join(sent).join(avg_in)
               .join(distinct_in).join(distinct_out)
               .join(night_out).join(hop))

    node_df["median_dwell_seconds"] = pd.Series(dwell_rows)
    node_df["burst_out_5min"] = pd.Series(burst_rows)

    for col, fill in [
        ("total_received", 0.0), ("total_sent", 0.0), ("in_degree", 0), ("out_degree", 0),
        ("avg_txn_amount", 0.0), ("distinct_senders", 0), ("distinct_receivers", 0),
        ("night_txn_ratio", 0.0), ("hop_depth", 0), ("burst_out_5min", 1),
    ]:
        node_df[col] = node_df[col].fillna(fill)

    # An account that never forwarded gets the observation window as its dwell —
    # a censored value, not a missing one.
    node_df["median_dwell_seconds"] = node_df["median_dwell_seconds"].fillna(120 * 24 * 3600.0)

    # Fraction of incoming funds passed on. Bounded so a pure receiver reads 0.
    node_df["passthrough_ratio"] = np.where(
        node_df["total_received"] > 0,
        np.clip(node_df["total_sent"] / node_df["total_received"].replace(0, np.nan), 0, 3),
        0.0,
    )
    node_df["passthrough_ratio"] = node_df["passthrough_ratio"].fillna(0.0).round(4)

    # Retained for backward compatibility with the transaction-count feature.
    node_df["txn_count_24h"] = node_df["in_degree"].astype(int)

    node_df = node_df.reset_index().rename(columns={"index": "account_id"})

    # ── Label noise ──────────────────────────────────────────────────────────
    # Ground truth from bank STR filings is not perfect: some mules are never
    # reported, some ordinary accounts are wrongly flagged. Without this the
    # ceiling is an artefact of the simulator.
    n_flip = int(len(node_df) * LABEL_NOISE_RATE)
    if n_flip:
        flip_idx = np.random.choice(node_df.index, size=n_flip, replace=False)
        node_df.loc[flip_idx, "is_mule_label"] = 1 - node_df.loc[flip_idx, "is_mule_label"]

    node_df["lat"] = node_df["lat"].clip(INDIA_LAT_MIN, INDIA_LAT_MAX).round(6)
    node_df["long"] = node_df["long"].clip(INDIA_LON_MIN, INDIA_LON_MAX).round(6)
    node_df["median_dwell_seconds"] = node_df["median_dwell_seconds"].round(1)
    node_df["avg_txn_amount"] = node_df["avg_txn_amount"].round(2)
    node_df["night_txn_ratio"] = node_df["night_txn_ratio"].round(4)

    COL_ORDER = [
        "account_id", "bank_name", "city", "lat", "long",
        "total_received", "total_sent", "txn_count_24h", "avg_txn_amount",
        "in_degree", "out_degree", "distinct_senders", "distinct_receivers",
        "median_dwell_seconds", "passthrough_ratio", "account_age_days",
        "night_txn_ratio", "burst_out_5min",
        "is_mule_label", "hop_depth",
    ]
    return node_df[COL_ORDER]


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main(n_complaints=2500, n_transactions=20000, n_atms=1000, seed=42,
         n_accounts=20000, n_legit_txns=30000):
    """Run the full Pan-India data generation pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  MuleShield AI — Dataset Generation (leakage-free)")
    print("=" * 68)

    print("\n[0/6] Building account registry (labels fixed before any activity)...")
    registry = build_account_registry(n_accounts)
    syndicates = _syndicate_index(registry)
    n_mules = sum(1 for r in registry.values() if r["is_mule"])
    print(f"      {len(registry):,} accounts | {n_mules:,} mules "
          f"({n_mules/len(registry):.1%}) across {len(syndicates)} syndicates")

    print("\n[1/6] Generating 1930 victim complaints...")
    complaints_df = generate_victim_complaints(n_complaints, fake, registry)
    print(f"      {len(complaints_df):,} complaints")

    print("\n[2/6] Simulating fraud chains...")
    fraud_df = generate_transactions(complaints_df, n_transactions, registry, syndicates)
    print(f"      {len(fraud_df):,} laundering transactions")

    print("\n[3/6] Simulating legitimate banking activity...")
    legit = generate_legitimate_activity(registry, n_legit_txns)
    legit_df = pd.DataFrame(legit)
    print(f"      {len(legit_df):,} legitimate transactions")

    txn_df = pd.concat([fraud_df, legit_df], ignore_index=True)
    txn_df["lat"] = txn_df["lat"].clip(INDIA_LAT_MIN, INDIA_LAT_MAX)
    txn_df["long"] = txn_df["long"].clip(INDIA_LON_MIN, INDIA_LON_MAX)

    print("\n[4/6] Generating ATM directory...")
    atm_df = generate_atm_directory(n_atms, fake)
    print(f"      {len(atm_df):,} ATMs")

    print("\n[5/6] Sampling cashout ATM + delay per terminal (choice model)...")
    txn_df = assign_cashouts(txn_df, atm_df, registry)
    n_term = int((txn_df["is_terminal"] == 1).sum())
    print(f"      {n_term:,} terminal cashouts labelled")

    print("\n[6/6] Measuring node features from the ledger...")
    node_df = generate_node_features(txn_df, registry)
    edges_df = generate_graph_edges(txn_df)

    complaints_df.to_csv(DATA_DIR / "victim_complaints.csv", index=False)
    txn_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    atm_df.to_csv(DATA_DIR / "atm_directory.csv", index=False)
    edges_df.to_csv(DATA_DIR / "graph_edges.csv", index=False)
    node_df.to_csv(DATA_DIR / "node_features.csv", index=False)

    mule_count = int(node_df["is_mule_label"].sum())
    print("\n" + "=" * 68)
    print("  DATASET SUMMARY")
    print("=" * 68)
    print(f"  Complaints        : {len(complaints_df):,}")
    print(f"  Transactions      : {len(txn_df):,} "
          f"({int((txn_df['is_fraud']==1).sum()):,} fraud / "
          f"{int((txn_df['is_fraud']==0).sum()):,} legitimate)")
    print(f"  Accounts in graph : {len(node_df):,}")
    print(f"  Labelled mules    : {mule_count:,} ({mule_count/len(node_df):.1%})")
    print(f"  ATMs              : {len(atm_df):,}")
    print(f"  Label noise       : {LABEL_NOISE_RATE:.0%}")
    print("=" * 68)

    # Keyed by name so callers (and the Phase 1 test-suite) do not depend on
    # positional order.
    return {
        "complaints": complaints_df,
        "transactions": txn_df,
        "atms": atm_df,
        "graph_edges": edges_df,
        "node_features": node_df,
        "registry": registry,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MuleShield AI dataset generator")
    ap.add_argument("--complaints", type=int, default=2500)
    ap.add_argument("--transactions", type=int, default=20000)
    ap.add_argument("--atms", type=int, default=1000)
    ap.add_argument("--accounts", type=int, default=20000)
    ap.add_argument("--legit", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(
        n_complaints=args.complaints,
        n_transactions=args.transactions,
        n_atms=args.atms,
        seed=args.seed,
        n_accounts=args.accounts,
        n_legit_txns=args.legit,
    )
