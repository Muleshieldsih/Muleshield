# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 2a: Node Embedding Generator
SIH26184 | MHA / I4C

Loads the trained GraphSAGE model and generates 64-dim embeddings
for all accounts in the graph. Saves them to:
    embeddings/node_embeddings.pkl

Also provides get_embeddings_for_complaint() for real-time inference
(<2 seconds per new complaint graph).

Usage:
    python engine/embed.py                    # generate all embeddings
    python engine/embed.py --complaint <id>   # embed a single complaint subgraph
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from gnn_model import GraphSAGEMule, FEATURE_COLS, IN_CHANNELS
from train_gnn import load_pyg_data

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EMBEDDINGS_DIR = ROOT / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "graphsage_mule.pt"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "node_embeddings.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_model(
    model_path: Path = MODEL_PATH,
) -> tuple[GraphSAGEMule, StandardScaler, list[str]]:
    """
    Load the saved GraphSAGE checkpoint.

    Returns:
        (model, scaler, account_ids)
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = GraphSAGEMule()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_  = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])

    account_ids = checkpoint["account_ids"]
    return model, scaler, account_ids


# ─────────────────────────────────────────────────────────────────────────────
# FULL GRAPH EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_embeddings(
    model: GraphSAGEMule,
    scaler: StandardScaler,
    account_ids: list[str],
    transactions_path: Path = DATA_DIR / "transactions.csv",
    node_features_path: Path = DATA_DIR / "node_features.csv",
) -> dict[str, np.ndarray]:
    """
    Generate 64-dim embeddings for every account in the full graph.

    Returns:
        dict: {account_id: np.ndarray of shape (64,)}
    """
    data, returned_ids, _, _ = load_pyg_data(
        transactions_path=transactions_path,
        node_features_path=node_features_path,
    )

    # Re-scale using the saved scaler (not the one from load_pyg_data)
    x_scaled = scaler.transform(
        data.x.numpy()  # already float32, just re-normalize with saved params
    ).astype(np.float32)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float)

    embeddings = model.get_embeddings(x_tensor, data.edge_index)  # [N, 64]
    emb_np = embeddings.numpy()

    return {acc: emb_np[i] for i, acc in enumerate(returned_ids)}


def save_embeddings(embeddings_dict: dict[str, np.ndarray], path: Path = EMBEDDINGS_PATH):
    """Serialize embeddings dict to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    """Load embeddings from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE COMPLAINT EMBEDDINGS (real-time inference)
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings_for_complaint(
    complaint_id: str,
    model: GraphSAGEMule,
    scaler: StandardScaler,
    transactions_df: pd.DataFrame,
    node_features_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Generate 64-dim embeddings for all accounts in a single complaint subgraph.
    Designed to run in <2 seconds on a small subgraph.

    Args:
        complaint_id:    Complaint ticket ID (from victim_complaints.csv)
        model:           Trained GraphSAGEMule
        scaler:          Fitted StandardScaler from training
        transactions_df: Full transactions DataFrame (filtered internally)
        node_features_df: Full node_features DataFrame

    Returns:
        dict: {account_id: np.ndarray (64,)}
    """
    t0 = time.time()

    # Filter to this complaint's transactions only
    sub_txn = transactions_df[transactions_df["complaint_id"] == complaint_id].copy()
    if sub_txn.empty:
        return {}

    # Get all unique accounts in this subgraph
    all_accounts = pd.concat([
        sub_txn["src_account"], sub_txn["dst_account"]
    ]).unique().tolist()
    account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
    N = len(all_accounts)

    # Build feature matrix for subgraph nodes
    node_lookup = node_features_df.set_index("account_id")
    x_list = []
    for acc in all_accounts:
        if acc in node_lookup.index:
            row = node_lookup.loc[acc]
            feats = [float(row.get(col, 0.0)) for col in FEATURE_COLS]
        else:
            feats = [0.0] * IN_CHANNELS
        x_list.append(feats)

    x_raw = np.array(x_list, dtype=np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float)

    # Build subgraph edge_index
    src_idx = [account_to_idx[s] for s in sub_txn["src_account"]]
    dst_idx = [account_to_idx[d] for d in sub_txn["dst_account"]]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    # Generate embeddings
    embeddings = model.get_embeddings(x_tensor, edge_index)  # [N, 64]
    emb_np = embeddings.numpy()

    elapsed = (time.time() - t0) * 1000
    print(f"  Complaint {complaint_id[:8]}... embedded {N} nodes in {elapsed:.1f}ms")

    return {acc: emb_np[i] for i, acc in enumerate(all_accounts)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(complaint_id: str = None):
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("        Run: python engine/train_gnn.py first")
        return

    print("=" * 60)
    print("  MuleShield AI - Embedding Generator  |  SIH26184")
    print("=" * 60)

    print("\n[1/3] Loading trained model...")
    model, scaler, account_ids = load_trained_model()
    print(f"      [OK] Model loaded ({len(account_ids)} accounts in training graph)")

    if complaint_id:
        print(f"\n[2/3] Embedding complaint subgraph: {complaint_id}")
        txn_df = pd.read_csv(DATA_DIR / "transactions.csv")
        node_df = pd.read_csv(DATA_DIR / "node_features.csv")
        t0 = time.time()
        emb = get_embeddings_for_complaint(complaint_id, model, scaler, txn_df, node_df)
        elapsed = (time.time() - t0) * 1000
        print(f"      [OK] {len(emb)} node embeddings in {elapsed:.1f}ms")
        assert elapsed < 2000, f"Embedding took {elapsed:.1f}ms — exceeds 2s AC!"
        print("      [OK] AC verified: embedding time < 2000ms")
    else:
        print("\n[2/3] Generating embeddings for ALL accounts...")
        t0 = time.time()
        embeddings = generate_all_embeddings(model, scaler, account_ids)
        elapsed = (time.time() - t0) * 1000
        print(f"      [OK] {len(embeddings)} embeddings generated in {elapsed:.1f}ms")

        print(f"\n[3/3] Saving to {EMBEDDINGS_PATH}...")
        save_embeddings(embeddings)
        print(f"      [OK] Saved {len(embeddings)} x 64-dim embeddings")

        # Quick verification
        sample_key = next(iter(embeddings))
        sample_emb = embeddings[sample_key]
        assert sample_emb.shape == (64,), f"Wrong embedding dim: {sample_emb.shape}"
        print(f"      [OK] Embedding shape verified: {sample_emb.shape}")

    print("\n" + "=" * 60)
    print("  Embedding generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuleShield AI -- Generate Node Embeddings")
    parser.add_argument("--complaint", type=str, default=None,
                        help="Embed a single complaint subgraph (optional)")
    args = parser.parse_args()
    main(complaint_id=args.complaint)
