# -*- coding: utf-8 -*-
"""
Calibration transfer: training forward pass vs cached-embedding serving path.
SIH26184 | MHA / I4C

The console scores an account from its cached 64-dim embedding and the trained
classifier head, then maps that through the isotonic curve fitted during
training. That only means anything if the cached embedding is the same vector
the training forward pass produced.

For a while it was not. engine/embed.py re-applied the StandardScaler to a
`data.x` that load_pyg_data had already standardized, so every cached embedding
came from doubly-normalized inputs and deviated from the training pass by up to
167.9. The served [0.70, 0.95) band then held 3,988 held-out accounts at a 0.58%
true mule rate, against 89.9% for the same band computed correctly. Ranking
inside a traced chain still looked sensible, which is exactly why it survived:
nothing checks out as wrong until a band is compared against labels.

These tests compare the two paths directly and check the bands against ground
truth, so the displayed number has to keep meaning what it says.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

CKPT = ROOT / "models" / "graphsage_mule.pt"
EMB = ROOT / "embeddings" / "node_embeddings.pkl"

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def paths():
    if not CKPT.exists() or not EMB.exists():
        pytest.skip("run: python engine/train_gnn.py && python engine/embed.py")

    import pickle

    from gnn_model import build_model
    from train_gnn import load_pyg_data

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    data, ids, _, _ = load_pyg_data()
    model = build_model(in_channels=data.x.shape[1])
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        fresh = model.get_embeddings(data.x, data.edge_index).numpy()
        p_fwd = torch.sigmoid(model(data.x, data.edge_index).squeeze(-1)).numpy()

    with open(EMB, "rb") as fh:
        cached_map = pickle.load(fh)

    keep = [i for i, a in enumerate(ids) if a in cached_map]
    cached = np.array([cached_map[ids[i]] for i in keep])

    w = sd["classifier.weight"].numpy().reshape(-1)
    b = float(sd["classifier.bias"].numpy().reshape(-1)[0])
    p_cached = 1.0 / (1.0 + np.exp(-(cached @ w + b)))

    cal = ckpt.get("calibration")
    return {
        "fresh": fresh[keep],
        "cached": cached,
        "p_fwd": p_fwd[keep],
        "p_cached": p_cached,
        "y": data.y.numpy()[keep],
        "test_mask": np.isin(np.array(keep), np.array(ckpt["split"]["test_idx"])),
        "cal": (np.array(cal["x"]), np.array(cal["y"])) if cal else None,
    }


class TestEmbeddingPathAgreement:
    """The cached vector must BE the training vector, not merely resemble it."""

    def test_cached_embeddings_match_the_training_forward_pass(self, paths):
        dev = np.abs(paths["cached"] - paths["fresh"]).max()
        assert dev < 1e-4, (
            f"cached embeddings deviate from the training pass by {dev:.4g}. "
            "Check that engine/embed.py is not re-scaling an already-scaled data.x."
        )

    def test_probabilities_agree_between_paths(self, paths):
        dev = np.abs(paths["p_cached"] - paths["p_fwd"]).max()
        assert dev < 1e-5, f"served probability differs from training by {dev:.4g}"

    def test_every_graph_account_has_an_embedding(self, paths):
        assert len(paths["cached"]) == len(paths["fresh"])


class TestCalibrationOnHeldOut:
    """
    A band that says 70-95% must contain accounts that are mules 70-95% of the
    time. Checked on the checkpoint's own test split, which the isotonic curve
    was never fitted on.
    """

    def _calibrated(self, paths, p):
        if paths["cal"] is None:
            pytest.skip("checkpoint carries no calibration curve")
        return np.interp(p, paths["cal"][0], paths["cal"][1])

    def test_high_band_is_mostly_mules(self, paths):
        m = paths["test_mask"]
        pc = self._calibrated(paths, paths["p_cached"])[m]
        y = paths["y"][m]
        band = (pc >= 0.70) & (pc < 0.95)
        if band.sum() < 20:
            pytest.skip("too few held-out accounts in the band to assert a rate")
        rate = y[band].mean()
        assert rate >= 0.60, (
            f"[0.70,0.95) holds {band.sum()} held-out accounts at a {rate:.2%} "
            "true mule rate — the served scores are not calibrated probabilities."
        )

    def test_low_band_is_mostly_clean(self, paths):
        m = paths["test_mask"]
        pc = self._calibrated(paths, paths["p_cached"])[m]
        y = paths["y"][m]
        band = pc < 0.30
        assert band.sum() > 0
        assert y[band].mean() <= 0.05

    def test_calibration_is_monotonic(self, paths):
        if paths["cal"] is None:
            pytest.skip("no calibration curve")
        ys = paths["cal"][1]
        assert np.all(np.diff(ys) >= -1e-9), "isotonic output must be non-decreasing"

    def test_bands_agree_across_paths(self, paths):
        """The console and the training report must place accounts identically."""
        m = paths["test_mask"]
        a = self._calibrated(paths, paths["p_cached"])[m]
        b = self._calibrated(paths, paths["p_fwd"])[m]
        assert np.abs(a - b).max() < 1e-5
