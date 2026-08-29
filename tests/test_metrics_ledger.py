# -*- coding: utf-8 -*-
"""
MuleShield AI -- Metrics Ledger Test Suite
SIH26184 | MHA / I4C

Guards the bug class that put three different numbers for one quantity into the
repository: a measured figure retyped by hand into a place that no retrain
touches. The card said 0.8501 for the best non-graph baseline, the console said
0.8463, and the measured value was 0.8423.

The rule these tests enforce: training and evaluation scripts WRITE figures,
everything else READS them.

Run:
    python -m pytest tests/test_metrics_ledger.py -v
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from metrics_io import METRICS_PATH, FRONTEND_STATS_PATH, read_metrics  # noqa: E402

MATRIX_SCRIPT = ROOT / "scripts" / "generate_model_matrix.py"
CONSTANTS_JS = ROOT / "frontend" / "src" / "utils" / "constants.js"

REQUIRED = {
    "detection": ["test_f1", "threshold"],
    "detection_baselines": ["best_non_graph", "best_non_graph_model"],
    "location": ["zone_containment", "zone_containment_baseline_nearest3",
                 "zone_median_atms", "zone_median_radius_km", "n_atms",
                 "time_mae_minutes", "time_baseline_mae"],
}


@pytest.fixture(scope="module")
def ledger():
    if not METRICS_PATH.exists():
        pytest.skip("data/metrics.json not built; run scripts/export_metrics.py")
    return read_metrics()


# ── The ledger itself ────────────────────────────────────────────────────────

@pytest.mark.parametrize("section,keys", REQUIRED.items())
def test_ledger_section_is_populated(ledger, section, keys):
    """Every figure the card and console publish has been measured."""
    assert section in ledger, f"missing section '{section}'"
    for k in keys:
        assert k in ledger[section], f"missing '{section}.{k}'"


def test_every_section_records_how_to_regenerate_it(ledger):
    """A stale figure must be able to name the command that refreshes it."""
    for name, sec in ledger.items():
        if name.startswith("_"):
            continue
        assert sec.get("source"), f"section '{name}' has no 'source' command"
        assert sec.get("measured_utc"), f"section '{name}' has no timestamp"


def test_best_non_graph_is_a_real_max(ledger):
    """
    The published baseline is the maximum over the non-graph models.

    It used to be read off as bars[-2] — always the random forest — while being
    labelled "best non-graph". That holds only while the forest happens to lead.
    """
    bl = ledger["detection_baselines"]
    per_model = bl["per_model_f1"]
    assert bl["best_non_graph"] == pytest.approx(max(per_model.values()))
    assert per_model[bl["best_non_graph_model"]] == pytest.approx(bl["best_non_graph"])


def test_gnn_still_beats_the_best_non_graph_baseline(ledger):
    """The headline claim, checked against measured numbers rather than prose."""
    gnn = ledger["detection"]["test_f1"]
    best = ledger["detection_baselines"]["best_non_graph"]
    assert gnn > best, f"GNN F1 {gnn:.4f} does not beat baseline {best:.4f}"


def test_zone_beats_its_naive_baselines(ledger):
    """Containment is only a claim if it clears the naive zones at equal radius."""
    loc = ledger["location"]
    assert loc["zone_containment"] > loc["zone_containment_baseline_nearest3"]
    assert loc["zone_containment"] > loc["zone_containment_baseline_mule"]


def test_countdown_beats_the_mean_predictor(ledger):
    loc = ledger["location"]
    assert loc["time_mae_minutes"] < loc["time_baseline_mae"]


def test_countdown_scatter_reproduces_the_published_mae(ledger):
    """
    The deck's countdown panel plots the same evaluation the card quotes.

    The panel used to re-score the shipped regressor on a split of its own
    making, so it drew a 12.35 min MAE underneath a card that said 11.8.
    """
    import numpy as np
    pairs = ROOT / "data" / "countdown_eval.npz"
    if not pairs.exists():
        pytest.skip("countdown_eval.npz not built; run engine/train_xgb.py")
    cd = np.load(pairs)
    mae = float(np.mean(np.abs(cd["predicted"] - cd["actual"])))
    assert mae == pytest.approx(ledger["location"]["time_mae_minutes"], abs=1e-6)

    base = float(np.mean(np.abs(cd["actual"] - cd["train_mean"])))
    assert base == pytest.approx(ledger["location"]["time_baseline_mae"], abs=1e-6)


def test_detection_sections_agree_on_the_gnn_score(ledger):
    """
    Both halves of the detection story quote the same F1.

    train_gnn.py writes it from the checkpoint; evaluate_baselines.py reads the
    checkpoint to compute lift. If these drift, one of them is stale.
    """
    assert ledger["detection"]["test_f1"] == pytest.approx(
        ledger["detection_baselines"]["gnn_test_f1"], abs=1e-9)


# ── The console's generated copy ─────────────────────────────────────────────

def test_frontend_stats_match_the_ledger(ledger):
    """The bundled console figures are a faithful projection, not a snapshot."""
    if not FRONTEND_STATS_PATH.exists():
        pytest.skip("model_stats.json not built; run scripts/export_metrics.py")
    stats = json.loads(FRONTEND_STATS_PATH.read_text(encoding="utf-8"))
    assert stats["zoneContainment"] == pytest.approx(
        ledger["location"]["zone_containment"])
    assert stats["countdownMae"] == pytest.approx(
        ledger["location"]["time_mae_minutes"])
    assert stats["gnnF1"] == pytest.approx(ledger["detection"]["test_f1"])
    assert stats["gnnBaseline"] == pytest.approx(
        ledger["detection_baselines"]["best_non_graph"])


# ── Nothing may retype a measured figure ─────────────────────────────────────

def test_console_constants_carry_no_literal_figures():
    """
    constants.js formats numbers; it must not contain any.

    This is the exact file where the retracted 98.5% / 1.2 s / 0.9996 figures
    outlived every corrected document.
    """
    src = CONSTANTS_JS.read_text(encoding="utf-8")
    body = src[src.index("export const MODEL_STATS"):]
    body = body[:body.index("}") + 1]
    offenders = re.findall(r"'\s*\d+\.\d+\s*%?\s*(?:min)?'|\b0\.\d{3,}\b", body)
    assert not offenders, f"hardcoded figures in MODEL_STATS: {offenders}"


def test_matrix_script_has_no_hardcoded_zone_dict():
    """The zone figures were a literal dict that no retrain ever touched."""
    src = MATRIX_SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"zone\s*=\s*dict\(\s*model\s*=\s*\.?\d", src), \
        "generate_model_matrix.py has re-introduced a hardcoded zone dict"


def test_matrix_card_reads_every_headline_from_data():
    """
    No card literal. The first card's big number was the string "87.4%" while
    its own subtitle was formatted from the data beside it, so updating one
    moved the baseline and left the headline behind.
    """
    src = MATRIX_SCRIPT.read_text(encoding="utf-8")
    cards = src[src.index("    cards = ["):]
    cards = cards[:cards.index("    ]") + 5]
    offenders = re.findall(r'"\s*\d+\.\d+\s*%', cards)
    assert not offenders, f"hardcoded headline on a card: {offenders}"
    assert "of 1,000" not in cards, "ATM denominator hardcoded on the card"
