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


def test_forward_forecast_beats_historical_density(ledger):
    """The differentiating claim, as a number rather than as prose.

    I4C already runs Pratibimb, which maps cybercrime geographically. Ranking
    cells by decayed historical cash-out density IS that system, functionally.
    If the forward forecast does not clear it, this component has rebuilt
    something the judges already own and the deck must not claim otherwise.
    """
    h = ledger["hotspot"]
    fwd = h["pai_at_k"]["5"]
    base = h["baseline_historical_density"]["pai_at_5"]
    assert fwd > base, (
        f"forward forecast PAI@5 {fwd} does not beat historical density {base} "
        f"-- the differentiating claim has failed and the pitch must change"
    )


def test_prior_does_not_dominate_the_surface(ledger):
    """The historical term may reorder cells; it may never carry them.

    prior_share_national is MEASURED from a real surface at the busiest epoch,
    not computed algebraically from prior_weight -- an algebraic value would
    make this test tautological, which is the failure mode that let the
    retracted figures survive for weeks.
    """
    h = ledger["hotspot"]
    assert h["prior_share_national"] <= h["prior_weight"] + 1e-6, (
        f"prior carries {h['prior_share_national']:.4f} of the surface, above "
        f"the {h['prior_weight']} cap -- the forecast has drifted into being a "
        f"density map"
    )


def test_hotspot_reports_the_baseline_it_does_not_beat(ledger):
    """The nearest-cell baseline must stay in the ledger even though it wins.

    Distance from the traced terminal account beats the forecast on per-complaint
    hit rate. That is the same ceiling OVERNIGHT_ML_AUDIT.md found for Top-K, and
    the project's standing rule is that a losing comparison is published rather
    than dropped. This test fails the build if someone removes it.
    """
    h = ledger["hotspot"]
    assert "baseline_nearest_cell" in h, "the strong baseline was removed"
    assert "beats_nearest_cell" in h
    assert isinstance(h["beats_nearest_cell"], bool)


def test_lead_time_is_reported_and_positive(ledger):
    """"In Advance" is the phrase the problem statement turns on. It needs a
    number attached to it, not an adjective."""
    h = ledger["hotspot"]
    assert h["lead_time_median_min"] > 0
    assert 0.0 <= h["lead_actionable_rate"] <= 1.0


# ── Base-rate sensitivity ────────────────────────────────────────────────────
#
# Added when the Model Performance screen grew an "operational deployment policy"
# table. A screen that tells an I4C desk what posture to adopt at a given mule
# rate is making an operational claim, so the arithmetic behind it has to be
# checked rather than trusted, and the one thing it must never do is describe a
# governance control the code does not implement.

class TestPrevalenceSensitivity:

    @pytest.fixture(scope="class")
    def prev(self):
        m = read_metrics()
        if "detection_prevalence" not in m:
            pytest.skip("run: python scripts/export_confusion.py")
        return m["detection_prevalence"]

    def test_tpr_and_fpr_match_the_confusion_matrix(self, prev):
        """The projection is only honest if it starts from the counted matrix."""
        c = read_metrics()["detection_confusion"]
        tpr = c["true_positives"] / (c["true_positives"] + c["false_negatives"])
        fpr = c["false_positives"] / (c["false_positives"] + c["true_negatives"])
        assert prev["tpr"] == pytest.approx(tpr, abs=1e-6)
        assert prev["fpr"] == pytest.approx(fpr, abs=1e-6)

    def test_precision_is_the_bayes_projection(self, prev):
        """precision(p) = TPR*p / (TPR*p + FPR*(1-p)), for every row."""
        tpr, fpr = prev["tpr"], prev["fpr"]
        for r in prev["scenarios"]:
            p = r["prevalence"]
            expected = (tpr * p) / (tpr * p + fpr * (1 - p))
            assert r["precision"] == pytest.approx(expected, abs=1e-4), r["scenario"]

    def test_the_measured_row_reproduces_the_test_set_precision(self, prev):
        """The corpus row must agree with what was actually observed, or the
        projection is describing a different detector than the one measured."""
        c = read_metrics()["detection_confusion"]
        observed = c["true_positives"] / (c["true_positives"] + c["false_positives"])
        row = next(r for r in prev["scenarios"] if r["is_measured"])
        assert row["precision"] == pytest.approx(observed, abs=1e-3)

    def test_precision_falls_as_prevalence_falls(self, prev):
        """The whole point of the table. If this ever reads the other way the
        arithmetic is wrong, not the world."""
        rows = sorted(prev["scenarios"], key=lambda r: -r["prevalence"])
        precisions = [r["precision"] for r in rows]
        assert precisions == sorted(precisions, reverse=True)

    def test_innocent_count_is_reported_and_dominates_at_low_prevalence(self, prev):
        """The column that decides deployability. At the lowest prevalence the
        queue must be overwhelmingly innocent -- if it is not, the numbers are
        wrong."""
        rows = sorted(prev["scenarios"], key=lambda r: r["prevalence"])
        lowest = rows[0]
        assert lowest["innocent_per_100k"] > lowest["true_mules_per_100k"] * 10

    def test_flagged_splits_into_true_and_innocent(self, prev):
        for r in prev["scenarios"]:
            assert r["flagged_per_100k"] == pytest.approx(
                r["true_mules_per_100k"] + r["innocent_per_100k"], abs=1.0), r["scenario"]

    def test_automation_is_only_called_safe_above_the_floor(self, prev):
        floor = prev["automation_floor"]
        for r in prev["scenarios"]:
            assert r["automation_safe"] == (r["prevalence"] >= floor), r["scenario"]

    def test_the_ledger_admits_the_policy_is_not_enforced(self, prev):
        """THE IMPORTANT ONE.

        backend/routers/freeze.py has no prevalence gate -- it freezes whatever
        it is given. The console prints a recommended posture per scenario, and
        it may only do so while this flag says the posture is advisory. If
        somebody implements the gate, they flip this and the screen's wording
        changes with it; until then, claiming enforcement would be a claim the
        code does not honour.
        """
        assert prev["actions_are_enforced"] is False

    def test_freeze_router_really_has_no_prevalence_gate(self):
        """Keeps the flag above honest by checking the code it describes."""
        src = (ROOT / "backend" / "routers" / "freeze.py").read_text(encoding="utf-8")
        assert "prevalence" not in src.lower(), (
            "freeze.py now mentions prevalence -- if a gate was implemented, set "
            "detection_prevalence.actions_are_enforced True and update the console")

    def test_console_reads_every_scenario_from_the_ledger(self):
        """No scenario may be typed into the page."""
        stats = json.loads(FRONTEND_STATS_PATH.read_text(encoding="utf-8"))
        ledger = read_metrics()["detection_prevalence"]
        assert stats["prevalenceScenarios"] == ledger["scenarios"]
        assert stats["automationFloor"] == ledger["automation_floor"]
        assert stats["actionsAreEnforced"] == ledger["actions_are_enforced"]


# ── Zone significance ────────────────────────────────────────────────────────

class TestZoneSignificance:

    @pytest.fixture(scope="class")
    def sig(self):
        m = read_metrics()
        if "zone_significance" not in m:
            pytest.skip("run: python scripts/zone_significance.py")
        return m["zone_significance"]

    def test_it_was_measured_on_the_published_zone(self, sig):
        """A p-value computed on a different containment than the published one
        would be describing a different quantity. The script self-checks this;
        so does the ledger."""
        published = read_metrics()["location"]["zone_containment"]
        assert sig["zone_containment"] == pytest.approx(published, abs=1e-6)

    def test_both_naive_zones_are_compared(self, sig):
        keys = {c["baseline_key"] for c in sig["comparisons"]}
        assert keys == {"nearest3", "mule"}

    def test_discordant_counts_are_consistent(self, sig):
        for c in sig["comparisons"]:
            assert c["discordant"] == c["model_only"] + c["baseline_only"]

    def test_significance_flag_matches_the_p_value(self, sig):
        for c in sig["comparisons"]:
            assert c["significant_at_05"] == (c["p_value"] < sig["alpha"])

    def test_p_value_survives_the_json_round_trip(self, sig):
        """These run to 1e-27. An earlier revision rounded to 8 decimals and
        wrote a literal 0.0 into the console payload."""
        for c in sig["comparisons"]:
            if c["significant_at_05"]:
                assert c["p_value"] > 0.0, (
                    f"{c['baseline_key']} p-value collapsed to zero in the ledger")

    def test_console_reads_significance_from_the_ledger(self, sig):
        stats = json.loads(FRONTEND_STATS_PATH.read_text(encoding="utf-8"))
        assert stats["zoneSignificance"] == sig["comparisons"]
        assert stats["zoneSignificanceN"] == sig["n_cashouts"]
