# -*- coding: utf-8 -*-
"""
MuleShield AI — metrics ledger
SIH26184 | MHA / I4C

One file, data/metrics.json, that every measured figure is written to and every
consumer reads from.

The figures on the deck card used to live in three places at once: computed in
engine/train_xgb.py, retyped as a literal dict in scripts/generate_model_matrix.py,
and retyped again in frontend/src/utils/constants.js. Nothing kept them in step,
and they had already drifted — the mule-detection baseline read 0.8501 on the
card and 0.8463 in the console.

So: training writes, everything else reads. A figure that is not in the ledger
was not measured, and a consumer that cannot find its key says so loudly rather
than falling back to a number someone remembered.

Sections are merged, not overwritten, so running one trainer does not erase
another's numbers.

    from metrics_io import write_metrics, read_metrics, require

    write_metrics("location", {"zone_containment": 0.874, ...})
    m = read_metrics()
    hit = require(m, "location", "zone_containment")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "data" / "metrics.json"

# Display-shaped subset the console bundles at build time. The frontend has to
# render these on a venue network with no backend, so the numbers ship with it.
FRONTEND_STATS_PATH = ROOT / "frontend" / "src" / "data" / "model_stats.json"

_ABOUT = (
    "Measured figures for MuleShield AI (SIH26184). Written by the training "
    "and evaluation scripts; read by scripts/generate_model_matrix.py and the "
    "console. Do not hand-edit — regenerate with the commands in 'source'."
)


class MetricsMissing(KeyError):
    """A consumer asked for a figure that has not been measured yet."""


def _plain(obj: Any) -> Any:
    """Coerce numpy scalars and containers into JSON-native types."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    return obj


def read_metrics(path: Path = METRICS_PATH) -> dict:
    """Load the ledger. Returns an empty dict if nothing has been measured yet."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_metrics(section: str, payload: Mapping[str, Any], *,
                  source: str | None = None, path: Path = METRICS_PATH) -> dict:
    """
    Merge one section into the ledger, leaving every other section untouched.

    `source` records the command that regenerates this section, so a stale
    figure names its own fix.
    """
    ledger = read_metrics(path)
    ledger["_about"] = _ABOUT

    entry = _plain(dict(payload))
    entry["measured_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if source:
        entry["source"] = source
    ledger[section] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return ledger


def require(ledger: Mapping[str, Any], section: str, key: str) -> Any:
    """
    Fetch a measured figure, or fail with the command that produces it.

    Deliberately not `.get(key, <a number that looks plausible>)`. A silent
    fallback is how the retracted figures survived in the console long after
    every document had been corrected.
    """
    sec = ledger.get(section)
    if sec is None or key not in sec:
        cmd = {
            "detection": "python engine/train_gnn.py",
            "detection_baselines": "python scripts/evaluate_baselines.py",
            "location": "python engine/train_xgb.py",
            "ranking": "python scripts/topk_curve.py",
            "hotspot": "python scripts/evaluate_hotspots.py",
        }.get(section, "the training scripts")
        raise MetricsMissing(
            f"data/metrics.json has no '{section}.{key}'. Run: {cmd}"
        )
    return sec[key]


def write_frontend_stats(path: Path = FRONTEND_STATS_PATH,
                         ledger_path: Path = METRICS_PATH) -> dict:
    """
    Project the ledger into the small raw-number payload the console imports.

    Numbers stay numbers here; the console formats them for display, so there is
    exactly one place that decides what '87.4%' looks like.
    """
    m = read_metrics(ledger_path)
    stats = {
        "_about": "Generated from data/metrics.json — do not hand-edit. "
                  "Regenerate: python scripts/export_metrics.py",
        "zoneContainment": require(m, "location", "zone_containment"),
        "zoneBaselineNearest3": require(m, "location",
                                        "zone_containment_baseline_nearest3"),
        "zoneMedianAtms": require(m, "location", "zone_median_atms"),
        "zoneMedianRadiusKm": require(m, "location", "zone_median_radius_km"),
        "atmTotal": require(m, "location", "n_atms"),
        "countdownMae": require(m, "location", "time_mae_minutes"),
        "countdownBaseline": require(m, "location", "time_baseline_mae"),
        "gnnF1": require(m, "detection", "test_f1"),
        "gnnBaseline": require(m, "detection_baselines", "best_non_graph"),
        "gnnBaselineModel": require(m, "detection_baselines", "best_non_graph_model"),
        # Ranked candidate locations - the operating point the console exposes.
        "top5Containment": require(m, "ranking", "top5"),
        "top5BaselineDistance": require(m, "ranking", "top5_baseline_distance"),
        "top5SearchReduction": require(m, "ranking", "top5_search_reduction"),
        "operatingK": require(m, "ranking", "operating_k"),
        # Detection detail for the Model performance screen. These describe the
        # detector across a held-out test set; they say nothing about any one
        # case, which is why they no longer sit beside one.
        "precision": require(m, "detection", "test_precision"),
        "recall": require(m, "detection", "test_recall"),
        "rocAuc": require(m, "detection", "test_auc"),
        "prAuc": require(m, "detection", "test_pr_auc"),
        "threshold": require(m, "detection", "threshold"),
        "truePositives": require(m, "detection_confusion", "true_positives"),
        "falsePositives": require(m, "detection_confusion", "false_positives"),
        "falseNegatives": require(m, "detection_confusion", "false_negatives"),
        "trueNegatives": require(m, "detection_confusion", "true_negatives"),
        "nTestAccounts": require(m, "detection_confusion", "n_test_accounts"),
        "actualMules": require(m, "detection_confusion", "actual_mules"),
        "detectionBaselines": require(m, "detection_baselines", "per_model_f1"),
        "rankingCurve": require(m, "ranking", "curve"),
        "nTestCashouts": require(m, "ranking", "n_test_cashouts"),
        "top5BaselineDistance": require(m, "ranking", "top5_baseline_distance"),
        "zoneMedianErrorKm": require(m, "location", "zone_median_error_km"),
        "countdownR2": require(m, "location", "time_r2"),
        "leadTimeMedianMin": require(m, "location", "lead_time_median_min"),
        # Forward hotspot forecast. hotspotBaselinePai is the historical-density
        # ranking -- functionally what I4C's Pratibimb already provides -- and it
        # is projected alongside the headline on purpose: the console shows the
        # comparison, not just the number.
        "hotspotHitRateAt5": require(m, "hotspot", "hit_rate_at_k")["5"],
        "hotspotPaiAt5": require(m, "hotspot", "pai_at_k")["5"],
        "hotspotPeiAt5": require(m, "hotspot", "pei_at_k")["5"],
        "hotspotRupeesCoveredAt5": require(m, "hotspot", "rupees_covered_at_k")["5"],
        "hotspotBaselinePai": require(m, "hotspot", "baseline_historical_density")["pai_at_5"],
        "hotspotBaselineHitRate": require(m, "hotspot", "baseline_historical_density")["hit_rate_at_5"],
        "hotspotNearestCellHitRate": require(m, "hotspot", "baseline_nearest_cell")["hit_rate_at_5"],
        "hotspotLeadTimeMedianMin": require(m, "hotspot", "lead_time_median_min"),
        "hotspotLeadActionableRate": require(m, "hotspot", "lead_actionable_rate"),
        "hotspotCells": require(m, "hotspot", "n_cells"),
        "hotspotOperatingK": require(m, "hotspot", "operating_k_cells"),
        "hotspotPriorWeight": require(m, "hotspot", "prior_weight"),
        "hotspotPriorShare": require(m, "hotspot", "prior_share_national"),
        "hotspotFlaggedAtmShareAt5": require(m, "hotspot", "flagged_atm_share_at_k")["5"],
        "hotspotPrecisionCurve": require(m, "hotspot", "precision_coverage_curve"),
        "hotspotNTestCashouts": require(m, "hotspot", "n_test_cashouts"),
        "inferenceMeanMs": require(m, "location", "inference_mean_ms"),
        "measuredUtc": require(m, "detection", "measured_utc"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
        fh.write("\n")
    return stats
