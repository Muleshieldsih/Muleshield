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
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
        fh.write("\n")
    return stats
