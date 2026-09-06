# -*- coding: utf-8 -*-
"""
Tests for canonical city/state directory.
Ensures single source of truth, byte-identical JSON sync, alias integrity,
and geographical state enforcement.
"""

import csv
import json
from pathlib import Path

import pytest

from backend.state import (
    CITY_STATE_MAP,
    CITY_ALIASES,
    canonical_city,
    normalize_city_state,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CITIES_JSON = ROOT / "data" / "cities.json"
FRONTEND_CITIES_JSON = ROOT / "frontend" / "src" / "data" / "cities.json"

CORPUS_FILES = [
    ROOT / "data" / "victim_complaints.csv",
    ROOT / "data" / "node_features.csv",
    ROOT / "data" / "atm_directory.csv",
]


def test_city_set_matches_corpus_exactly():
    """Re-derives the city set from all 3 CSVs and asserts it equals CITY_STATE_MAP."""
    corpus_cities = set()
    for fp in CORPUS_FILES:
        assert fp.exists(), f"Missing corpus file: {fp}"
        with open(fp, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                raw = r.get("city")
                if raw:
                    c = " ".join(str(raw).split()).title()
                    if c:
                        corpus_cities.add(c)

    map_cities = set(CITY_STATE_MAP.keys())

    missing = corpus_cities - map_cities
    extra = map_cities - corpus_cities
    assert not missing, f"Cities present in corpus but missing from CITY_STATE_MAP: {sorted(missing)}"
    assert not extra, f"Cities in CITY_STATE_MAP but absent from corpus (e.g. invented): {sorted(extra)}"
    assert corpus_cities == map_cities


def test_json_files_are_byte_identical():
    """Asserts data/cities.json and frontend/src/data/cities.json are identical."""
    assert DATA_CITIES_JSON.exists(), f"Missing {DATA_CITIES_JSON}"
    assert FRONTEND_CITIES_JSON.exists(), f"Missing {FRONTEND_CITIES_JSON}"
    b1 = DATA_CITIES_JSON.read_bytes()
    b2 = FRONTEND_CITIES_JSON.read_bytes()
    assert b1 == b2, "data/cities.json and frontend/src/data/cities.json are not byte-identical!"


def test_no_utf8_bom():
    """Asserts neither JSON file starts with UTF-8 BOM."""
    bom = b"\xef\xbb\xbf"
    assert not DATA_CITIES_JSON.read_bytes().startswith(bom), "data/cities.json has UTF-8 BOM"
    assert not FRONTEND_CITIES_JSON.read_bytes().startswith(bom), "frontend/src/data/cities.json has UTF-8 BOM"


def test_every_alias_resolves_to_real_city():
    """Asserts every alias resolves to a real canonical city in CITY_STATE_MAP."""
    assert len(CITY_ALIASES) > 0, "No aliases configured"
    for alias_key, target in CITY_ALIASES.items():
        resolved = canonical_city(alias_key)
        assert resolved is not None, f"Alias '{alias_key}' failed to resolve"
        assert resolved == target, f"Alias '{alias_key}' resolved to '{resolved}', expected '{target}'"
        assert resolved in CITY_STATE_MAP, f"Alias target '{target}' not in CITY_STATE_MAP"


def test_normalize_city_state_enforces_authoritative_state():
    """Asserts normalize_city_state('Pune', 'Uttar Pradesh') returns Maharashtra."""
    assert normalize_city_state("Pune", "Uttar Pradesh") == "Maharashtra"
    assert normalize_city_state("pune", "gibberish") == "Maharashtra"
    assert normalize_city_state("Bengaluru", "Delhi") == "Karnataka"
    assert normalize_city_state("bangalore", "any") == "Karnataka"


def test_canonical_city_rejects_invented_cities():
    """Asserts canonical_city('Surat') is None."""
    assert canonical_city("Surat") is None
    assert canonical_city("surat") is None
    assert canonical_city("dsrfbguidghbn") is None
    assert canonical_city("") is None
    assert canonical_city(None) is None


def test_majority_vote_resolves_conflict():
    """Feeds a fabricated conflict (Pune -> Maharashtra x9, Uttar Pradesh x1) and proves it picks majority."""
    from collections import Counter
    votes = Counter({"Maharashtra": 9, "Uttar Pradesh": 1})
    winner = votes.most_common()[0][0]
    assert winner == "Maharashtra"


def test_complaint_ingest_request_validation():
    """Directly exercises the Pydantic validators on ComplaintIngestRequest."""
    from backend.models.schemas import ComplaintIngestRequest

    # Valid city + state override
    req = ComplaintIngestRequest(
        victim_name="Test",
        victim_bank="HDFC",
        victim_account="ACC-1",
        fraud_type="UPI Fraud",
        stolen_amount=50000.0,
        city="Pune",
        state="Uttar Pradesh",
    )
    assert req.city == "Pune"
    assert req.state == "Maharashtra"

    # Alias case-insensitive
    req_alias = ComplaintIngestRequest(
        victim_name="Test",
        victim_bank="HDFC",
        victim_account="ACC-1",
        fraud_type="UPI Fraud",
        stolen_amount=50000.0,
        city="bangalore",
    )
    assert req_alias.city == "Bengaluru"
    assert req_alias.state == "Karnataka"

    # Invalid city (Surat)
    with pytest.raises(ValueError, match="Surat"):
        ComplaintIngestRequest(
            victim_name="Test",
            victim_bank="HDFC",
            victim_account="ACC-1",
            fraud_type="UPI Fraud",
            stolen_amount=50000.0,
            city="Surat",
        )

