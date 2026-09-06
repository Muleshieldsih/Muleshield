# -*- coding: utf-8 -*-
"""
Generate the canonical city/state directory for MuleShield.

Reads victim_complaints.csv, node_features.csv, and atm_directory.csv,
resolves authoritative states via majority vote, validates aliases,
and writes byte-identical JSON outputs for both backend and frontend.

Rule: "generated, never typed".
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORPUS_FILES = [
    ROOT / "data" / "victim_complaints.csv",
    ROOT / "data" / "node_features.csv",
    ROOT / "data" / "atm_directory.csv",
]

TARGET_FILES = [
    ROOT / "data" / "cities.json",
    ROOT / "frontend" / "src" / "data" / "cities.json",
]

# Hand-maintained aliases for backward-compatibility.
# Target must resolve to a canonical city name present in the corpus.
HAND_ALIASES = {
    "bangalore": "Bengaluru",
    "new delhi": "Delhi",
    "gurugram": "Gurgaon",
    "prayagraj": "Allahabad",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "mysore": "Mysuru",
    "trichy": "Tiruchirappalli",
    "trivandrum": "Thiruvananthapuram",
}


def normalize_city_name(raw: str) -> str:
    return " ".join(str(raw or "").split()).title()


def normalize_state_name(raw: str) -> str:
    # Collapse whitespace, preserve verbatim casing (e.g. "NCT of Delhi")
    return " ".join(str(raw or "").split())


def main():
    cities_by_file = {}
    city_state_votes = defaultdict(Counter)

    for filepath in CORPUS_FILES:
        rel_path = filepath.relative_to(ROOT).as_posix()
        if not filepath.exists():
            print(f"ERROR: Corpus file not found: {rel_path}", file=sys.stderr)
            sys.exit(1)

        file_cities = set()
        with open(filepath, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            has_state = "state" in (reader.fieldnames or [])
            for row in reader:
                city_raw = row.get("city")
                if not city_raw:
                    continue
                city = normalize_city_name(city_raw)
                if not city:
                    continue
                file_cities.add(city)

                if has_state:
                    state_raw = row.get("state")
                    if state_raw:
                        state = normalize_state_name(state_raw)
                        if state:
                            city_state_votes[city][state] += 1

        cities_by_file[rel_path] = file_cities

    all_cities = sorted(set().union(*cities_by_file.values()))

    print("=== Corpus File City Contribution ===")
    for rel_path, cset in cities_by_file.items():
        other_sets = [s for p, s in cities_by_file.items() if p != rel_path]
        other_union = set().union(*other_sets) if other_sets else set()
        unique_to_file = cset - other_union
        print(
            f"  {rel_path}: {len(cset)} cities contributed "
            f"({len(unique_to_file)} unique to this file)"
        )
    print(f"Total union of unique cities: {len(all_cities)}")

    # Resolve state by majority vote
    city_to_state = {}
    missing_state_cities = []

    for city in all_cities:
        votes = city_state_votes.get(city)
        if not votes:
            missing_state_cities.append(city)
            continue

        most_common = votes.most_common()
        if len(most_common) > 1:
            print(f"WARNING: City '{city}' has conflicting state entries across corpus:")
            for state_val, count in most_common:
                print(f"    - '{state_val}': {count} votes")

        resolved_state = most_common[0][0]
        city_to_state[city] = resolved_state

    if missing_state_cities:
        print(
            f"FATAL: The following cities have no state recorded in any file: "
            f"{missing_state_cities}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate aliases against canonical city set
    valid_aliases = {}
    for alias_key, target in sorted(HAND_ALIASES.items()):
        if target not in city_to_state:
            print(
                f"WARNING: Alias target '{target}' for alias '{alias_key}' "
                f"is not in city_to_state directory. Dropping alias."
            )
        else:
            valid_aliases[alias_key] = target

    payload = {
        "total": len(all_cities),
        "cities": all_cities,
        "city_to_state": city_to_state,
        "aliases": valid_aliases,
    }

    content_bytes = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    print("=== Exporting Canonical Directory ===")
    for target in TARGET_FILES:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            f.write(content_bytes)
        print(f"  Written: {target.relative_to(ROOT).as_posix()} ({len(content_bytes)} bytes)")

    print(f"SUCCESS: Exported {len(all_cities)} cities and {len(valid_aliases)} aliases.")


if __name__ == "__main__":
    main()
