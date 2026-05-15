#!/usr/bin/env python3
"""
Fetch 2024 US House results from MIT Election Lab (Harvard Dataverse) and
compute D margin per state for catalist_2024.csv.

Dataset: U.S. House 1976–2022 Elections (periodically updated; check for 2024 version)
DOI: https://doi.org/10.7910/DVN/IG0UN2

Margin formula: (total_D_votes - total_R_votes) / (total_D_votes + total_R_votes) * 100
  - General elections only (stage == "gen"); runoffs and specials excluded.
  - Uncontested districts (one major party absent) are flagged in the notes column
    because they inflate the statewide margin and should be reviewed before use.
    They are still included in the aggregate — caller decides whether to impute.

Usage:
    python data/fetch_catalist.py
    python data/fetch_catalist.py --year 2022   # test with an older year
    python data/fetch_catalist.py --local path/to/1976-2024-house.csv
"""

import argparse
import csv
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATASET_DOI = "doi:10.7910/DVN/IG0UN2"
DATAVERSE_API = "https://dataverse.harvard.edu/api"
OUTPUT_FILE = Path(__file__).parent / "catalist_2024.csv"
TARGET_YEAR = 2024

# Parties that count as Democratic or Republican votes
D_LABELS = {"DEMOCRAT", "DEMOCRATIC", "DEM", "D"}
R_LABELS = {"REPUBLICAN", "REP", "R"}


def fetch_dataset_metadata(doi: str) -> list[dict]:
    url = f"{DATAVERSE_API}/datasets/:persistentId/?persistentId={doi}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching dataset metadata: {e.reason}") from e
    return data["data"]["latestVersion"]["files"]


def find_house_file(files: list[dict]) -> tuple[int, str]:
    """Return (datafile_id, filename) for the House results CSV."""
    candidates = [
        f for f in files
        if "house" in f["dataFile"]["filename"].lower()
        and f["dataFile"]["filename"].lower().endswith(".csv")
    ]
    if not candidates:
        names = [f["dataFile"]["filename"] for f in files]
        raise RuntimeError(
            f"No house CSV found in dataset. Available files:\n  " + "\n  ".join(names)
        )
    # Prefer the file with the widest year range (longer filename usually)
    best = max(candidates, key=lambda f: len(f["dataFile"]["filename"]))
    return best["dataFile"]["id"], best["dataFile"]["filename"]


def download_datafile(file_id: int) -> str:
    url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    print(f"  Downloading file id={file_id} from {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading file {file_id}: {e.reason}") from e


def load_local_file(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def compute_margins(csv_text: str, year: int) -> list[dict]:
    reader = csv.DictReader(io.StringIO(csv_text))

    # state -> {abbr, d_votes, r_votes, uncontested_districts}
    states: dict[str, dict] = {}
    # district-level tallies for uncontested detection: (state, district) -> {d, r}
    district_votes: dict[tuple, dict] = {}

    for row in reader:
        try:
            row_year = int(row.get("year", 0))
        except ValueError:
            continue
        if row_year != year:
            continue

        stage = (row.get("stage") or "").strip().lower()
        if stage != "gen":
            continue

        # Skip runoffs and specials
        if (row.get("runoff") or "").strip().upper() == "TRUE":
            continue
        if (row.get("special") or "").strip().upper() == "TRUE":
            continue
        # Skip write-ins
        if (row.get("writein") or "").strip().upper() == "TRUE":
            continue

        state = (row.get("state") or "").strip().title()
        abbr = (row.get("state_po") or "").strip().upper()
        district = (row.get("district") or "0").strip()
        party_raw = (row.get("party") or "").strip().upper()

        try:
            votes = int(float(row.get("candidatevotes") or 0))
        except (ValueError, TypeError):
            votes = 0

        if not state or not abbr:
            continue

        if state not in states:
            states[state] = {"abbr": abbr, "d": 0, "r": 0}

        dk = (state, district)
        if dk not in district_votes:
            district_votes[dk] = {"d": 0, "r": 0}

        if party_raw in D_LABELS:
            states[state]["d"] += votes
            district_votes[dk]["d"] += votes
        elif party_raw in R_LABELS:
            states[state]["r"] += votes
            district_votes[dk]["r"] += votes

    if not states:
        return []

    # Identify uncontested districts (one major party absent from a district)
    uncontested_states: set[str] = set()
    for (state, _district), dv in district_votes.items():
        if dv["d"] == 0 or dv["r"] == 0:
            uncontested_states.add(state)

    results = []
    for state in sorted(states):
        v = states[state]
        dr_total = v["d"] + v["r"]
        if dr_total == 0:
            continue
        margin = (v["d"] - v["r"]) / dr_total * 100
        sign = "+" if margin >= 0 else ""
        notes = "uncontested_district_present" if state in uncontested_states else ""
        results.append({
            "state": state,
            "abbr": v["abbr"],
            "d_margin_2024": f"{sign}{margin:.1f}",
            "notes": notes,
        })

    return results


def write_output(rows: list[dict]) -> None:
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["state", "abbr", "d_margin_2024", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=TARGET_YEAR,
                        help=f"Election year to extract (default: {TARGET_YEAR})")
    parser.add_argument("--local", metavar="PATH",
                        help="Use a local CSV file instead of downloading")
    args = parser.parse_args()

    if args.local:
        print(f"Loading local file: {args.local}")
        csv_text = load_local_file(args.local)
    else:
        print(f"Fetching dataset metadata from Harvard Dataverse ({DATASET_DOI}) ...")
        try:
            files = fetch_dataset_metadata(DATASET_DOI)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(files)} files in dataset.")
        try:
            file_id, filename = find_house_file(files)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Identified house results file: {filename}")
        try:
            csv_text = download_datafile(file_id)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Computing state-level D margins for {args.year} ...")
    rows = compute_margins(csv_text, args.year)

    if not rows:
        print(
            f"ERROR: No {args.year} general-election data found in the dataset.",
            file=sys.stderr,
        )
        print(
            "The MIT Election Lab dataset may not yet include this year. Check:\n"
            f"  https://dataverse.harvard.edu/dataset.xhtml?persistentId={DATASET_DOI}",
            file=sys.stderr,
        )
        sys.exit(1)

    uncontested = [r["state"] for r in rows if r["notes"]]
    if uncontested:
        print(
            f"\nWARNING: Uncontested districts present in {len(uncontested)} states:\n"
            f"  {', '.join(uncontested)}\n"
            "  These inflate the statewide D margin. Review before using as SSOT."
        )

    print(f"\nResults ({len(rows)} states):")
    for r in rows:
        flag = " ← UNCONTESTED" if r["notes"] else ""
        print(f"  {r['abbr']:2s}  {r['d_margin_2024']:>7s}{flag}")

    write_output(rows)


if __name__ == "__main__":
    main()
