#!/usr/bin/env python3
"""Build the observation-matched AOTS2Action Table I evaluation corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


HORIZONS = (6, 12, 24, 48, 72, 96)
TRACK_ALIASES = {
    "FUNGWONG": "FUNG-WONG",
    "WONG": "FUNG-WONG",
}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).replace(tzinfo=None)


def canonical_track_id(value: str) -> str:
    track_id = value.strip().upper()
    return TRACK_ALIASES.get(track_id, track_id)


def load_forecast_cases(path: Path) -> list[dict[str, object]]:
    cases: dict[tuple[str, str, int], dict[str, object]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            horizon = int(row["LEAD_TIME"])
            if horizon not in HORIZONS:
                continue
            track_id = canonical_track_id(row["TRACK_ID"])
            forecast_time = parse_time(row["FORECAST_TIME"])
            valid_time = parse_time(row["VALID_TIME"])
            key = (track_id, forecast_time.isoformat(), horizon)
            case = cases.setdefault(
                key,
                {
                    "cyclone_id": track_id,
                    "forecast_time": forecast_time,
                    "valid_time": valid_time,
                    "horizon_h": horizon,
                    "members": set(),
                },
            )
            if case["valid_time"] != valid_time:
                raise ValueError(f"Multiple valid times for forecast case {key}")
            case["members"].add(row["ENSEMBLE_MEMBER"])
    return list(cases.values())


def load_observations(path: Path) -> dict[str, list[dict[str, object]]]:
    observations: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if not row.get("SID", "").strip() or not row.get("ISO_TIME", "").strip():
                continue
            name = canonical_track_id(row["NAME"])
            if not row.get("LAT", "").strip() or not row.get("LON", "").strip():
                continue
            observations[name].append(
                {
                    "sid": row["SID"].strip(),
                    "observation_time": parse_time(row["ISO_TIME"]),
                    "observed_lat": float(row["LAT"]),
                    "observed_lon": float(row["LON"]),
                }
            )
    for records in observations.values():
        records.sort(key=lambda record: record["observation_time"])
    return observations


def match_cases(
    cases: list[dict[str, object]],
    observations: dict[str, list[dict[str, object]]],
    tolerance_hours: float,
) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for case in cases:
        candidates = observations.get(case["cyclone_id"], [])
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda record: abs(record["observation_time"] - case["valid_time"]),
        )
        mismatch_hours = abs(
            (nearest["observation_time"] - case["valid_time"]).total_seconds()
        ) / 3600
        if mismatch_hours > tolerance_hours:
            continue
        matched.append(
            {
                **case,
                **nearest,
                "member_count": len(case["members"]),
                "temporal_mismatch_h": mismatch_hours,
            }
        )
    return matched


def write_corpus(path: Path, matched: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "cyclone_id",
        "sid",
        "forecast_time",
        "horizon_h",
        "valid_time",
        "observation_time",
        "temporal_mismatch_h",
        "observed_lat",
        "observed_lon",
        "member_count",
    )
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            matched,
            key=lambda item: (item["horizon_h"], item["cyclone_id"], item["forecast_time"]),
        ):
            writer.writerow({field: row[field] for field in fields})


def write_summary(path: Path, matched: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["metric", *[f"{h} h" for h in HORIZONS]])
        writer.writerow(
            ["Verifying pairs n_h", *[sum(row["horizon_h"] == h for row in matched) for h in HORIZONS]]
        )
        writer.writerow(
            [
                "Distinct cyclones",
                *[
                    len({row["sid"] for row in matched if row["horizon_h"] == h})
                    for h in HORIZONS
                ],
            ]
        )
        writer.writerow(["Zero-realized-exposure cases", *(["unavailable"] * len(HORIZONS))])
        writer.writerow(["Cases with |R_i,h| < 3", *(["unavailable"] * len(HORIZONS))])


def write_metadata(path: Path, cases: list[dict[str, object]], matched: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "total_cyclones": len({case["cyclone_id"] for case in cases}),
        "total_forecast_cycles": len(
            {(case["cyclone_id"], case["forecast_time"]) for case in cases}
        ),
        "maximum_ensemble_size_M_full": max(len(case["members"]) for case in cases),
        "requested_horizons_h": list(HORIZONS),
        "temporal_tolerance_h": 3,
        "forecast_cases_at_requested_horizons": len(cases),
        "matched_verifying_cases": len(matched),
        "canonical_track_aliases": TRACK_ALIASES,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tolerance-hours", type=float, default=3.0)
    args = parser.parse_args()

    cases = load_forecast_cases(args.forecasts)
    observations = load_observations(args.observations)
    matched = match_cases(cases, observations, args.tolerance_hours)
    write_corpus(args.corpus, matched)
    write_summary(args.summary, matched)
    write_metadata(args.metadata, cases, matched)

    print(f"Forecast cases at requested horizons: {len(cases)}")
    print(f"Matched verifying cases: {len(matched)}")
    print(f"Maximum ensemble size M_full: {max(len(case['members']) for case in cases)}")


if __name__ == "__main__":
    main()