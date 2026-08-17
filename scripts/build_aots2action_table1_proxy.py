#!/usr/bin/env python3
"""Compute explicitly marked proxy-only exposure rows for AOTS2Action Table I."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


HORIZONS = (6, 12, 24, 48, 72, 96)
TRACK_ALIASES = {"FUNGWONG": "FUNG-WONG", "WONG": "FUNG-WONG"}
EARTH_RADIUS_KM = 6371.0


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).replace(tzinfo=None)


def canonical_track_id(value: str) -> str:
    track_id = value.strip().upper()
    return TRACK_ALIASES.get(track_id, track_id)


def distances_km(lat: np.ndarray, lon: np.ndarray, point_lat: float, point_lon: float) -> np.ndarray:
    lat_radians = np.radians(lat)
    point_lat_radians = np.radians(point_lat)
    delta_lat = np.radians(point_lat - lat)
    delta_lon = np.radians(point_lon - lon)
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_radians)
        * np.cos(point_lat_radians)
        * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def load_matched_cases(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as source:
        return [
            {
                **row,
                "forecast_time_parsed": parse_time(row["forecast_time"]),
                "horizon_h_parsed": int(row["horizon_h"]),
                "observed_lat_parsed": float(row["observed_lat"]),
                "observed_lon_parsed": float(row["observed_lon"]),
            }
            for row in csv.DictReader(source)
        ]


def load_forecast_positions(
    path: Path, case_keys: set[tuple[str, datetime, int]]
) -> dict[tuple[str, datetime, int], list[tuple[float, float]]]:
    positions: dict[tuple[str, datetime, int], list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            horizon = int(row["LEAD_TIME"])
            if horizon not in HORIZONS:
                continue
            key = (
                canonical_track_id(row["TRACK_ID"]),
                parse_time(row["FORECAST_TIME"]),
                horizon,
            )
            if key in case_keys:
                positions[key].append((float(row["LATITUDE"]), float(row["LONGITUDE"])))
    return positions


def load_proxy_grid(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        "lat": np.array([float(row["lat"]) for row in rows]),
        "lon": np.array([float(row["lon"]) for row in rows]),
        "population": np.array([float(row["population"]) for row in rows]),
        "region_id": np.array([row["region_id"] for row in rows], dtype=object),
    }


def compute_proxy_cases(
    cases: list[dict[str, object]],
    positions: dict[tuple[str, datetime, int], list[tuple[float, float]]],
    grid: dict[str, np.ndarray],
    radius_km: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        key = (
            case["cyclone_id"],
            case["forecast_time_parsed"],
            case["horizon_h_parsed"],
        )
        predicted_mask = np.zeros(len(grid["lat"]), dtype=bool)
        for member_lat, member_lon in positions[key]:
            predicted_mask |= distances_km(
                grid["lat"], grid["lon"], member_lat, member_lon
            ) <= radius_km
        realized_mask = distances_km(
            grid["lat"],
            grid["lon"],
            case["observed_lat_parsed"],
            case["observed_lon_parsed"],
        ) <= radius_km
        union_regions = set(grid["region_id"][predicted_mask | realized_mask])
        results.append(
            {
                "cyclone_id": case["cyclone_id"],
                "forecast_time": case["forecast_time"],
                "horizon_h": case["horizon_h_parsed"],
                "proxy_realized_population": float(grid["population"][realized_mask].sum()),
                "proxy_realized_cells": int(realized_mask.sum()),
                "proxy_nonzero_union_regions": len(union_regions),
            }
        )
    return results


def write_case_results(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def write_summary(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = "PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE"
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["marker", "metric", *[f"{h} h" for h in HORIZONS]])
        writer.writerow(
            [marker, "Zero-realized-exposure cases"]
            + [
                sum(
                    row["horizon_h"] == horizon
                    and row["proxy_realized_population"] == 0
                    for row in results
                )
                for horizon in HORIZONS
            ]
        )
        writer.writerow(
            [marker, "Cases with |R_i,h| < 3"]
            + [
                sum(
                    row["horizon_h"] == horizon
                    and row["proxy_nonzero_union_regions"] < 3
                    for row in results
                )
                for horizon in HORIZONS
            ]
        )


def write_full_marked_table(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["marker", "metric", *[f"{h} h" for h in HORIZONS]])
        writer.writerow(
            ["OBSERVATION_MATCHED", "Verifying pairs n_h"]
            + [sum(row["horizon_h"] == horizon for row in results) for horizon in HORIZONS]
        )
        writer.writerow(
            ["OBSERVATION_MATCHED", "Distinct cyclones"]
            + [
                len(
                    {
                        row["cyclone_id"]
                        for row in results
                        if row["horizon_h"] == horizon
                    }
                )
                for horizon in HORIZONS
            ]
        )
        writer.writerow(
            ["PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE", "Zero-realized-exposure cases"]
            + [
                sum(
                    row["horizon_h"] == horizon
                    and row["proxy_realized_population"] == 0
                    for row in results
                )
                for horizon in HORIZONS
            ]
        )
        writer.writerow(
            ["PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE", "Cases with |R_i,h| < 3"]
            + [
                sum(
                    row["horizon_h"] == horizon
                    and row["proxy_nonzero_union_regions"] < 3
                    for row in results
                )
                for horizon in HORIZONS
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--proxy-grid", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--full-marked-table", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--impact-radius-km", type=float, default=25.0)
    args = parser.parse_args()

    cases = load_matched_cases(args.corpus)
    case_keys = {
        (case["cyclone_id"], case["forecast_time_parsed"], case["horizon_h_parsed"])
        for case in cases
    }
    positions = load_forecast_positions(args.forecasts, case_keys)
    missing_positions = case_keys - positions.keys()
    if missing_positions:
        raise ValueError(f"Missing forecast positions for {len(missing_positions)} cases")
    grid = load_proxy_grid(args.proxy_grid)
    results = compute_proxy_cases(cases, positions, grid, args.impact_radius_km)
    write_case_results(args.case_output, results)
    write_summary(args.summary, results)
    write_full_marked_table(args.full_marked_table, results)
    args.metadata.write_text(
        json.dumps(
            {
                "marker": "PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE",
                "impact_radius_km_assumed": args.impact_radius_km,
                "proxy_grid": str(args.proxy_grid),
                "proxy_grid_resolution_degrees": 0.75,
                "proxy_population_definition": "forecast-track sample density^1.15 * 800",
                "proxy_vulnerability": 0.5,
                "proxy_region_definition": "10-degree latitude/longitude bins",
                "matched_cases": len(results),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()