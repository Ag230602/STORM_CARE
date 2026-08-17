#!/usr/bin/env python3
"""Calculate AOTS2Action RQ1 calibration, sharpness, and track error."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import chi2


EARTH_RADIUS_KM = 6371.0
HORIZONS = (6, 12, 24, 48, 72, 96)
LEVELS = (0.5, 0.9)
TRACK_ALIASES = {"FUNGWONG": "FUNG-WONG", "WONG": "FUNG-WONG"}
ASSUMPTION_MARKER = "ASSUMED_NOT_ORIGINALLY_PREREGISTERED"
PRIMARY_REPRESENTATION = "P90 covariance ellipse"
PRIMARY_BASELINE = "P90 percentile cone"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).replace(tzinfo=None)


def canonical_track_id(value: str) -> str:
    track_id = value.strip().upper()
    return TRACK_ALIASES.get(track_id, track_id)


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    haversine = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))


def local_xy_km(lat: np.ndarray, lon: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    x = EARTH_RADIUS_KM * math.cos(math.radians(center_lat)) * np.radians(lon - center_lon)
    y = EARTH_RADIUS_KM * np.radians(lat - center_lat)
    return np.column_stack((x, y))


def disc_area_km2(radius_km: float) -> float:
    return 2.0 * math.pi * EARTH_RADIUS_KM**2 * (1.0 - math.cos(radius_km / EARTH_RADIUS_KM))


def load_matched_cases(path: Path) -> dict[tuple[str, datetime, int], dict[str, object]]:
    cases: dict[tuple[str, datetime, int], dict[str, object]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["cyclone_id"], parse_time(row["forecast_time"]), int(row["horizon_h"]))
            cases[key] = {
                "cyclone_id": row["cyclone_id"],
                "forecast_time": parse_time(row["forecast_time"]),
                "horizon_h": int(row["horizon_h"]),
                "observed_lat": float(row["observed_lat"]),
                "observed_lon": float(row["observed_lon"]),
            }
    return cases


def load_positions(
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


def evaluate_cases(
    matched: dict[tuple[str, datetime, int], dict[str, object]],
    positions: dict[tuple[str, datetime, int], list[tuple[float, float]]],
    fixed_radius_km: float,
    cone_buffer_km: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    fixed_area = disc_area_km2(fixed_radius_km)
    for key, case in matched.items():
        member_positions = np.asarray(positions[key], dtype=float)
        mean_lat = float(member_positions[:, 0].mean())
        mean_lon = float(member_positions[:, 1].mean())
        observed_lat = float(case["observed_lat"])
        observed_lon = float(case["observed_lon"])
        track_error = great_circle_km(mean_lat, mean_lon, observed_lat, observed_lon)
        member_distances = np.array(
            [great_circle_km(mean_lat, mean_lon, lat, lon) for lat, lon in member_positions]
        )
        base = {
            "cyclone_id": case["cyclone_id"],
            "forecast_time": case["forecast_time"].isoformat(sep=" "),
            "horizon_h": case["horizon_h"],
            "member_count": len(member_positions),
            "mean_track_error_km": track_error,
        }
        output.append(
            {
                **base,
                "representation": "Fixed-radius region",
                "nominal_level": "",
                "covered": int(track_error <= fixed_radius_km),
                "area_km2": fixed_area,
            }
        )
        for level in LEVELS:
            label = f"P{int(level * 100)}"
            cone_radius = float(np.quantile(member_distances, level)) + cone_buffer_km
            output.append(
                {
                    **base,
                    "representation": f"{label} percentile cone",
                    "nominal_level": level,
                    "covered": int(track_error <= cone_radius),
                    "area_km2": disc_area_km2(cone_radius),
                }
            )
            if len(member_positions) < 3:
                continue
            local_members = local_xy_km(
                member_positions[:, 0], member_positions[:, 1], mean_lat, mean_lon
            )
            covariance = np.cov(local_members, rowvar=False, ddof=1)
            determinant = float(np.linalg.det(covariance))
            if determinant <= 1e-12:
                continue
            observed_xy = local_xy_km(
                np.array([observed_lat]), np.array([observed_lon]), mean_lat, mean_lon
            )[0]
            mahalanobis_squared = float(observed_xy @ np.linalg.solve(covariance, observed_xy))
            chi_squared = float(chi2.ppf(level, df=2))
            output.append(
                {
                    **base,
                    "representation": f"{label} covariance ellipse",
                    "nominal_level": level,
                    "covered": int(mahalanobis_squared <= chi_squared),
                    "area_km2": math.pi * chi_squared * math.sqrt(determinant),
                }
            )
    return output


def storm_bootstrap_coverage_ci(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> tuple[float, float]:
    by_storm: dict[str, tuple[int, int]] = {}
    for cyclone_id in sorted({str(row["cyclone_id"]) for row in rows}):
        storm_rows = [row for row in rows if row["cyclone_id"] == cyclone_id]
        by_storm[cyclone_id] = (sum(int(row["covered"]) for row in storm_rows), len(storm_rows))
    counts = np.asarray(list(by_storm.values()), dtype=int)
    random = np.random.default_rng(seed)
    samples = random.integers(0, len(counts), size=(replicates, len(counts)))
    selected = counts[samples]
    bootstrap_coverage = selected[:, :, 0].sum(axis=1) / selected[:, :, 1].sum(axis=1)
    return tuple(float(value) for value in np.quantile(bootstrap_coverage, [0.025, 0.975]))


def summarize(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    representations = (
        "Fixed-radius region",
        "P50 percentile cone",
        "P90 percentile cone",
        "P50 covariance ellipse",
        "P90 covariance ellipse",
    )
    for representation in representations:
        for horizon in HORIZONS:
            selected = [
                row
                for row in rows
                if row["representation"] == representation and row["horizon_h"] == horizon
            ]
            coverage = float(np.mean([row["covered"] for row in selected]))
            ci_low, ci_high = storm_bootstrap_coverage_ci(
                selected, replicates, seed + horizon + representations.index(representation) * 100
            )
            nominal = selected[0]["nominal_level"]
            summaries.append(
                {
                    "representation": representation,
                    "horizon_h": horizon,
                    "nominal_level": nominal,
                    "empirical_coverage": coverage,
                    "coverage_ci95_low": ci_low,
                    "coverage_ci95_high": ci_high,
                    "absolute_calibration_error": (
                        "" if nominal == "" else abs(coverage - float(nominal))
                    ),
                    "mean_area_km2": float(np.mean([row["area_km2"] for row in selected])),
                    "median_area_km2": float(np.median([row["area_km2"] for row in selected])),
                    "verifying_cases": len(selected),
                }
            )
    return summaries


def summarize_track_errors(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fixed_rows = [row for row in rows if row["representation"] == "Fixed-radius region"]
    return [
        {
            "horizon_h": horizon,
            "mean_track_error_km": float(
                np.mean([row["mean_track_error_km"] for row in fixed_rows if row["horizon_h"] == horizon])
            ),
            "median_track_error_km": float(
                np.median([row["mean_track_error_km"] for row in fixed_rows if row["horizon_h"] == horizon])
            ),
            "verifying_cases": sum(row["horizon_h"] == horizon for row in fixed_rows),
        }
        for horizon in HORIZONS
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--track-output", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fixed-radius-km", type=float, default=25.0)
    parser.add_argument("--cone-buffer-km", type=float, default=25.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    matched = load_matched_cases(args.corpus)
    positions = load_positions(args.forecasts, set(matched))
    missing = set(matched) - positions.keys()
    if missing:
        raise ValueError(f"Missing ensemble positions for {len(missing)} cases")
    case_rows = evaluate_cases(
        matched, positions, args.fixed_radius_km, args.cone_buffer_km
    )
    table_rows = summarize(case_rows, args.bootstrap_replicates, args.seed)
    track_rows = summarize_track_errors(case_rows)
    proposed = next(
        row
        for row in table_rows
        if row["representation"] == PRIMARY_REPRESENTATION and row["horizon_h"] == 48
    )
    baseline = next(
        row
        for row in table_rows
        if row["representation"] == PRIMARY_BASELINE and row["horizon_h"] == 48
    )
    area_reduction = 100.0 * (
        float(baseline["mean_area_km2"]) - float(proposed["mean_area_km2"])
    ) / float(baseline["mean_area_km2"])
    endpoint = {
        "marker": ASSUMPTION_MARKER,
        "selected_p90_representation": PRIMARY_REPRESENTATION,
        "empirical_p90_coverage": proposed["empirical_coverage"],
        "coverage_ci95_low": proposed["coverage_ci95_low"],
        "coverage_ci95_high": proposed["coverage_ci95_high"],
        "mean_region_area_km2": proposed["mean_area_km2"],
        "comparison_representation": PRIMARY_BASELINE,
        "baseline_mean_area_km2": baseline["mean_area_km2"],
        "percentage_area_reduction": area_reduction,
    }
    write_csv(args.case_output, case_rows)
    write_csv(args.table_output, table_rows)
    write_csv(args.track_output, track_rows)
    write_csv(args.endpoint_output, [endpoint])
    args.metadata.write_text(
        json.dumps(
            {
                "marker": ASSUMPTION_MARKER,
                "fixed_radius_km_assumed": args.fixed_radius_km,
                "cone_base_buffer_km_assumed": args.cone_buffer_km,
                "primary_representation_fixed_before_execution": PRIMARY_REPRESENTATION,
                "primary_baseline_fixed_before_execution": PRIMARY_BASELINE,
                "coverage_ci_method": "storm-level percentile bootstrap",
                "bootstrap_replicates": args.bootstrap_replicates,
                "random_seed": args.seed,
                "ellipse_covariance": "sample covariance, ddof=1",
                "ellipse_projection": "local equirectangular centered on component-wise ensemble mean",
                "disc_area": "exact spherical-cap area",
                "matched_cases": len(matched),
                "ellipse_ineligible_cases_by_horizon": {
                    str(horizon): len(matched)
                    - sum(
                        row["representation"] == "P90 covariance ellipse"
                        and row["horizon_h"] == horizon
                        for row in case_rows
                    )
                    - sum(case["horizon_h"] != horizon for case in matched.values())
                    for horizon in HORIZONS
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()