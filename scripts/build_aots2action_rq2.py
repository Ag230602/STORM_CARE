#!/usr/bin/env python3
"""Calculate marked-proxy AOTS2Action RQ2 exposure-estimation fidelity."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


EARTH_RADIUS_KM = 6371.0
HORIZONS = (24, 48, 72)
ESTIMATORS = ("Deterministic mean-track", "P90 envelope", "Ensemble probability-weighted")
TRACK_ALIASES = {"FUNGWONG": "FUNG-WONG", "WONG": "FUNG-WONG"}
MARKER = "PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).replace(tzinfo=None)


def canonical_track_id(value: str) -> str:
    track_id = value.strip().upper()
    return TRACK_ALIASES.get(track_id, track_id)


def distances_km(lat: np.ndarray, lon: np.ndarray, point_lat: float, point_lon: float) -> np.ndarray:
    lat_radians = np.radians(lat)
    point_lat_radians = math.radians(point_lat)
    delta_lat = np.radians(point_lat - lat)
    delta_lon = np.radians(point_lon - lon)
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_radians)
        * math.cos(point_lat_radians)
        * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(distances_km(np.array([lat1]), np.array([lon1]), lat2, lon2)[0])


def load_cases(path: Path) -> dict[tuple[str, datetime, int], dict[str, object]]:
    cases: dict[tuple[str, datetime, int], dict[str, object]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            horizon = int(row["horizon_h"])
            if horizon not in HORIZONS:
                continue
            key = (row["cyclone_id"], parse_time(row["forecast_time"]), horizon)
            cases[key] = {
                "cyclone_id": row["cyclone_id"],
                "forecast_time": parse_time(row["forecast_time"]),
                "horizon_h": horizon,
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


def load_grid(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        "lat": np.array([float(row["lat"]) for row in rows]),
        "lon": np.array([float(row["lon"]) for row in rows]),
        "weight": np.array(
            [float(row["population"]) * float(row["inform_risk"]) for row in rows]
        ),
    }


def evaluate_cases(
    cases: dict[tuple[str, datetime, int], dict[str, object]],
    positions: dict[tuple[str, datetime, int], list[tuple[float, float]]],
    grid: dict[str, np.ndarray],
    impact_radius_km: float,
    cone_buffer_km: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, case in cases.items():
        members = np.asarray(positions[key], dtype=float)
        mean_lat = float(members[:, 0].mean())
        mean_lon = float(members[:, 1].mean())
        member_mean_distances = np.array(
            [great_circle_km(mean_lat, mean_lon, lat, lon) for lat, lon in members]
        )
        p90_radius = float(np.quantile(member_mean_distances, 0.9)) + cone_buffer_km
        realized_mask = distances_km(
            grid["lat"], grid["lon"], float(case["observed_lat"]), float(case["observed_lon"])
        ) <= impact_radius_km
        deterministic_mask = distances_km(
            grid["lat"], grid["lon"], mean_lat, mean_lon
        ) <= impact_radius_km
        p90_mask = distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon) <= (
            p90_radius + impact_radius_km
        )
        ensemble_probability = np.zeros(len(grid["lat"]), dtype=float)
        for member_lat, member_lon in members:
            ensemble_probability += (
                distances_km(grid["lat"], grid["lon"], member_lat, member_lon)
                <= impact_radius_km
            )
        ensemble_probability /= len(members)

        realized = float(grid["weight"][realized_mask].sum())
        predictions = {
            "Deterministic mean-track": float(grid["weight"][deterministic_mask].sum()),
            "P90 envelope": float(grid["weight"][p90_mask].sum()),
            "Ensemble probability-weighted": float(
                np.sum(grid["weight"] * ensemble_probability)
            ),
        }
        for estimator, predicted in predictions.items():
            output.append(
                {
                    "marker": MARKER,
                    "cyclone_id": case["cyclone_id"],
                    "forecast_time": case["forecast_time"].isoformat(sep=" "),
                    "horizon_h": case["horizon_h"],
                    "member_count": len(members),
                    "estimator": estimator,
                    "predicted_vw_exposure": predicted,
                    "realized_vw_exposure": realized,
                    "absolute_error": abs(predicted - realized),
                }
            )
    return output


def cluster_bootstrap_ci(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> tuple[float, float]:
    storms = sorted({str(row["cyclone_id"]) for row in rows})
    values = [
        np.array([float(row["absolute_error"]) for row in rows if row["cyclone_id"] == storm])
        for storm in storms
    ]
    random = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = random.integers(0, len(storms), size=len(storms))
        bootstrap[replicate] = np.concatenate([values[index] for index in selected]).mean()
    return tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))


def estimator_summary(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for estimator_index, estimator in enumerate(ESTIMATORS):
        for horizon in HORIZONS:
            selected = [
                row
                for row in rows
                if row["estimator"] == estimator and row["horizon_h"] == horizon
            ]
            ci_low, ci_high = cluster_bootstrap_ci(
                selected, replicates, seed + estimator_index * 100 + horizon
            )
            output.append(
                {
                    "marker": MARKER,
                    "estimator": estimator,
                    "horizon_h": horizon,
                    "mean_absolute_error": float(
                        np.mean([float(row["absolute_error"]) for row in selected])
                    ),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "verifying_cases": len(selected),
                    "cyclones": len({row["cyclone_id"] for row in selected}),
                }
            )
    return output


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = (
        ("det - ens", "Deterministic mean-track"),
        ("P90 - ens", "P90 envelope"),
    )
    output: list[dict[str, object]] = []
    for label, comparator in comparisons:
        for horizon in HORIZONS:
            selected = [row for row in rows if row["horizon_h"] == horizon]
            by_case = defaultdict(dict)
            for row in selected:
                by_case[(row["cyclone_id"], row["forecast_time"])][row["estimator"]] = float(
                    row["absolute_error"]
                )
            storm_differences: dict[str, list[float]] = defaultdict(list)
            for (cyclone_id, _), errors in by_case.items():
                storm_differences[cyclone_id].append(
                    errors[comparator] - errors["Ensemble probability-weighted"]
                )
            paired = np.array(
                [np.mean(differences) for differences in storm_differences.values()], dtype=float
            )
            raw_p = (
                1.0
                if np.allclose(paired, 0)
                else float(wilcoxon(paired, zero_method="wilcox", alternative="two-sided").pvalue)
            )
            output.append(
                {
                    "marker": MARKER,
                    "comparison": label,
                    "horizon_h": horizon,
                    "mean_cycle_level_difference": float(
                        np.mean(
                            [
                                errors[comparator] - errors["Ensemble probability-weighted"]
                                for errors in by_case.values()
                            ]
                        )
                    ),
                    "mean_storm_level_difference": float(paired.mean()),
                    "paired_cyclones": len(paired),
                    "wilcoxon_raw_p": raw_p,
                }
            )
    adjusted = holm_adjust(np.array([row["wilcoxon_raw_p"] for row in output]))
    for row, adjusted_p in zip(output, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
        row["significant_after_holm_0_05"] = bool(adjusted_p < 0.05)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_table_ii(
    path: Path,
    estimators: dict[tuple[str, int], dict[str, object]],
    pairs: dict[tuple[str, int], dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["marker", "Result", "24 h", "48 h", "72 h"])
        for label, estimator in (
            ("Deterministic mean-track AE", "Deterministic mean-track"),
            ("P90-envelope AE", "P90 envelope"),
            ("Ensemble probability-weighted AE", "Ensemble probability-weighted"),
        ):
            values = []
            for horizon in HORIZONS:
                row = estimators[(estimator, horizon)]
                values.append(
                    f"{row['mean_absolute_error']:.6f} [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"
                )
            writer.writerow([MARKER, label, *values])
        for comparison in ("det - ens", "P90 - ens"):
            writer.writerow(
                [MARKER, comparison]
                + [f"{pairs[(comparison, horizon)]['mean_cycle_level_difference']:.6f}" for horizon in HORIZONS]
            )
            writer.writerow(
                [MARKER, "Holm-adjusted p-value"]
                + [f"{pairs[(comparison, horizon)]['holm_adjusted_p']:.12g}" for horizon in HORIZONS]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--proxy-grid", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--estimator-output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--improvement-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--impact-radius-km", type=float, default=25.0)
    parser.add_argument("--cone-buffer-km", type=float, default=25.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    cases = load_cases(args.corpus)
    positions = load_positions(args.forecasts, set(cases))
    missing = set(cases) - positions.keys()
    if missing:
        raise ValueError(f"Missing ensemble positions for {len(missing)} cases")
    grid = load_grid(args.proxy_grid)
    case_rows = evaluate_cases(
        cases, positions, grid, args.impact_radius_km, args.cone_buffer_km
    )
    estimator_rows = estimator_summary(
        case_rows, args.bootstrap_replicates, args.seed
    )
    estimator_lookup = {
        (row["estimator"], row["horizon_h"]): row for row in estimator_rows
    }
    paired_rows = paired_summary(case_rows)
    pair_lookup = {(row["comparison"], row["horizon_h"]): row for row in paired_rows}
    improvement_rows = []
    for horizon in HORIZONS:
        deterministic = float(
            estimator_lookup[("Deterministic mean-track", horizon)]["mean_absolute_error"]
        )
        p90 = float(estimator_lookup[("P90 envelope", horizon)]["mean_absolute_error"])
        ensemble = float(
            estimator_lookup[("Ensemble probability-weighted", horizon)]["mean_absolute_error"]
        )
        improvement_rows.append(
            {
                "marker": MARKER,
                "horizon_h": horizon,
                "improvement_vs_deterministic_percent": 100.0 * (deterministic - ensemble) / deterministic,
                "improvement_vs_p90_percent": 100.0 * (p90 - ensemble) / p90,
                "det_vs_ens_holm_p": pair_lookup[("det - ens", horizon)]["holm_adjusted_p"],
                "det_vs_ens_significant": pair_lookup[("det - ens", horizon)]["significant_after_holm_0_05"],
                "p90_vs_ens_holm_p": pair_lookup[("P90 - ens", horizon)]["holm_adjusted_p"],
                "p90_vs_ens_significant": pair_lookup[("P90 - ens", horizon)]["significant_after_holm_0_05"],
            }
        )
    write_csv(args.case_output, case_rows)
    write_csv(args.estimator_output, estimator_rows)
    write_csv(args.paired_output, paired_rows)
    write_table_ii(args.table_output, estimator_lookup, pair_lookup)
    write_csv(args.improvement_output, improvement_rows)
    args.metadata.write_text(
        json.dumps(
            {
                "marker": MARKER,
                "impact_radius_km_assumed": args.impact_radius_km,
                "p90_cone_buffer_km_assumed": args.cone_buffer_km,
                "proxy_grid": str(args.proxy_grid),
                "vulnerability_weight": "population * inform_risk; proxy inform_risk=0.5",
                "p90_envelope_radius": "empirical P90 member-to-mean distance + cone buffer + impact radius",
                "bootstrap": "cyclone cluster bootstrap; sampled cyclone contributes all cycles",
                "bootstrap_replicates": args.bootstrap_replicates,
                "random_seed": args.seed,
                "paired_inference": "cycle differences averaged within cyclone, two-sided Wilcoxon signed-rank",
                "holm_family_size": 6,
                "primary_horizons_h": list(HORIZONS),
                "matched_cases": len(cases),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()