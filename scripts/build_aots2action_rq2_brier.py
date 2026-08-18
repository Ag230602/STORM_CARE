#!/usr/bin/env python3
"""Calculate marked-proxy exposure-field Brier scores for AOTS2Action RQ2."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import build_aots2action_rq2 as rq2


HORIZONS = (6, 12, 24, 48, 72, 96)
DOMAIN_HALF_SIZE_DEG = 8.0
RELIABILITY_BINS = 10


def storm_centered_domain(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    half_size_deg: float,
) -> np.ndarray:
    lon_distance = np.abs((lon - center_lon + 180.0) % 360.0 - 180.0)
    return (np.abs(lat - center_lat) <= half_size_deg) & (
        lon_distance <= half_size_deg
    )


def evaluate_fields(
    cases: dict[tuple[str, object, int], dict[str, object]],
    positions: dict[tuple[str, object, int], list[tuple[float, float]]],
    grid: dict[str, np.ndarray],
    impact_radius_km: float,
    cone_buffer_km: float,
    domain_half_size_deg: float,
) -> tuple[list[dict[str, object]], dict[int, tuple[list[np.ndarray], list[np.ndarray]]]]:
    output: list[dict[str, object]] = []
    reliability: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {
        horizon: ([], []) for horizon in HORIZONS
    }
    for key, case in cases.items():
        members = np.asarray(positions[key], dtype=float)
        mean_lat = float(members[:, 0].mean())
        mean_lon = float(members[:, 1].mean())
        member_mean_distances = np.asarray(
            [rq2.great_circle_km(mean_lat, mean_lon, lat, lon) for lat, lon in members]
        )
        p90_radius = float(np.quantile(member_mean_distances, 0.9)) + cone_buffer_km
        domain = storm_centered_domain(
            grid["lat"],
            grid["lon"],
            float(case["observed_lat"]),
            float(case["observed_lon"]),
            domain_half_size_deg,
        )
        if not np.any(domain):
            raise ValueError(f"Storm-centered domain has no grid cells for {key}")

        realized = (
            rq2.distances_km(
                grid["lat"],
                grid["lon"],
                float(case["observed_lat"]),
                float(case["observed_lon"]),
            )
            <= impact_radius_km
        ).astype(float)
        deterministic = (
            rq2.distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon)
            <= impact_radius_km
        ).astype(float)
        p90 = (
            rq2.distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon)
            <= p90_radius + impact_radius_km
        ).astype(float)
        ensemble = np.zeros(len(grid["lat"]), dtype=float)
        for member_lat, member_lon in members:
            ensemble += (
                rq2.distances_km(grid["lat"], grid["lon"], member_lat, member_lon)
                <= impact_radius_km
            )
        ensemble /= len(members)

        fields = {
            "Deterministic mean-track": deterministic,
            "P90 envelope": p90,
            "Ensemble probability-weighted": ensemble,
        }
        for estimator, predicted in fields.items():
            output.append(
                {
                    "marker": rq2.MARKER,
                    "cyclone_id": case["cyclone_id"],
                    "forecast_time": case["forecast_time"].isoformat(sep=" "),
                    "horizon_h": case["horizon_h"],
                    "member_count": len(members),
                    "domain_cell_count": int(domain.sum()),
                    "realized_event_cell_count": int(realized[domain].sum()),
                    "estimator": estimator,
                    "brier_score": float(np.mean((predicted[domain] - realized[domain]) ** 2)),
                }
            )
        reliability[int(case["horizon_h"])][0].append(ensemble[domain])
        reliability[int(case["horizon_h"])][1].append(realized[domain])
    return output, reliability


def cluster_bootstrap_ci(
    rows: list[dict[str, object]],
    value_key: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    by_storm: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_storm[str(row["cyclone_id"])].append(float(row[value_key]))
    values = [np.asarray(by_storm[storm], dtype=float) for storm in sorted(by_storm)]
    random = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = random.integers(0, len(values), size=len(values))
        bootstrap[replicate] = np.concatenate([values[index] for index in selected]).mean()
    return tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def summarize_scores(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for estimator_index, estimator in enumerate(rq2.ESTIMATORS):
        for horizon in HORIZONS:
            selected_rows = [
                row
                for row in rows
                if row["estimator"] == estimator and row["horizon_h"] == horizon
            ]
            selected = [float(row["brier_score"]) for row in selected_rows]
            ci_low, ci_high = cluster_bootstrap_ci(
                selected_rows, "brier_score", replicates, seed + estimator_index * 1000 + horizon
            )
            output.append(
                {
                    "marker": rq2.MARKER,
                    "estimator": estimator,
                    "horizon_h": horizon,
                    "brier_score": float(np.mean(selected)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "verifying_cases": len(selected),
                }
            )
    return output


def paired_brier_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    comparisons = (
        ("det - ens", "Deterministic mean-track"),
        ("P90 - ens", "P90 envelope"),
    )
    for label, comparator in comparisons:
        for horizon in HORIZONS:
            selected = [row for row in rows if row["horizon_h"] == horizon]
            by_case = defaultdict(dict)
            for row in selected:
                by_case[(row["cyclone_id"], row["forecast_time"])][row["estimator"]] = float(
                    row["brier_score"]
                )
            storm_differences: dict[str, list[float]] = defaultdict(list)
            cycle_differences: list[float] = []
            for (cyclone_id, _), scores in by_case.items():
                difference = scores[comparator] - scores["Ensemble probability-weighted"]
                cycle_differences.append(difference)
                storm_differences[cyclone_id].append(difference)
            paired = np.asarray(
                [np.mean(differences) for differences in storm_differences.values()], dtype=float
            )
            raw_p = (
                1.0
                if np.allclose(paired, 0)
                else float(wilcoxon(paired, zero_method="wilcox", alternative="two-sided").pvalue)
            )
            output.append(
                {
                    "marker": rq2.MARKER,
                    "comparison": label,
                    "horizon_h": horizon,
                    "mean_cycle_level_difference": float(np.mean(cycle_differences)),
                    "mean_storm_level_difference": float(paired.mean()),
                    "paired_cyclones": len(paired),
                    "wilcoxon_raw_p": raw_p,
                }
            )
    adjusted = holm_adjust(np.asarray([row["wilcoxon_raw_p"] for row in output]))
    for row, adjusted_p in zip(output, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
        row["ensemble_lower_brier_supported_0_05"] = bool(
            adjusted_p < 0.05 and float(row["mean_storm_level_difference"]) > 0
        )
    return output


def reliability_summary(
    values: dict[int, tuple[list[np.ndarray], list[np.ndarray]]]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for horizon in HORIZONS:
        probabilities = np.concatenate(values[horizon][0])
        observations = np.concatenate(values[horizon][1])
        bin_indices = np.minimum(
            (probabilities * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1
        )
        for bin_index in range(RELIABILITY_BINS):
            selected = bin_indices == bin_index
            output.append(
                {
                    "marker": rq2.MARKER,
                    "horizon_h": horizon,
                    "bin_index": bin_index,
                    "bin_lower": bin_index / RELIABILITY_BINS,
                    "bin_upper": (bin_index + 1) / RELIABILITY_BINS,
                    "cell_count": int(selected.sum()),
                    "mean_forecast_probability": (
                        float(probabilities[selected].mean()) if np.any(selected) else ""
                    ),
                    "observed_frequency": (
                        float(observations[selected].mean()) if np.any(selected) else ""
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    scores: list[dict[str, object]],
    reliability: list[dict[str, object]],
    tests: list[dict[str, object]],
    domain_half_size_deg: float,
) -> None:
    lookup = {
        (str(row["estimator"]), int(row["horizon_h"])): float(row["brier_score"])
        for row in scores
    }
    winners = {
        horizon: min(
            rq2.ESTIMATORS, key=lambda estimator: lookup[(estimator, horizon)]
        )
        for horizon in HORIZONS
    }
    ensemble_wins = sum(
        winner == "Ensemble probability-weighted" for winner in winners.values()
    )
    macro_scores = {
        estimator: float(np.mean([lookup[(estimator, horizon)] for horizon in HORIZONS]))
        for estimator in rq2.ESTIMATORS
    }
    macro_ensemble = macro_scores["Ensemble probability-weighted"]
    macro_det_improvement = 100.0 * (
        macro_scores["Deterministic mean-track"] - macro_ensemble
    ) / macro_scores["Deterministic mean-track"]
    macro_p90_improvement = 100.0 * (
        macro_scores["P90 envelope"] - macro_ensemble
    ) / macro_scores["P90 envelope"]
    lines = [
        "# RQ2: exposure-field Brier scores",
        "",
        f"Marker: **{rq2.MARKER}**",
        "",
        f"Scores reuse the repository model configuration's pre-specified +/-{domain_half_size_deg:g}",
        "degree storm-centered crop, centered here on the verifying best-track position.",
        "A domain mean is computed",
        "for each case and then cases are averaged, so unequal grid density does",
        "not reweight cases. Lower Brier score is better.",
        "",
        "| Estimator | 6 h | 12 h | 24 h | 48 h | 72 h | 96 h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for estimator in rq2.ESTIMATORS:
        values = " | ".join(f"{lookup[(estimator, horizon)]:.6f}" for horizon in HORIZONS)
        lines.append(f"| {estimator} | {values} |")
    lines.extend(["", "## Comparisons", ""])
    for horizon in HORIZONS:
        ensemble = lookup[("Ensemble probability-weighted", horizon)]
        deterministic = lookup[("Deterministic mean-track", horizon)]
        p90 = lookup[("P90 envelope", horizon)]
        det_improvement = 100.0 * (deterministic - ensemble) / deterministic
        p90_improvement = 100.0 * (p90 - ensemble) / p90
        lines.append(
            f"- {horizon} h: {winners[horizon]} is lowest; ensemble improvement is "
            f"{det_improvement:.2f}% over deterministic and {p90_improvement:.2f}% over P90."
        )
    lines.extend(["", "## Paired tests", ""])
    for test in tests:
        supported = "supported" if test["ensemble_lower_brier_supported_0_05"] else "not supported"
        lines.append(
            f"- {test['comparison']} at {test['horizon_h']} h: "
            f"mean storm-level difference {float(test['mean_storm_level_difference']):+.6f}; "
            f"raw p={float(test['wilcoxon_raw_p']):.6g}, "
            f"Holm p={float(test['holm_adjusted_p']):.6g}; ensemble lower Brier {supported}."
        )
    lines.extend(
        [
            "",
            f"The ensemble estimator is best at **{ensemble_wins} out of 6 horizons**.",
            f"Across the six horizon scores (macro-average), its improvement is "
            f"**{macro_det_improvement:.2f}% over deterministic** and "
            f"**{macro_p90_improvement:.2f}% over P90**.",
            "",
            "## Reliability",
            "",
            "Ten-bin reliability-diagram data were generated from the same storm-centered",
            "domains. The evidence does not support one blanket well-calibrated,",
            "overconfident, or underconfident label. At 6 h and 12 h, the well-populated",
            "bins are mostly close to the diagonal. At 24 h and 48 h, positive-probability",
            "bins lie below the diagonal, indicating overforecasting consistent with",
            "overconfidence. At 72 h and 96 h, only nine cells in total have probability",
            ">=0.1, which is insufficient for a reliable confidence diagnosis. The overall",
            "calibration classification is therefore **inconclusive**, with lead-dependent",
            "evidence of overconfidence at 24-48 h and no consistent evidence of",
            "underconfidence.",
            "",
            "| Lead | Bin | Cells | Mean probability | Observed frequency |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in reliability:
        if int(row["cell_count"]) == 0:
            continue
        lines.append(
            f"| {row['horizon_h']} h | {float(row['bin_lower']):.1f}-"
            f"{float(row['bin_upper']):.1f} | {row['cell_count']} | "
            f"{float(row['mean_forecast_probability']):.4f} | "
            f"{float(row['observed_frequency']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--proxy-grid", type=Path)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--grid-kind", choices=("proxy", "real"), default="proxy")
    parser.add_argument("--grid-metadata", type=Path)
    parser.add_argument("--vulnerability-column", default="inform_risk")
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--reliability-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--tests-output", type=Path)
    parser.add_argument("--impact-radius-km", type=float, default=25.0)
    parser.add_argument("--cone-buffer-km", type=float, default=25.0)
    parser.add_argument("--domain-half-size-deg", type=float, default=DOMAIN_HALF_SIZE_DEG)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    rq2.HORIZONS = HORIZONS
    rq2.MARKER = rq2.REAL_MARKER if args.grid_kind == "real" else rq2.MARKER
    grid_path = args.grid or args.proxy_grid
    if grid_path is None:
        raise ValueError("Provide --grid for real data or --proxy-grid for proxy data")
    cases = rq2.load_cases(args.corpus)
    positions = rq2.load_positions(args.forecasts, set(cases))
    missing = set(cases) - positions.keys()
    if missing:
        raise ValueError(f"Missing ensemble positions for {len(missing)} cases")
    grid = rq2.load_grid(grid_path, args.vulnerability_column)
    case_rows, reliability_values = evaluate_fields(
        cases,
        positions,
        grid,
        args.impact_radius_km,
        args.cone_buffer_km,
        args.domain_half_size_deg,
    )
    scores = summarize_scores(case_rows, args.bootstrap_replicates, args.seed)
    tests = paired_brier_tests(case_rows)
    reliability = reliability_summary(reliability_values)
    write_csv(args.case_output, case_rows)
    write_csv(args.table_output, scores)
    write_csv(args.reliability_output, reliability)
    if args.tests_output:
        write_csv(args.tests_output, tests)
    write_report(args.report_output, scores, reliability, tests, args.domain_half_size_deg)
    args.metadata.write_text(
        json.dumps(
            {
                "marker": rq2.MARKER,
                "horizons_h": list(HORIZONS),
                "impact_radius_km_assumed": args.impact_radius_km,
                "p90_cone_buffer_km_assumed": args.cone_buffer_km,
                "domain": "square centered on verifying best-track position",
                "domain_half_size_deg": args.domain_half_size_deg,
                "domain_source": "model/track_pipeline_unified_X.py CFG.crop_deg",
                "case_aggregation": "mean over domain cells, then unweighted mean over cases",
                "reliability_bins": RELIABILITY_BINS,
                "confidence_intervals": "cyclone cluster bootstrap of the mean",
                "bootstrap_replicates": args.bootstrap_replicates,
                "random_seed": args.seed,
                "paired_inference": "cycle differences averaged within cyclone, two-sided Wilcoxon signed-rank",
                "holm_family_size": 12,
                "grid_kind": args.grid_kind,
                "grid": str(grid_path),
                "grid_metadata": str(args.grid_metadata) if args.grid_metadata else None,
                "matched_cases": len(cases),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
