#!/usr/bin/env python3
"""Calculate marked-proxy AOTS2Action RQ3 regional ranking performance."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

import build_aots2action_rq2 as rq2


HORIZONS = (24, 48, 72)
METRICS = (
    ("nDCG@10", "ndcg_at_10"),
    ("nDCG@5", "ndcg_at_5"),
    ("Recall@10", "recall_at_10"),
    ("Recall@5", "recall_at_5"),
    ("Spearman", "spearman"),
)


def load_regional_grid(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    region_ids = np.asarray([row["region_id"] for row in rows], dtype=object)
    unique_regions = np.asarray(sorted(set(region_ids)), dtype=object)
    region_lookup = {region: index for index, region in enumerate(unique_regions)}
    return {
        "lat": np.asarray([float(row["lat"]) for row in rows]),
        "lon": np.asarray([float(row["lon"]) for row in rows]),
        "weight": np.asarray(
            [float(row["population"]) * float(row["inform_risk"]) for row in rows]
        ),
        "region_ids": unique_regions,
        "region_index": np.asarray([region_lookup[region] for region in region_ids]),
    }


def aggregate_regions(grid: dict[str, np.ndarray], field: np.ndarray) -> np.ndarray:
    return np.bincount(
        grid["region_index"],
        weights=grid["weight"] * field,
        minlength=len(grid["region_ids"]),
    )


def positive_ranking(scores: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    positive = np.flatnonzero(scores > 0)
    return np.asarray(
        sorted(positive, key=lambda index: (-scores[index], str(region_ids[index]))),
        dtype=int,
    )


def ndcg_at_k(
    predicted: np.ndarray, realized: np.ndarray, region_ids: np.ndarray, k: int
) -> float:
    ideal_order = positive_ranking(realized, region_ids)[:k]
    if len(ideal_order) == 0:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    ideal = float(np.sum(realized[ideal_order] * discounts[: len(ideal_order)]))
    predicted_order = positive_ranking(predicted, region_ids)[:k]
    actual = float(
        np.sum(realized[predicted_order] * discounts[: len(predicted_order)])
    )
    return actual / ideal


def recall_at_k(
    predicted: np.ndarray, realized: np.ndarray, region_ids: np.ndarray, k: int
) -> tuple[float, int, int]:
    realized_top = positive_ranking(realized, region_ids)[:k]
    if len(realized_top) == 0:
        return float("nan"), 0, 0
    predicted_top = positive_ranking(predicted, region_ids)[:k]
    overlap = len(set(realized_top) & set(predicted_top))
    return overlap / len(realized_top), overlap, len(realized_top)


def spearman_nonzero_union(predicted: np.ndarray, realized: np.ndarray) -> tuple[float, int]:
    union = (predicted > 0) | (realized > 0)
    union_size = int(union.sum())
    if union_size < 3:
        return float("nan"), union_size
    predicted_union = predicted[union]
    realized_union = realized[union]
    if np.all(predicted_union == predicted_union[0]) or np.all(
        realized_union == realized_union[0]
    ):
        return float("nan"), union_size
    return float(spearmanr(predicted_union, realized_union).statistic), union_size


def evaluate_cases(
    cases: dict[tuple[str, object, int], dict[str, object]],
    positions: dict[tuple[str, object, int], list[tuple[float, float]]],
    grid: dict[str, np.ndarray],
    impact_radius_km: float,
    cone_buffer_km: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, case in cases.items():
        members = np.asarray(positions[key], dtype=float)
        mean_lat = float(members[:, 0].mean())
        mean_lon = float(members[:, 1].mean())
        spread = np.asarray(
            [rq2.great_circle_km(mean_lat, mean_lon, lat, lon) for lat, lon in members]
        )
        p90_radius = float(np.quantile(spread, 0.9)) + cone_buffer_km
        realized_field = (
            rq2.distances_km(
                grid["lat"],
                grid["lon"],
                float(case["observed_lat"]),
                float(case["observed_lon"]),
            )
            <= impact_radius_km
        ).astype(float)
        deterministic_field = (
            rq2.distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon)
            <= impact_radius_km
        ).astype(float)
        p90_field = (
            rq2.distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon)
            <= p90_radius + impact_radius_km
        ).astype(float)
        ensemble_field = np.zeros(len(grid["lat"]), dtype=float)
        for member_lat, member_lon in members:
            ensemble_field += (
                rq2.distances_km(grid["lat"], grid["lon"], member_lat, member_lon)
                <= impact_radius_km
            )
        ensemble_field /= len(members)

        realized = aggregate_regions(grid, realized_field)
        predicted_fields = {
            "Deterministic mean-track": deterministic_field,
            "P90 envelope": p90_field,
            "Ensemble probability-weighted": ensemble_field,
        }
        for estimator, field in predicted_fields.items():
            predicted = aggregate_regions(grid, field)
            recall_10, overlap_10, target_10 = recall_at_k(
                predicted, realized, grid["region_ids"], 10
            )
            recall_5, overlap_5, target_5 = recall_at_k(
                predicted, realized, grid["region_ids"], 5
            )
            spearman, union_size = spearman_nonzero_union(predicted, realized)
            output.append(
                {
                    "marker": rq2.MARKER,
                    "cyclone_id": case["cyclone_id"],
                    "forecast_time": case["forecast_time"].isoformat(sep=" "),
                    "horizon_h": case["horizon_h"],
                    "estimator": estimator,
                    "realized_positive_regions": int((realized > 0).sum()),
                    "predicted_positive_regions": int((predicted > 0).sum()),
                    "nonzero_union_regions": union_size,
                    "ndcg_at_10": ndcg_at_k(predicted, realized, grid["region_ids"], 10),
                    "ndcg_at_5": ndcg_at_k(predicted, realized, grid["region_ids"], 5),
                    "recall_at_10": recall_10,
                    "recall_at_5": recall_5,
                    "recall_at_10_overlap": overlap_10,
                    "recall_at_10_target": target_10,
                    "recall_at_5_overlap": overlap_5,
                    "recall_at_5_target": target_5,
                    "spearman": spearman,
                }
            )
    return output


def storm_values(
    rows: list[dict[str, object]], estimator: str, horizon: int, metric: str
) -> dict[str, float]:
    by_storm: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if (
            row["estimator"] == estimator
            and row["horizon_h"] == horizon
            and np.isfinite(value)
        ):
            by_storm[str(row["cyclone_id"])].append(value)
    return {storm: float(np.mean(values)) for storm, values in by_storm.items()}


def bootstrap_mean_ci(
    values: dict[str, float], replicates: int, seed: int
) -> tuple[float, float, float]:
    ordered = np.asarray([values[storm] for storm in sorted(values)], dtype=float)
    random = np.random.default_rng(seed)
    indices = random.integers(0, len(ordered), size=(replicates, len(ordered)))
    bootstrap = ordered[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(ordered.mean()), float(low), float(high)


def summarize_metrics(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric_index, (metric_label, metric) in enumerate(METRICS):
        for estimator_index, estimator in enumerate(rq2.ESTIMATORS):
            for horizon in HORIZONS:
                values = storm_values(rows, estimator, horizon, metric)
                mean, low, high = bootstrap_mean_ci(
                    values,
                    replicates,
                    seed + metric_index * 1000 + estimator_index * 100 + horizon,
                )
                output.append(
                    {
                        "marker": rq2.MARKER,
                        "metric": metric_label,
                        "metric_key": metric,
                        "estimator": estimator,
                        "horizon_h": horizon,
                        "storm_level_mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "contributing_storms": len(values),
                        "contributing_cases": sum(
                            row["estimator"] == estimator
                            and row["horizon_h"] == horizon
                            and np.isfinite(float(row[metric]))
                            for row in rows
                        ),
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


def headline_tests(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> list[dict[str, object]]:
    ensemble = storm_values(rows, "Ensemble probability-weighted", 48, "ndcg_at_10")
    output: list[dict[str, object]] = []
    for index, comparator in enumerate(("Deterministic mean-track", "P90 envelope")):
        baseline = storm_values(rows, comparator, 48, "ndcg_at_10")
        storms = sorted(set(ensemble) & set(baseline))
        differences = np.asarray(
            [ensemble[storm] - baseline[storm] for storm in storms], dtype=float
        )
        difference_values = dict(zip(storms, differences))
        mean, low, high = bootstrap_mean_ci(
            difference_values, replicates, seed + 20_000 + index
        )
        raw_p = (
            1.0
            if np.allclose(differences, 0)
            else float(
                wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue
            )
        )
        output.append(
            {
                "marker": rq2.MARKER,
                "comparison": f"Ensemble - {comparator}",
                "mean_storm_level_difference": mean,
                "difference_ci95_low": low,
                "difference_ci95_high": high,
                "paired_storms": len(storms),
                "wilcoxon_raw_p": raw_p,
            }
        )
    adjusted = holm_adjust(np.asarray([row["wilcoxon_raw_p"] for row in output]))
    for row, adjusted_p in zip(output, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
        row["superiority_supported"] = bool(
            adjusted_p < 0.05 and float(row["difference_ci95_low"]) > 0
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
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    tests: list[dict[str, object]],
) -> None:
    lookup = {
        (str(row["metric"]), str(row["estimator"]), int(row["horizon_h"])): row
        for row in summary
    }
    lines = [
        "# RQ3: regional-prioritization performance",
        "",
        f"Marker: **{rq2.MARKER}**",
        "",
        "Values are equal-weight storm means with 95% cyclone-bootstrap confidence",
        "intervals (10,000 replicates; seed 20260817). Regions are proxy 10-degree",
        "latitude/longitude bins. nDCG uses linear realized vulnerability-weighted",
        "exposure gain. Zero-realized cases are excluded from nDCG and recall.",
        "Spearman uses only the estimator-specific nonzero union and requires at least",
        "three regions.",
        "",
        "| Metric | Estimator | 24 h | 48 h | 72 h |",
        "|---|---|---:|---:|---:|",
    ]
    for metric, _ in METRICS:
        for estimator in rq2.ESTIMATORS:
            cells = []
            for horizon in HORIZONS:
                row = lookup[(metric, estimator, horizon)]
                cells.append(
                    f"{float(row['storm_level_mean']):.4f} "
                    f"[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}]"
                )
            lines.append(f"| {metric} | {estimator} | " + " | ".join(cells) + " |")

    ensemble = lookup[("nDCG@10", "Ensemble probability-weighted", 48)]
    deterministic = lookup[("nDCG@10", "Deterministic mean-track", 48)]
    p90 = lookup[("nDCG@10", "P90 envelope", 48)]
    recall = lookup[("Recall@10", "Ensemble probability-weighted", 48)]
    eligible_recall = [
        row
        for row in rows
        if row["estimator"] == "Ensemble probability-weighted"
        and row["horizon_h"] == 48
        and int(row["recall_at_10_target"]) > 0
    ]
    total_overlap = sum(int(row["recall_at_10_overlap"]) for row in eligible_recall)
    total_target = sum(int(row["recall_at_10_target"]) for row in eligible_recall)
    lines.extend(
        [
            "",
            "## Headline: nDCG@10 at 48 h",
            "",
            f"- Ensemble: **{float(ensemble['storm_level_mean']):.4f}** "
            f"(95% CI {float(ensemble['ci95_low']):.4f}-"
            f"{float(ensemble['ci95_high']):.4f})",
            f"- Deterministic: {float(deterministic['storm_level_mean']):.4f}",
            f"- P90 envelope: {float(p90['storm_level_mean']):.4f}",
        ]
    )
    for test in tests:
        supported = "supported" if test["superiority_supported"] else "not supported"
        lines.append(
            f"- {test['comparison']}: {float(test['mean_storm_level_difference']):+.4f} "
            f"(95% CI {float(test['difference_ci95_low']):+.4f} to "
            f"{float(test['difference_ci95_high']):+.4f}); raw p="
            f"{float(test['wilcoxon_raw_p']):.6g}, Holm p="
            f"{float(test['holm_adjusted_p']):.6g}; superiority {supported}."
        )
    lines.extend(
        [
            f"- Ensemble Recall@10: {float(recall['storm_level_mean']):.4f} "
            f"(95% CI {float(recall['ci95_low']):.4f}-"
            f"{float(recall['ci95_high']):.4f})",
            f"- Identified realized top-10 regions: {total_overlap}/{total_target} "
            f"({total_overlap / total_target:.2%}) across {len(eligible_recall)} eligible cases.",
            "",
            "Because the 25 km footprint and coarse proxy regions usually produce fewer",
            "than ten positive realized regions, the overlap denominator is the number of",
            "available positive regions up to ten, not ten artificial zero-relevance ties.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--proxy-grid", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--tests-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--impact-radius-km", type=float, default=25.0)
    parser.add_argument("--cone-buffer-km", type=float, default=25.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    rq2.HORIZONS = HORIZONS
    cases = rq2.load_cases(args.corpus)
    positions = rq2.load_positions(args.forecasts, set(cases))
    missing = set(cases) - positions.keys()
    if missing:
        raise ValueError(f"Missing ensemble positions for {len(missing)} cases")
    grid = load_regional_grid(args.proxy_grid)
    case_rows = evaluate_cases(
        cases, positions, grid, args.impact_radius_km, args.cone_buffer_km
    )
    summary = summarize_metrics(case_rows, args.bootstrap_replicates, args.seed)
    tests = headline_tests(case_rows, args.bootstrap_replicates, args.seed)
    write_csv(args.case_output, case_rows)
    write_csv(args.summary_output, summary)
    write_csv(args.tests_output, tests)
    write_report(args.report_output, case_rows, summary, tests)
    args.metadata.write_text(
        json.dumps(
            {
                "marker": rq2.MARKER,
                "horizons_h": list(HORIZONS),
                "impact_radius_km_assumed": args.impact_radius_km,
                "p90_cone_buffer_km_assumed": args.cone_buffer_km,
                "proxy_region_definition": "10-degree latitude/longitude bins",
                "region_count": len(grid["region_ids"]),
                "ndcg_gain": "linear realized vulnerability-weighted exposure",
                "ranking_ties": "ascending region_id",
                "predicted_ranking": "positive predicted scores only",
                "zero_realized_ndcg_recall": "excluded as undefined",
                "spearman_regions": "predicted > 0 or realized > 0; minimum 3",
                "summary": "equal-weight mean of within-cyclone case means",
                "confidence_intervals": "cyclone bootstrap of storm means",
                "bootstrap_replicates": args.bootstrap_replicates,
                "random_seed": args.seed,
                "headline_tests": "two-sided Wilcoxon on paired cyclone means",
                "holm_family_size": 2,
                "matched_cases": len(cases),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()