"""Build NeurIPS STORM-World workshop result artifacts.

This script keeps workshop-specific tables separate from the existing
STORM-CARE/AAAI outputs. It reads frozen metrics already committed under the
main repository and writes derived world-model evaluation tables under
neurips_storm_world/results/.
"""
from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "neurips_storm_world" / "results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def rankdata(values: list[float]) -> list[float]:
    """Ascending ranks with average ranks for ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a: list[float], b: list[float]) -> float:
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0.0 or vb == 0.0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def kendall_tau(pred_ranks: list[float], expected_ranks: list[float]) -> tuple[float, int, int]:
    concordant = 0
    discordant = 0
    for i in range(len(pred_ranks)):
        for j in range(i + 1, len(pred_ranks)):
            pred_delta = pred_ranks[i] - pred_ranks[j]
            expected_delta = expected_ranks[i] - expected_ranks[j]
            if pred_delta == 0.0 or expected_delta == 0.0:
                continue
            if pred_delta * expected_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else float("nan")
    return tau, concordant, discordant


def sign_label(delta: float) -> str:
    if delta > 0:
        return "increase"
    if delta < 0:
        return "decrease"
    return "no_change"


def consistency_rate(
    by_sequence: dict[int, dict[str, dict[str, str]]],
    scenario: str,
    metric: str,
    expected_sign: int,
) -> tuple[int, int, float]:
    good = 0
    deltas: list[float] = []
    for scenario_rows in by_sequence.values():
        baseline = as_float(scenario_rows["baseline"], metric)
        value = as_float(scenario_rows[scenario], metric)
        delta = value - baseline
        deltas.append(delta)
        if expected_sign < 0 and delta < 0:
            good += 1
        if expected_sign > 0 and delta > 0:
            good += 1
    mean_delta = sum(deltas) / len(deltas)
    return good, len(deltas), mean_delta


def build_counterfactual_tables() -> dict[str, float]:
    summary_rows = read_csv(ROOT / "metrics" / "counterfactual" / "counterfactual_outcomes.csv")
    long_rows = read_csv(ROOT / "metrics" / "counterfactual" / "counterfactual_outcomes_long.csv")
    dose_rows = read_csv(ROOT / "metrics" / "counterfactual" / "dose_response_verdict.csv")
    adjacent_rows = read_csv(ROOT / "metrics" / "counterfactual" / "dose_response_adjacent_tests.csv")

    summary = {row["scenario"]: row for row in summary_rows}
    by_sequence: dict[int, dict[str, dict[str, str]]] = {}
    for row in long_rows:
        by_sequence.setdefault(int(row["sequence_id"]), {})[row["scenario"]] = row

    expected = [
        {
            "scenario": "earlier_evacuation",
            "intervention_family": "evacuation_timing",
            "primary_metric": "peak_exposure",
            "expected_direction": "decrease",
            "expected_sign": -1,
            "assessment": "supported",
            "assessment_rationale": "Peak exposure decreases consistently as evacuation lead time increases.",
        },
        {
            "scenario": "earlier_evacuation_24h",
            "intervention_family": "evacuation_timing",
            "primary_metric": "peak_exposure",
            "expected_direction": "decrease",
            "expected_sign": -1,
            "assessment": "supported",
            "assessment_rationale": "Peak exposure decreases consistently as evacuation lead time increases.",
        },
        {
            "scenario": "earlier_evacuation_36h",
            "intervention_family": "evacuation_timing",
            "primary_metric": "peak_exposure",
            "expected_direction": "decrease",
            "expected_sign": -1,
            "assessment": "supported",
            "assessment_rationale": "Peak exposure decreases consistently as evacuation lead time increases.",
        },
        {
            "scenario": "delayed_evacuation",
            "intervention_family": "evacuation_timing",
            "primary_metric": "peak_exposure",
            "expected_direction": "increase",
            "expected_sign": 1,
            "assessment": "supported",
            "assessment_rationale": "Peak exposure increases consistently under delayed evacuation.",
        },
        {
            "scenario": "intensity_increase",
            "intervention_family": "exogenous_storm_perturbation",
            "primary_metric": "peak_exposure",
            "expected_direction": "increase",
            "expected_sign": 1,
            "assessment": "partial",
            "assessment_rationale": "Exposure sign is consistent, but magnitude is small and hazard-response fidelity remains weak.",
        },
        {
            "scenario": "intensity_decrease",
            "intervention_family": "exogenous_storm_perturbation",
            "primary_metric": "peak_exposure",
            "expected_direction": "decrease",
            "expected_sign": -1,
            "assessment": "partial",
            "assessment_rationale": "Exposure sign is consistent, but magnitude is small and hazard-response fidelity remains weak.",
        },
        {
            "scenario": "road_blockage",
            "intervention_family": "infrastructure_failure",
            "primary_metric": "peak_exposure",
            "expected_direction": "increase",
            "expected_sign": 1,
            "assessment": "failure",
            "assessment_rationale": "Peak exposure decreases instead of increasing.",
        },
        {
            "scenario": "shelter_failure",
            "intervention_family": "infrastructure_failure",
            "primary_metric": "resource_deficit",
            "expected_direction": "increase",
            "expected_sign": 1,
            "assessment": "failure",
            "assessment_rationale": "Resource-deficit sign is only partially consistent and exposure does not show a reliable failure response.",
        },
        {
            "scenario": "hospital_failure",
            "intervention_family": "infrastructure_failure",
            "primary_metric": "infra_damage_final",
            "expected_direction": "increase",
            "expected_sign": 1,
            "assessment": "failure",
            "assessment_rationale": "Infrastructure damage does not increase reliably after the intervention.",
        },
        {
            "scenario": "additional_emergency_resources",
            "intervention_family": "resource_intervention",
            "primary_metric": "resource_deficit",
            "expected_direction": "decrease",
            "expected_sign": -1,
            "assessment": "failure",
            "assessment_rationale": "Resource-deficit reduction is tiny, so this is treated as approximately null.",
        },
    ]

    fidelity_rows: list[dict[str, object]] = []
    rates: list[float] = []
    for item in expected:
        scenario = str(item["scenario"])
        metric = str(item["primary_metric"])
        good, total, mean_delta = consistency_rate(
            by_sequence,
            scenario,
            metric,
            int(item["expected_sign"]),
        )
        rate = good / total
        rates.append(rate)
        baseline_value = as_float(summary["baseline"], metric)
        scenario_value = as_float(summary[scenario], metric)
        fidelity_rows.append(
            {
                "scenario": scenario,
                "intervention_family": item["intervention_family"],
                "primary_metric": metric,
                "baseline_value": round(baseline_value, 6),
                "scenario_value": round(scenario_value, 6),
                "aggregate_delta_vs_baseline": round(mean_delta, 6),
                "expected_direction": item["expected_direction"],
                "observed_direction": sign_label(mean_delta),
                "sequence_consistency": f"{good}/{total}",
                "intervention_consistency_rate": round(rate, 6),
                "assessment": item["assessment"],
                "assessment_rationale": item["assessment_rationale"],
                "source": "metrics/counterfactual/counterfactual_outcomes_long.csv",
            }
        )

    write_csv(TABLES / "intervention_fidelity_matrix.csv", fidelity_rows)

    evac_order = [
        "earlier_evacuation_36h",
        "earlier_evacuation_24h",
        "earlier_evacuation",
        "baseline",
        "delayed_evacuation",
    ]
    expected_ranks = [1, 2, 3, 4, 5]

    sequence_rank_rows: list[dict[str, object]] = []
    spearman_values: list[float] = []
    tau_values: list[float] = []
    top1_correct = 0
    strict_order_correct = 0
    pairwise_correct = 0
    pairwise_total = 0

    for sequence_id, scenario_rows in sorted(by_sequence.items()):
        values = [as_float(scenario_rows[scenario], "peak_exposure") for scenario in evac_order]
        pred_ranks = rankdata(values)
        spearman = pearson(pred_ranks, expected_ranks)
        tau, concordant, discordant = kendall_tau(pred_ranks, expected_ranks)
        spearman_values.append(spearman)
        tau_values.append(tau)
        top1 = evac_order[values.index(min(values))]
        strict_order = all(values[i] < values[i + 1] for i in range(len(values) - 1))
        top1_is_correct = top1 == evac_order[0]
        top1_correct += int(top1_is_correct)
        strict_order_correct += int(strict_order)
        pairwise_correct += concordant
        pairwise_total += concordant + discordant
        sequence_rank_rows.append(
            {
                "sequence_id": sequence_id,
                "spearman_rank_correlation": round(spearman, 6),
                "kendall_tau": round(tau, 6),
                "top1_predicted_action": top1,
                "top1_matches_expected": top1_is_correct,
                "pairwise_correct": concordant,
                "pairwise_total": concordant + discordant,
                "strict_full_order_matches_expected": strict_order,
            }
        )

    write_csv(TABLES / "policy_ranking_by_sequence.csv", sequence_rank_rows)

    aggregate_values = [as_float(summary[scenario], "peak_exposure") for scenario in evac_order]
    policy_rows = [
        {
            "metric": "aggregate_expected_order",
            "value": " > ".join(evac_order),
            "interpretation": "Lower peak exposure is better; expected order runs best to worst.",
        },
        {
            "metric": "aggregate_observed_order",
            "value": " > ".join([x for _, x in sorted(zip(aggregate_values, evac_order))]),
            "interpretation": "Observed mean peak-exposure order across 24 held-out sequences.",
        },
        {
            "metric": "mean_spearman_rank_correlation",
            "value": round(sum(spearman_values) / len(spearman_values), 6),
            "interpretation": "Sequence-level correlation between predicted and expected evacuation order.",
        },
        {
            "metric": "mean_kendall_tau",
            "value": round(sum(tau_values) / len(tau_values), 6),
            "interpretation": "Sequence-level pairwise ranking consistency.",
        },
        {
            "metric": "top1_action_accuracy",
            "value": round(top1_correct / len(by_sequence), 6),
            "interpretation": f"{top1_correct}/{len(by_sequence)} sequences select 36h earlier evacuation as best.",
        },
        {
            "metric": "pairwise_intervention_ranking_accuracy",
            "value": round(pairwise_correct / pairwise_total, 6),
            "interpretation": f"{pairwise_correct}/{pairwise_total} pairwise comparisons match expected order.",
        },
        {
            "metric": "strict_full_order_accuracy",
            "value": round(strict_order_correct / len(by_sequence), 6),
            "interpretation": f"{strict_order_correct}/{len(by_sequence)} sequences match the full five-action order.",
        },
    ]
    write_csv(TABLES / "policy_ranking_preservation.csv", policy_rows)

    dose_out = [
        {
            "scenario": row["scenario"],
            "mean_peak_exposure": round(float(row["mean"]), 6),
            "ci95_lo": round(float(row["ci95_lo"]), 6),
            "ci95_hi": round(float(row["ci95_hi"]), 6),
            "source": "metrics/counterfactual/dose_response_verdict.csv",
        }
        for row in dose_rows
    ]
    write_csv(TABLES / "evacuation_dose_response.csv", dose_out)

    adjacent_out = [
        {
            "comparison": row.get("comparison", row.get("step", "")),
            "mean_delta": round(float(row.get("mean_delta", row.get("mean_diff", "nan"))), 8),
            "p_value": row.get("p_value", row.get("wilcoxon_p_one_sided_less", "")),
            "holm_p": row.get("holm_p", row.get("p_holm", "")),
            "significant_after_holm": row.get("significant_after_holm", row.get("claimable_at_0.05", "")),
            "source": "metrics/counterfactual/dose_response_adjacent_tests.csv",
        }
        for row in adjacent_rows
    ]
    write_csv(TABLES / "evacuation_adjacent_significance.csv", adjacent_out)

    return {
        "n_sequences": float(len(by_sequence)),
        "macro_intervention_consistency": sum(rates) / len(rates),
        "mean_spearman": sum(spearman_values) / len(spearman_values),
        "mean_kendall_tau": sum(tau_values) / len(tau_values),
        "top1_accuracy": top1_correct / len(by_sequence),
        "pairwise_accuracy": pairwise_correct / pairwise_total,
        "strict_order_accuracy": strict_order_correct / len(by_sequence),
    }


def build_physics_uncertainty_tables() -> None:
    physics_rows = read_csv(ROOT / "metrics" / "physics" / "physics_full_vs_ablation.csv")
    physics_out = []
    for row in physics_rows:
        physics_out.append(
            {
                "variant": row["run"],
                "validation_track_rmse_normalized": round(float(row["final_val_track_rmse"]), 8),
                "weighted_validation_physics_loss": round(float(row["final_val_L_phys"]), 8),
                "advection_residual": round(float(row["final_val_R_adv"]), 10),
                "diffusion_residual": round(float(row["final_val_R_diff"]), 10),
                "mass_residual": round(float(row["final_val_R_mass"]), 10),
                "wind_pressure_residual": round(float(row["final_val_R_wp"]), 10),
                "continuity_residual": round(float(row["final_val_R_cont"]), 10),
                "energy_residual": round(float(row["final_val_R_nrg"]), 10),
                "source": "metrics/physics/physics_full_vs_ablation.csv",
            }
        )
    write_csv(TABLES / "physical_fidelity_ablation.csv", physics_out)

    calibration_rows = read_csv(ROOT / "tables" / "table_calibration_cone_coverage.csv")
    calibration_out = []
    for row in calibration_rows:
        p50 = float(row["cone_p50_ep20"])
        p90 = float(row["cone_p90_ep20"])
        calibration_out.append(
            {
                "lead_time_h": row["lead_time_h"],
                "p50_empirical_coverage": round(p50, 6),
                "p90_empirical_coverage": round(p90, 6),
                "p50_abs_error": round(abs(p50 - float(row["ideal_p50"])), 6),
                "p90_abs_error": round(abs(p90 - float(row["ideal_p90"])), 6),
                "interpretation": "short_horizon_calibrated" if int(row["lead_time_h"]) <= 48 else "long_horizon_uncertainty_drift",
                "source": row["source_csv"],
            }
        )
    write_csv(TABLES / "uncertainty_drift_by_horizon.csv", calibration_out)


def copy_workshop_figures() -> None:
    copies = [
        (ROOT / "figures" / "physics_residuals_full_vs_ablation.png", FIGURES / "physics_residuals_full_vs_ablation.png"),
        (ROOT / "figures" / "physics_residuals_full_vs_ablation.pdf", FIGURES / "physics_residuals_full_vs_ablation.pdf"),
        (ROOT / "figures" / "calibration.png", FIGURES / "calibration.png"),
        (ROOT / "figures" / "calibration.pdf", FIGURES / "calibration.pdf"),
    ]
    FIGURES.mkdir(parents=True, exist_ok=True)
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def build_scorecard(metrics: dict[str, float]) -> None:
    rows = [
        {
            "dimension": "Predictive world fidelity",
            "headline_result": "Use existing forecast/baseline tables; do not claim all-horizon superiority over Persistence.",
            "primary_artifact": "tables/table1_track_error_vs_baselines.csv",
            "workshop_use": "Background evidence, not the main novelty.",
        },
        {
            "dimension": "Physical fidelity",
            "headline_result": "Physics residuals are reported as full-vs-no-physics ablation.",
            "primary_artifact": "neurips_storm_world/results/tables/physical_fidelity_ablation.csv",
            "workshop_use": "Supports physically grounded long-horizon rollout framing.",
        },
        {
            "dimension": "Probabilistic fidelity",
            "headline_result": "P50/P90 coverage degrades at 72-120h, giving an uncertainty-drift story.",
            "primary_artifact": "neurips_storm_world/results/tables/uncertainty_drift_by_horizon.csv",
            "workshop_use": "Shows why stochastic future worlds must be evaluated over horizon.",
        },
        {
            "dimension": "Intervention consistency",
            "headline_result": f"Macro consistency rate = {metrics['macro_intervention_consistency']:.3f}; evacuation and intensity exposure signs are strongest, infrastructure channels fail.",
            "primary_artifact": "neurips_storm_world/results/tables/intervention_fidelity_matrix.csv",
            "workshop_use": "New world-model-specific evaluation metric.",
        },
        {
            "dimension": "Decision fidelity",
            "headline_result": f"Evacuation ranking: Spearman = {metrics['mean_spearman']:.3f}, Kendall tau = {metrics['mean_kendall_tau']:.3f}, pairwise accuracy = {metrics['pairwise_accuracy']:.3f}.",
            "primary_artifact": "neurips_storm_world/results/tables/policy_ranking_preservation.csv",
            "workshop_use": "Strongest novelty-facing result.",
        },
        {
            "dimension": "Rollout horizon fidelity",
            "headline_result": "Additional script reports RSSM rollout error versus lead time and compares it against direct persistence/linear predictors.",
            "primary_artifact": "neurips_storm_world/results/tables/rollout_fidelity_vs_horizon.csv",
            "workshop_use": "Addresses world-model/direct-predictor guideline item.",
        },
        {
            "dimension": "Stochastic future modeling",
            "headline_result": "Additional script compares deterministic RSSM mean-path rollouts with stochastic sample-mean rollouts and reports coverage/sharpness.",
            "primary_artifact": "neurips_storm_world/results/tables/deterministic_vs_stochastic_rollout.csv",
            "workshop_use": "Addresses deterministic-vs-stochastic rollout guideline item.",
        },
        {
            "dimension": "Intervention-conditioning ablation",
            "headline_result": "Additional script removes branch-state conditioning by reusing baseline warm-up states for all intervention labels.",
            "primary_artifact": "neurips_storm_world/results/tables/intervention_conditioning_ablation.csv",
            "workshop_use": "Tests whether intervention responses require the branch intervention.",
        },
        {
            "dimension": "AOTS2Action bridge",
            "headline_result": "Additional script summarizes real-geospatial AOTS exposure/ranking metrics as a downstream evaluator bridge.",
            "primary_artifact": "neurips_storm_world/results/tables/aots2action_bridge_summary.csv",
            "workshop_use": "Useful bridge evidence, but not a full STORM rollout-to-geospatial integration.",
        },
    ]
    write_csv(TABLES / "workshop_evaluation_scorecard.csv", rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = build_counterfactual_tables()
    build_physics_uncertainty_tables()
    copy_workshop_figures()
    build_scorecard(metrics)

    manifest = [
        {"artifact": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
        for path in sorted(OUT.rglob("*"))
        if path.is_file()
    ]
    write_csv(TABLES / "source_manifest.csv", manifest)
    print(f"Wrote NeurIPS STORM-World results to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
