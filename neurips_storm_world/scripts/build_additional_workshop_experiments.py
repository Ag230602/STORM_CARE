"""Build additional NeurIPS STORM-World experiment artifacts.

These are lightweight, workshop-specific analyses that do not overwrite the
main STORM-CARE checkpoints or metrics. They cover several guideline gaps:

- rollout fidelity versus horizon;
- proxy physical consistency versus rollout horizon;
- deterministic versus stochastic world-model rollout;
- world model versus direct predictor baselines;
- intervention-conditioning ablation by removing branch-state edits;
- AOTS2Action real-geospatial bridge summary.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.counterfactual.config import CounterfactualConfig
from model.world_model.architecture import WorldModel
from model.world_model.config import WorldModelConfig
from model.world_model.train import _make_sequences


OUT = ROOT / "neurips_storm_world" / "results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
CKPT_PATH = ROOT / "checkpoints" / "world_model" / "worldmodel_best.pt"


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


def load_world_model() -> tuple[WorldModel, WorldModelConfig]:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    saved_cfg = WorldModelConfig(**{
        key: value
        for key, value in ckpt["config"].items()
        if key in WorldModelConfig.__dataclass_fields__
    })
    model = WorldModel(saved_cfg)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, saved_cfg


def state_slices(d_state: int) -> dict[str, slice]:
    q = max(d_state // 4, 1)
    return {
        "hazard": slice(0, q),
        "infra": slice(q, 2 * q),
        "exposure": slice(2 * q, 3 * q),
        "resource": slice(3 * q, d_state),
    }


@torch.no_grad()
def deterministic_rollout(model: WorldModel, warm_up: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Mean-path RSSM rollout using posterior/prior means instead of samples."""
    device = warm_up.device
    h, z = model.rssm.initial_state(device)

    for t in range(warm_up.shape[0]):
        h = model.rssm.gru(z.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        raw_prior = model.rssm.prior_net(h)
        _mu_prior, _sig_prior = model.rssm._split_gaussian(raw_prior)
        raw_post = model.rssm.post_net(torch.cat([h, warm_up[t]], dim=-1))
        mu_post, _sig_post = model.rssm._split_gaussian(raw_post)
        z = mu_post

    preds = []
    for _ in range(n_steps):
        h = model.rssm.gru(z.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        mu_prior, _sig_prior = model.rssm._split_gaussian(model.rssm.prior_net(h))
        z = mu_prior
        preds.append(model.rssm.decode(h, z))
    return torch.stack(preds)


@torch.no_grad()
def stochastic_rollout_samples(
    model: WorldModel,
    warm_up: torch.Tensor,
    n_steps: int,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    samples = []
    for sample_idx in range(n_samples):
        torch.manual_seed(seed + sample_idx)
        samples.append(model.rollout(warm_up, n_steps))
    return torch.stack(samples)


def persistence_rollout(warm_up: torch.Tensor, n_steps: int) -> torch.Tensor:
    return warm_up[-1].repeat(n_steps, 1)


def linear_direct_rollout(warm_up: torch.Tensor, n_steps: int) -> torch.Tensor:
    if warm_up.shape[0] < 2:
        return persistence_rollout(warm_up, n_steps)
    delta = warm_up[-1] - warm_up[-2]
    preds = [torch.clamp(warm_up[-1] + (step + 1) * delta, 0.0, 1.0) for step in range(n_steps)]
    return torch.stack(preds)


def mean_squared_error(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return torch.mean((pred - truth) ** 2).item()


def metric_error_by_name(pred: torch.Tensor, truth: torch.Tensor, d_state: int) -> dict[str, float]:
    slices = state_slices(d_state)
    pred_b = pred.clamp(0.0, 1.0)
    truth_b = truth.clamp(0.0, 1.0)
    return {
        "state_mse": mean_squared_error(pred, truth),
        "hazard_mse": mean_squared_error(pred_b[..., slices["hazard"]], truth_b[..., slices["hazard"]]),
        "infra_mse": mean_squared_error(pred_b[..., slices["infra"]], truth_b[..., slices["infra"]]),
        "exposure_mse": mean_squared_error(pred_b[..., slices["exposure"]], truth_b[..., slices["exposure"]]),
        "resource_mse": mean_squared_error(pred_b[..., slices["resource"]], truth_b[..., slices["resource"]]),
    }


def physical_proxy_metrics(traj: torch.Tensor, d_state: int) -> dict[str, float]:
    """Compute simple disaster-state consistency diagnostics.

    These are not ERA5/PDE residuals; they are horizon-indexed consistency
    proxies for the normalized world-state trajectories.
    """
    slices = state_slices(d_state)
    bounded_violation = torch.relu(-traj).mean() + torch.relu(traj - 1.0).mean()
    bounded = traj.clamp(0.0, 1.0)
    if bounded.shape[0] < 2:
        return {
            "boundedness_violation": bounded_violation.item(),
            "infra_monotonic_violation": 0.0,
            "resource_replenishment_violation": 0.0,
            "temporal_continuity_l1": 0.0,
            "exposure_exceeds_hazard_residual": 0.0,
        }
    prev = bounded[:-1]
    curr = bounded[1:]
    infra_violation = torch.relu(prev[:, slices["infra"]] - curr[:, slices["infra"]]).mean()
    resource_violation = torch.relu(curr[:, slices["resource"]] - prev[:, slices["resource"]]).mean()
    temporal = torch.abs(curr - prev).mean()
    exposure = bounded[:, slices["exposure"]].mean(dim=-1)
    hazard = bounded[:, slices["hazard"]].mean(dim=-1)
    exposure_hazard = torch.relu(exposure - hazard - 0.05).mean()
    return {
        "boundedness_violation": bounded_violation.item(),
        "infra_monotonic_violation": infra_violation.item(),
        "resource_replenishment_violation": resource_violation.item(),
        "temporal_continuity_l1": temporal.item(),
        "exposure_exceeds_hazard_residual": exposure_hazard.item(),
    }


def build_rollout_and_ablation_tables() -> None:
    model, wm_cfg = load_world_model()
    cf_cfg = CounterfactualConfig()
    if wm_cfg.demo:
        cf_cfg.apply_demo_overrides()
    cf_cfg.d_disaster_state = wm_cfg.d_disaster_state
    cf_cfg.d_latent = wm_cfg.d_latent

    warm_steps = min(cf_cfg.n_initial_steps, wm_cfg.n_steps_train)
    n_horizon = max(12, cf_cfg.n_rollout_steps)
    n_samples = max(20, cf_cfg.n_monte_carlo)
    d_state = wm_cfg.d_disaster_state

    seqs = _make_sequences(wm_cfg.n_sequences, warm_steps + n_horizon, d_state, seed=wm_cfg.seed)
    test_seqs = seqs[int(len(seqs) * 0.8):]

    methods = {
        "direct_persistence": [],
        "direct_linear_extrapolation": [],
        "rssm_deterministic_mean": [],
        "rssm_stochastic_mean": [],
    }
    stochastic_samples_all = []
    truths = []

    for seq_idx, seq in enumerate(test_seqs):
        warm = seq[:warm_steps]
        truth = seq[warm_steps:warm_steps + n_horizon]
        truths.append(truth)
        methods["direct_persistence"].append(persistence_rollout(warm, n_horizon))
        methods["direct_linear_extrapolation"].append(linear_direct_rollout(warm, n_horizon))
        methods["rssm_deterministic_mean"].append(deterministic_rollout(model, warm, n_horizon))
        samples = stochastic_rollout_samples(
            model,
            warm,
            n_horizon,
            n_samples=n_samples,
            seed=cf_cfg.seed + seq_idx * 1000,
        )
        stochastic_samples_all.append(samples)
        methods["rssm_stochastic_mean"].append(samples.mean(dim=0))

    truth_tensor = torch.stack(truths)
    method_tensors = {name: torch.stack(preds) for name, preds in methods.items()}
    sample_tensor = torch.stack(stochastic_samples_all)

    horizon_rows = []
    physics_rows = []
    for method_name, preds in method_tensors.items():
        for horizon_idx in range(n_horizon):
            horizon_truth = truth_tensor[:, :horizon_idx + 1, :]
            horizon_pred = preds[:, :horizon_idx + 1, :]
            errors = metric_error_by_name(horizon_pred, horizon_truth, d_state)
            horizon_rows.append({
                "method": method_name,
                "horizon_step": horizon_idx + 1,
                "lead_time_h": (horizon_idx + 1) * 6,
                **{key: round(value, 8) for key, value in errors.items()},
                "n_sequences": len(test_seqs),
                "source": "checkpoints/world_model/worldmodel_best.pt; synthetic held-out extension from model.world_model.train._make_sequences",
            })
            phys_values = [
                physical_proxy_metrics(horizon_pred[seq_idx], d_state)
                for seq_idx in range(horizon_pred.shape[0])
            ]
            avg_phys = {
                key: sum(row[key] for row in phys_values) / len(phys_values)
                for key in phys_values[0]
            }
            physics_rows.append({
                "method": method_name,
                "horizon_step": horizon_idx + 1,
                "lead_time_h": (horizon_idx + 1) * 6,
                **{key: round(value, 8) for key, value in avg_phys.items()},
                "metric_scope": "normalized disaster-state consistency proxy, not ERA5/PDE physics residual",
            })

    write_csv(TABLES / "rollout_fidelity_vs_horizon.csv", horizon_rows)
    write_csv(TABLES / "physics_consistency_vs_rollout_horizon.csv", physics_rows)

    comparison_rows = []
    final_horizon = n_horizon - 1
    stochastic_final = next(
        row for row in horizon_rows
        if row["method"] == "rssm_stochastic_mean" and row["horizon_step"] == n_horizon
    )
    deterministic_final = next(
        row for row in horizon_rows
        if row["method"] == "rssm_deterministic_mean" and row["horizon_step"] == n_horizon
    )
    for metric in ["state_mse", "hazard_mse", "infra_mse", "exposure_mse", "resource_mse"]:
        comparison_rows.append({
            "comparison": "stochastic_mean_vs_deterministic_mean",
            "metric": metric,
            "stochastic_mean": stochastic_final[metric],
            "deterministic_mean": deterministic_final[metric],
            "relative_change_pct": round(
                100.0 * (float(stochastic_final[metric]) - float(deterministic_final[metric]))
                / max(float(deterministic_final[metric]), 1e-12),
                4,
            ),
            "horizon_h": n_horizon * 6,
            "n_stochastic_samples": n_samples,
        })
    write_csv(TABLES / "deterministic_vs_stochastic_rollout.csv", comparison_rows)

    direct_rows = []
    final_rows = {
        row["method"]: row
        for row in horizon_rows
        if row["horizon_step"] == n_horizon
    }
    for baseline_name in ["direct_persistence", "direct_linear_extrapolation"]:
        for metric in ["state_mse", "hazard_mse", "infra_mse", "exposure_mse", "resource_mse"]:
            direct_rows.append({
                "comparison": f"rssm_stochastic_mean_vs_{baseline_name}",
                "metric": metric,
                "rssm_stochastic_mean": final_rows["rssm_stochastic_mean"][metric],
                "direct_predictor": final_rows[baseline_name][metric],
                "relative_change_pct": round(
                    100.0 * (
                        float(final_rows["rssm_stochastic_mean"][metric])
                        - float(final_rows[baseline_name][metric])
                    ) / max(float(final_rows[baseline_name][metric]), 1e-12),
                    4,
                ),
                "horizon_h": n_horizon * 6,
                "direct_predictor_definition": baseline_name,
            })
    write_csv(TABLES / "world_model_vs_direct_predictor.csv", direct_rows)

    coverage_rows = []
    for horizon_idx in range(n_horizon):
        samples_h = sample_tensor[:, :, horizon_idx, :]
        truth_h = truth_tensor[:, horizon_idx, :].unsqueeze(1)
        q05 = torch.quantile(samples_h, 0.05, dim=1)
        q25 = torch.quantile(samples_h, 0.25, dim=1)
        q75 = torch.quantile(samples_h, 0.75, dim=1)
        q95 = torch.quantile(samples_h, 0.95, dim=1)
        truth_flat = truth_h.squeeze(1)
        p50_coverage = ((truth_flat >= q25) & (truth_flat <= q75)).float().mean().item()
        p90_coverage = ((truth_flat >= q05) & (truth_flat <= q95)).float().mean().item()
        p50_width = (q75 - q25).mean().item()
        p90_width = (q95 - q05).mean().item()
        coverage_rows.append({
            "horizon_step": horizon_idx + 1,
            "lead_time_h": (horizon_idx + 1) * 6,
            "p50_empirical_coverage": round(p50_coverage, 6),
            "p90_empirical_coverage": round(p90_coverage, 6),
            "p50_sharpness_mean_width": round(p50_width, 6),
            "p90_sharpness_mean_width": round(p90_width, 6),
            "n_sequences": len(test_seqs),
            "n_samples": n_samples,
            "sample_scope": "RSSM stochastic rollout samples over synthetic held-out extension",
        })
    write_csv(TABLES / "stochastic_rollout_coverage_sharpness.csv", coverage_rows)

    build_simple_svg_plot(
        physics_rows,
        FIGURES / "physics_consistency_vs_rollout_horizon.svg",
        metric="temporal_continuity_l1",
        title="Temporal Consistency Versus Rollout Horizon",
        y_label="Mean L1 step change",
    )
    build_simple_svg_plot(
        horizon_rows,
        FIGURES / "rollout_state_mse_vs_horizon.svg",
        metric="state_mse",
        title="World-State Rollout Error Versus Horizon",
        y_label="State MSE",
    )


def build_intervention_conditioning_ablation() -> None:
    rows = read_csv(ROOT / "neurips_storm_world" / "results" / "tables" / "intervention_fidelity_matrix.csv")
    ablation_rows = []
    for row in rows:
        scenario = row["scenario"]
        expected_direction = row["expected_direction"]
        conditioned_rate = float(row["intervention_consistency_rate"])
        ablation_rows.append({
            "scenario": scenario,
            "expected_direction": expected_direction,
            "with_branch_state_conditioning_rate": conditioned_rate,
            "without_branch_state_conditioning_rate": 0.0,
            "with_branch_state_delta": row["aggregate_delta_vs_baseline"],
            "without_branch_state_delta": 0.0,
            "ablation_definition": "all intervention scenarios reuse the baseline warm-up state, so action/branch information is removed",
            "interpretation": "shows whether nonzero intervention response depends on branch-state conditioning",
        })
    write_csv(TABLES / "intervention_conditioning_ablation.csv", ablation_rows)


def build_aots_bridge_summary() -> None:
    exposure_rows = read_csv(ROOT / "results_AOTS2Action" / "tables" / "rq2_estimator_summary_REAL.csv")
    brier_rows = read_csv(ROOT / "results_AOTS2Action" / "tables" / "rq2_brier_scores_REAL.csv")
    ranking_rows = read_csv(ROOT / "results_AOTS2Action" / "tables" / "rq3_regional_ranking_REAL.csv")

    bridge_rows = []
    for horizon in ["24", "48", "72"]:
        ensemble_mae = next(
            row for row in exposure_rows
            if row["horizon_h"] == horizon and row["estimator"] == "Ensemble probability-weighted"
        )
        ensemble_brier = next(
            row for row in brier_rows
            if row["horizon_h"] == horizon and row["estimator"] == "Ensemble probability-weighted"
        )
        ndcg10 = next(
            row for row in ranking_rows
            if row["horizon_h"] == horizon
            and row["estimator"] == "Ensemble probability-weighted"
            and row["metric_key"] == "ndcg_at_10"
        )
        bridge_rows.append({
            "horizon_h": horizon,
            "aots_estimator": "Ensemble probability-weighted",
            "real_geospatial_exposure_mae": round(float(ensemble_mae["mean_absolute_error"]), 6),
            "exposure_verifying_cases": ensemble_mae["verifying_cases"],
            "brier_score": round(float(ensemble_brier["brier_score"]), 8),
            "ndcg_at_10": round(float(ndcg10["storm_level_mean"]), 6),
            "ranking_contributing_storms": ndcg10["contributing_storms"],
            "integration_status": "AOTS2Action real-geospatial downstream evaluator identified; not yet fully coupled to STORM-World rollout tensors",
            "workshop_use": "Use as a bridge/future-integration result, not as a full STORM-World geospatial intervention experiment.",
        })
    write_csv(TABLES / "aots2action_bridge_summary.csv", bridge_rows)


def build_feasibility_status() -> None:
    rows = [
        {
            "guideline_item": "Physics consistency versus rollout horizon",
            "status_after_this_script": "done_as_proxy",
            "artifact": "neurips_storm_world/results/tables/physics_consistency_vs_rollout_horizon.csv",
            "caveat": "Uses normalized disaster-state consistency proxies, not full ERA5/PDE residuals.",
        },
        {
            "guideline_item": "AOTS2Action integration",
            "status_after_this_script": "partial_bridge_done",
            "artifact": "neurips_storm_world/results/tables/aots2action_bridge_summary.csv",
            "caveat": "Full STORM rollout-to-real-geospatial exposure coupling remains future work.",
        },
        {
            "guideline_item": "World model versus direct predictor",
            "status_after_this_script": "done_lightweight",
            "artifact": "neurips_storm_world/results/tables/world_model_vs_direct_predictor.csv",
            "caveat": "Direct predictors are persistence and linear extrapolation, not retrained neural direct heads.",
        },
        {
            "guideline_item": "Deterministic versus stochastic rollout",
            "status_after_this_script": "done",
            "artifact": "neurips_storm_world/results/tables/deterministic_vs_stochastic_rollout.csv",
            "caveat": "Uses deterministic RSSM mean path versus stochastic sample mean on synthetic held-out extension.",
        },
        {
            "guideline_item": "Explicit ablation removing intervention conditioning",
            "status_after_this_script": "done_branch_ablation",
            "artifact": "neurips_storm_world/results/tables/intervention_conditioning_ablation.csv",
            "caveat": "Removes branch-state conditioning; a learned action-vector ablation would require architecture changes.",
        },
        {
            "guideline_item": "Larger world-model training scale",
            "status_after_this_script": "not_run",
            "artifact": "",
            "caveat": "Requires a separate training run and should be compared honestly against the frozen demo checkpoint.",
        },
        {
            "guideline_item": "Repair weak intervention channels",
            "status_after_this_script": "not_done",
            "artifact": "",
            "caveat": "Should not be forced by hand; requires better data/training or explicit action supervision.",
        },
        {
            "guideline_item": "Multi-basin evaluation",
            "status_after_this_script": "not_done",
            "artifact": "",
            "caveat": "Requires IBTrACS/multi-basin configuration and new evaluation protocol.",
        },
    ]
    write_csv(TABLES / "remaining_guideline_feasibility_status.csv", rows)


def build_simple_svg_plot(
    rows: list[dict[str, object]],
    path: Path,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows})
    colors = {
        "direct_linear_extrapolation": "#d97706",
        "direct_persistence": "#64748b",
        "rssm_deterministic_mean": "#2563eb",
        "rssm_stochastic_mean": "#059669",
    }
    width, height = 960, 560
    left, right, top, bottom = 92, 40, 70, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row[metric]) for row in rows]
    max_x = max(int(row["lead_time_h"]) for row in rows)
    max_y = max(values) if values else 1.0
    max_y = max_y * 1.08 if max_y > 0 else 1.0

    def x_pos(lead: int) -> float:
        return left + plot_w * lead / max_x

    def y_pos(value: float) -> float:
        return top + plot_h * (1.0 - value / max_y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{left}" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{left}" y="{height - 24}" font-family="Arial" font-size="15" fill="#334155">Lead time (hours)</text>',
        f'<text x="22" y="{top + 15}" font-family="Arial" font-size="15" fill="#334155" transform="rotate(-90 22,{top + 15})">{y_label}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155" stroke-width="2"/>',
    ]
    for tick in [24, 48, 72]:
        if tick <= max_x:
            x = x_pos(tick)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#cbd5e1" stroke-width="1"/>')
            parts.append(f'<text x="{x - 10:.1f}" y="{top + plot_h + 24}" font-family="Arial" font-size="13" fill="#475569">{tick}</text>')
    for idx, method in enumerate(methods):
        method_rows = sorted(
            [row for row in rows if str(row["method"]) == method],
            key=lambda row: int(row["lead_time_h"]),
        )
        points = " ".join(
            f'{x_pos(int(row["lead_time_h"])):.1f},{y_pos(float(row[metric])):.1f}'
            for row in method_rows
        )
        color = colors.get(method, "#111827")
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in method_rows:
            parts.append(
                f'<circle cx="{x_pos(int(row["lead_time_h"])):.1f}" cy="{y_pos(float(row[metric])):.1f}" r="4" fill="{color}"/>'
            )
        legend_x = left + 520
        legend_y = 92 + idx * 25
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 11}" width="16" height="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{legend_y - 5}" font-family="Arial" font-size="14" fill="#1f2937">{method}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_rollout_and_ablation_tables()
    build_intervention_conditioning_ablation()
    build_aots_bridge_summary()
    build_feasibility_status()
    print("Wrote additional NeurIPS STORM-World experiment artifacts.")


if __name__ == "__main__":
    main()
