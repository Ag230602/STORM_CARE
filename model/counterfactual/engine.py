"""
CounterfactualEngine — generates and compares scenario trajectories (Module 5).

Usage (called from run.py):
    engine = CounterfactualEngine(world_model, cfg)
    report = engine.compare(warm_up_seq)
"""
from __future__ import annotations

import time
from typing import Dict, List

import torch
from torch import Tensor

from .config import CounterfactualConfig
from .scenarios import SCENARIOS, SCENARIO_DESCRIPTIONS
from ..world_model.architecture import WorldModel


class CounterfactualEngine:
    """
    Generates baseline + counterfactual trajectories and computes outcome metrics.

    Outcome metrics (per trajectory)
    ---------------------------------
    peak_exposure        max over time of mean(exposure_dims)
    shelter_shortfall    fraction of steps where resource dims < 0.3 threshold
    infra_damage_final   mean(infra_dims) at the last time step
    resource_deficit     mean over time of max(0, 0.5 − mean(resource_dims))
    mean_hazard          mean over time and hazard dims (storm severity)
    """

    RESOURCE_THRESHOLD = 0.3   # below this = shortfall
    EXPOSURE_THRESHOLD = 0.5   # above this = high exposure step

    def __init__(self, world_model: WorldModel, cfg: CounterfactualConfig):
        self.model = world_model
        self.cfg   = cfg
        self.model.eval()

    @torch.no_grad()
    def _rollout_once(
        self,
        warm_up: Tensor,
        z_override: Tensor | None = None,
    ) -> Tensor:
        """Return one rollout trajectory (n_rollout_steps, d_state)."""
        return self.model.rollout(
            warm_up, self.cfg.n_rollout_steps, z_override
        )

    @torch.no_grad()
    def _rollout_mc(
        self,
        warm_up: Tensor,
        z_override: Tensor | None = None,
    ) -> Tensor:
        """
        Monte Carlo rollout: average cfg.n_monte_carlo stochastic rollouts.
        Returns mean trajectory (n_rollout_steps, d_state).
        """
        samples = torch.stack([
            self._rollout_once(warm_up, z_override)
            for _ in range(self.cfg.n_monte_carlo)
        ])                                         # (MC, T, d)
        return samples.mean(dim=0)                 # (T, d)

    def _compute_metrics(self, traj: Tensor) -> Dict[str, float]:
        """
        Compute outcome metrics from a trajectory tensor (T, d_state).
        All values in [0, 1].
        """
        cfg = self.cfg
        T   = traj.shape[0]

        exp  = traj[:, cfg.exposure_dims].mean(dim=-1)     # (T,)
        res  = traj[:, cfg.resource_dims].mean(dim=-1)     # (T,)
        infr = traj[:, cfg.infra_dims].mean(dim=-1)        # (T,)
        haz  = traj[:, cfg.hazard_dims].mean(dim=-1)       # (T,)

        peak_exposure       = exp.max().item()
        shelter_shortfall   = (res < self.RESOURCE_THRESHOLD).float().mean().item()
        infra_damage_final  = infr[-1].item()
        resource_deficit    = torch.clamp(0.5 - res, min=0).mean().item()
        mean_hazard         = haz.mean().item()

        return {
            "peak_exposure":      round(peak_exposure,      4),
            "shelter_shortfall":  round(shelter_shortfall,  4),
            "infra_damage_final": round(infra_damage_final, 4),
            "resource_deficit":   round(resource_deficit,   4),
            "mean_hazard":        round(mean_hazard,        4),
        }

    def compare(self, warm_up_seq: Tensor) -> Dict[str, dict]:
        """
        Run all 6 trajectories (baseline + 5 interventions).

        warm_up_seq : (T_warm, d_disaster_state)

        Returns dict: scenario_name → {"metrics": ..., "trajectory": tensor}
        """
        results = {}
        for name, fn in SCENARIOS.items():
            modified_warm_up = fn(warm_up_seq, self.cfg)
            traj    = self._rollout_mc(modified_warm_up)
            metrics = self._compute_metrics(traj)
            results[name] = {
                "description": SCENARIO_DESCRIPTIONS[name],
                "metrics":     metrics,
                "trajectory":  traj,
            }
        return results

    @staticmethod
    def print_report(results: Dict[str, dict]) -> None:
        """Print a formatted comparison table to stdout."""
        metric_names = [
            "peak_exposure", "shelter_shortfall",
            "infra_damage_final", "resource_deficit", "mean_hazard",
        ]
        col_w = 20

        print()
        print("═" * 130)
        print("  Module 5 — Counterfactual Reasoning Engine  |  Scenario Comparison")
        print("═" * 130)

        # Header
        header = f"  {'Scenario':<30}" + "".join(f"{m:>{col_w}}" for m in metric_names)
        print(header)
        print("  " + "─" * 128)

        baseline = results["baseline"]["metrics"]

        for name, res in results.items():
            m   = res["metrics"]
            row = f"  {name:<30}"
            for mn in metric_names:
                val   = m[mn]
                delta = val - baseline[mn]
                sign  = "+" if delta > 0 else ""
                cell  = f"{val:.3f} ({sign}{delta:+.3f})" if name != "baseline" else f"{val:.3f}"
                row  += f"{cell:>{col_w}}"
            print(row)

        print()
        print("  Metric guide:")
        print("    peak_exposure       — highest population exposure fraction (lower is better)")
        print("    shelter_shortfall   — fraction of steps with resource below threshold (lower)")
        print("    infra_damage_final  — infrastructure damage at end of horizon (lower)")
        print("    resource_deficit    — mean supply-demand gap (lower is better)")
        print("    mean_hazard         — average storm hazard level (not directly controllable)")
        print()
        print("  (+) = worse than baseline  |  (−) = better than baseline")
        print("═" * 130)
