"""
CounterfactualEngine — generates and compares scenario trajectories (Module 5).

Usage (called from run.py):
    engine = CounterfactualEngine(world_model, cfg)
    report = engine.compare(warm_up_seq)
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from .config import CounterfactualConfig
from .scenarios import (
    InterventionSpec,
    SCENARIO_DESCRIPTIONS,
    SCENARIO_INTERVENTIONS,
)
from ..world_model.architecture import WorldModel


class CounterfactualEngine:
    """
    Generates baseline + counterfactual trajectories by modifying the observed
    branch state, encoding that branch through the RSSM posterior, and rolling
    forward with the learned prior and decoder.  The engine does not edit decoded
    output trajectories.

    Outcome metrics (per trajectory)
    ---------------------------------
    peak_exposure        max over time of mean(exposure_dims)
    shelter_shortfall    exposure-weighted unmet resource proxy computed as
                         mean(exposure * (1 - resource))
    infra_damage_final   mean(infra_dims) at the last time step
    resource_deficit     mean over time of max(0, initial_resource - resource)
    mean_hazard          mean over time and hazard dims (sanity check)
    """

    def __init__(self, world_model: WorldModel, cfg: CounterfactualConfig):
        self.model = world_model
        self.cfg   = cfg
        self.model.eval()

    def _state_slices(self) -> Dict[str, List[int]]:
        d = self.cfg.d_disaster_state
        q = max(d // 4, 1)
        return {
            "hazard": list(range(0, q)),
            "infra": list(range(q, 2 * q)),
            "exposure": list(range(2 * q, 3 * q)),
            "resource": list(range(3 * q, d)),
        }

    def _apply_intervention(
        self,
        warm_up: Tensor,
        spec: InterventionSpec,
    ) -> Tensor:
        """
        Apply a branch-point state intervention before posterior encoding.

        The perturbation is ramped over the final warm-up steps, then the world
        model converts the intervened state history into h/z and propagates it
        through learned latent dynamics.
        """
        if spec == InterventionSpec():
            return warm_up

        out = warm_up.clone()
        slices = self._state_slices()
        n_steps = max(1, round(out.shape[0] * spec.warmup_fraction))
        start = out.shape[0] - n_steps
        ramps = torch.linspace(
            1.0 / n_steps,
            1.0,
            n_steps,
            dtype=out.dtype,
            device=out.device,
        )
        deltas = {
            "hazard": spec.hazard_delta,
            "infra": spec.infra_delta,
            "exposure": spec.exposure_delta,
            "resource": spec.resource_delta,
        }
        for offset, scale in enumerate(ramps):
            t = start + offset
            for name, delta in deltas.items():
                if abs(delta) > 0:
                    out[t, slices[name]] = out[t, slices[name]] + scale * delta
        return out.clamp(0.0, 1.0)

    @torch.no_grad()
    def _rollout_once(
        self,
        warm_up: Tensor,
        intervention: InterventionSpec | None = None,
    ) -> Tensor:
        """Return one rollout trajectory (n_rollout_steps, d_state)."""
        branch = warm_up if intervention is None else self._apply_intervention(warm_up, intervention)
        return self.model.rollout(branch, self.cfg.n_rollout_steps)

    @torch.no_grad()
    def _rollout_mc(
        self,
        warm_up: Tensor,
        intervention: InterventionSpec | None = None,
    ) -> Tensor:
        """
        Monte Carlo rollout: average cfg.n_monte_carlo stochastic rollouts.
        Returns mean trajectory (n_rollout_steps, d_state).
        """
        samples = []
        for i in range(self.cfg.n_monte_carlo):
            # Common random numbers make scenario deltas less noisy while still
            # using stochastic RSSM rollouts.
            torch.manual_seed(self.cfg.seed + i)
            samples.append(self._rollout_once(warm_up, intervention))
        samples = torch.stack(samples)             # (MC, T, d)
        return samples.mean(dim=0)                 # (T, d)

    def _compute_metrics(self, traj: Tensor) -> Dict[str, float]:
        """
        Compute outcome metrics from a trajectory tensor (T, d_state).
        All values in [0, 1].
        """
        cfg = self.cfg
        T   = traj.shape[0]

        slices = self._state_slices()
        bounded = traj.clamp(0.0, 1.0)
        exp  = bounded[:, slices["exposure"]].mean(dim=-1)     # (T,)
        res  = bounded[:, slices["resource"]].mean(dim=-1)     # (T,)
        infr = bounded[:, slices["infra"]].mean(dim=-1)        # (T,)
        haz  = bounded[:, slices["hazard"]].mean(dim=-1)       # (T,)

        peak_exposure       = exp.max().item()
        # Exposure-weighted unmet-resource proxy.  The previous hard threshold
        # was degenerate for the demo RSSM because normalized resource values
        # stayed high for every scenario.  This continuous proxy remains an
        # outcome of the learned rollout state: exposure increases shortfall,
        # and available resources reduce it.
        unmet_resource      = torch.clamp(1.0 - res, min=0.0, max=1.0)
        shelter_shortfall   = (exp * unmet_resource).mean().item()
        infra_damage_final  = infr[-1].item()
        resource_deficit    = (0.5 * (exp + haz) * unmet_resource).mean().item()
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
        Run all scenarios on one warm-up sequence.
        Returns dict: name → {description, metrics, trajectory}.
        """
        results = {}
        for name, fn in SCENARIO_INTERVENTIONS.items():
            intervention = fn(self.cfg)
            traj    = self._rollout_mc(warm_up_seq, intervention)
            metrics = self._compute_metrics(traj)
            results[name] = {
                "description": SCENARIO_DESCRIPTIONS[name],
                "metrics":     metrics,
                "trajectory":  traj,
                "intervention": intervention,
            }
        return results

    def compare_multi_storm(
        self, warm_up_seqs: List[Tensor], return_per_sequence: bool = False,
    ):
        """Run all scenarios on each storm and average metrics.

        If return_per_sequence, also returns a long-format list of
        {sequence_id, scenario, <metric>: value, ...} rows (one per
        storm x scenario), needed for storm-level bootstrap uncertainty
        on scenario outcome deltas (E3/E4).
        """
        accum: Dict[str, List] = {name: [] for name in SCENARIO_INTERVENTIONS}
        per_sequence_rows: List[Dict[str, object]] = []
        for seq_idx, warm_up in enumerate(warm_up_seqs):
            for name, res in self.compare(warm_up).items():
                accum[name].append(res["metrics"])
                if return_per_sequence:
                    per_sequence_rows.append({
                        "sequence_id": seq_idx, "scenario": name,
                        **res["metrics"],
                    })
        averaged = {}
        for name in SCENARIO_INTERVENTIONS:
            rows = accum[name]
            keys = list(rows[0].keys())
            averaged[name] = {
                "description": SCENARIO_DESCRIPTIONS[name],
                "metrics":     {k: round(sum(r[k] for r in rows)/len(rows), 4) for k in keys},
                "n_storms":    len(warm_up_seqs),
            }
        if return_per_sequence:
            return averaged, per_sequence_rows
        return averaged

    def direct_mirror_diagnostics(self, results: Dict[str, dict]) -> List[Dict[str, object]]:
        """Check that scenario metric deltas are not equal to input deltas."""
        baseline = results["baseline"]["metrics"]
        checks = []
        scenario_to_metric = {
            "earlier_evacuation": ("peak_exposure", -self.cfg.evac_exposure_delta),
            "earlier_evacuation_24h": ("peak_exposure", -self.cfg.evac_24h_exposure_delta),
            "earlier_evacuation_36h": ("peak_exposure", -self.cfg.evac_36h_exposure_delta),
            "delayed_evacuation": ("peak_exposure", self.cfg.delayed_evac_exposure_delta),
            "shelter_failure": ("resource_deficit", self.cfg.shelter_failure_resource_delta),
            "hospital_failure": ("infra_damage_final", self.cfg.hospital_failure_infra_delta),
            "road_blockage": ("peak_exposure", self.cfg.road_blockage_exposure_delta),
            "intensity_increase": ("mean_hazard", self.cfg.intensity_delta),
            "intensity_decrease": ("mean_hazard", -self.cfg.intensity_delta),
            "additional_emergency_resources": ("resource_deficit", -self.cfg.additional_resource_delta),
        }
        for name, (metric, input_delta) in scenario_to_metric.items():
            observed_delta = results[name]["metrics"][metric] - baseline[metric]
            checks.append({
                "scenario": name,
                "metric": metric,
                "input_delta": round(input_delta, 6),
                "observed_delta": round(observed_delta, 6),
                "mirrors_input": abs(observed_delta - input_delta) < 1e-6,
            })
        return checks

    @staticmethod
    def print_report(results: Dict[str, dict], n_storms: int = 1) -> None:
        metric_names = [
            "peak_exposure", "shelter_shortfall",
            "infra_damage_final", "resource_deficit", "mean_hazard",
        ]
        col_w = 22
        sep   = "═" * 140
        print()
        print(sep)
        tag = f"averaged over {n_storms} test storms" if n_storms > 1 else "single storm"
        print(f"  Module 5 — Counterfactual Reasoning Engine  ({tag})")
        print(sep)
        header = f"  {'Scenario':<34}" + "".join(f"{m:>{col_w}}" for m in metric_names)
        print(header)
        print("  " + "─" * 138)
        baseline = results["baseline"]["metrics"]
        order = [
            "baseline",
            "earlier_evacuation",
            "earlier_evacuation_24h",
            "earlier_evacuation_36h",
            "delayed_evacuation",
            "additional_emergency_resources",
            "shelter_failure",
            "hospital_failure",
            "road_blockage",
            "intensity_increase",
            "intensity_decrease",
        ]
        prev_g = None
        for name in order:
            if name not in results: continue
            g = ("beneficial" if name in {"earlier_evacuation", "earlier_evacuation_24h", "earlier_evacuation_36h", "additional_emergency_resources", "intensity_decrease"}
                 else "adverse" if name != "baseline" else "ref")
            if g != prev_g and prev_g is not None:
                print("  " + "·" * 138)
            prev_g = g
            m   = results[name]["metrics"]
            row = f"  {name:<34}"
            for mn in metric_names:
                val   = m[mn]
                delta = val - baseline[mn]
                if name == "baseline":
                    cell = f"{val:.4f}"
                else:
                    sign = "↑" if delta > 0 else "↓"
                    cell = f"{val:.4f} ({sign}{abs(delta):.4f})"
                row += f"{cell:>{col_w}}"
            print(row)
        print()
        print("  Metric guide (all lower = better):")
        print("    peak_exposure      — max population exposure across horizon")
        print("    shelter_shortfall  — exposure-weighted unmet-resource proxy")
        print("    infra_damage_final — infrastructure damage at end of rollout")
        print("    resource_deficit   — hazard/exposure-weighted unmet-resource proxy")
        print("    mean_hazard        — mean storm hazard (sanity check only)")
        print("  ↑ = worse than baseline  |  ↓ = better than baseline")
        print(sep)
        if {"earlier_evacuation", "earlier_evacuation_24h", "earlier_evacuation_36h", "delayed_evacuation"}.issubset(results):
            e12 = results["earlier_evacuation"]["metrics"]["peak_exposure"]
            e24 = results["earlier_evacuation_24h"]["metrics"]["peak_exposure"]
            e36 = results["earlier_evacuation_36h"]["metrics"]["peak_exposure"]
            late = results["delayed_evacuation"]["metrics"]["peak_exposure"]
            monotonic = e36 <= e24 <= e12 <= baseline["peak_exposure"] <= late
            print(
                f"\n  Evacuation lead-time ordering  "
                f"36h_early={e36:.4f}  24h_early={e24:.4f}  12h_early={e12:.4f}  "
                f"baseline={baseline['peak_exposure']:.4f}  delayed={late:.4f}"
                f"  monotonic={'yes' if monotonic else 'no'}\n"
            )
