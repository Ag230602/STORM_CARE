"""
Counterfactual scenario definitions for Module 5.

Each function takes the initial warm-up sequence and returns a modified
copy that represents the "what-if" intervention.  The modifications are
applied in the disaster-state space (d_disaster_state-dim vectors).

Semantic layout of the disaster state vector
--------------------------------------------
The WorldModel is trained without enforcing a fixed layout, but by convention
CounterfactualConfig.hazard_dims / infra_dims / exposure_dims / resource_dims
index into the d_disaster_state-dim vector.  When the disaster state comes
from DisasterGNN.state_head (global mean pool), these dimensions carry soft
interpretable meaning learned during training.

Scenarios
---------
1. early_evacuation       People leave 40 % sooner → exposure dims reduced
2. shelter_failure        One shelter offline → resource dims reduced
3. storm_intensification  Storm 20 % stronger → hazard dims scaled up
4. extra_resources        More hospitals + supplies → resource dims boosted
5. route_failure          Roads fail → infra dims degraded
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import CounterfactualConfig


def early_evacuation(
    seq: Tensor,
    cfg: CounterfactualConfig,
) -> Tensor:
    """
    Scenario 1 — What if evacuation starts 12 hours earlier?

    Effect: Population in the exposure dimensions is reduced by evac_boost
    for the first n_initial_steps (evacuation already happened).
    """
    seq = seq.clone()
    for t in range(min(cfg.n_initial_steps + 2, seq.shape[0])):
        seq[t, cfg.exposure_dims] *= (1.0 - cfg.evac_boost)
    return seq


def shelter_failure(
    seq: Tensor,
    cfg: CounterfactualConfig,
) -> Tensor:
    """
    Scenario 2 — What if a shelter becomes unavailable?

    Effect: Resource dimensions reduced (capacity offline).
    The first resource_dim entry is zeroed as a proxy for one shelter failing.
    """
    seq = seq.clone()
    if cfg.resource_dims:
        fail_dim = cfg.resource_dims[cfg.shelter_fail_idx % len(cfg.resource_dims)]
        seq[:, fail_dim] = 0.0
    return seq


def storm_intensification(
    seq: Tensor,
    cfg: CounterfactualConfig,
) -> Tensor:
    """
    Scenario 3 — What if storm intensity increases by 20 %?

    Effect: Hazard dimensions scaled by storm_intensity_scale.
    """
    seq = seq.clone()
    seq[:, cfg.hazard_dims] *= cfg.storm_intensity_scale
    return seq


def extra_resources(
    seq: Tensor,
    cfg: CounterfactualConfig,
) -> Tensor:
    """
    Scenario 4 — What if additional emergency resources are deployed?

    Effect: Resource dimensions boosted (hospitals + shelters reinforced).
    Hospital boost applied to first half of resource dims,
    shelter boost to second half.
    """
    seq = seq.clone()
    rd  = cfg.resource_dims
    half = max(len(rd) // 2, 1)
    hosp_dims = rd[:half]
    shlt_dims = rd[half:]
    seq[:, hosp_dims] *= (1.0 + cfg.resource_hospital_boost)
    seq[:, shlt_dims] *= (1.0 + cfg.resource_shelter_boost)
    return seq


def route_failure(
    seq: Tensor,
    cfg: CounterfactualConfig,
) -> Tensor:
    """
    Scenario 5 — What if transportation routes fail?

    Effect: Infrastructure dimensions degraded (transport-dependent capacity lost).
    """
    seq = seq.clone()
    seq[:, cfg.infra_dims] *= cfg.route_fail_scale
    return seq


# Registry: scenario name → function
SCENARIOS = {
    "baseline":             lambda seq, cfg: seq.clone(),
    "early_evacuation":     early_evacuation,
    "shelter_failure":      shelter_failure,
    "storm_intensification": storm_intensification,
    "extra_resources":      extra_resources,
    "route_failure":        route_failure,
}

SCENARIO_DESCRIPTIONS = {
    "baseline":              "No intervention (reference trajectory)",
    "early_evacuation":      "Evacuation starts 12 h earlier (−40 % exposure)",
    "shelter_failure":       "One shelter becomes unavailable (flooding)",
    "storm_intensification": "Storm intensity increases by 20 %",
    "extra_resources":       "Extra hospital (+50 %) and shelter (+30 %) capacity",
    "route_failure":         "Transportation routes fail (roads blocked)",
}
