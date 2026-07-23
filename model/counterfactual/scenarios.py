"""
Counterfactual scenario definitions.

Scenarios are encoded as branch-point interventions on the observed disaster
state before RSSM posterior encoding.  The intervention never edits decoded
rollout outputs.  Outcome changes must therefore propagate through:

    intervened warm-up state -> posterior latent state -> prior rollout -> decoder
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import CounterfactualConfig


@dataclass(frozen=True)
class InterventionSpec:
    """Additive do-operator applied to semantic slices of the warm-up state."""

    hazard_delta: float = 0.0
    infra_delta: float = 0.0
    exposure_delta: float = 0.0
    resource_delta: float = 0.0
    warmup_fraction: float = 1.0


def baseline(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec()


def earlier_evacuation(cfg: CounterfactualConfig) -> InterventionSpec:
    """~12h earlier evacuation order."""
    return InterventionSpec(
        exposure_delta=-cfg.evac_exposure_delta,
        resource_delta=-0.25 * cfg.evac_exposure_delta,
        warmup_fraction=1.0,
    )


def earlier_evacuation_24h(cfg: CounterfactualConfig) -> InterventionSpec:
    """~24h earlier evacuation order. Larger exposure reduction than the
    12h case (earlier_evacuation) to test lead-time monotonicity."""
    return InterventionSpec(
        exposure_delta=-cfg.evac_24h_exposure_delta,
        resource_delta=-0.25 * cfg.evac_24h_exposure_delta,
        warmup_fraction=1.0,
    )


def delayed_evacuation(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec(
        exposure_delta=cfg.delayed_evac_exposure_delta,
        resource_delta=-0.15 * cfg.delayed_evac_exposure_delta,
        warmup_fraction=1.0,
    )


def shelter_failure(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec(
        infra_delta=0.04,
        exposure_delta=0.06,
        resource_delta=-cfg.shelter_failure_resource_delta,
        warmup_fraction=1.0,
    )


def hospital_failure(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec(
        infra_delta=cfg.hospital_failure_infra_delta,
        resource_delta=-cfg.hospital_failure_resource_delta,
        warmup_fraction=1.0,
    )


def road_blockage(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec(
        infra_delta=cfg.road_blockage_infra_delta,
        exposure_delta=cfg.road_blockage_exposure_delta,
        resource_delta=-cfg.road_blockage_resource_delta,
        warmup_fraction=1.0,
    )


def intensity_increase(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec(
        hazard_delta=cfg.intensity_delta,
        infra_delta=0.25 * cfg.intensity_delta,
        exposure_delta=0.35 * cfg.intensity_delta,
        warmup_fraction=1.0,
    )


def intensity_decrease(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec(
        hazard_delta=-cfg.intensity_delta,
        infra_delta=-0.20 * cfg.intensity_delta,
        exposure_delta=-0.30 * cfg.intensity_delta,
        warmup_fraction=1.0,
    )


def additional_emergency_resources(cfg: CounterfactualConfig) -> InterventionSpec:
    return InterventionSpec(
        exposure_delta=-0.20 * cfg.additional_resource_delta,
        resource_delta=cfg.additional_resource_delta,
        warmup_fraction=1.0,
    )


SCENARIO_INTERVENTIONS = {
    "baseline": baseline,
    "earlier_evacuation": earlier_evacuation,
    "earlier_evacuation_24h": earlier_evacuation_24h,
    "delayed_evacuation": delayed_evacuation,
    "shelter_failure": shelter_failure,
    "hospital_failure": hospital_failure,
    "road_blockage": road_blockage,
    "intensity_increase": intensity_increase,
    "intensity_decrease": intensity_decrease,
    "additional_emergency_resources": additional_emergency_resources,
}


SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "baseline": "No intervention; learned world-model rollout from observed warm-up",
    "earlier_evacuation": "Earlier evacuation branch-state intervention (~12h lead)",
    "earlier_evacuation_24h": "Earlier evacuation branch-state intervention (~24h lead)",
    "delayed_evacuation": "Delayed evacuation branch-state intervention",
    "shelter_failure": "Shelter capacity failure branch-state intervention",
    "hospital_failure": "Hospital service failure branch-state intervention",
    "road_blockage": "Road blockage / transport disruption branch-state intervention",
    "intensity_increase": "Storm intensity increase branch-state intervention",
    "intensity_decrease": "Storm intensity decrease branch-state intervention",
    "additional_emergency_resources": "Additional emergency resources branch-state intervention",
}
