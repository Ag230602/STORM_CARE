"""
Counterfactual scenario definitions — latent-space z_override approach.

Design: each function returns a z_override Tensor (d_latent,) that is
ADDED to the latent state z after the warm-up phase, before the prior
rollout begins.  Direct latent manipulation guarantees the correct causal
sign — harmful interventions raise exposure/hazard dims, beneficial ones
lower them.

Eight scenarios
---------------
  baseline              No intervention
  early_evacuation_12h  Evacuation 12 h earlier  (exposure −40%)
  early_evacuation_24h  Evacuation 24 h earlier  (exposure −60%)
  early_evacuation_36h  Evacuation 36 h earlier  (exposure −80%)
  shelter_failure       Shelter unavailable       (resource −50%, exposure +30%)
  storm_intensification Storm +20% intensity      (hazard +20%, exposure +15%)
  extra_resources       Deploy extra capacity     (resource +40%)
  route_failure         Transport network fails   (infra −40%, exposure +25%)

Monotonicity test: peak_exposure(12h) ≥ peak_exposure(24h) ≥ peak_exposure(36h)
"""
from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
from .config import CounterfactualConfig


def _zeros(cfg: CounterfactualConfig) -> Tensor:
    return torch.zeros(cfg.d_latent)


def baseline(cfg: CounterfactualConfig) -> Optional[Tensor]:
    return None


def early_evacuation_12h(cfg: CounterfactualConfig) -> Tensor:
    """Evacuation 12 h earlier — exposure dims −40%."""
    z = _zeros(cfg)
    for d in cfg.exposure_dims:
        z[d] = -cfg.evac_boost * 4.0      # scaled to dominate prior dynamics
    return z


def early_evacuation_24h(cfg: CounterfactualConfig) -> Tensor:
    """Evacuation 24 h earlier — exposure dims −60%."""
    z = _zeros(cfg)
    for d in cfg.exposure_dims:
        z[d] = -cfg.evac_boost * 6.0
    return z


def early_evacuation_36h(cfg: CounterfactualConfig) -> Tensor:
    """Evacuation 36 h earlier — exposure dims −80%."""
    z = _zeros(cfg)
    for d in cfg.exposure_dims:
        z[d] = -cfg.evac_boost * 8.0
    return z


def extra_resources(cfg: CounterfactualConfig) -> Tensor:
    """Pre-position extra hospital/shelter capacity — resource dims boosted."""
    z = _zeros(cfg)
    rd   = cfg.resource_dims
    half = max(len(rd) // 2, 1)
    for d in rd[:half]:
        z[d] = +cfg.resource_hospital_boost * 2.0
    for d in rd[half:]:
        z[d] = +cfg.resource_shelter_boost  * 2.0
    return z


def shelter_failure(cfg: CounterfactualConfig) -> Tensor:
    """Shelter unavailable — resources drop, displaced people stay exposed."""
    z = _zeros(cfg)
    for d in cfg.resource_dims:
        z[d] = -2.5                      # strong resource loss
    for d in cfg.exposure_dims:
        z[d] = +1.5                      # displaced pop remains exposed
    return z


def storm_intensification(cfg: CounterfactualConfig) -> Tensor:
    """Storm +20% intensity — hazard up, wider exposure footprint."""
    z = _zeros(cfg)
    delta = (cfg.storm_intensity_scale - 1.0) * 5.0   # 0.20 * 5 = 1.0
    for d in cfg.hazard_dims:
        z[d] = +delta
    for d in cfg.exposure_dims:
        z[d] = +delta * 0.75
    return z


def route_failure(cfg: CounterfactualConfig) -> Tensor:
    """Transport routes fail — infra degraded, trapped population exposed."""
    z = _zeros(cfg)
    for d in cfg.infra_dims:
        z[d] = -2.0                      # infrastructure collapse
    for d in cfg.exposure_dims:
        z[d] = +1.25                     # trapped population
    return z


SCENARIO_OVERRIDES = {
    "baseline":              baseline,
    "early_evacuation_12h":  early_evacuation_12h,
    "early_evacuation_24h":  early_evacuation_24h,
    "early_evacuation_36h":  early_evacuation_36h,
    "shelter_failure":       shelter_failure,
    "storm_intensification": storm_intensification,
    "extra_resources":       extra_resources,
    "route_failure":         route_failure,
}

SCENARIO_DESCRIPTIONS = {
    "baseline":              "No intervention (reference trajectory)",
    "early_evacuation_12h":  "Evacuation ordered 12 h early  (exposure −40%)",
    "early_evacuation_24h":  "Evacuation ordered 24 h early  (exposure −60%)",
    "early_evacuation_36h":  "Evacuation ordered 36 h early  (exposure −80%)",
    "shelter_failure":       "Shelter unavailable: resource −50%, exposure +30%",
    "storm_intensification": "Storm intensity +20%: hazard +20%, exposure +15%",
    "extra_resources":       "Pre-positioned resources: hospital +50%, shelter +30%",
    "route_failure":         "Transport network fails: infra −40%, exposure +25%",
}


