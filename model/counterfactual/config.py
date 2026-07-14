"""
CounterfactualConfig — configuration for the Counterfactual Reasoning Engine
(Module 5).

The engine takes a trained World Model (Module 4), generates a baseline
trajectory from an initial disaster state, then replays the same initial state
under five different interventions and compares outcomes.

Scenarios simulated
-------------------
  1. early_evacuation       Mobility of population clusters increased 40 %
                            for the first 4 time steps (people leave sooner).
  2. shelter_failure        One shelter's capacity drops to zero (e.g. flooding).
  3. storm_intensification  Hazard component of the latent state scaled ×1.20.
  4. extra_resources        Hospital capacity +50 %, shelter supplies +30 %.
  5. route_failure          Transportation connectivity removed from the latent
                            state (road network fails due to storm damage).

Outcome metrics computed per trajectory
-----------------------------------------
  peak_exposure        Maximum population exposure across all steps
  shelter_shortfall    Proportion of steps where demand exceeds capacity
  infra_damage_final   Infrastructure damage level at the end of the rollout
  resource_deficit     Mean resource-demand gap across the trajectory
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class CounterfactualConfig:
    # ── World Model interface ──────────────────────────────────────────────────
    d_disaster_state: int = 32   # must match WorldModelConfig.d_disaster_state
    d_latent:         int = 32   # must match WorldModelConfig.d_latent

    # ── Rollout ────────────────────────────────────────────────────────────────
    n_rollout_steps: int = 20    # forecast horizon for each scenario
    n_initial_steps: int = 3     # warm-up steps before the branch point
    n_monte_carlo:   int = 10    # MC samples per scenario (stochastic rollout)

    # ── Scenario intervention strengths ───────────────────────────────────────
    evac_boost:           float = 0.40   # mobility increase for early evacuation
    shelter_fail_idx:     int   = 0      # which shelter fails (0-indexed)
    storm_intensity_scale: float = 1.20  # multiplicative scale on hazard dims
    resource_hospital_boost: float = 0.50
    resource_shelter_boost:  float = 0.30
    route_fail_scale:     float = 0.0    # transport dims set to this fraction

    # ── Latent space semantic slices (assume d_latent=32, split into 4×8) ──────
    # These tell the engine which dimensions of z correspond to each concept.
    # If d_latent changes, adjust these accordingly.
    hazard_dims:    List[int] = field(default_factory=lambda: list(range(0, 8)))
    infra_dims:     List[int] = field(default_factory=lambda: list(range(8, 16)))
    exposure_dims:  List[int] = field(default_factory=lambda: list(range(16, 24)))
    resource_dims:  List[int] = field(default_factory=lambda: list(range(24, 32)))

    # ── Output ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/world_model"
    metrics_dir:    str = "metrics/counterfactual"

    seed: int = 42
    demo: bool = False

    def apply_demo_overrides(self) -> "CounterfactualConfig":
        self.demo             = True
        self.d_latent         = 16
        self.n_rollout_steps  = 12
        self.n_monte_carlo    = 5
        # Re-slice for d_latent=16 (4×4)
        self.hazard_dims   = list(range(0, 4))
        self.infra_dims    = list(range(4, 8))
        self.exposure_dims = list(range(8, 12))
        self.resource_dims = list(range(12, 16))
        return self

    def __str__(self) -> str:
        tag = "[DEMO] " if self.demo else ""
        return (f"{tag}CounterfactualConfig | "
                f"horizon={self.n_rollout_steps} steps | "
                f"MC samples={self.n_monte_carlo} | 5 scenarios")
