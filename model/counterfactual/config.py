"""
CounterfactualConfig — configuration for the Counterfactual Reasoning Engine
(Module 5).

The engine takes a trained World Model (Module 4), generates a baseline
trajectory from an initial disaster state, then replays the same initial state
under branch-point interventions and compares outcomes.

Scenarios simulated
-------------------
  1. earlier_evacuation (~12h earlier lead time)
  2. earlier_evacuation_24h (~24h earlier lead time; monotonicity check vs. #1)
  3. earlier_evacuation_36h (~36h earlier lead time; dose-response sweep, E3)
  4. delayed_evacuation
  5. shelter_failure
  6. hospital_failure
  7. road_blockage
  8. intensity_increase
  9. intensity_decrease
  10. additional_emergency_resources

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
    n_test_sequences: int | None = None  # None means use complete held-out split

    # ── Scenario intervention strengths ───────────────────────────────────────
    # Additive normalized branch-state interventions.  These are not reported as
    # expected percentage outcome changes; they are inputs to the learned RSSM.
    evac_exposure_delta: float = 0.12          # earlier_evacuation, ~12h lead
    evac_24h_exposure_delta: float = 0.20       # earlier_evacuation_24h, ~24h lead
    evac_36h_exposure_delta: float = 0.28        # earlier_evacuation_36h, ~36h lead (E3 dose-response)
    delayed_evac_exposure_delta: float = 0.12
    shelter_failure_resource_delta: float = 0.18
    hospital_failure_infra_delta: float = 0.12
    hospital_failure_resource_delta: float = 0.10
    road_blockage_infra_delta: float = 0.09
    road_blockage_exposure_delta: float = 0.08
    road_blockage_resource_delta: float = 0.08
    intensity_delta: float = 0.10
    additional_resource_delta: float = 0.14

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
                f"MC samples={self.n_monte_carlo} | 9 interventions + baseline")
