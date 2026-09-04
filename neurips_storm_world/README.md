# NeurIPS 2026 STORM-World Workshop Package

This folder isolates the workshop-specific reframing and results for:

```text
STORM-World: A Physics-Informed Intervention-Aware World Model for Counterfactual Storm Simulation
```

It does not overwrite the existing STORM-CARE/AAAI manuscript artifacts.

## Central Novelty

The workshop paper should not be framed as another hurricane-forecasting paper. The novelty is:

- an explicit action-conditioned latent world-model formulation;
- branching the same observed storm-disaster state into alternative intervention-conditioned futures;
- evaluating physical fidelity, uncertainty drift, intervention consistency, and decision-ranking preservation;
- reporting failed intervention channels as evidence that forecast accuracy alone is not enough.

Recommended central message:

```text
A useful physical world model must do more than forecast accurately: it must preserve physical dynamics, represent uncertainty, respond coherently to interventions, and preserve meaningful action rankings.
```

## Folder Layout

```text
neurips_storm_world/
  manuscript/
    STORM_World_workshop_draft.md
  results/
    figures/
      storm_world_architecture.svg
      physics_residuals_full_vs_ablation.png
      physics_residuals_full_vs_ablation.pdf
      calibration.png
      calibration.pdf
    tables/
      workshop_evaluation_scorecard.csv
      intervention_fidelity_matrix.csv
      policy_ranking_preservation.csv
      policy_ranking_by_sequence.csv
      evacuation_dose_response.csv
      evacuation_adjacent_significance.csv
      physical_fidelity_ablation.csv
      physics_consistency_vs_rollout_horizon.csv
      rollout_fidelity_vs_horizon.csv
      deterministic_vs_stochastic_rollout.csv
      stochastic_rollout_coverage_sharpness.csv
      world_model_vs_direct_predictor.csv
      intervention_conditioning_ablation.csv
      aots2action_bridge_summary.csv
      remaining_guideline_feasibility_status.csv
      uncertainty_drift_by_horizon.csv
      source_manifest.csv
  scripts/
    build_workshop_results.py
    build_additional_workshop_experiments.py
```

## Strongest Result To Lead With

Evacuation timing gives the cleanest world-model controllability and decision-fidelity result:

```text
evacuation 36 h earlier > evacuation 24 h earlier > evacuation 12 h earlier > baseline > delayed evacuation
```

Lower peak exposure is better. The observed aggregate ordering matches the expected ordering.

Decision-ranking preservation:

- Spearman rank correlation: `0.975`
- Kendall tau: `0.958`
- top-1 action accuracy: `21/24 = 0.875`
- pairwise ranking accuracy: `235/240 = 0.979`
- strict full-order accuracy: `21/24 = 0.875`

Source table: `results/tables/policy_ranking_preservation.csv`

## Failure Story To Keep

Do not hide the weak channels. The current model responds coherently to evacuation timing and exposure-level storm-intensity perturbations, but infrastructure/resource interventions remain unreliable or approximately null.

Use this as the workshop thesis:

```text
Predictive accuracy is not enough. Physical world models should be evaluated for physical consistency, uncertainty calibration, intervention controllability, and decision-ranking fidelity, including explicit failure modes.
```

Source table: `results/tables/intervention_fidelity_matrix.csv`

## Regenerate Results

From the repository root:

```bash
python3 neurips_storm_world/scripts/build_workshop_results.py
python3 neurips_storm_world/scripts/build_additional_workshop_experiments.py
python3 neurips_storm_world/scripts/build_workshop_results.py
```

The first script reads existing frozen artifacts under `metrics/`, `tables/`, and `figures/`, then writes workshop-specific outputs under `neurips_storm_world/results/`. The additional experiment script adds lightweight horizon, stochasticity, direct-predictor, intervention-conditioning, and AOTS bridge analyses without overwriting the main STORM-CARE checkpoint.

## Guideline Coverage

Added after the initial package:

- physics consistency versus rollout horizon: `results/tables/physics_consistency_vs_rollout_horizon.csv`
- rollout error versus horizon: `results/tables/rollout_fidelity_vs_horizon.csv`
- deterministic versus stochastic rollout: `results/tables/deterministic_vs_stochastic_rollout.csv`
- stochastic coverage and sharpness: `results/tables/stochastic_rollout_coverage_sharpness.csv`
- world model versus direct predictors: `results/tables/world_model_vs_direct_predictor.csv`
- branch-state intervention-conditioning ablation: `results/tables/intervention_conditioning_ablation.csv`
- AOTS2Action real-geospatial bridge summary: `results/tables/aots2action_bridge_summary.csv`

Still not fully completed:

- larger world-model training scale;
- repairing weak intervention channels;
- true multi-basin evaluation;
- full STORM-World rollout tensors coupled into AOTS2Action's real geospatial exposure pipeline.

## Writing Instructions

Use the workshop draft as the starting point:

```text
neurips_storm_world/manuscript/STORM_World_workshop_draft.md
```

Keep the manuscript to three contributions:

1. Physically grounded storm world model.
2. Intervention-conditioned imagination.
3. Evaluation beyond forecasting accuracy.

Avoid these claims unless new causal identification experiments are added:

- the model estimates real-world causal evacuation effects;
- all interventions work;
- STORM-World beats persistence or operational baselines at all forecast horizons.
