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
      uncertainty_drift_by_horizon.csv
      source_manifest.csv
  scripts/
    build_workshop_results.py
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
```

The script reads existing frozen artifacts under `metrics/`, `tables/`, and `figures/`, then writes workshop-specific outputs under `neurips_storm_world/results/`.

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

