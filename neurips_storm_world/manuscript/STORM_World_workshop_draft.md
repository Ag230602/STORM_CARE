# STORM-World: A Physics-Informed Intervention-Aware World Model for Counterfactual Storm Simulation

Target workshop: World Models in Physical AI, NeurIPS 2026  
Status: workshop-specific draft built from existing STORM-CARE and AOTS2Action artifacts  
Result folder: `neurips_storm_world/results/`

## Abstract

World models are increasingly expected to do more than predict the next state of a physical environment: they must support reasoning about how the environment would evolve under alternative actions. Yet predictive accuracy alone does not establish that an imagined future is physically plausible, uncertainty-aware, or intervention-faithful. Tropical cyclones provide a challenging testbed because atmospheric dynamics interact with infrastructure, population exposure, and human response over long horizons. We present STORM-World, a physics-informed, intervention-aware latent world model for simulating alternative storm-impact futures. STORM-World encodes a coupled storm-disaster state, rolls it forward with recurrent latent dynamics, and branches a common observed state into intervention-conditioned imagined futures. We evaluate the model beyond forecast error, using physical residuals, uncertainty drift, intervention consistency, and decision-ranking preservation. On the current held-out demo checkpoint, evacuation timing produces a stable dose-response: 36 h earlier evacuation yields lower peak exposure than 24 h earlier, 12 h earlier, baseline, and delayed evacuation. Sequence-level policy-ranking preservation is high, with Spearman correlation 0.975, Kendall tau 0.958, and 235/240 correct pairwise action rankings. However, several infrastructure/resource interventions fail or are approximately null. These mixed findings support the central thesis that useful physical world models must be evaluated for controllability and intervention fidelity, not forecast accuracy alone.

## 1. Introduction

Most data-driven cyclone models are evaluated as forecasters: given an observed storm state, they predict future track, intensity, or downstream impact. That is necessary but incomplete for physical AI. Decision-makers often need to ask a different question: how would the coupled storm-disaster state evolve under alternative actions or perturbations?

This paper reframes STORM-CARE as STORM-World: a learned physical world model that imagines alternative futures from the same observed warm-up state. The model is not claimed to identify real-world causal effects of evacuation, shelter failure, or road blockage from observational data. We use counterfactual to denote model-generated intervention scenarios: alternative rollouts inside a learned latent dynamics model after an intervention-conditioned branch.

The central question is:

Can a learned world model simulate physically plausible storm-disaster futures and preserve meaningful intervention effects under alternative actions?

Our contributions are:

1. Physically grounded storm world model. We formulate coupled storm, infrastructure, population, and response dynamics as a structured latent world model for open-loop multi-step simulation.
2. Intervention-conditioned imagination. We branch a common observed world state into alternative future trajectories under evacuation timing and environmental/system perturbations.
3. Evaluation beyond forecasting accuracy. We evaluate predictive fidelity, physical consistency, uncertainty drift, intervention consistency, action-ranking preservation, and explicit failure modes.

## 2. Related Work

STORM-World sits at the intersection of latent world models, physical inductive bias, neural operators, AI weather forecasting, and probabilistic ensemble evaluation. General world models learn compact latent dynamics for imagined rollouts; action-conditioned variants use those rollouts for planning. AI weather systems such as FourCastNet, Pangu-Weather, GraphCast, GenCast, NeuralGCM, and Aurora primarily advance deterministic or probabilistic forecast skill. STORM-World takes a complementary position: it asks whether a physics-informed storm-disaster model preserves intervention directionality and action rankings when rolled forward under alternative decisions.

## 3. Problem Formulation

Let the coupled world state be:

```text
W_t = {H_t, I_t, P_t, R_t}
```

where `H_t` is hazard/storm state, `I_t` is infrastructure state, `P_t` is population/exposure state, and `R_t` is response/resource state.

The model encodes:

```text
z_t = E(W_t)
```

and transitions according to:

```text
z_{t+1} ~ p_theta(z_{t+1} | z_t, a_t, c_t)
```

where `a_t` denotes a human intervention or response action and `c_t` denotes exogenous forcing such as storm-intensity perturbation or external system failure. A decoded future is:

```text
W_hat_{t+1} = D(z_{t+1})
```

An intervention-conditioned rollout is:

```text
tau_hat^(a) = {W_hat_{t+1}^(a), ..., W_hat_{t+H}^(a)}
```

All action branches start from the same observed warm-up state. The paper compares factual/baseline rollouts against multiple intervention-conditioned rollouts.

## 4. STORM-World Method

STORM-World reuses the strongest defensible pieces of STORM-CARE:

- frozen storm-level train/validation/test splits;
- physics-informed graph/operator representation;
- pretrained encoder and latent recurrent state-space model;
- open-loop rollout evaluation;
- physics ablations;
- uncertainty calibration;
- evacuation dose-response experiments;
- explicit intervention failure cases.

The workshop-specific view is shown in `neurips_storm_world/results/figures/storm_world_architecture.svg`. The key visual story is one observed world branching into multiple imagined futures under different actions.

The current implementation uses branch-state interventions rather than direct edits to decoded outputs. Each intervention modifies the warm-up branch state, encodes the intervened sequence through the RSSM posterior, rolls forward through learned latent dynamics, and decodes the future state. This prevents the result from merely mirroring manually injected output percentages.

## 5. Experimental Design

We evaluate five dimensions.

Predictive world fidelity measures whether rollouts remain accurate as horizon increases. Existing forecast/baseline tables should be used carefully: current results do not support an all-horizon superiority claim over Persistence. The workshop package also includes a lightweight RSSM rollout-versus-horizon analysis in `neurips_storm_world/results/tables/rollout_fidelity_vs_horizon.csv`.

Physical fidelity measures whether physics-informed learning reduces implausible dynamics. The workshop package reports the full-vs-no-physics residual table in `neurips_storm_world/results/tables/physical_fidelity_ablation.csv` and a horizon-indexed disaster-state consistency proxy in `neurips_storm_world/results/tables/physics_consistency_vs_rollout_horizon.csv`.

Probabilistic fidelity measures uncertainty quality over horizon. The package reports P50/P90 coverage drift in `neurips_storm_world/results/tables/uncertainty_drift_by_horizon.csv` and stochastic rollout coverage/sharpness in `neurips_storm_world/results/tables/stochastic_rollout_coverage_sharpness.csv`.

Intervention fidelity measures whether action-conditioned rollouts change in the expected direction. We define intervention consistency:

```text
C(a) = (1 / N) sum_i 1[sign(Delta_i^a) = s_a]
```

where `s_a` is the expected sign of the intervention effect.

Decision fidelity measures whether the world model preserves action rankings. For an intervention family, a useful model should choose and rank policies similarly to a reference ordering even if state prediction is imperfect.

The package includes two additional workshop ablations. First, `neurips_storm_world/results/tables/world_model_vs_direct_predictor.csv` compares the RSSM rollout to direct persistence and linear extrapolation predictors. Second, `neurips_storm_world/results/tables/intervention_conditioning_ablation.csv` removes branch-state conditioning by reusing the baseline warm-up state for all intervention labels.

## 6. Results

### 6.1 Evacuation Dose-Response

Evacuation timing is the strongest intervention story. Mean peak exposure decreases monotonically as evacuation happens earlier:

| Scenario | Mean peak exposure | 95% CI |
| --- | ---: | --- |
| Delayed evacuation | 0.2969 | reported in source outcome table |
| Baseline | 0.2915 | [0.2842, 0.2990] |
| Evacuation 12 h earlier | 0.2831 | [0.2754, 0.2910] |
| Evacuation 24 h earlier | 0.2787 | [0.2715, 0.2864] |
| Evacuation 36 h earlier | 0.2754 | [0.2689, 0.2824] |

The workshop-ready dose-response table is `neurips_storm_world/results/tables/evacuation_dose_response.csv`, with adjacent significance tests in `neurips_storm_world/results/tables/evacuation_adjacent_significance.csv`.

### 6.2 Policy-Ranking Preservation

For evacuation timing, the expected best-to-worst order is:

```text
evacuation 36 h earlier > evacuation 24 h earlier > evacuation 12 h earlier > baseline > delayed evacuation
```

The observed aggregate order matches this exactly. Across 24 held-out sequences, policy-ranking preservation is:

| Metric | Value |
| --- | ---: |
| Mean Spearman rank correlation | 0.975 |
| Mean Kendall tau | 0.958 |
| Top-1 action accuracy | 0.875 |
| Pairwise intervention-ranking accuracy | 0.979 |
| Strict full-order accuracy | 0.875 |

These values are in `neurips_storm_world/results/tables/policy_ranking_preservation.csv`, with per-sequence details in `neurips_storm_world/results/tables/policy_ranking_by_sequence.csv`.

This is the strongest novelty-facing result: the model preserves a meaningful decision ordering, not just a scalar forecast value.

### 6.3 Intervention Consistency and Failure Modes

The macro intervention consistency rate is 0.771 across the currently tested actions. Evacuation-timing interventions are fully consistent across all 24 held-out sequences. Storm-intensity perturbations are directionally consistent for exposure but weak in magnitude, so we mark them partial. Road blockage, shelter failure, hospital failure, and additional-resource interventions are reported as failures or approximately null responses.

The complete intervention-fidelity matrix is `neurips_storm_world/results/tables/intervention_fidelity_matrix.csv`.

This failure audit should be framed as a scientific contribution. The current model can forecast, preserve some physical structure, and respond coherently to evacuation timing while still failing to encode several infrastructure/resource intervention channels. That is exactly the workshop-relevant point: predictive skill is not sufficient evidence of an intervention-faithful physical world model.

### 6.4 Physical and Probabilistic Fidelity

The physics table in `neurips_storm_world/results/tables/physical_fidelity_ablation.csv` reports residual components for the full physics model and no-physics ablation. Use this to support the claim that physical grounding is measured directly, not assumed from forecast accuracy.

The horizon consistency table in `neurips_storm_world/results/tables/physics_consistency_vs_rollout_horizon.csv` adds rollout-step diagnostics for boundedness, monotonic infrastructure damage, resource-replenishment violations, temporal continuity, and exposure-hazard consistency. These are normalized disaster-state consistency proxies, not full ERA5/PDE residuals.

The uncertainty table in `neurips_storm_world/results/tables/uncertainty_drift_by_horizon.csv` shows that calibration is stronger at short horizons and degrades at 72-120 h. This should be described as long-horizon world-model uncertainty drift. The stochastic rollout table in `neurips_storm_world/results/tables/deterministic_vs_stochastic_rollout.csv` compares the deterministic RSSM mean path against stochastic sample-mean futures.

### 6.5 AOTS2Action Bridge

The file `neurips_storm_world/results/tables/aots2action_bridge_summary.csv` summarizes real-geospatial AOTS2Action exposure, Brier, and regional ranking results for the ensemble probability-weighted estimator. This should be used as bridge evidence that a downstream real-geospatial evaluator exists. It should not be described as a fully coupled STORM-World rollout-to-geospatial intervention experiment unless that coupling is implemented later.

## 7. Discussion and Limitations

STORM-World should not claim causal identification of real-world intervention effects. It supports model-internal intervention-conditioned rollouts. The evacuation dose-response and ranking preservation results show controllability in one action family, while infrastructure/resource failures show that not all interventions are faithfully represented.

The current checkpoint is demo-scale, so effect magnitudes should be interpreted as model-behavior evidence rather than policy estimates. The paper should avoid claiming that earlier evacuation has a measured real-world causal effect of a particular size. The defensible claim is that the learned rollout mechanism preserves the expected evacuation-timing order under controlled model-generated branches.

## 8. Conclusion

STORM-World uses tropical cyclones as a physical-AI testbed for a broader world-model question: can a learned physical model imagine alternative futures that remain plausible, uncertainty-aware, controllable, and useful for decisions? The current evidence supports a careful answer. Evacuation timing produces coherent dose-response and strong ranking preservation; several other interventions fail. This combination makes the paper a contribution to world-model evaluation rather than another hurricane-forecasting paper.
