# Counterfactual World Model Audit

Date: 2026-07-16

## Scope

This audit verifies that Module 5 counterfactual outcomes are generated through
the learned RSSM world model rather than by directly editing decoded outputs.
All metrics below were regenerated after code changes.

## Root Cause

The previous Module 5 runner used `compare_analytic_multi_storm()`, which rolled
out the baseline trajectory once and then applied direct proportional edits to
decoded trajectory tensors. That made outcomes numerically mirror intervention
percentages, for example exposure reductions matching hard-coded evacuation
factors.

Affected files:

- `model/counterfactual/run.py`
- `model/counterfactual/engine.py`
- `model/counterfactual/scenarios.py`
- `model/world_model/architecture.py`

## Fix

Counterfactual interventions are now branch-state do-operators:

1. Start from the same held-out warm-up disaster-state sequence.
2. Apply the scenario intervention to the final warm-up state history.
3. Encode the intervened warm-up sequence through the RSSM posterior.
4. Roll forward using the learned RSSM prior.
5. Decode outcomes from the evolved latent state.

No scenario edits decoded rollout outputs. The world-model `rollout()` API no
longer accepts `z_override`, preventing persistent latent injection during
counterfactual evaluation.

## Scenarios Tested

- `earlier_evacuation`
- `delayed_evacuation`
- `shelter_failure`
- `hospital_failure`
- `road_blockage`
- `intensity_increase`
- `intensity_decrease`
- `additional_emergency_resources`

The baseline plus all eight interventions were evaluated on the complete
held-out split for the loaded demo world-model checkpoint: 24 test sequences.
The loaded checkpoint is demo-sized, so the final run used a 12-step rollout
horizon and 5 Monte Carlo samples per scenario.

## Commands Rerun

```bash
.venv/bin/python -m model.counterfactual.run --metrics-dir metrics/counterfactual
.venv/bin/python -c "from scripts.generate_submission_outputs import regenerate_counterfactual_table, ROOT; regenerate_counterfactual_table(ROOT/'metrics', ROOT/'tables')"
.venv/bin/python -m py_compile model/world_model/architecture.py model/counterfactual/__init__.py model/counterfactual/config.py model/counterfactual/scenarios.py model/counterfactual/engine.py model/counterfactual/run.py scripts/run_ablations.py
```

## Regenerated Artifacts

- `metrics/counterfactual/counterfactual_outcomes.csv`
- `metrics/counterfactual/counterfactual_mirror_diagnostics.csv`
- `tables/table_counterfactual_outcomes.csv`
- `results/module5_counterfactual/metrics/counterfactual_outcomes.csv`
- `results/module5_counterfactual/metrics/counterfactual_mirror_diagnostics.csv`
- `results/module5_counterfactual/tables/table_counterfactual_outcomes.csv`

## Results

Final averaged metrics over 24 held-out sequences:

| Scenario | Peak Exposure | Resource Deficit | Mean Hazard |
| --- | ---: | ---: | ---: |
| baseline | 0.2915 | 0.0313 | 0.3403 |
| earlier_evacuation | 0.2831 | 0.0309 | 0.3407 |
| delayed_evacuation | 0.2969 | 0.0317 | 0.3394 |
| shelter_failure | 0.2842 | 0.0315 | 0.3393 |
| hospital_failure | 0.2843 | 0.0313 | 0.3398 |
| road_blockage | 0.2902 | 0.0316 | 0.3393 |
| intensity_increase | 0.2970 | 0.0317 | 0.3394 |
| intensity_decrease | 0.2854 | 0.0309 | 0.3409 |
| additional_emergency_resources | 0.2916 | 0.0312 | 0.3406 |

Mirror diagnostics:

| Scenario | Diagnostic Metric | Input Delta | Observed Delta | Mirrors Input |
| --- | --- | ---: | ---: | --- |
| earlier_evacuation | peak_exposure | -0.1200 | -0.0084 | False |
| delayed_evacuation | peak_exposure | 0.1200 | 0.0054 | False |
| shelter_failure | resource_deficit | 0.1800 | 0.0002 | False |
| hospital_failure | infra_damage_final | 0.1200 | 0.0000 | False |
| road_blockage | peak_exposure | 0.0800 | -0.0013 | False |
| intensity_increase | mean_hazard | 0.1000 | -0.0009 | False |
| intensity_decrease | mean_hazard | -0.1000 | 0.0006 | False |
| additional_emergency_resources | resource_deficit | -0.1400 | -0.0001 | False |

## Scientific Conclusion

The counterfactual outputs are no longer injected directly into decoder outputs
and no longer numerically mirror intervention percentages. They emerge through
RSSM posterior encoding, learned latent-state evolution, and decoder prediction.

The corrected demo world model does not support strong claims about all
intervention signs. Earlier evacuation lowers peak exposure and delayed
evacuation raises peak exposure, but several infrastructure/resource scenarios
have weak or counterintuitive effects under the current short demo checkpoint.
Those signs should be reported as learned-model outputs and limitations, not
overridden with analytic edits.

## Addendum (2026-07-21): Lead-Time Monotonicity, Re-Verified Signs, Noise Diagnostic

A reviewer re-flagged the wrong-sign issue with numbers (shelter failure
-11.0%, route disruption -11.7%, intensification -12.6% all *reducing* peak
exposure; `shelter_shortfall` saturated at 1.0) that do not match any
committed artifact in this repo — they predate the fix documented above (the
committed `counterfactual_outcomes.csv` already shows `shelter_shortfall` at a
non-degenerate ~0.023, and `intensity_increase` already raises peak exposure,
+1.89%). Re-running the current pipeline reproduces the committed numbers
exactly, confirming they reflect the code as it stands.

Two remaining items from that review were addressed:

1. **Added `earlier_evacuation_24h`** (`model/counterfactual/scenarios.py`,
   `config.py`) with a larger branch-state exposure reduction than the
   existing ~12h `earlier_evacuation`, to test lead-time monotonicity.
   Re-running `model.counterfactual.run` over the full 24-sequence held-out
   split gives:

   | Scenario | Peak Exposure |
   | --- | ---: |
   | earlier_evacuation_24h | 0.2787 |
   | earlier_evacuation (12h) | 0.2831 |
   | baseline | 0.2915 |
   | delayed_evacuation | 0.2969 |

   Monotonic (24h < 12h < baseline < delayed), printed by
   `CounterfactualEngine.print_report` as `monotonic=yes`.

2. **Diagnosed whether the remaining weak/wrong-sign scenarios
   (`shelter_failure`, `hospital_failure`, `road_blockage`) are a Monte Carlo
   noise artifact or a structural issue.** Reran `compare_multi_storm` with
   `n_monte_carlo` raised from 5 to 60 (12x): the signs and magnitudes barely
   moved (e.g. `shelter_failure` peak-exposure delta: -0.0073 at MC=5 vs.
   -0.0074 at MC=60). This rules out sampling noise as the cause — the effect
   is a deterministic property of this specific short-demo `WorldModel`
   checkpoint (small `d_latent`, 20 training epochs, 120 synthetic
   sequences), most likely a spurious infra/resource-to-exposure correlation
   learned from the small synthetic training set rather than the intended
   causal direction. The do-operator wiring itself (branch-state edit ->
   posterior encoding -> prior rollout -> decode, no `z_override` injection)
   was re-confirmed correct, so this is not a pipeline bug. Fixing the
   remaining signs would require retraining `WorldModel` (Module 4) longer
   and/or on more data — out of scope for this pass, and already disclosed
   as a limitation above and in the README.

3. **Re-run over all test storms** was already satisfied before this pass:
   `model/counterfactual/run.py` calls `compare_multi_storm` over the
   complete held-out split (24 sequences for the loaded demo checkpoint), not
   a single event.

All regenerated artifacts (`metrics/counterfactual/counterfactual_outcomes.csv`,
`counterfactual_mirror_diagnostics.csv`, `tables/table_counterfactual_outcomes.csv`,
and the `results/module5_counterfactual` mirrors) now include the
`earlier_evacuation_24h` row.
