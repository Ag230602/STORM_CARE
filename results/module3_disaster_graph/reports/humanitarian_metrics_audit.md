# Humanitarian Metrics Audit

## Root Causes Addressed
- exposed-children MAPE now compares predicted and true exposed-child counts, not counts versus fractions.
- exposed-children MAPE is computed on scenario-level peak exposure to avoid near-zero early-step denominators.
- school AUC is pooled across held-out school nodes instead of averaged per scenario with one-class folds.
- damage simulator no longer saturates all infrastructure targets to one after a single step.
- humanitarian heads are supervised directly during Module 3 training.
- train and test synthetic scenarios use disjoint seeds.

## Validity Scope
- targets are simulator-derived proxy labels, not observed disaster outcomes.
- metrics are valid for the synthetic demo protocol only.

## Regenerated Metrics
| metric                     |   STORM-CARE-M3 |   RF_baseline |   MLP_baseline | units   | notes                                                         |
|:---------------------------|----------------:|--------------:|---------------:|:--------|:--------------------------------------------------------------|
| exposed_children_MAPE      |        469.396  |      721.505  |      1070.21   | %       | Lower is better; test seed 999; baselines trained on seed 123 |
| school_disruption_AUC      |          0.8724 |        0.6827 |         0.5494 | [0,1]   | Higher is better; pooled over all held-out school nodes       |
| hospital_accessibility_MAE |          0.0256 |        0.0332 |         0.4174 | [0,1]   | Lower is better                                               |
| recovery_priority_spearman |         -0.0212 |       -0.0631 |         0.0144 | [-1,1]  | Higher is better                                              |

## Label Audit
- Test school positive rate: 0.0146
- Test hospital access range: 0.6860 to 1.0000
- Test exposed-child total mean: 182.6731
