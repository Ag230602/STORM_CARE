# Humanitarian Metrics Audit

## Root Causes Addressed
- exposed-children MAPE now compares predicted and true exposed-child counts, not counts versus fractions.
- exposed-children MAPE is computed on scenario-level peak exposure to avoid near-zero early-step denominators.
- school AUC is pooled across held-out school nodes instead of averaged per scenario with one-class folds.
- damage simulator no longer saturates all infrastructure targets to one after a single step.
- humanitarian heads are supervised directly during Module 3 training.
- train and test synthetic scenarios use disjoint seeds.
- (E7) headline exposed-child error metric switched from MAPE to sMAPE, plus a raw MAE-in-counts row, since MAPE remained dominated by near-zero true-count denominators even after the scenario-level fix above.

## Validity Scope
- targets are simulator-derived proxy labels, not observed disaster outcomes.
- metrics are valid for the synthetic demo protocol only.
- (E7) under sMAPE and MAE-in-counts, STORM-CARE-M3 does NOT beat RF/XGB on exposed-child error (sMAPE 118.7% vs RF 97.7%/XGB 97.3%; MAE 298.5 vs RF 267.6/XGB 263.5) — the reverse of the old MAPE ranking, which favored STORM-CARE-M3 only because MAPE penalizes baseline under-prediction more harshly near zero. This is reported plainly as an open problem, not hidden by keeping the old headline metric..

## Regenerated Metrics
| metric                      |   STORM-CARE-M3 |   RF_baseline |   MLP_baseline |   XGB_baseline | units            | notes                                                                                                                                                                                                  |
|:----------------------------|----------------:|--------------:|---------------:|---------------:|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| exposed_children_sMAPE      |        118.659  |       97.6975 |       110.569  |        97.278  | %                | Lower is better; headline metric (E7): symmetric MAPE, bounded [0,200]%, robust to the near-zero true-count denominators that made plain MAPE degenerate; test seed 999; baselines trained on seed 123 |
| exposed_children_MAE_counts |        298.524  |      267.617  |       290.146  |       263.533  | children (count) | Lower is better; mean absolute error on peak exposed-child count, unnormalized                                                                                                                         |
| exposed_children_MAPE       |        469.396  |      721.505  |      1070.21   |       779.415  | %                | Lower is better; retained for reference only — dominated by near-zero true-count denominators, see exposed_children_sMAPE for the headline metric; test seed 999; baselines trained on seed 123        |
| school_disruption_AUC       |          0.8724 |        0.6827 |         0.5494 |         0.6255 | [0,1]            | Higher is better; pooled over all held-out school nodes                                                                                                                                                |
| hospital_accessibility_MAE  |          0.0256 |        0.0332 |         0.4174 |         0.0327 | [0,1]            | Lower is better                                                                                                                                                                                        |
| recovery_priority_spearman  |         -0.0212 |       -0.0631 |         0.0144 |        -0.0347 | [-1,1]           | Higher is better                                                                                                                                                                                       |

## Label Audit
- Test school positive rate: 0.0146
- Test hospital access range: 0.6860 to 1.0000
- Test exposed-child total mean: 182.6731
