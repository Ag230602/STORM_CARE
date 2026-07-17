# Module 3 Disaster Graph Humanitarian Metrics

This folder contains the corrected Module 3 humanitarian evaluation artifacts.

Key artifacts:

- `checkpoints/disaster_gnn_best.pt` - retrained multitask checkpoint
- `metrics/train_log.csv` - per-epoch multitask training/validation losses
- `metrics/humanitarian_eval_metrics.csv` - corrected held-out metrics
- `metrics/humanitarian_label_audit.json` - proxy-label distribution and leakage audit
- `tables/table2_humanitarian_impact.csv` - regenerated manuscript table
- `reports/humanitarian_metrics_audit.md` - root-cause and validation report

Supported conclusion: the corrected synthetic demo now reports finite,
unit-consistent humanitarian metrics.  School disruption AUC and hospital access
MAE improve over the included sklearn baselines, but exposed-child peak MAPE
remains high and recovery-priority ranking is not meaningfully positive.  These
are simulator-derived proxy-label metrics, not observed disaster-outcome claims.
