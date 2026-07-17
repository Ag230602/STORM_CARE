# Module 3 Baseline Track Benchmark

This folder contains the corrected LSTM, Transformer, GNO+DynGNN, and
Persistence case-study benchmark artifacts regenerated from the corrected
pipeline.

Key artifacts:

- `metrics/inference_test_metrics_summary.csv` - benchmark metrics from the corrected held-out test split
- `metrics/inference_test_predictions_all_models.csv` - decoded lat/lon predictions for all models
- `metrics/baseline_input_audit.csv` - ERA5 availability and missing-value audit
- `metrics/baseline_split_manifest.csv` - deterministic train/val/test split membership
- `tables/table_case_study_track_error.csv` - manuscript-ready case-study table
- `tables/table_fullset_ci_detail.csv` - generated `not_evaluated` audit row for the disabled unequal-input full-set learned-baseline protocol
- `tables/table_forecast_performance_audit.csv` - horizon-level supported-claim audit
- `figures/track_error_vs_lead.png` - regenerated benchmark track-error curve
- `figures/case_study_ian_combined.png` - regenerated Ian case-study figure
- `checkpoints/*.pt` - corrected validation-selected learned-baseline checkpoints
- `reports/baseline_audit.md` - root-cause and validation notes

Supported conclusion: on the corrected Irma/Ian ERA5-complete window benchmark,
Persistence has the lowest mean 6/12/24/48 h track error.  The learned models are
reported as a small case study only, not as storm-held-out generalization.
GNO+DynGNN does not outperform Persistence at any reported case-study horizon.
