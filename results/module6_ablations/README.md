# Module 6 Ablation Study Results

This folder mirrors the regenerated ablation study outputs.

- `tables/table3_ablations.csv` contains the submission Table 3 ablation audit with no blank cells.
- `metrics/foundation_ablation_metrics.csv` contains the same-split foundation full vs random-init evaluation.
- `metrics/graph_ablation_metrics.csv` contains the rerun graph-edge ablations on train seed 123 and test seed 999.
- `metrics/no_physics_runtime.json` and `metrics/no_world_model_runtime.json` contain targeted runtime measurements.
- `reports/ablation_study_audit.md` documents root causes, fixes, and validity limits.

Cells marked `not_applicable_to_changed_component` are intentional scientific status values, not missing data.
