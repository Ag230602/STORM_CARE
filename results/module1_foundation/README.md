# Module 1 Foundation Results

This folder contains the validation-selected foundation-model artifacts.

## Selected Checkpoint

- Checkpoint: `checkpoints/foundation_best.pt`
- Epoch: 2
- Selection metric: `mean_track_err_km`
- Selection score: `1005.24854442106`

`foundation_best.pt` is selected by validation performance, not training loss.
`foundation_final.pt` is retained for reproducibility but is not used for
manuscript-facing tables or figures unless it is also the selected checkpoint.

## Split Audit

The train/validation split is grouped by storm identity:

`basin|year|storm_name`

Audit summary:

- Train records: 324
- Validation records: 76
- Train groups: 260
- Validation groups: 65
- Storm ID overlap: 0
- Group-key overlap: 0

See `metrics/foundation_split_audit.json` and
`metrics/foundation_split_manifest.csv`.

## Main Files

- `metrics/foundation_eval_metrics.csv` — all evaluated epochs with one selected row
- `metrics/foundation_train_log.csv` — training/evaluation log
- `tables/table_foundation_model_training.csv` — selected-checkpoint table
- `tables/table_calibration_cone_coverage.csv` — selected-checkpoint calibration table
- `figures/calibration.png` — regenerated from selected-checkpoint outputs
- `reports/foundation_checkpoint_audit.md` — root-cause and validation report

## Supported Result

The selected 2-epoch CPU-demo checkpoint has:

- 6h track error: `143.4044 km`
- 24h track error: `582.0694 km`
- 120h track error: `2326.3788 km`
- linear probe accuracy: `0.8750`
- reconstruction MSE: `0.2037`

This is a reproducibility/demo result, not a full-scale pretraining claim.
