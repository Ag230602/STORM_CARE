# Dataset Integrity Report

- Status: **PASS**
- Storm split counts: `{'test': 107, 'train': 342, 'val': 70}`
- Baseline case-study sample counts: `{'test': 9, 'train': 27, 'val': 9}`
- Foundation record counts: `{'train': 324, 'val': 76}`
- Foundation window counts: `{'train': 837, 'val': 238}`
- Prediction rows: `36` across `4` models and `9` storm/t0 windows.
- Foundation train/val group overlap count: `0`

## Issues
- None.


## Generated CSV Audits
- `metrics/dataset_integrity/split_map_counts.csv`
- `metrics/dataset_integrity/baseline_case_study_counts.csv`
- `metrics/dataset_integrity/baseline_era5_counts.csv`
- `metrics/dataset_integrity/foundation_record_counts.csv`
- `metrics/dataset_integrity/foundation_window_counts.csv`
- `metrics/dataset_integrity/prediction_window_counts.csv`
