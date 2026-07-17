# Reproducibility Report

## Required Regeneration Commands

```bash
.venv/bin/python scripts/create_splits.py
.venv/bin/python -m model.foundation.pretrain --demo --epochs 2
.venv/bin/python benchmark.py --metrics-dir metrics
.venv/bin/python scripts/eval_humanitarian.py
.venv/bin/python -m model.counterfactual.run --demo
.venv/bin/python scripts/run_ablations.py
.venv/bin/python scripts/audit_calibration_consistency.py
.venv/bin/python scripts/audit_dataset_integrity.py
.venv/bin/python scripts/case_study_ian.py
.venv/bin/python scripts/generate_submission_outputs.py
.venv/bin/python scripts/sync_manuscript.py
```

## Integrity Status

- Dataset integrity: `PASS`
- Calibration consistency: `PASS`
- Manuscript source hashes are listed in `reports/experiment_log.md`.
