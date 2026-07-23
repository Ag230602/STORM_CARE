# Foundation Model Checkpoint Audit

Date: 2026-07-16

## Scope

This audit verifies the Module 1 self-supervised foundation-model pretraining
protocol, checkpoint selection, split isolation, and downstream artifact
consistency. All reported metrics were regenerated after code changes.

## Root Causes

1. `foundation_best.pt` was selected by training loss.
   - Source: `model/foundation/pretrain.py`
   - Risk: a training-loss checkpoint can diverge from validation tables and
     figures, enabling checkpoint cherry-picking.

2. Foundation evaluation used invalid future horizons.
   - Source: `model/foundation/evaluation.py`
   - Risk: horizon targets beyond the available storm track were included in
     track, CRPS, and cone metrics.

3. The train/validation split was record-level.
   - Source: `model/foundation/pretrain.py`
   - Risk: if the same natural storm appears in multiple sources, source-level
     records could leak across train and validation.

4. Manuscript tables and calibration figures could include multiple epochs.
   - Source: `scripts/generate_submission_outputs.py`
   - Risk: tables/figures could silently use a different checkpoint than the
     selected checkpoint.

## Fixes

- Split pretraining data by storm identity group: `basin|year|storm_name`.
- Persisted `foundation_split_manifest.csv` and `foundation_split_audit.json`.
- Evaluated every epoch under the same protocol.
- Computed validation selection score as mean valid track error across all
  configured lead times.
- Saved `foundation_best.pt` only when validation selection score improves.
- Marked exactly one row in `foundation_eval_metrics.csv` with
  `selected_checkpoint=True`.
- Regenerated foundation tables and calibration figures from that selected row
  only.
- Filtered validation metrics by `horizon_valid`.
- Made the linear-probe split deterministic with the experiment seed.

## Commands Rerun

```bash
.venv/bin/python -m model.foundation.pretrain --demo --epochs 2
.venv/bin/python -c "from scripts.generate_submission_outputs import regenerate_foundation_tables, regenerate_calibration_figure, ROOT; regenerate_foundation_tables(ROOT/'metrics', ROOT/'tables'); regenerate_calibration_figure(ROOT/'tables', ROOT/'figures')"
.venv/bin/python -m py_compile model/foundation/evaluation.py model/foundation/pretrain.py scripts/generate_submission_outputs.py
```

## Split Audit

From `metrics/foundation/foundation_split_audit.json`:

| Field | Value |
| --- | ---: |
| Split unit | `basin|year|storm_name` |
| Seed | 42 |
| Records | 400 |
| Groups | 325 |
| Train records | 324 |
| Validation records | 76 |
| Train groups | 260 |
| Validation groups | 65 |
| Storm ID overlap | 0 |
| Group-key overlap | 0 |

## Selected Checkpoint

The selected checkpoint is:

- `checkpoints/foundation/foundation_best.pt`
- epoch: 2
- selection metric: `mean_track_err_km`
- selection score: 1005.24854442106

The final checkpoint is also saved as `foundation_final.pt`, but tables, figures,
and manuscript-facing metrics use the selected checkpoint row only.

## Selected Validation Metrics

From `metrics/foundation/foundation_eval_metrics.csv`, selected epoch 2:

| Metric | Value |
| --- | ---: |
| `track_err_km_6h` | 143.4044 |
| `track_err_km_12h` | 287.9981 |
| `track_err_km_24h` | 582.0694 |
| `track_err_km_48h` | 1125.2586 |
| `track_err_km_72h` | 1566.3819 |
| `track_err_km_120h` | 2326.3788 |
| `linear_probe_acc` | 0.8750 |
| `recon_mse` | 0.2037 |
| `contrast_align` | 0.9839 |

## Regenerated Artifacts

- `metrics/foundation/foundation_eval_metrics.csv`
- `metrics/foundation/foundation_train_log.csv`
- `metrics/foundation/foundation_split_manifest.csv`
- `metrics/foundation/foundation_split_audit.json`
- `tables/table_foundation_model_training.csv`
- `tables/table_calibration_cone_coverage.csv`
- `figures/calibration.png`
- `figures/calibration.pdf`
- `results/module1_foundation/`

## Scientific Conclusion

The foundation-model checkpoint is now selected by validation performance and is
used consistently across tables, figures, and documentation. The split audit
shows no train/validation overlap by storm ID or storm identity group. The demo
run remains a short CPU validation-selected run, not a full-scale foundation
pretraining result.

## Limitations

- ERA5 coverage remains sparse in the capped demo subset: 86 ERA5-enhanced
  observations out of 11,087 total observations, or 0.8%.
- The selected checkpoint is from a 2-epoch CPU demo. It should not be compared
  to older longer-run claims.

## Addendum (2026-07-23): Extended 20-Epoch Rerun

Reran pretraining for 20 epochs instead of 2 (`.venv/bin/python -m model.foundation.pretrain --demo --epochs 20`) to check for cheap gains from longer training, per reviewer request. Validation selection score improved every single epoch, from `1018.6468` (epoch 1) to `860.1958` (epoch 20) — no overfitting observed in this run length. Epoch 20 is now the selected checkpoint.

Selected validation metrics, epoch 20 (`metrics/foundation/foundation_eval_metrics.csv`):

| Metric | Epoch 2 (previous) | Epoch 20 (current) |
| --- | ---: | ---: |
| `track_err_km_6h` | 143.4044 | 106.8845 |
| `track_err_km_12h` | 287.9981 | 231.0135 |
| `track_err_km_24h` | 582.0694 | 509.3300 |
| `track_err_km_48h` | 1125.2586 | 954.1142 |
| `track_err_km_72h` | 1566.3819 | 1357.8206 |
| `track_err_km_120h` | 2326.3788 | 2002.0120 |
| `cone_p90_6h` | 0.8320 | 0.8908 |
| `cone_p90_12h` | 0.7013 | 0.9134 |
| `cone_p90_24h` | 0.2968 | 0.8858 |
| `cone_p90_48h` | 0.1082 | 0.8454 |
| `cone_p90_72h` | 0.0710 | 0.7160 |
| `cone_p90_120h` | 0.0420 | 0.4874 |

Track error improved at every horizon on this rerun (the 20-epoch checkpoint
strictly beats the 2-epoch checkpoint here, unlike Module 2/5 where longer
runs traded off different metrics). P90 cone coverage improved dramatically at
every horizon and is now much closer to nominal calibration (target 0.90) at
6-48h, though it still under-covers materially beyond 48h (0.72 at 72h, 0.49
at 120h vs. target 0.90). This is a substantially better-calibrated checkpoint
than the 2-epoch demo, but is still a short CPU demo, not the full ~50-epoch
run described in `model/foundation/pretrain.py`'s computational-requirements
docstring (~4-8 GPU-hours on an A100).

Regenerated: `metrics/foundation/*`, `tables/table_foundation_model_training.csv`,
`tables/table_calibration_cone_coverage.csv`, `figures/calibration.png/.pdf`,
`results/module1_foundation/`. Calibration consistency audit re-passed for the
new selected epoch (`reports/calibration_consistency_audit.md`).
