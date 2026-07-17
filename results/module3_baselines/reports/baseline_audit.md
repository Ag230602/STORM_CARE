# Baseline Track Benchmark Audit

## Root Cause

The impossible LSTM, Transformer, and GNO+DynGNN trajectory errors came from
multiple implementation/protocol issues:

- `model/track_pipeline_unified_X.py` built history windows ending at `t0 - 6h`,
  so the models did not receive the current storm position.
- The learned models trained directly on raw absolute latitude/longitude targets
  and raw ERA5/meta magnitudes, making the Gaussian NLL ill-conditioned.
- `benchmark.py` evaluated a different 80/20 split than the training script used
  for checkpoint selection.
- `scripts/run_full_test_ci.py` compared storms with real ERA5 against storms
  with zero-filled ERA5, an unequal-input full-HURDAT2 protocol.
- Prediction writers treated model outputs as absolute coordinates; corrected
  outputs are normalized displacements decoded back to lat/lon before metrics.

## Fixes

- Current `t0` is included in each history window.
- ERA5 patches are normalized by per-sample, per-channel z-scores.
- History coordinates are scaled as `lat/90`, `lon/180`.
- Metadata is normalized as `vmax/150` and centered/scaled pressure.
- Learned targets are normalized future displacements from current `t0`.
- Metrics and prediction CSVs decode normalized outputs back to physical
  latitude/longitude.
- Checkpoints are selected only by validation mean track error and then reported
  on a held-out test split.
- The unequal-input full-HURDAT2 learned-baseline runner is disabled.
- `tables/table_fullset_ci_detail.csv` is regenerated as a `not_evaluated`
  audit row rather than retaining unsupported zero-filled-ERA5 metrics.

## Regenerated Protocol

- ERA5-complete common subset: 45 windows.
- Split: 27 train, 9 validation, 9 test.
- Irma: 48 candidate windows, 21 ERA5-complete windows, 27 skipped because ERA5
  crop/time coverage was incomplete.
- Ian: 24 candidate windows, 24 ERA5-complete windows, 0 skipped.
- Raw and normalized ERA5 non-finite values: 0.

## Regenerated Test Metrics

| Model | 6h km | 12h km | 24h km | 48h km | Mean km |
|---|---:|---:|---:|---:|---:|
| Persistence | 29.247 | 72.842 | 197.041 | 527.727 | 206.714 |
| GNO+DynGNN | 119.690 | 156.353 | 322.895 | 598.080 | 299.255 |
| Transformer | 170.140 | 211.689 | 381.798 | 611.512 | 343.785 |
| LSTM | 83.445 | 286.149 | 442.575 | 873.994 | 421.541 |

These values are copied from regenerated
`metrics/inference_test_metrics_summary.csv` and
`tables/table_case_study_track_error.csv`; no CSVs or figures were manually
edited.

## Supported Claim

The corrected implementation removes the impossible trajectory errors and makes
the learned baselines comparable on identical ERA5-complete inputs.  It does not
support a learned-model superiority claim on this small two-storm case study:
Persistence remains the best mean-error baseline.

The horizon-level claim audit is in
`tables/table_forecast_performance_audit.csv` and
`reports/forecast_performance_audit.md`.  It shows that GNO+DynGNN beats
Transformer at 6/12/24/48 h and LSTM at 12/24/48 h, but loses to LSTM at 6 h
and loses to Persistence at every reported case-study horizon.
