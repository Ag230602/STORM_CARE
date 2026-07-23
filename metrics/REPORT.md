# Experimental Results Report (Track Forecasting)

## Setup
- Task: Hurricane track forecasting (Irma 2017 + Ian 2022)
- Split: 80% train / 20% test (seeded)
- Outputs: probabilistic mean track + uncertainty (P50/P90 cone)

## Models evaluated
- Persistence (constant-velocity)
- LSTM baseline (past track + ERA5 patch)
- Transformer baseline (past track + ERA5 patch)
- Primary: GNO+DynGNN (operator-style ERA5 encoder + dynamic GNN)

## Key summary (from inference_test_metrics_summary.csv)
**Best mean track error:** Persistence (mean=206.71 km across 6/12/24/48h)

| model       |   mean_track_km |   track_6h_km |   track_12h_km |   track_24h_km |   track_48h_km |   landfall_time_err_hours |   cone_cov50_24h |   cone_cov90_24h |
|:------------|----------------:|--------------:|---------------:|---------------:|---------------:|--------------------------:|-----------------:|-----------------:|
| Persistence |         206.714 |       29.2474 |        72.8421 |        197.041 |        527.727 |                       2.4 |              nan |              nan |
| GNO+DynGNN  |         299.255 |      119.69   |       156.353  |        322.895 |        598.08  |                       1   |                1 |                1 |
| Transformer |         343.785 |      170.14   |       211.689  |        381.798 |        611.512 |                       1   |                1 |                1 |
| DCRNN       |         357.771 |       89.5152 |       269.963  |        335.053 |        736.555 |                       1.2 |                1 |                1 |
| LSTM        |         421.541 |       83.4453 |       286.149  |        442.575 |        873.994 |                       1.2 |                1 |                1 |

## Visualizations
- Track error vs lead: `plots/track_error_vs_lead.png`
- P50 cone coverage: `plots/cone_coverage_p50.png`
- P90 cone coverage: `plots/cone_coverage_p90.png`
- Landfall time error proxy: `plots/landfall_time_error.png`

## Data-grounded interpretation
- This report is generated from `inference_test_metrics_summary.csv` and `inference_test_predictions_all_models.csv`; it does not add manual performance claims.
- The lowest mean track error on this regenerated benchmark is `Persistence` (206.71 km over 6/12/24/48h).
- Treat this as an Irma/Ian window-level case-study benchmark unless the upstream experiment is rerun with storm-held-out splits and training data disjoint from test storms.
- Cone coverage is reported only for probabilistic models with finite sigma columns; deterministic baselines should have missing coverage.

## Per-storm breakdown (24h/48h mean error from predictions CSV)

| model       | storm_tag   |   err_km_24h |   err_km_48h |
|:------------|:------------|-------------:|-------------:|
| DCRNN       | ian         |      348.253 |      739.37  |
| DCRNN       | irma        |      324.492 |      734.302 |
| GNO+DynGNN  | ian         |      356.237 |      643.602 |
| GNO+DynGNN  | irma        |      296.221 |      561.662 |
| LSTM        | ian         |      402.88  |      834.568 |
| LSTM        | irma        |      474.331 |      905.535 |
| Persistence | ian         |      175.28  |      461.958 |
| Persistence | irma        |      214.449 |      580.343 |
| Transformer | ian         |      391.289 |      683.146 |
| Transformer | irma        |      374.205 |      554.205 |

## Cone coverage recomputed from predictions CSV (mean)

| model       |   cov50_@6h |   cov50_@12h |   cov50_@24h |   cov50_@48h |   cov90_@6h |   cov90_@12h |   cov90_@24h |   cov90_@48h |
|:------------|------------:|-------------:|-------------:|-------------:|------------:|-------------:|-------------:|-------------:|
| DCRNN       |           1 |            1 |            1 |            1 |           1 |            1 |            1 |            1 |
| GNO+DynGNN  |           1 |            1 |            1 |            1 |           1 |            1 |            1 |            1 |
| LSTM        |           1 |            1 |            1 |            1 |           1 |            1 |            1 |            1 |
| Persistence |           0 |            0 |            0 |            0 |           0 |            0 |            0 |            0 |
| Transformer |           1 |            1 |            1 |            1 |           1 |            1 |            1 |            1 |

## Files used
- `inference_test_metrics_summary.csv`
- `inference_test_predictions_all_models.csv`
