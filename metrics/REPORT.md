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
**Best mean track error:** Persistence (mean=230.52 km across 6/12/24/48h)

| model       |   mean_track_km |   track_6h_km |   track_12h_km |   track_24h_km |   track_48h_km |   landfall_time_err_hours |   cone_cov50_24h |   cone_cov90_24h |
|:------------|----------------:|--------------:|---------------:|---------------:|---------------:|--------------------------:|-----------------:|-----------------:|
| Persistence |         230.517 |       117.743 |        135.141 |        202.756 |        466.427 |                       2.4 |              nan |              nan |
| GNO+DynGNN  |         755.17  |       619.579 |        587.14  |        878.028 |        935.935 |                      10   |                1 |                1 |
| LSTM        |        1650.09  |      3440.87  |        740.634 |       1251.56  |       1167.32  |                      10   |                0 |                0 |
| Transformer |        6059.06  |      7820.24  |       2829.76  |      12566.2   |       1020.05  |                     nan   |                0 |                0 |

## Visualizations
- Track error vs lead: `plots/track_error_vs_lead.png`
- P50 cone coverage: `plots/cone_coverage_p50.png`
- P90 cone coverage: `plots/cone_coverage_p90.png`
- Landfall time error proxy: `plots/landfall_time_error.png`

## Notes / Interpretation (fill in after you inspect plots)
- Persistence should degrade quickly beyond 12h.
- LSTM and Transformer should improve accuracy over Persistence.
- GNO+DynGNN should generally perform best at longer horizons and give smoother uncertainty.

## Per-storm breakdown (24h/48h mean error from predictions CSV)

| model       | storm_tag   |   err_km_24h |   err_km_48h |
|:------------|:------------|-------------:|-------------:|
| GNO+DynGNN  | ian         |      846.245 |      891.894 |
| GNO+DynGNN  | irma        |      903.454 |      971.168 |
| LSTM        | ian         |     1026.71  |      999.015 |
| LSTM        | irma        |     1431.43  |     1301.96  |
| Persistence | ian         |      202.769 |      439.81  |
| Persistence | irma        |      202.746 |      487.72  |
| Transformer | ian         |    12624.6   |     1293.08  |
| Transformer | irma        |    12519.4   |      801.629 |

## Cone coverage recomputed from predictions CSV (mean)

| model       |   cov50_@6h |   cov50_@12h |   cov50_@24h |   cov50_@48h |   cov90_@6h |   cov90_@12h |   cov90_@24h |   cov90_@48h |
|:------------|------------:|-------------:|-------------:|-------------:|------------:|-------------:|-------------:|-------------:|
| GNO+DynGNN  |           1 |            1 |            1 |            1 |           1 |            1 |            1 |            1 |
| LSTM        |           0 |            0 |            0 |            1 |           0 |            0 |            0 |            1 |
| Persistence |           0 |            0 |            0 |            0 |           0 |            0 |            0 |            0 |
| Transformer |           0 |            0 |            0 |            0 |           0 |            0 |            0 |            0 |

## Files used
- `inference_test_metrics_summary.csv`
- `inference_test_predictions_all_models.csv`
