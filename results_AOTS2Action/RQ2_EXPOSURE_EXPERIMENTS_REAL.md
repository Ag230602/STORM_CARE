# RQ2 real-data exposure experiments

Marker: **REAL_HUMANITARIAN_GEOSPATIAL_DATA**

This rerun uses the all-horizon matched corpus (`n=1,035`) and the existing cyclone forecast grid, with real population and country-level socioeconomic vulnerability harmonized onto that grid.

## Data and harmonization

- Population: WorldPop Total Population 1 km ImageServer, year 2020; 2,192 RQ2-active cells queried.
- Vulnerability/socioeconomic weight: World Bank country income classification mapped as LIC=1.00, LMC=0.75, UMC=0.50, HIC=0.25; fallback 0.50.
- Administrative boundaries: Natural Earth country boundaries via `datasets/geo-countries`; coastal cells use country-envelope intersection when the cell center is offshore.
- Infrastructure: Natural Earth 1:10m airports v5.0.0 and ports v5.0.0 are harmonized onto the grid as 100 km counts and nearest-distance context fields. The primary RQ2 exposure weight remains the prespecified `population * inform_risk`.
- Inactive cells outside every RQ2 realized/deterministic/P90/member footprint were set to zero population because they cannot affect exposure AE, signed error, or exposure ratio; Brier scores are unweighted over grid cells.

Confidence intervals are 95% cyclone-cluster bootstrap intervals with 10,000 replicates and seed 20260817. Paired comparisons average cycle-level differences within cyclone and use two-sided Wilcoxon signed-rank tests with Holm correction.

## Absolute error

| Estimator | 6 h | 12 h | 24 h | 48 h | 72 h | 96 h |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic mean-track | 18,472.92 [5,644.04, 36,261.66] | 25,001.47 [6,264.27, 51,535.43] | 38,650.93 [11,329.68, 89,390.54] | 54,499.39 [16,148.96, 116,840.47] | 61,506.30 [9,328.10, 168,116.90] | 14,143.87 [2,742.30, 30,755.35] |
| P90 envelope | 919,870.60 [287,796.49, 1,915,604.58] | 1,091,270.50 [399,537.76, 2,134,440.87] | 1,950,951.93 [781,057.70, 3,906,800.78] | 3,512,657.50 [1,054,773.84, 7,539,209.55] | 5,814,748.89 [1,973,176.80, 12,409,655.31] | 8,691,406.46 [3,488,472.53, 18,344,666.27] |
| Ensemble probability-weighted | 25,575.72 [10,089.82, 48,192.75] | 25,742.86 [11,216.22, 47,584.37] | 45,893.90 [15,314.86, 94,175.03] | 43,208.85 [14,496.01, 87,703.46] | 29,850.23 [9,123.70, 64,474.89] | 28,580.58 [11,504.62, 57,547.25] |

## Signed error

| Estimator | 6 h | 12 h | 24 h | 48 h | 72 h | 96 h |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic mean-track | -3,739.18 [-11,715.32, 2,822.95] | 3,705.85 [-2,435.03, 12,730.58] | 10,438.20 [-11,766.69, 48,176.93] | 31,971.18 [-1,369.80, 86,801.64] | 50,536.26 [2,687.33, 157,721.40] | -428.72 [-12,702.34, 20,755.25] |
| P90 envelope | 919,870.60 [292,587.23, 1,891,020.68] | 1,091,270.50 [400,239.20, 2,190,566.54] | 1,950,951.93 [776,185.27, 3,836,940.25] | 3,511,059.69 [1,040,912.76, 7,441,532.22] | 5,814,748.89 [2,009,796.84, 12,350,551.80] | 8,691,406.46 [3,529,501.50, 18,619,414.54] |
| Ensemble probability-weighted | 6,715.56 [-1,022.33, 22,278.79] | 7,916.08 [-1,108.92, 24,733.95] | 24,437.86 [-4,217.29, 68,898.79] | 24,408.13 [275.58, 68,142.36] | 21,840.54 [2,076.72, 57,074.32] | 18,938.86 [-6.86, 51,921.01] |

## Exposure ratio

| Estimator | 6 h | 12 h | 24 h | 48 h | 72 h | 96 h |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic mean-track | 0.498 [0.267, 0.700]; median 0.306 | 0.423 [0.266, 0.570]; median 0.000 | 0.184 [0.053, 0.312]; median 0.000 | 0.293 [0.077, 0.488]; median 0.000 | 0.000 [0.000, 0.000]; median 0.000 | 0.004 [0.000, 0.009]; median 0.000 |
| P90 envelope | 11,790.524 [20.762, 31,948.910]; median 18.366 | 11,657.087 [17.120, 31,734.080]; median 19.696 | 12,666.143 [29.732, 32,492.211]; median 28.667 | 52,318.443 [54.285, 120,285.498]; median 49.713 | 75,850.635 [94.465, 156,638.949]; median 66.309 | 95,460.111 [207.882, 225,481.997]; median 88.892 |
| Ensemble probability-weighted | 40.317 [0.285, 108.893]; median 0.305 | 57.391 [0.371, 155.782]; median 0.403 | 22.718 [0.314, 56.652]; median 0.462 | 1,216.681 [0.598, 2,835.608]; median 0.515 | 40.695 [0.248, 86.814]; median 0.571 | 1,155.249 [0.474, 2,815.088]; median 0.443 |

## Brier score

| Estimator | 6 h | 12 h | 24 h | 48 h | 72 h | 96 h |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic mean-track | 0.002194 [0.001647, 0.002920] | 0.002691 [0.001951, 0.003446] | 0.002475 [0.001875, 0.003151] | 0.002570 [0.001885, 0.003356] | 0.002653 [0.001855, 0.003432] | 0.002667 [0.001671, 0.003550] |
| P90 envelope | 0.052310 [0.038163, 0.067733] | 0.057435 [0.043111, 0.073271] | 0.076280 [0.058244, 0.096759] | 0.138650 [0.104426, 0.172936] | 0.240488 [0.179443, 0.308809] | 0.387230 [0.272082, 0.509672] |
| Ensemble probability-weighted | 0.001563 [0.001173, 0.002039] | 0.001536 [0.001150, 0.001907] | 0.001264 [0.000855, 0.001720] | 0.001191 [0.000766, 0.001625] | 0.001285 [0.000715, 0.001835] | 0.001407 [0.000783, 0.001954] |

## Absolute-error paired comparisons

| Comparison | Horizon | Mean cycle diff | Mean storm diff | Holm p | Significant 0.05 |
|---|---:|---:|---:|---:|---|
| det - ens | 6 h | -7,102.80 | -12,271.32 | 1 | False |
| det - ens | 12 h | -741.39 | -589.45 | 1 | False |
| det - ens | 24 h | -7,242.96 | 10,512.38 | 1 | False |
| det - ens | 48 h | 11,290.54 | 37,904.30 | 1 | False |
| det - ens | 72 h | 31,656.08 | 33,683.85 | 1 | False |
| det - ens | 96 h | -14,436.72 | -18,806.11 | 0.557352 | False |
| P90 - ens | 6 h | 894,294.88 | 1,146,916.14 | 0.0506203 | False |
| P90 - ens | 12 h | 1,065,527.64 | 1,535,383.60 | 0.0506203 | False |
| P90 - ens | 24 h | 1,905,058.03 | 2,374,930.25 | 0.0401474 | True |
| P90 - ens | 48 h | 3,469,448.64 | 4,704,424.92 | 0.0401474 | True |
| P90 - ens | 72 h | 5,784,898.67 | 7,849,321.62 | 0.0506203 | False |
| P90 - ens | 96 h | 8,662,825.88 | 11,895,387.48 | 0.0538006 | False |

## Brier-score paired comparisons

| Comparison | Horizon | Mean cycle diff | Mean storm diff | Holm p | Ensemble lower Brier supported |
|---|---:|---:|---:|---:|---|
| det - ens | 6 h | 0.000631 | 0.000954 | 0.0136719 | True |
| det - ens | 12 h | 0.001155 | 0.000926 | 0.0161133 | True |
| det - ens | 24 h | 0.001211 | 0.002436 | 0.00292969 | True |
| det - ens | 48 h | 0.001379 | 0.001665 | 0.00341797 | True |
| det - ens | 72 h | 0.001368 | 0.001142 | 0.0146484 | True |
| det - ens | 96 h | 0.001260 | 0.001341 | 0.0161133 | True |
| P90 - ens | 6 h | 0.050748 | 0.060097 | 0.00292969 | True |
| P90 - ens | 12 h | 0.055900 | 0.068939 | 0.00292969 | True |
| P90 - ens | 24 h | 0.075016 | 0.088892 | 0.00292969 | True |
| P90 - ens | 48 h | 0.137459 | 0.154580 | 0.00292969 | True |
| P90 - ens | 72 h | 0.239203 | 0.279992 | 0.00585938 | True |
| P90 - ens | 96 h | 0.385823 | 0.439401 | 0.00976562 | True |

## Interpretation guardrails

- Ensemble probability weighting has the lowest Brier score at every horizon and the paired Brier tests support lower Brier scores versus both deterministic and P90 methods after Holm correction.
- Absolute-error superiority is not supported versus deterministic mean-track at any horizon. The ensemble method significantly reduces absolute error versus the P90 envelope at 24 h and 48 h only after Holm correction.
- Exposure-ratio means are unstable when realized exposure is small; ratio rows exclude zero-realized cases, and medians are reported alongside means.
