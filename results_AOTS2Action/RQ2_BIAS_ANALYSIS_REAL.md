# RQ2: direction and magnitude of exposure-estimation bias

Marker: **REAL_HUMANITARIAN_GEOSPATIAL_DATA**

Positive signed error means overestimation and negative signed error means
underestimation. Exposure ratios greater than 1 indicate overestimation and
ratios below 1 indicate underestimation. Brackets contain cyclone-cluster
bootstrap 95% confidence intervals for means (10,000 replicates; seed
20260817). Signed errors retain zero-realized-exposure cases; ratios exclude
them. Exposure values are real-data vulnerability-weighted population units.

| Estimator | Lead | Mean signed error [95% CI] | Median signed error | Mean ratio [95% CI] | Median ratio | n signed / ratio |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic mean-track | 6 h | -3,739.18 [-11,715.32, 2,822.95] | 0.00 | 0.498 [0.267, 0.700] | 0.306 | 219 / 19 |
| Deterministic mean-track | 12 h | 3,705.85 [-2,435.03, 12,730.58] | 0.00 | 0.423 [0.266, 0.570] | 0.000 | 214 / 20 |
| Deterministic mean-track | 24 h | 10,438.20 [-11,766.69, 48,176.93] | 0.00 | 0.184 [0.053, 0.312] | 0.000 | 201 / 18 |
| Deterministic mean-track | 48 h | 31,971.18 [-1,369.80, 86,801.64] | 0.00 | 0.293 [0.077, 0.488] | 0.000 | 168 / 14 |
| Deterministic mean-track | 72 h | 50,536.26 [2,687.33, 157,721.40] | 0.00 | 0.000 [0.000, 0.000] | 0.000 | 133 / 10 |
| Deterministic mean-track | 96 h | -428.72 [-12,702.34, 20,755.25] | 0.00 | 0.004 [0.000, 0.009] | 0.000 | 100 / 9 |
| P90 envelope | 6 h | 919,870.60 [292,587.23, 1,891,020.68] | 1.19 | 11790.524 [20.762, 31948.910] | 18.366 | 219 / 19 |
| P90 envelope | 12 h | 1,091,270.50 [400,239.20, 2,190,566.54] | 19.52 | 11657.087 [17.120, 31734.080] | 19.696 | 214 / 20 |
| P90 envelope | 24 h | 1,950,951.93 [776,185.27, 3,836,940.25] | 718.16 | 12666.143 [29.732, 32492.211] | 28.667 | 201 / 18 |
| P90 envelope | 48 h | 3,511,059.69 [1,040,912.76, 7,441,532.22] | 32,368.36 | 52318.443 [54.285, 120285.498] | 49.713 | 168 / 14 |
| P90 envelope | 72 h | 5,814,748.89 [2,009,796.84, 12,350,551.80] | 929,767.96 | 75850.635 [94.465, 156638.949] | 66.309 | 133 / 10 |
| P90 envelope | 96 h | 8,691,406.46 [3,529,501.50, 18,619,414.54] | 3,103,594.80 | 95460.111 [207.882, 225481.997] | 88.892 | 100 / 9 |
| Ensemble probability-weighted | 6 h | 6,715.56 [-1,022.33, 22,278.79] | 0.00 | 40.317 [0.285, 108.893] | 0.305 | 219 / 19 |
| Ensemble probability-weighted | 12 h | 7,916.08 [-1,108.92, 24,733.95] | 0.00 | 57.391 [0.371, 155.782] | 0.403 | 214 / 20 |
| Ensemble probability-weighted | 24 h | 24,437.86 [-4,217.29, 68,898.79] | 0.00 | 22.718 [0.314, 56.652] | 0.462 | 201 / 18 |
| Ensemble probability-weighted | 48 h | 24,408.13 [275.58, 68,142.36] | 0.10 | 1216.681 [0.598, 2835.608] | 0.515 | 168 / 14 |
| Ensemble probability-weighted | 72 h | 21,840.54 [2,076.72, 57,074.32] | 22.30 | 40.695 [0.248, 86.814] | 0.571 | 133 / 10 |
| Ensemble probability-weighted | 96 h | 18,938.86 [-6.86, 51,921.01] | 5,742.11 | 1155.249 [0.474, 2815.088] | 0.443 | 100 / 9 |
