# RQ1: calibration, sharpness, and track error

## Status and assumptions

Marker: **ASSUMED_NOT_ORIGINALLY_PREREGISTERED**

The manuscript leaves `r0`, cone buffer `b`, and the O2 primary P90
representation as placeholders. Before executing this RQ1 calculation, this run
fixed `r0 = 25 km`, `b = 25 km`, P90 covariance ellipse as the primary
representation, and P90 percentile cone as its comparison. The 25 km values
come from the pre-existing UNICEF workflow configuration. This is a transparent
analysis assumption, not a claim that the paper originally preregistered it.

Coverage intervals are 95% percentile intervals from 10,000 storm-level
bootstrap replicates with seed 20260817. Covariance ellipses require at least
three members and nonsingular sample covariance; ineligible cases are excluded
from ellipse rows only.

## Calibration and sharpness

| Representation | Horizon | Coverage (95% CI) | Absolute calibration error | Mean area (km2) | Median area (km2) | n |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-radius region | 6 h | 0.425 [0.326, 0.507] | N/A | 1,963 | 1,963 | 219 |
| Fixed-radius region | 12 h | 0.290 [0.203, 0.359] | N/A | 1,963 | 1,963 | 214 |
| Fixed-radius region | 24 h | 0.199 [0.144, 0.254] | N/A | 1,963 | 1,963 | 201 |
| Fixed-radius region | 48 h | 0.119 [0.047, 0.196] | N/A | 1,963 | 1,963 | 168 |
| Fixed-radius region | 72 h | 0.045 [0.000, 0.116] | N/A | 1,963 | 1,963 | 133 |
| Fixed-radius region | 96 h | 0.020 [0.000, 0.047] | N/A | 1,963 | 1,963 | 100 |
| P50 percentile cone | 6 h | 0.904 [0.836, 0.948] | 0.404 | 15,279 | 12,177 | 219 |
| P50 percentile cone | 12 h | 0.902 [0.825, 0.951] | 0.402 | 17,753 | 14,221 | 214 |
| P50 percentile cone | 24 h | 0.851 [0.752, 0.915] | 0.351 | 25,199 | 20,665 | 201 |
| P50 percentile cone | 48 h | 0.821 [0.693, 0.912] | 0.321 | 50,353 | 44,416 | 168 |
| P50 percentile cone | 72 h | 0.722 [0.556, 0.871] | 0.222 | 98,345 | 80,823 | 133 |
| P50 percentile cone | 96 h | 0.620 [0.434, 0.809] | 0.120 | 176,425 | 143,328 | 100 |
| P90 percentile cone | 6 h | 0.959 [0.920, 0.988] | 0.059 | 42,446 | 31,403 | 219 |
| P90 percentile cone | 12 h | 0.967 [0.937, 0.990] | 0.067 | 47,890 | 35,321 | 214 |
| P90 percentile cone | 24 h | 0.975 [0.946, 0.995] | 0.075 | 75,747 | 59,642 | 201 |
| P90 percentile cone | 48 h | 0.982 [0.952, 1.000] | 0.082 | 158,047 | 139,859 | 168 |
| P90 percentile cone | 72 h | 0.970 [0.925, 1.000] | 0.070 | 335,710 | 283,801 | 133 |
| P90 percentile cone | 96 h | 0.960 [0.926, 1.000] | 0.060 | 695,107 | 539,933 | 100 |
| P50 covariance ellipse | 6 h | 0.741 [0.623, 0.843] | 0.241 | 7,421 | 4,847 | 212 |
| P50 covariance ellipse | 12 h | 0.686 [0.583, 0.760] | 0.186 | 8,676 | 6,348 | 210 |
| P50 covariance ellipse | 24 h | 0.680 [0.541, 0.767] | 0.180 | 13,400 | 10,370 | 200 |
| P50 covariance ellipse | 48 h | 0.665 [0.535, 0.765] | 0.165 | 31,136 | 27,493 | 167 |
| P50 covariance ellipse | 72 h | 0.669 [0.504, 0.850] | 0.169 | 63,437 | 57,604 | 130 |
| P50 covariance ellipse | 96 h | 0.640 [0.471, 0.860] | 0.140 | 118,043 | 99,956 | 100 |
| P90 covariance ellipse | 6 h | 0.925 [0.853, 0.969] | 0.025 | 24,651 | 16,101 | 212 |
| P90 covariance ellipse | 12 h | 0.929 [0.865, 0.967] | 0.029 | 28,820 | 21,086 | 210 |
| P90 covariance ellipse | 24 h | 0.940 [0.901, 0.971] | 0.040 | 44,515 | 34,448 | 200 |
| P90 covariance ellipse | 48 h | 0.988 [0.964, 1.000] | 0.088 | 103,430 | 91,331 | 167 |
| P90 covariance ellipse | 72 h | 0.954 [0.880, 1.000] | 0.054 | 210,732 | 191,355 | 130 |
| P90 covariance ellipse | 96 h | 0.940 [0.891, 1.000] | 0.040 | 392,130 | 332,048 | 100 |

## Ensemble-mean track error

| Horizon | Mean error (km) | Median error (km) | n |
|---:|---:|---:|---:|
| 6 h | 36.28 | 28.38 | 219 |
| 12 h | 41.50 | 35.07 | 214 |
| 24 h | 52.82 | 43.35 | 201 |
| 48 h | 82.96 | 77.07 | 168 |
| 72 h | 135.38 | 121.19 | 133 |
| 96 h | 208.31 | 172.85 | 100 |

## Primary 48-hour P90 endpoint

1. Selected representation: P90 covariance ellipse
2. Empirical coverage: 0.9880
3. 95% CI: [0.9635, 1.0000]
4. Mean region area: 103,430.49 km2
5. Comparison representation: P90 percentile cone
6. Baseline mean area: 158,047.33 km2
7. Area reduction: 34.56%

The selected P90 coverage is above nominal 0.90 by 0.0880 at 48 h, and its
bootstrap interval lies above 0.90. Across horizons its point coverage is above
0.90 throughout: it rises from 0.9245 at 6 h to 0.9880 at 48 h, then declines to
0.9538 at 72 h and 0.9400 at 96 h. The manuscript's tolerance for classifying a
result as "near nominal" remains an unresolved placeholder, so no tolerance-based
near/above label is asserted.