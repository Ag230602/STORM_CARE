# RQ2: direction and magnitude of exposure-estimation bias

Marker: **PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE**

Positive signed error means overestimation and negative signed error means
underestimation. Exposure ratios greater than 1 indicate overestimation and
ratios below 1 indicate underestimation. Brackets contain cyclone-cluster
bootstrap 95% confidence intervals for means (10,000 replicates; seed
20260817). Signed errors retain zero-realized-exposure cases; ratios exclude
them. Exposure values are proxy vulnerability-weighted units, not people.

| Estimator | Lead | Mean signed error [95% CI] | Median signed error | Mean ratio [95% CI] | Median ratio | n signed / ratio |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic mean-track | 6 h | 2,805.05 [-3,437.27, 9,673.47] | 0.00 | 1.011 [0.382, 2.536] | 0.000 | 219 / 59 |
| Deterministic mean-track | 12 h | 6,323.26 [-8,919.06, 26,243.26] | 0.00 | 0.675 [0.324, 1.408] | 0.000 | 214 / 59 |
| Deterministic mean-track | 24 h | 7,595.36 [-941.22, 17,220.32] | 0.00 | 0.462 [0.203, 0.824] | 0.000 | 201 / 49 |
| Deterministic mean-track | 48 h | 18,687.22 [-15,942.44, 73,776.63] | 0.00 | 0.228 [0.138, 0.338] | 0.000 | 168 / 40 |
| Deterministic mean-track | 72 h | 10,839.64 [-32,677.10, 66,308.92] | 0.00 | 0.400 [0.110, 0.918] | 0.000 | 133 / 34 |
| Deterministic mean-track | 96 h | -19,357.69 [-39,963.02, -665.26] | 0.00 | 0.158 [0.036, 0.281] | 0.000 | 100 / 29 |
| P90 envelope | 6 h | 660,400.54 [379,644.20, 913,201.93] | 451,385.74 | 10.673 [5.791, 22.352] | 5.638 | 219 / 59 |
| P90 envelope | 12 h | 755,860.09 [434,894.84, 1,067,341.42] | 501,511.11 | 15.014 [6.547, 36.318] | 6.560 | 214 / 59 |
| P90 envelope | 24 h | 990,938.73 [591,407.84, 1,371,259.36] | 690,970.80 | 19.214 [7.828, 45.490] | 7.313 | 201 / 49 |
| P90 envelope | 48 h | 1,615,650.72 [918,015.79, 2,344,880.68] | 1,095,041.90 | 13.754 [10.654, 18.859] | 10.975 | 168 / 40 |
| P90 envelope | 72 h | 2,376,482.85 [1,390,958.55, 3,477,032.44] | 2,114,541.70 | 122.766 [12.756, 368.799] | 14.566 | 133 / 34 |
| P90 envelope | 96 h | 3,090,150.71 [2,009,459.05, 4,508,548.55] | 2,757,636.11 | 650.984 [17.900, 1603.245] | 21.876 | 100 / 29 |
| Ensemble probability-weighted | 6 h | 6,724.92 [-4,199.04, 19,151.96] | 16,078.48 | 0.511 [0.284, 1.128] | 0.291 | 219 / 59 |
| Ensemble probability-weighted | 12 h | 3,389.61 [-8,559.53, 16,213.26] | 16,111.79 | 0.486 [0.276, 1.024] | 0.285 | 214 / 59 |
| Ensemble probability-weighted | 24 h | 4,864.98 [-11,008.86, 23,190.01] | 16,779.82 | 0.581 [0.256, 1.343] | 0.254 | 201 / 49 |
| Ensemble probability-weighted | 48 h | -5,178.32 [-22,856.40, 14,826.42] | 11,174.38 | 0.205 [0.176, 0.250] | 0.179 | 168 / 40 |
| Ensemble probability-weighted | 72 h | -20,031.53 [-40,120.01, 678.33] | 7,661.67 | 0.327 [0.121, 0.765] | 0.147 | 133 / 34 |
| Ensemble probability-weighted | 96 h | -20,221.00 [-41,127.35, 262.30] | 4,977.49 | 0.747 [0.092, 1.676] | 0.111 | 100 / 29 |

## Numerical interpretation

The deterministic mean-track estimator has small positive mean signed bias
from 6 h through 72 h (+2,805 to +18,687), but every interval in that
range crosses zero. At 96 h it changes to underestimation (-19,358; 95% CI
-39,963 to -665). Its median signed error is zero at every lead. Conditional
mean ratios are below 1 after 6 h, and the 24, 48, 72, and 96 h ratio
intervals are wholly below 1.

The P90 envelope overestimates at every lead. Mean signed bias grows from
+660,401 at 6 h to +3,090,151 at 96 h, and all signed-error intervals are
wholly above zero. Median ratios rise monotonically from 5.64 to 21.88,
showing progressively larger conditional overestimation with lead time.

The ensemble probability-weighted estimator has near-zero mean signed bias
relative to its uncertainty through 24 h (+3,390 to +6,725), then changes
to negative means from 48 h onward (-5,178 to -20,221). Every signed-error
interval still crosses zero. Conditional median ratios decline from 0.291 at
6 h to 0.111 at 96 h; ratio intervals are wholly below 1 at 48 h and 72 h.

Ratio sample sizes fall from 59 at 6 h to 29 at 96 h because 160, 155,
152, 128, 99, and 71 zero-realized cases are excluded by horizon. Ratio
results therefore describe only cases with positive realized proxy exposure.
The disparity between signed-error and ratio summaries is expected: signed
errors include zero-realized cases and preserve exposure magnitude, while
ratios condition on a nonzero denominator and weight each case equally.
