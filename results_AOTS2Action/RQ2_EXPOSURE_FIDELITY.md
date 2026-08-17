# RQ2: vulnerability-weighted exposure-estimation fidelity

## Status and assumptions

Marker: **PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE**

The calculation uses the same 25 km impact radius for forecast and realized
best-track exposure. It uses the UNICEF workflow's explicitly synthetic 0.75
degree grid, where population is derived from forecast-track density and the
vulnerability value is fixed at 0.5. Values are therefore proxy
vulnerability-weighted exposure units, not observed people or impacts.

The deterministic estimator applies the 25 km footprint to the ensemble-mean
position. The P90 estimator applies the footprint to the buffered P90 cone, so
its total radius is the empirical P90 member-to-mean distance plus the 25 km
cone buffer plus the 25 km impact radius. The ensemble estimator averages the
member-level 25 km exposure indicators before applying population and
vulnerability weights. Realized exposure applies the same 25 km operator to the
matched IBTrACS position.

Confidence intervals use 10,000 cyclone-cluster bootstrap replicates with seed
20260817; each sampled cyclone contributes all its cycles. Paired differences
are averaged within cyclone before two-sided Wilcoxon signed-rank tests. Holm
adjustment is across all six primary tests.

## Table II

Mean AE with storm-level bootstrap 95% confidence interval is shown in brackets.

| Result | 24 h | 48 h | 72 h |
|---|---:|---:|---:|
| Deterministic mean-track AE | 57,574.43 [32,286.25, 81,541.01] | 80,488.20 [32,935.62, 139,029.96] | 80,454.22 [38,780.49, 129,882.99] |
| P90-envelope AE | 990,938.73 [590,383.63, 1,371,294.08] | 1,615,675.35 [927,095.25, 2,329,453.39] | 2,376,482.85 [1,408,054.33, 3,463,953.71] |
| Ensemble probability-weighted AE | 53,615.46 [30,547.69, 76,130.72] | 54,577.05 [30,778.59, 76,945.74] | 57,132.10 [32,153.88, 81,924.60] |
| det - ens | 3,958.97 | 25,911.14 | 23,322.12 |
| Holm-adjusted p-value | 0.3818359375 | 0.119384765625 | 0.51953125 |
| P90 - ens | 937,323.27 | 1,561,098.30 | 2,319,350.76 |
| Holm-adjusted p-value | 0.00146484375 | 0.00146484375 | 0.00390625 |

The paired difference rows are differences between overall cycle-level mean
AEs. Wilcoxon testing uses the corresponding within-cyclone mean differences,
as required by the paper.

## Percentage reductions

| Horizon | Improvement vs deterministic | Holm significant | Improvement vs P90 | Holm significant |
|---:|---:|:---:|---:|:---:|
| 24 h | 6.88% | No | 94.59% | Yes |
| 48 h | 32.19% | No | 96.62% | Yes |
| 72 h | 28.99% | No | 97.60% | Yes |

The largest improvement versus deterministic is at 48 h; the pattern grows
from 24 h to 48 h and then attenuates slightly at 72 h, so it is not monotonic.
None of these deterministic comparisons remains significant after Holm
correction. Improvement versus P90 is largest at 72 h and grows monotonically
with lead time; all three P90 comparisons remain significant after Holm
correction.

The very large P90 errors reflect uniform exposure over a broad binary envelope
on a coarse synthetic grid. They must not be interpreted as publication-grade
humanitarian exposure findings.