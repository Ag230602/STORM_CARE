# RQ2: exposure-field Brier scores

Marker: **PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE**

Scores reuse the repository model configuration's pre-specified +/-8
degree storm-centered crop, centered here on the verifying best-track position.
A domain mean is computed
for each case and then cases are averaged, so unequal proxy-grid density does
not reweight cases. Lower Brier score is better.

| Estimator | 6 h | 12 h | 24 h | 48 h | 72 h | 96 h |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic mean-track | 0.002194 | 0.002691 | 0.002475 | 0.002570 | 0.002653 | 0.002667 |
| P90 envelope | 0.052310 | 0.057435 | 0.076280 | 0.138650 | 0.240488 | 0.387230 |
| Ensemble probability-weighted | 0.001563 | 0.001536 | 0.001264 | 0.001191 | 0.001285 | 0.001407 |

## Comparisons

- 6 h: Ensemble probability-weighted is lowest; ensemble improvement is 28.78% over deterministic and 97.01% over P90.
- 12 h: Ensemble probability-weighted is lowest; ensemble improvement is 42.93% over deterministic and 97.33% over P90.
- 24 h: Ensemble probability-weighted is lowest; ensemble improvement is 48.93% over deterministic and 98.34% over P90.
- 48 h: Ensemble probability-weighted is lowest; ensemble improvement is 53.65% over deterministic and 99.14% over P90.
- 72 h: Ensemble probability-weighted is lowest; ensemble improvement is 51.57% over deterministic and 99.47% over P90.
- 96 h: Ensemble probability-weighted is lowest; ensemble improvement is 47.25% over deterministic and 99.64% over P90.

The ensemble estimator is best at **6 out of 6 horizons**.
Across the six horizon scores (macro-average), its improvement is **45.93% over deterministic** and **99.13% over P90**.

## Reliability

Ten-bin reliability-diagram data were generated from the same storm-centered
domains. The evidence does not support one blanket well-calibrated,
overconfident, or underconfident label. At 6 h and 12 h, the well-populated
bins are mostly close to the diagonal. At 24 h and 48 h, positive-probability
bins lie below the diagonal, indicating overforecasting consistent with
overconfidence. At 72 h and 96 h, only nine cells in total have probability
>=0.1, which is insufficient for a reliable confidence diagnosis. The overall
calibration classification is therefore **inconclusive**, with lead-dependent
evidence of overconfidence at 24-48 h and no consistent evidence of
underconfidence.

| Lead | Bin | Cells | Mean probability | Observed frequency |
|---:|---:|---:|---:|---:|
| 6 h | 0.0-0.1 | 45209 | 0.0006 | 0.0005 |
| 6 h | 0.1-0.2 | 102 | 0.1472 | 0.1275 |
| 6 h | 0.2-0.3 | 63 | 0.2426 | 0.2698 |
| 6 h | 0.3-0.4 | 11 | 0.3418 | 0.2727 |
| 6 h | 0.4-0.5 | 7 | 0.4423 | 0.4286 |
| 6 h | 0.5-0.6 | 2 | 0.5000 | 0.0000 |
| 6 h | 0.9-1.0 | 2 | 1.0000 | 0.0000 |
| 12 h | 0.0-0.1 | 44641 | 0.0007 | 0.0007 |
| 12 h | 0.1-0.2 | 142 | 0.1441 | 0.1408 |
| 12 h | 0.2-0.3 | 34 | 0.2400 | 0.1765 |
| 12 h | 0.3-0.4 | 7 | 0.3286 | 0.2857 |
| 12 h | 0.4-0.5 | 5 | 0.4196 | 0.2000 |
| 12 h | 0.9-1.0 | 1 | 1.0000 | 0.0000 |
| 24 h | 0.0-0.1 | 43340 | 0.0009 | 0.0007 |
| 24 h | 0.1-0.2 | 138 | 0.1372 | 0.1232 |
| 24 h | 0.2-0.3 | 18 | 0.2281 | 0.1111 |
| 24 h | 0.3-0.4 | 3 | 0.3349 | 0.0000 |
| 48 h | 0.0-0.1 | 38383 | 0.0012 | 0.0009 |
| 48 h | 0.1-0.2 | 40 | 0.1209 | 0.1000 |
| 48 h | 0.2-0.3 | 3 | 0.2500 | 0.0000 |
| 72 h | 0.0-0.1 | 31706 | 0.0012 | 0.0011 |
| 72 h | 0.1-0.2 | 6 | 0.1202 | 0.0000 |
| 72 h | 0.2-0.3 | 1 | 0.2500 | 0.0000 |
| 96 h | 0.0-0.1 | 23956 | 0.0012 | 0.0012 |
| 96 h | 0.1-0.2 | 2 | 0.1000 | 0.0000 |
