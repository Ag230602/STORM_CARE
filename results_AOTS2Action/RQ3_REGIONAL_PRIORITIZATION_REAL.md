# RQ3: regional-prioritization performance

Marker: **REAL_HUMANITARIAN_GEOSPATIAL_DATA**

Values are equal-weight storm means with 95% cyclone-bootstrap confidence
intervals (10,000 replicates; seed 20260817). nDCG uses linear realized
vulnerability-weighted exposure gain. Zero-realized cases are excluded from
nDCG and recall.
Regions are actual administrative units assigned during real-data harmonization.
Spearman uses only the estimator-specific nonzero union and requires at least
three regions.

| Metric | Estimator | 24 h | 48 h | 72 h |
|---|---|---:|---:|---:|
| nDCG@10 | Deterministic mean-track | 0.1667 [0.0417, 0.3333] | 0.1548 [0.0357, 0.3095] | 0.0000 [0.0000, 0.0000] |
| nDCG@10 | P90 envelope | 1.0000 [1.0000, 1.0000] | 0.7863 [0.5006, 0.9649] | 0.8343 [0.6686, 1.0000] |
| nDCG@10 | Ensemble probability-weighted | 0.9583 [0.8750, 1.0000] | 0.8308 [0.5451, 1.0000] | 0.8760 [0.7520, 1.0000] |
| nDCG@5 | Deterministic mean-track | 0.1667 [0.0417, 0.3333] | 0.1548 [0.0357, 0.3095] | 0.0000 [0.0000, 0.0000] |
| nDCG@5 | P90 envelope | 1.0000 [1.0000, 1.0000] | 0.7863 [0.5006, 0.9649] | 0.8343 [0.6686, 1.0000] |
| nDCG@5 | Ensemble probability-weighted | 0.9583 [0.8750, 1.0000] | 0.8308 [0.5451, 1.0000] | 0.8760 [0.7520, 1.0000] |
| Recall@10 | Deterministic mean-track | 0.1667 [0.0417, 0.3125] | 0.1548 [0.0357, 0.3095] | 0.0000 [0.0000, 0.0000] |
| Recall@10 | P90 envelope | 1.0000 [1.0000, 1.0000] | 0.8571 [0.5714, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Recall@10 | Ensemble probability-weighted | 0.9583 [0.8750, 1.0000] | 0.8571 [0.5714, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Recall@5 | Deterministic mean-track | 0.1667 [0.0417, 0.3333] | 0.1548 [0.0357, 0.3095] | 0.0000 [0.0000, 0.0000] |
| Recall@5 | P90 envelope | 1.0000 [1.0000, 1.0000] | 0.8571 [0.5714, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Recall@5 | Ensemble probability-weighted | 0.9583 [0.8750, 1.0000] | 0.8571 [0.5714, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Spearman | Deterministic mean-track | NA (insufficient regions) | NA (insufficient regions) | NA (insufficient regions) |
| Spearman | P90 envelope | 0.8660 [0.8660, 0.8660] | 0.0748 [-0.2582, 0.4078] | 0.0319 [-0.2582, 0.3220] |
| Spearman | Ensemble probability-weighted | NA (insufficient regions) | 0.8660 [0.8660, 0.8660] | -0.2450 [-0.8660, 0.3761] |

## Headline: nDCG@10 at 48 h

- Ensemble: **0.8308** (95% CI 0.5451-1.0000)
- Deterministic: 0.1548
- P90 envelope: 0.7863
- Ensemble - Deterministic mean-track: +0.6760 (95% CI +0.3878 to +0.9167); raw p=0.0260145, Holm p=0.052029; superiority not supported.
- Ensemble - P90 envelope: +0.0445 (95% CI +0.0000 to +0.1148); raw p=0.179712, Holm p=0.179712; superiority not supported.
- Ensemble Recall@10: 0.8571 (95% CI 0.5714-1.0000)
- Identified realized top-10 regions: 13/14 (92.86%) across 14 eligible cases.

Because the 25 km footprint and regional grid usually produce fewer
than ten positive realized regions, the overlap denominator is the number of
available positive regions up to ten, not ten artificial zero-relevance ties.
