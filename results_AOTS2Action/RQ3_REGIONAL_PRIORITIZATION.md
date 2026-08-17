# RQ3: regional-prioritization performance

Marker: **PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE**

Values are equal-weight storm means with 95% cyclone-bootstrap confidence
intervals (10,000 replicates; seed 20260817). Regions are proxy 10-degree
latitude/longitude bins. nDCG uses linear realized vulnerability-weighted
exposure gain. Zero-realized cases are excluded from nDCG and recall.
Spearman uses only the estimator-specific nonzero union and requires at least
three regions.

| Metric | Estimator | 24 h | 48 h | 72 h |
|---|---|---:|---:|---:|
| nDCG@10 | Deterministic mean-track | 0.2419 [0.1212, 0.3616] | 0.2257 [0.0908, 0.3748] | 0.3929 [0.1143, 0.6929] |
| nDCG@10 | P90 envelope | 0.9627 [0.8956, 1.0000] | 0.9328 [0.8223, 1.0000] | 0.9455 [0.8822, 0.9947] |
| nDCG@10 | Ensemble probability-weighted | 0.9858 [0.9650, 1.0000] | 0.8885 [0.7338, 1.0000] | 0.9455 [0.8822, 0.9947] |
| nDCG@5 | Deterministic mean-track | 0.2419 [0.1207, 0.3636] | 0.2257 [0.0927, 0.3723] | 0.3929 [0.1143, 0.6857] |
| nDCG@5 | P90 envelope | 0.9627 [0.8956, 1.0000] | 0.9328 [0.8294, 1.0000] | 0.9455 [0.8822, 0.9947] |
| nDCG@5 | Ensemble probability-weighted | 0.9858 [0.9650, 1.0000] | 0.8885 [0.7361, 1.0000] | 0.9455 [0.8822, 0.9947] |
| Recall@10 | Deterministic mean-track | 0.2419 [0.1187, 0.3642] | 0.2257 [0.0908, 0.3733] | 0.3929 [0.1214, 0.7000] |
| Recall@10 | P90 envelope | 1.0000 [1.0000, 1.0000] | 0.9500 [0.8500, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Recall@10 | Ensemble probability-weighted | 1.0000 [1.0000, 1.0000] | 0.9500 [0.8500, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Recall@5 | Deterministic mean-track | 0.2419 [0.1192, 0.3616] | 0.2257 [0.0927, 0.3742] | 0.3929 [0.1143, 0.6929] |
| Recall@5 | P90 envelope | 1.0000 [1.0000, 1.0000] | 0.9500 [0.8500, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Recall@5 | Ensemble probability-weighted | 1.0000 [1.0000, 1.0000] | 0.9500 [0.8500, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Spearman | Deterministic mean-track | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Spearman | P90 envelope | 0.8203 [0.7746, 0.8660] | 0.8355 [0.7746, 0.8660] | 0.6148 [0.2039, 0.8660] |
| Spearman | Ensemble probability-weighted | 0.8660 [0.8660, 0.8660] | 0.0457 [-0.7746, 0.8660] | 0.8660 [0.8660, 0.8660] |

## Headline: nDCG@10 at 48 h

- Ensemble: **0.8885** (95% CI 0.7338-1.0000)
- Deterministic: 0.2257
- P90 envelope: 0.9328
- Ensemble - Deterministic mean-track: +0.6628 (95% CI +0.5276 to +0.8043); raw p=0.00195312, Holm p=0.00390625; superiority supported.
- Ensemble - P90 envelope: -0.0444 (95% CI -0.1674 to +0.0285); raw p=1, Holm p=1; superiority not supported.
- Ensemble Recall@10: 0.9500 (95% CI 0.8500-1.0000)
- Identified realized top-10 regions: 39/40 (97.50%) across 40 eligible cases.

Because the 25 km footprint and coarse proxy regions usually produce fewer
than ten positive realized regions, the overlap denominator is the number of
available positive regions up to ten, not ten artificial zero-relevance ties.
