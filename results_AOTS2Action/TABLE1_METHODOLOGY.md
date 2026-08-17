# Table I evaluation corpus

## Inputs

- Forecasts: `../../UNICEF_DATA/AOTS_DATA_SHARE (5).csv`
- Observations: NOAA IBTrACS v04r01 last-three-years CSV, downloaded 2026-08-17 from
  `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.last3years.list.v04r01.csv`
- Observation SHA-256: `4b9277db5ed3f5b0456c4b4c742e3907f9b14b3421659ed5f48557d86e839507`

## Construction

Forecast ensemble rows are reduced to one case per canonical cyclone, forecast
cycle, and requested horizon. Each case is matched to the nearest IBTrACS record
for the same cyclone. Matches farther than 3 hours are excluded. All 1,035
retained matches are exact-time matches.

IBTrACS identifies `FUNG-WONG`, `FUNGWONG`, and `WONG` as the same 2025 cyclone
(`SID=2025308N09144`), so the latter two forecast labels are normalized to
`FUNG-WONG`. This produces 13 canonical cyclones and does not merge any forecast
cycles.

## Outputs

- `csv/table1_evaluation_corpus.csv`: one row per verifying pair
- `tables/table1_evaluation_corpus.csv`: per-horizon Table I counts
- `csv/table1_corpus_metadata.json`: corpus totals and matching configuration

## Unavailable exposure rows

The manuscript defines zero realized exposure from the observed-track footprint
on a population grid and defines `|R_i,h|` from forecast or realized nonzero
vulnerability-weighted scores over administrative regions. These rows cannot be
computed because the manuscript still contains placeholders for impact radius
`r`, population data, vulnerability data, and administrative boundaries. No
corresponding analysis-ready spatial inputs are present. Substituting the
UNICEF project's proxy exposure grid would change the paper's evaluation
protocol and is therefore not used.

## Marked proxy calculation

At the user's request, a separate non-publication-grade calculation is provided
in `tables/table1_evaluation_corpus_MARKED_PROXY.csv`. The exposure-dependent
rows carry the marker `PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE`. The calculation
assumes a 25 km impact radius from the UNICEF workflow's configured base buffer
and uses its explicitly synthetic 0.75-degree grid: population is derived from
forecast-track sample density, vulnerability is fixed at 0.5, and regions are
10-degree bins. These proxy values must not replace the unavailable cells in the
publication-grade table.

## Reproduction

```sh
python3 scripts/build_aots2action_table1.py \
  --forecasts '../UNICEF_DATA/AOTS_DATA_SHARE (5).csv' \
  --observations results_AOTS2Action/data/ibtracs.last3years.list.v04r01.csv \
  --corpus results_AOTS2Action/csv/table1_evaluation_corpus.csv \
  --summary results_AOTS2Action/tables/table1_evaluation_corpus.csv \
  --metadata results_AOTS2Action/csv/table1_corpus_metadata.json
```

The marked proxy rows are reproduced with:

```sh
python3 scripts/build_aots2action_table1_proxy.py \
  --forecasts '../UNICEF_DATA/AOTS_DATA_SHARE (5).csv' \
  --corpus results_AOTS2Action/csv/table1_evaluation_corpus.csv \
  --proxy-grid ../UNICEF_DATA/outputs/proxy_external_grid_from_aots.csv \
  --case-output results_AOTS2Action/csv/table1_exposure_cases_PROXY_ASSUMPTION.csv \
  --summary results_AOTS2Action/tables/table1_exposure_rows_PROXY_ASSUMPTION.csv \
  --full-marked-table results_AOTS2Action/tables/table1_evaluation_corpus_MARKED_PROXY.csv \
  --metadata results_AOTS2Action/csv/table1_proxy_assumption_metadata.json \
  --impact-radius-km 25
```