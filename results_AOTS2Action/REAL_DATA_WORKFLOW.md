# Real Humanitarian/Geospatial Data Workflow for RQ2 and RQ3

This workflow replaces the marked proxy exposure grid with a harmonized
real-data grid while preserving the existing RQ2/RQ3 evaluation logic.

## Supported Sources

The harmonized grid can use any vetted source that can be represented as CSV,
GeoJSON, or shapefile inputs:

- Population: WorldPop or GHSL/GHS-POP, reduced to cyclone-grid cells.
- Vulnerability/socioeconomic indicators: CDC/ATSDR SVI for U.S. work, INFORM,
  DHS, World Bank, or another documented socioeconomic layer.
- Administrative boundaries: county, district, state, tract, or equivalent
  polygons appropriate to the study geography.
- Infrastructure: point or polygon layers for schools, hospitals, shelters,
  roads, utilities, or other available facilities.

Every output row retains source names in `population_source`,
`vulnerability_source`, `admin_source`, and `infrastructure_source`. The sidecar
metadata JSON records paths, source URLs, harmonization choices, and row/feature
counts.

## Optional Public Downloads

Use the downloader only after confirming that the selected year/geography fits
the evaluation design:

```bash
python3 scripts/download_public_geospatial_sources.py \
  --output-dir data_cache/public_geodata \
  --cdc-svi county \
  --worldpop-iso3 USA \
  --tiger-year 2024 \
  --tiger-geography county
```

This can fetch:

- CDC/ATSDR SVI 2022 county or tract GeoJSON from the CDC ArcGIS REST service.
- U.S. Census TIGER county or tract boundary zip files.
- WorldPop population catalog metadata for selecting a population raster/source.

Large population rasters should be reviewed and reduced deliberately before
publication. Alternatively, `build_real_humanitarian_grid.py --worldpop-api`
can query WorldPop API v2 per cyclone-grid cell for small study areas.

## Build the Harmonized Grid

Use an existing cyclone forecast grid to keep RQ2/RQ3 aligned with the forecast
domain:

```bash
python3 scripts/build_real_humanitarian_grid.py \
  --forecast-grid path/to/cyclone_forecast_grid.csv \
  --population-csv path/to/population_points_or_grid.csv \
  --population-source-name "WorldPop 2020 100m constrained population" \
  --population-source-url "https://www.worldpop.org/" \
  --vulnerability-vector data_cache/public_geodata/cdc_svi_2022_us_county.geojson \
  --vulnerability-value-field RPL_THEMES \
  --vulnerability-source-name "CDC/ATSDR SVI 2022 county overall percentile" \
  --admin-boundaries path/to/admin_boundaries.shp \
  --admin-id-field GEOID \
  --admin-name-field NAME \
  --admin-source-name "U.S. Census TIGER/Line counties" \
  --infrastructure-csv path/to/infrastructure_points.csv \
  --infrastructure-source-name "HDX/OpenStreetMap/user-vetted facilities" \
  --output results_AOTS2Action/csv/humanitarian_grid_REAL.csv \
  --metadata-output results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json
```

If no fixed forecast grid exists, the script can build one from forecast/member
positions and verifying corpus points:

```bash
python3 scripts/build_real_humanitarian_grid.py \
  --forecasts path/to/member_forecasts.csv \
  --corpus path/to/matched_cases.csv \
  --grid-resolution-deg 0.25 \
  --grid-buffer-deg 2.0 \
  --population-csv path/to/population_points_or_grid.csv \
  --output results_AOTS2Action/csv/humanitarian_grid_REAL.csv \
  --metadata-output results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json
```

## Run RQ2 With Real Data

```bash
python3 scripts/build_aots2action_rq2.py \
  --forecasts path/to/member_forecasts.csv \
  --corpus path/to/matched_cases.csv \
  --grid results_AOTS2Action/csv/humanitarian_grid_REAL.csv \
  --grid-kind real \
  --grid-metadata results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json \
  --case-output results_AOTS2Action/csv/rq2_case_errors_REAL.csv \
  --estimator-output results_AOTS2Action/tables/rq2_estimator_summary_REAL.csv \
  --paired-output results_AOTS2Action/tables/rq2_paired_inference_REAL.csv \
  --table-output results_AOTS2Action/tables/table2_rq2_exposure_fidelity_REAL.csv \
  --improvement-output results_AOTS2Action/tables/rq2_percentage_improvements_REAL.csv \
  --metadata results_AOTS2Action/csv/rq2_metadata_REAL.json
```

## Run RQ3 With Real Data

```bash
python3 scripts/build_aots2action_rq3.py \
  --forecasts path/to/member_forecasts.csv \
  --corpus path/to/matched_cases.csv \
  --grid results_AOTS2Action/csv/humanitarian_grid_REAL.csv \
  --grid-kind real \
  --grid-metadata results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json \
  --case-output results_AOTS2Action/csv/rq3_case_metrics_REAL.csv \
  --summary-output results_AOTS2Action/tables/rq3_regional_ranking_REAL.csv \
  --tests-output results_AOTS2Action/csv/rq3_headline_tests_REAL.csv \
  --metadata results_AOTS2Action/csv/rq3_metadata_REAL.json \
  --report-output results_AOTS2Action/RQ3_REGIONAL_PRIORITIZATION_REAL.md
```

## Publication Guardrail

Only files marked `REAL_HUMANITARIAN_GEOSPATIAL_DATA` should be used for
publication-grade RQ2/RQ3 humanitarian claims. Files marked
`PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE` remain useful for reproducibility and
pipeline testing, but they should not be described as real exposure or impact
results.
