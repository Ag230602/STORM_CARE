# AOTS2Action Results Folder

Status: **FINAL REAL-DATA RQ2/RQ3/RQ4 DELIVERABLES AVAILABLE**

This folder contains the final real-data outputs for the AOTS2Action
humanitarian/geospatial reruns. The primary handoff bundle is:

- `AOTS2Action_REAL_DELIVERABLES.zip`
- `MANIFEST_REAL_DELIVERABLES.md`
- `csv/manifest_real_deliverables.csv`

The real-data outputs are marked with:

`REAL_HUMANITARIAN_GEOSPATIAL_DATA`

## Exposure Radius and P90 Formulation

The implemented member-level exposure radius is:

`r = 25.0 km`

This same `impact_radius_km=25.0` is used for realized exposure, deterministic
mean-track exposure, each ensemble-member exposure footprint, RQ2, RQ3, and
RQ4.

The P90 exposure baseline uses:

`q90(h) = empirical 90th percentile of member-to-ensemble-mean distances`

`p90_radius = q90(h) + b`

`P90 exposure mask = distance(grid cell, ensemble mean) <= p90_radius + r`

Final constants:

- `r = 25.0 km`
- `b = 25.0 km`
- implemented P90 threshold = `q90(h) + 50.0 km`

So, if the manuscript defines `r0.9(h)` as ensemble-dispersion P90 only, the
implemented formula is:

`r0.9(h) + b + r`

If the manuscript defines `r0.9(h)` as dispersion P90 plus the cone buffer, the
implemented formula is:

`r0.9(h) + r`

## REAL Geospatial Dataset Details

### Population

- Source/name: WorldPop Total Population 1 km ArcGIS ImageServer
- Year: 2020
- Spatial resolution: 1 km native population raster
- API endpoint:
  `https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Total_Population_1km/ImageServer/computeStatisticsHistograms`
- Harmonization: WorldPop population was queried as WGS84 (`wkid=4326`) envelope
  statistics over each harmonized cyclone-grid cell.
- Aggregation: population values are ImageServer sums over cyclone-grid cell
  envelopes.
- Final grid size: `2863` cyclone-grid cells.
- RQ2-active cells queried/cached from WorldPop: `2192`.

### Vulnerability / Socioeconomic Indicator

- Source/name: World Bank country income classification.
- API endpoint:
  `https://api.worldbank.org/v2/country?format=json&per_page=400`
- Administrative level: country.
- Normalization/mapping to vulnerability weight:
  - `LIC = 1.00`
  - `LMC = 0.75`
  - `UMC = 0.50`
  - `HIC = 0.25`
  - fallback / unclassified = `0.50`
- Primary RQ2/RQ3 exposure weight:
  `population * inform_risk`

### Administrative Boundaries / Regions

- Source/name: Natural Earth country boundaries via `datasets/geo-countries`.
- Source URL:
  `https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson`
- Administrative level: country/territory.
- Region identifier: ISO3-style `region_id`; unassigned/offshore cells use
  `UNASSIGNED`.
- RQ3 region count: `37`.
- Spatial join/preprocessing:
  - First assigns each cyclone-grid cell by point-in-polygon using the grid-cell
    center.
  - For coastal/offshore cells, falls back to cell-envelope intersection.
  - Coordinates are handled in WGS84 longitude/latitude.
- Unassigned/ocean cells after coastal-intersection assignment: `1990`.

### Infrastructure

- Included: yes.
- Source/name:
  - Natural Earth 1:10m airports, version 5.0.0
  - Natural Earth 1:10m ports, version 5.0.0
- Spatial type/resolution: global point vector layers, 1:10m scale.
- Source files retained:
  - `data/ne_10m_airports_REAL.geojson`
  - `data/ne_10m_ports_REAL.geojson`
- Features loaded:
  - airports: `893`
  - ports: `1081`
- Harmonization/preprocessing:
  - Airport and port points are counted within `100 km` of each harmonized
    cyclone-grid cell center.
  - Nearest airport and nearest port distances are retained in kilometers.
  - `infrastructure_access_score = airport_count_100km + port_count_100km`
  - `infrastructure_access_score_norm` is normalized by the maximum access score
    observed in the final grid.
- Note: infrastructure columns are retained as real spatial context and
  provenance. They are not mixed into the primary RQ2/RQ3 exposure target, which
  remains `population * inform_risk`.

## Main Final Outputs

- RQ2 report: `RQ2_EXPOSURE_EXPERIMENTS_REAL.md`
- RQ2 bias report: `RQ2_BIAS_ANALYSIS_REAL.md`
- RQ2 Brier report: `RQ2_BRIER_SCORES_REAL.md`
- RQ3 report: `RQ3_REGIONAL_PRIORITIZATION_REAL.md`
- RQ4 report: `RQ4_SCALABILITY_REAL.md`
- Dataset details: `DATASET_EXPERIMENT_DETAILS_REAL.md`
- Final settings/config: `config/real_rerun_settings.json`
- Harmonized real grid: `csv/humanitarian_grid_REAL.csv`
- Harmonized real-grid metadata: `csv/humanitarian_grid_REAL.metadata.json`

## Summary Tables and Plots

- `tables/*_REAL.csv` contains final summary tables for RQ2, RQ3, and RQ4.
- `csv/*_REAL.csv` contains case-level and diagnostic outputs.
- `figures/*_REAL.png` and `figures/*_REAL.pdf` contain manuscript-check plots.

The manifest files list every deliverable and SHA-256 hash for verification.
