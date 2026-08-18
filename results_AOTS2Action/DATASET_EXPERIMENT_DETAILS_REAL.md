# Dataset and Experiment Details for Placeholder Removal

Marker: **REAL_HUMANITARIAN_GEOSPATIAL_DATA**

## Ensemble Forecast Source

- Local source file used for all AOTS2Action reruns: `../UNICEF_DATA/AOTS_DATA_SHARE (5).csv`.
- Local dataset documentation labels this file as **AOTS tropical cyclone ensemble forecasts** from the **Advanced Operational Tropical-cyclone Simulation** dataset.
- Required columns used: `FORECAST_TIME`, `TRACK_ID`, `ENSEMBLE_MEMBER`, `VALID_TIME`, `LEAD_TIME`, `LATITUDE`, `LONGITUDE`.
- The CSV also contains intensity and wind-field fields including pressure, wind speed, radii of maximum winds, quadrant wind radii, and wind-field polygons.
- Public-provider note: the available local documentation identifies the dataset as AOTS / Advanced Operational Tropical-cyclone Simulation. It does not encode a separate public institutional provider label such as ECMWF or GEFS.

## Cyclones, Basins, Years, and Cycles

- Canonical cyclones: 13.
- Names/list: DITWAH, DUDZAI, FINA, FUNG-WONG, GEZANI, KALMAEGI, KOTO, MELISSA, MONTHA, NOKAEN, PENHA, SENYAR, SONIA.
- Years represented by forecast initialization times: 2025, 2026.
- Basin coverage: Australian / South Pacific region, Eastern Indian Ocean / maritime Southeast Asia, Eastern/Central Pacific, North Atlantic, North Indian Ocean, Southwest Indian Ocean, Western North Pacific, Western North Pacific / maritime Southeast Asia.
- Forecast cycles: 222 canonical cyclone/forecast-time cycles.
- Requested verification horizons: 6, 12, 24, 48, 72, 96 h for RQ2; 24, 48, 72 h for RQ3 and scalability.
- Maximum/full ensemble size: M_full = 51.

| Cyclone | Basin/region | Forecast cycles | Lat range | Lon range |
|---|---|---:|---:|---:|
| DITWAH | North Indian Ocean | 22 | 1.8 to 21.9 | 67.0 to 92.0 |
| DUDZAI | Southwest Indian Ocean | 1 | -37.0 to -14.5 | 53.9 to 75.1 |
| FINA | Australian / South Pacific region | 26 | -24.7 to -3.6 | 111.5 to 150.7 |
| FUNG-WONG | Western North Pacific | 32 | 7.5 to 48.8 | 105.8 to 177.5 |
| GEZANI | Southwest Indian Ocean | 16 | -50.9 to -11.2 | 27.1 to 81.3 |
| KALMAEGI | Western North Pacific | 23 | 7.9 to 25.0 | 87.4 to 138.8 |
| KOTO | Western North Pacific / maritime Southeast Asia | 32 | 1.4 to 19.5 | 87.5 to 121.2 |
| MELISSA | North Atlantic | 32 | 11.4 to 62.2 | -93.0 to -8.3 |
| MONTHA | North Indian Ocean | 12 | 11.5 to 27.6 | 76.7 to 95.6 |
| NOKAEN | Western North Pacific | 1 | 8.4 to 23.1 | 117.4 to 142.0 |
| PENHA | Eastern Indian Ocean / maritime Southeast Asia | 3 | -0.3 to 14.8 | 89.4 to 130.0 |
| SENYAR | Eastern Indian Ocean / maritime Southeast Asia | 7 | 0.4 to 18.8 | 95.3 to 116.9 |
| SONIA | Eastern/Central Pacific | 15 | 8.8 to 17.7 | -149.8 to -116.9 |

## Exposure and Regional-Ranking Parameters

- Impact radius for exposure: 25.0 km.
- P90 cone buffer: 25.0 km.
- P90 envelope radius: empirical P90 member-to-mean distance + cone buffer + impact radius.
- RQ2 confidence intervals: cyclone-cluster bootstrap, 10,000 replicates, seed 20260817.
- RQ2 paired comparisons: cycle differences averaged within cyclone, two-sided Wilcoxon signed-rank, Holm correction.
- RQ3 region definition: administrative boundaries from harmonized grid; region count = 37.
- RQ3 gain: linear realized vulnerability-weighted exposure.
- RQ3 zero-realized cases: excluded from nDCG and recall because those metrics are undefined.

## Real Geospatial Dataset Details

- Harmonized real grid: `results_AOTS2Action/csv/humanitarian_grid_REAL.csv`.
- Full harmonized grid size used by the exposure/scalability kernels: N_X = 2863 cyclone-grid cells.
- RQ2-active WorldPop cells queried/summarized: 2192.
- Native population source: WorldPop Total Population 1 km ImageServer; data year 2020; native resolution 1km.
- WorldPop endpoint: https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Total_Population_1km/ImageServer/computeStatisticsHistograms.
- Population values were obtained as ImageServer `computeStatisticsHistograms` sums over cyclone-grid cell envelopes.
- Vulnerability/socioeconomic source: World Bank country income classification.
- World Bank country API: https://api.worldbank.org/v2/country?format=json&per_page=400.
- Vulnerability mapping: {"HIC": 0.25, "LIC": 1.0, "LMC": 0.75, "UMC": 0.5}; fallback 0.5.
- Administrative boundaries: Natural Earth country boundaries via geo-countries.
- Administrative boundary URL: https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson.
- Unassigned/ocean cells after coastal-intersection assignment: 1990.
- Infrastructure sources: Natural Earth 1:10m airports v5.0.0 and Natural Earth 1:10m ports v5.0.0.
- Infrastructure source files retained in this handoff:
  - `results_AOTS2Action/data/ne_10m_airports_REAL.geojson`
  - `results_AOTS2Action/data/ne_10m_ports_REAL.geojson`
- Infrastructure harmonization: airport and port points are counted within 100 km of each harmonized cyclone-grid cell center; nearest airport/port distances are retained in km.
- Infrastructure columns retained in the real grid: `airport_count_100km`, `port_count_100km`, `nearest_airport_km`, `nearest_port_km`, `infrastructure_access_score`, `infrastructure_access_score_norm`, and `infrastructure_source`.
- Infrastructure feature counts loaded: 893 airports and 1081 ports.
- Primary exposure-weight note: RQ2/RQ3 continue to use `population * inform_risk` as the prespecified vulnerability-weighted exposure target; infrastructure is retained as real spatial context/provenance rather than mixed into the primary exposure metric.

## Scalability Experiment Details

- Real scalability report: `results_AOTS2Action/RQ4_SCALABILITY_REAL.md`.
- Grid file: `results_AOTS2Action/csv/humanitarian_grid_REAL.csv`.
- Full grid size: N_X = 2863.
- M values: 5, 10, 20, 40, M_full; M_full = 51.
- Spatial fractions/sizes: [(0.1, 286), (0.25, 716), (0.5, 1432), (0.75, 2147), (1.0, 2863)].
- Repeats per configuration: 10.
- Eligible cases for fixed-M_full timing: 298 out of 502 loaded cases.
- Runtime scope: per forecast case, derived from total eligible-case batch runtime / eligible_case_count.
- Throughput formula: M * |X| / runtime_s.

## Machine Specifications

- CPU: Apple M1 Pro.
- RAM: 16.00 GB.
- OS: macOS-26.5.2-arm64-arm-64bit.
- Python: 3.9.6.
- NumPy: 2.0.2.

## Largest Real Scalability Configuration

- Configuration: M_full=51, N_X=2863.
- Mean runtime: 0.005151 s per forecast case.
- Median runtime: 0.005122 s per forecast case.
- Runtime std: 0.000049 s.
- Peak memory: 0.0314 GB.
- Mean throughput: 28,348,666.61 items/s.
