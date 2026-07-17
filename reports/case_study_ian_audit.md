# Hurricane Ian Case Study Figure Audit

## Root Causes Addressed
- The previous case-study plot inverted longitude, which reverses west-east geography.
- Uncertainty cones were only shown as a final-lead diagnostic instead of explicit P50/P90 probability regions.
- Impact and intervention visualizations were not separated into publication figures.

## Regenerated Outputs
- `figures/case_study/ian_noaa_track_map.png`
- `figures/case_study/ian_noaa_track_map.pdf`
- `figures/case_study/ian_uncertainty_cones.png`
- `figures/case_study/ian_uncertainty_cones.pdf`
- `figures/case_study/ian_trajectory_errors.png`
- `figures/case_study/ian_trajectory_errors.pdf`
- `figures/case_study/ian_impact_map.png`
- `figures/case_study/ian_impact_map.pdf`
- `figures/case_study/ian_intervention_map.png`
- `figures/case_study/ian_intervention_map.pdf`
- `figures/case_study/ian_publication_multipanel.png`
- `figures/case_study/ian_publication_multipanel.pdf`
- `figures/case_study_ian_combined.png`
- `figures/case_study_ian_combined.pdf`
- `figures/case_study_ian_panel1_track.png`
- `figures/case_study_ian_panel2_error.png`
- `figures/case_study_ian_panel3_cone.png`
- `figures/case_study_ian_panel4_humanitarian.png`

## Protocol
- Source predictions: `metrics/inference_test_predictions_all_models.csv`
- Ian prediction rows: 16
- Ian forecast windows: 4
- Axes use true latitude and longitude; western longitudes remain negative and are not inverted.
- Coastlines are schematic offline context layers, not operational NOAA GIS products.
- Humanitarian and intervention maps are synthetic/proxy research visualizations and must not be interpreted as observed damage maps.
