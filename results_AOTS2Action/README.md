# AOTS2Action — Results Output Folder (scaffold)

Status: **BLOCKED — manuscript found, data still missing.**

This folder is reserved for the retrospective evaluation results supporting
"AOTS2Action: Uncertainty-Aware Big Data Analytics for Humanitarian Cyclone
Risk" (Adrija Ghosh, Yugyung Lee, IEEE Big Data 2026).

Update 2026-08-17: the manuscript was located at
`~/Downloads/IEEE_BigData_2026_Storm.pdf` (outside this project directory,
not in STORM_CARE). It has been read in full. `csv/placeholder_inventory.csv`
in this folder is a verbatim extraction of every bracketed placeholder in
the paper (Abstract through Conclusion), tagged by category:

- **A** = data/configuration placeholder, fixed before analysis (e.g.
  ensemble source, N, basin, years, r0, r, base buffer b, J, N_X, the O1/O2
  pre-registration choices, hardware specs)
- **B** = computed result, produced by running the pipeline on real data
- **C** = interpretive sentence, written only after B values exist

None of the underlying datasets have been located. A broad filename search
of `~/Downloads` (not just this repo) for IBTrACS/HURDAT/ATCF/GEFS/ECMWF/
WorldPop/GHS-POP/ensemble/best-track found nothing besides the manuscript
PDF itself. This working directory's own data (`your-repo/data/...
weather_ensemble.csv`, `facility_svi.csv`, `data_cache/cb_2023_us_county_500k`)
belongs to a different, unrelated project (STORM-CARE-FM) and is synthetic /
not storm-track-indexed, so it cannot substitute for AOTS2Action's real
ensemble-forecast and best-track inputs without violating the "no
fabricated/estimated results" rule.

Still required before this folder can be populated with real numbers:

- Ensemble forecast source (multi-member cyclone track forecasts, with
  forecast-cycle/valid-time structure) — Sec. IV-A [ENSEMBLE SOURCE]
- Best-track/observational source (IBTrACS / HURDAT2 / JTWC) — Sec. IV-A
- Population dataset (e.g. WorldPop, GHS-POP)
- Vulnerability dataset
- Administrative-boundary dataset
- Infrastructure dataset (optional per manuscript)
- Pre-specified, frozen-before-analysis parameters: r0, r, base buffer b,
  O1 (calibration tolerance), O2 (primary RQ1 representation), M subsample
  policy, spatial-volume reduction method, workstation CPU/RAM/OS

Subfolders:

- `tables/` — Tables I–IV and placeholder replacement sheet, as CSV/Markdown (pending data)
- `csv/` — `placeholder_inventory.csv` (done); machine-readable intermediate/final results (pending data)
- `figures/` — coverage-vs-horizon, area-vs-horizon, scalability plots (pending data)
- `logs/` — run logs / reproducibility manifests (pending data)
