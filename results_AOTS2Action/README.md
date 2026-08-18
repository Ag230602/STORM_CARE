# AOTS2Action — Results Output Folder (scaffold)

Status: **DATA INPUTS STILL REQUIRED — real-data harmonization workflow added.**

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

Added 2026-08-18: `scripts/build_real_humanitarian_grid.py` now creates a
publication-grade RQ2/RQ3 grid from real population, vulnerability,
administrative-boundary, and optional infrastructure layers, retaining source
names/URLs in row columns and a metadata sidecar. `scripts/build_aots2action_rq2.py`
and `scripts/build_aots2action_rq3.py` now accept `--grid-kind real` and mark
those outputs as `REAL_HUMANITARIAN_GEOSPATIAL_DATA`. See
`REAL_DATA_WORKFLOW.md` for commands and source notes.

Final real-data handoff files:

- `MANIFEST_REAL_DELIVERABLES.md` — verifier-facing index of real-data result files.
- `csv/manifest_real_deliverables.csv` — machine-readable manifest with file sizes and SHA-256 hashes.
- `config/real_rerun_settings.json` — configuration, data sources, horizons, tests, and rerun commands.
- `tables/*_REAL.csv` — summary tables for real-data RQ2/RQ3/RQ4.
- `csv/*_REAL.csv` and `csv/*_REAL.json` — case-level outputs, metadata, reliability bins, scalability diagnostics, and real-grid source data.
- `figures/*_REAL.{png,pdf}` — manuscript-check plots for real-data RQ2/RQ3/RQ4.
- `AOTS2Action_REAL_DELIVERABLES.zip` — portable bundle of the real-data reports, CSVs, tables, settings, and plots.
