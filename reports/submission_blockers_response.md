# Submission Blockers — Response

This document answers the "highest-priority submission blockers" and
lower-priority open items raised against the draft paper, with every value
pulled directly from the live repository code, saved checkpoint configs, and
regenerated metrics/tables as of this pass — not from the paper draft or
memory. Where the codebase genuinely does not support a value (e.g. a data
source that isn't wired in), that is stated explicitly rather than
approximated.

---

## Highest-priority submission blockers

### 1. Proxy-label generator (S2.2)

Source: `model/disaster_graph/schema.py::humanitarian_targets()` (formulas) +
`metrics/humanitarian/humanitarian_label_audit.json` (distributions). Exact
equations, from the frozen checkpoint's config (`n_pop=5, n_schools=6,
n_hospitals=2, n_shelters=3`, `school_disruption_threshold=0.15`):

- `exposed_children_count = pop_feat[:,0] * pop_feat[:,3] * child_exposure_frac * 20,000`,
  where `child_exposure_frac = clamp(damage_score_pop, 0, 1)` — `pop_feat[:,0]`
  is a normalized population-cluster weight, `pop_feat[:,3]` is child-fraction,
  `20,000` is a fixed per-cluster population scale constant (not derived from
  any real-population source — see caveat below).
- `school_disrupted = 1[school_damage > 0.15]`,
  `school_damage = clamp(damage_score_school, 0, 1)`.
- `hospital_access = clamp(1 - damage_score_hospital, 0, 1)` — a `[0,1]` index,
  unitless.
- `shelter_demand = clamp(0.55·damage_score_shelter + 0.45·shelter_capacity_feature, 0, 1)`.
- `recovery_priority = clamp(damage_score_region_infra, 0, 1)`.

**Caveat needed for S2.2**: none of this reads WorldPop, CDC SVI, or an HDX
facility snapshot. `damage_score_*` comes entirely from a synthetic
Rankine-vortex scenario generator (`build_dataset()` in the same file, seeded
via `np.random.default_rng`). There is no WorldPop file anywhere in the repo,
and grepping for `worldpop|HDX` across all `.py` files returns nothing. Real
CDC SVI 2022 data *does* exist on disk
(`your-repo/data/data/processed/facility_svi.csv`, 7,711 rows: 351 hospital /
1,978 shelter / 5,381 school, joined to `RPL_THEME1-4`; also a tract-level
`vulnerability_grid_clean.csv`, 5,123 Florida FIPS rows) but it is **not
wired into Module 3's label generator at all** — it's used only by Module 1's
foundation-model vulnerability features. If S2.2 currently implies Table 2's
labels are derived from real facility/SVI/population data, that is not
accurate for the current code and needs correcting, or the pipeline needs to
be built.

### 2. Final model configs (S3, S6, S7)

Pulled directly from the saved checkpoints' embedded `cfg` dicts (the actual
frozen run, not class defaults):

- **M1** (`checkpoints/foundation/foundation_best.pt`, epoch 20 selected):
  `d_model=128, n_heads=4, n_layers=3, d_ff=512`, `batch_size=16`, AdamW
  `lr=2e-4, weight_decay=1e-4`, LambdaLR warm-up (`warmup_epochs=5` of
  `epochs=20`, linear warm-up then decay — see `_lr_lambda` in `pretrain.py`).
- **M2** (`checkpoints/physics/full/pigno_best.pt`): `n_gno_layers=2`,
  `n_fno_layers=2`, `n_modes_x=n_modes_y=6`, `fno_width=32`, `d_v=32,
  d_hidden=64`, AdamW `lr=1e-3, weight_decay=1e-4`, LambdaLR.
- **M3** (`checkpoints/disaster_graph/disaster_gnn_best.pt`):
  `n_gnn_layers=2`, `n_node_types=6`, `n_edge_types=4`, `d_hidden=32,
  d_type_emb=8, d_edge_emb=8`, AdamW `lr=1e-3, weight_decay=1e-4`,
  CosineAnnealingLR (`T_max=n_epochs, eta_min=lr*0.01`). Five output heads
  (`model/disaster_graph/architecture.py`), all the same 2-layer MLP shape
  `Linear(d,d/2)→GELU→Linear(d/2,1)→activation`: `damage_head` (Sigmoid),
  `recovery_head` (Sigmoid), `child_exposure_head` (Sigmoid),
  `school_disruption_head` (Sigmoid), `hospital_access_head` (Sigmoid, then
  `1-x`), `shelter_demand_head` (Sigmoid). Plus a separate `state_head`
  (2-layer MLP, no final activation) projecting to `d_disaster_state=32` for
  the World Model handoff.
- **M4** (`checkpoints/world_model/worldmodel_best.pt`): `d_latent=16,
  d_hidden=32, d_enc_hidden=32, d_dec_hidden=32`, `n_steps_train=8,
  n_forecast=8`, `beta_kl=0.1, beta_pred=0.5`, AdamW `lr=1e-3,
  weight_decay=1e-4`, CosineAnnealingLR.
- **LSTM** (`model/track_pipeline_unified_X.py`): 1-layer
  `nn.LSTM(input_size=32, hidden_size=64)` + `OperatorEncoder(width=32,
  out_dim=96)` for ERA5.
- **Transformer**: `d_model=64, nhead=4, layers=2`, `dim_feedforward=128,
  dropout=0.1`.
- **DCRNN**: DCGRU cell, `hidden_ch=8, k_hops=2`, bidirectional diffusion
  convolution over an 8-neighbor k-NN grid graph.
- **GNO+DynGNN**: `OperatorEncoder(width=48, out_dim=128)` +
  `DynamicGNN(node_dim=32, hidden=64, layers=2)`.
- **RF/XGBoost/MLP** (humanitarian baselines, `scripts/eval_humanitarian.py`):
  `RandomForestRegressor(n_estimators=50, random_state=42)`,
  `MLPRegressor(hidden_layer_sizes=(32,16), max_iter=200, random_state=42)`,
  `XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
  random_state=42)`. Feature vector is 63-dim: mean/std/max of 7 raw node
  features (21) + per-node-type mean over 6 node types (42), all label-free.

### 3. Irma/Ian case-study numeric table (S6)

`tables/table_case_study_track_error.csv`, full contents:

| Model | 6h | 12h | 24h | 48h | mean(6-48h) |
|---|---:|---:|---:|---:|---:|
| Persistence | 29.247 | 72.842 | 197.041 | 527.727 | 206.714 |
| GNO+DynGNN | 119.690 | 156.353 | 322.895 | 598.080 | 299.255 |
| Transformer | 170.140 | 211.689 | 381.798 | 611.512 | 343.785 |
| DCRNN | 89.515 | 269.963 | 335.053 | 736.555 | 357.771 |
| LSTM | 83.445 | 286.149 | 442.575 | 873.994 | 421.541 |

(km, source `metrics/inference_test_metrics_summary.csv`, protocol note:
"window-level Irma/Ian case study" — confirms the S9 caveat about not
treating this as held-out-storm generalization.)

### 4. Data provenance (S1)

- **HURDAT2**: file runs through 2024-11-18 (last record), no embedded
  revision-date field — NHC HURDAT2 files don't carry one internally; cite it
  as "2024 Atlantic HURDAT2, as released by NHC" — no exact download
  timestamp is recoverable from the file itself.
- **IBTrACS**: **not used in the frozen run** — `ibtracs_path=None` in the
  saved checkpoint config, so the pipeline fell back to HURDAT2 + a synthetic
  global-storm generator. Any IBTrACS version claim would be false as
  currently configured.
- **ERA5**: variables `u, v, z` (zonal wind, meridional wind, geopotential)
  at pressure levels **850 hPa and 500 hPa**, 0.25°×0.25° resolution,
  ECMWF/`ecmf` institution, downloaded via `cfgrib` (GRIB history timestamp
  `2026-01-07`). Test-set ERA5 coverage: **1 of 107 test storms** has any
  ERA5 file (`metrics/test_coverage/test_coverage_manifest.csv`); overall
  dataset-wide coverage is 86/11,087 observations = 0.8%.
- **WorldPop**: not present anywhere in the repo, not referenced in any code
  path. Cannot supply a year/resolution because it isn't used.
- **CDC SVI**: real data, **2022 release, Florida geography**, both raw
  (`Florida_SVI_2022.zip`, `SVI2022_FLORIDA_tract.gdb`) and processed
  tract-level (`vulnerability_grid_clean.csv`, 5,123 FIPS rows, standard
  4-theme + composite `RPL_THEMES`) — but only consumed by Module 1, not
  Module 3 (see item 1).
- **HDX facility snapshot**: `facility_svi.csv` has facility IDs like
  `shelter_0`, `hospital_1` — these are **synthetic identifiers**, not a real
  HDX facility list. No HDX source file or download script exists anywhere in
  the repo. Post-filter counts in that file: 351 hospital / 1,978 shelter /
  5,381 school rows, but these cannot be verified to trace to an actual HDX
  snapshot rather than being generated alongside the synthetic scenario data.

### 5. Physics implementation detail (S4.1)

Source: `model/physics/physics_kernels.py`. State channels are `[u850, v850,
u500, v500, z500, T2m, MSLP]` (7 channels) — there is no `q` (humidity),
`rho`, or `W` channel in this repo; `rho` is a constant (`rho_air=1.225
kg/m³`), not a state variable. If S4.1's notation assumes those channels
exist, that is a mismatch with the actual code.

Wind-pressure ("gradient wind balance") residual, evaluated on the
**predicted** next state:

```
R_wp = V²/r + f·V + (1/ρ)·∂p/∂r,   V = √(u²+v²)
```

computed via `ops.radial_gradient(p, u, v)`; `r` is physical radius from
domain center in meters (`domain_radius_deg × 111,000`). There is no single
named matrix `B(v,p)` or `S` in the code — normalization is per-residual
scalars, not a matrix: `_accel_scale()` for advection/wind-pressure,
`_tendency_scale()` for diffusion, and an explicit per-channel vector
`[wind_scale_ms×4, geopotential_scale_m, temperature_scale_k,
pressure_scale_hpa]` for temporal continuity. `kappa_diffusivity = 1000.0
m²/s` — confirmed, this is κ in the temperature advection-diffusion equation.

### 6. Uncertainty-cone implementation (S8.1)

Axis-aligned (not full-covariance) Gaussian ellipse: `((lat-mu_lat)/sigma_lat)^2
+ ((lon-mu_lon)/sigma_lon)^2 <= z^2`, independent sigma per axis, no
cross-term — same convention in both `track_pipeline_unified_X.py` and
`foundation/evaluation.py`. `z` from 2-D chi-square cutoffs: **z_P50 = 1.177,
z_P90 = 2.146**. Effective per-horizon calibration counts (selected epoch 20,
`foundation_eval_metrics.csv`):

| Horizon | n_valid |
|---|---:|
| 6h | 238 |
| 12h | 231 |
| 24h | 219 |
| 48h | 194 |
| 72h | 169 |
| 120h | 119 |

### 7. Exact Holm-adjusted p-values (S9.2)

`metrics/counterfactual/dose_response_adjacent_tests.csv` — `scripts/check_dose_response.py`
now applies the same `holm_adjust()` step-down correction used by
`scripts/compute_significance.py` (imported directly, not reimplemented, so
the two scripts can't drift) to this 3-test adjacent-step family:

| Step | Mean diff | p (raw, one-sided) | p (Holm-adjusted) | Claimable at 0.05 |
|---|---:|---:|---:|---|
| earlier_evacuation vs baseline | -0.00843 | 9.08e-06 | 2.72e-05 | True |
| earlier_evacuation_24h vs earlier_evacuation | -0.00438 | 1.03e-05 | 2.72e-05 | True |
| earlier_evacuation_36h vs earlier_evacuation_24h | -0.00333 | 1.92e-05 | 2.72e-05 | True |

All three adjacent-step comparisons remain claimable after correction — the
raw p-values were small enough that Holm's step-down adjustment (multiply
each by its rank-descending count, 3/2/1) doesn't change the conclusion, it
just makes the reported number the defensible one.

### 8. Compute (S11.2)

Everything in this repo was trained **CPU-only** — no GPU was used anywhere;
`torch.cuda.is_available()` is `False` in this environment, so **total
accelerator-hours = 0**. Wall-clock for the actual frozen runs (CPU, this
machine):

| Module | Epochs | Wall-clock |
|---|---:|---|
| M1 (foundation) | 20 | 26.6 min real (55.9 min user-time, ~2 threads) |
| M2 (physics) | 20 | ~55s |
| M3 (disaster graph) | 20 | ~34s |
| M4 (world model) | 20 | 14.2s real (8.3s user-time) |

(M4 timed by rerunning `model.world_model.train --demo` with the frozen
checkpoint's exact seed=42 config; the rerun reproduced the checkpoint and
train log byte-for-byte, so the live checkpoint was left untouched.)

If S11.2 needs GPU-hours, the honest answer is there are none — the
"~4-8 GPU-hours on A100" figure in `reports/foundation_checkpoint_audit.md`
is a docstring *estimate* for the undone full-scale run, not a measured
number.

### 9. Anonymous code-release plan (S11.1)

This is a decision for the author/advisor, not something derivable from the
repo. An anonymized zip can be produced on request (scrubbed
names/paths/checkpoint metadata) — one was built earlier
(`STORM_CARE_anonymized_supplementary.zip`), but it predates the E1/E3/E5/E7
work in this pass and would need regenerating to be current.

---

## Lower-priority / optional items

- **Storm-category distribution by split**: not currently a tracked artifact
  (`foundation_split_manifest.csv` has basin/year/name, not Saffir-Simpson
  category). Computable by joining HURDAT2 status codes to
  `splits/storm_splits.json`, but not yet done.
- **Per-horizon random-init values**: already in
  `metrics/ablations/foundation_ablation_metrics.csv` (`no_ssl_random_init`
  row).
- **Per-scenario humanitarian error distributions**: not stored — only
  aggregate sMAPE/MAPE/MAE per model exist. Would need
  `scripts/eval_humanitarian.py` changed to emit per-scenario rows.
- **P50/other coverage points**: already in
  `metrics/foundation/foundation_eval_metrics.csv` — epoch-20 `cone_p50`
  values: 0.588 (6h) / 0.576 (12h) / 0.489 (24h) / 0.428 (48h) / 0.331 (72h) /
  0.210 (120h).
- **Canonical split manifest hash**: `splits/storm_splits.json` -> sha256
  `87bae628702eaf8bf887ba039b9f8aeff1e70123099c5b5a33b9d6a338f27959`.
