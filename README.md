# 🌪 STORM-CARE  
## Uncertainty-Aware Hurricane Forecasting & Child-Centered Decision Intelligence  
### Spatio-Temporal Graph Neural Networks for Humanitarian Planning

Dr. Yugyung Lee (Professor of Computer Science), Adrija Ghosh (CS Graduate Student)
School of Science and Engineering, University of Missouri–Kansas City

<p align="center">
  <b>Research Prototype | Probabilistic Forecasting | Human-Centered AI | Governance-Aware Visualization</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ST--GNN-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-ff4b4b)
![Geospatial](https://img.shields.io/badge/Geo-Spatial_Data-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</p>

---

# 🌍 Executive Summary

**STORM-CARE (Storm-Focused Child-Centered Actionable Risk Engine)** is a research and demonstration framework that transforms hurricane forecast uncertainty into actionable humanitarian planning intelligence using:

- 📈 Spatio-Temporal Graph Neural Networks (ST-GNN)  
- 📊 Probabilistic forecasting (P10 / P50 / P90)  
- 🌍 Multi-source geospatial data fusion  
- 🧒 Child-centered vulnerability modeling  
- 🗺 Governance-aware visualization  

Unlike traditional hurricane systems that stop at meteorological prediction, STORM-CARE translates uncertainty into impact-aware decision signals.

Data link:https://drive.google.com/drive/folders/1Qnki8L52euDNkveHjIheGZH-cz7gt1CB?usp=drive_link

 
---

# 🎯 Problem Statement

Emergency planners need more than:

- "Where will the storm go?"

They need:

- How many children could be exposed?
- Which schools are at risk?
- Where will healthcare access degrade?
- What does worst-case uncertainty imply?

STORM-CARE bridges:

Meteorology → Uncertainty Modeling → Human Impact → Planning Intelligence


---

# 🏗 System Architecture

     Multi-Source Data Ingestion
                 ↓
     Geospatial Harmonization
                 ↓
 Feature Engineering (Static + Dynamic)
                 ↓
 Graph Construction (Tract-Level Nodes)
                 ↓
Spatio-Temporal Graph Neural Network
↓
Probabilistic Forecast Outputs (P10/P50/P90)
↓
Child-Centered Impact Translation
↓
Interactive Decision Dashboard


---

# 🌪 Core Technical Contributions

## 1️⃣ Spatio-Temporal Graph Forecasting

- Census tract–level node modeling  
- Spatial adjacency via k-nearest neighbor graphs  
- Temporal storm progression encoding  
- Vulnerability-weighted hazard diffusion  
- Current regenerated results do **not** support an all-horizon forecast-superiority
  claim over Persistence; see `reports/forecast_performance_audit.md`.

---

## 2️⃣ Probabilistic Forecasting for Planning

Instead of a single deterministic track:

- **P50:** Median scenario  
- **P10:** Lower-bound scenario  
- **P90:** Upper-bound scenario  

This enables scenario-based resource allocation under uncertainty.

---

## 3️⃣ Child-Centered Impact Modeling

Derived indicators include:

- Children exposed (P50 & P90)  
- School disruption probability  
- Healthcare access degradation index  
- Infrastructure stress overlays  
- Recovery prioritization heatmaps  

---

# 🌍 Data Ecosystem

| Category | Dataset | Purpose |
|----------|----------|----------|
| Hurricane Tracks | NOAA HURDAT2 | Historical storm paths |
| Atmospheric Fields | ERA5 Reanalysis | Wind / pressure / precipitation |
| Social Vulnerability | CDC SVI (US) | Socioeconomic exposure proxy |
| Population | WorldPop | Demographic density |
| Infrastructure | HDX Facilities | Schools, hospitals, shelters |

---

## 🌎 Global Adaptation

For non-US deployment, vulnerability layers can be adapted using:

- INFORM Subnational Risk Index  
- DHS socio-economic indicators  
- World Bank poverty metrics  
- HDX humanitarian datasets  

---

# 📊 Evaluation Framework

## Deterministic Accuracy

- ADE – Average Displacement Error  
- FDE – Final Displacement Error  

Evaluated at 24h / 48h / 72h horizons.

---

## Probabilistic Quality

- Calibration analysis  
- Reliability diagrams  
- Sharpness vs dispersion trade-off  
- Empirical P50 / P90 coverage validation  

---

# Latest Foundation-Model Checkpoint Audit Results

The corrected Module 1 foundation-model rerun is mirrored in:

`results/module1_foundation/`

Key regenerated artifacts:

- `results/module1_foundation/checkpoints/foundation_best.pt` — selected by validation `mean_track_err_km`
- `results/module1_foundation/metrics/foundation_eval_metrics.csv` — all evaluated epochs with one selected row
- `results/module1_foundation/metrics/foundation_split_audit.json` — leakage audit
- `results/module1_foundation/tables/table_foundation_model_training.csv` — selected-checkpoint table
- `results/module1_foundation/figures/calibration.png` — selected-checkpoint calibration figure
- `results/module1_foundation/reports/foundation_checkpoint_audit.md` — root-cause and validation report

Scientifically supported conclusion from the extended 20-epoch CPU demo:

The selected checkpoint is epoch 20, chosen by validation mean track error (`selection_score=860.1958`, improved monotonically every epoch from 1018.6 at epoch 1). The grouped storm split has zero storm-id overlap and zero group-key overlap between train and validation. All foundation tables and calibration figures now use this one selected checkpoint consistently.

Track error at epoch 20: `106.9 / 231.0 / 509.3 / 954.1 / 1357.8 / 2002.0` km at 6/12/24/48/72/120h. Versus the epoch-2 checkpoint, 6h and 24h improved; 48/72/120h got worse — longer training sharpens short-horizon accuracy on this small demo set but does not uniformly help at long horizons. P90 cone coverage improved substantially at every horizon: `0.89/0.91/0.89/0.85/0.72/0.49` at 6/12/24/48/72/120h (versus `0.83/0.70/0.30/0.11/0.07/0.04` at epoch 2), still under-covering relative to the nominal 0.90 target especially beyond 48h.

---

# Latest Baseline Track Benchmark Audit Results

The corrected LSTM/Transformer/GNO baseline rerun is mirrored in:

`results/module3_baselines/`

Key regenerated artifacts:

- `results/module3_baselines/metrics/inference_test_metrics_summary.csv` — corrected held-out test metrics
- `results/module3_baselines/metrics/inference_test_predictions_all_models.csv` — decoded lat/lon predictions
- `results/module3_baselines/metrics/baseline_input_audit.csv` — ERA5 coverage and missing-value audit
- `results/module3_baselines/metrics/baseline_split_manifest.csv` — deterministic train/val/test split manifest
- `results/module3_baselines/tables/table_case_study_track_error.csv` — regenerated manuscript table
- `results/module3_baselines/figures/track_error_vs_lead.png` — regenerated benchmark figure
- `results/module3_baselines/reports/baseline_audit.md` — root-cause and validation report

Scientifically supported conclusion from the corrected Irma/Ian ERA5-complete case study:

All learned baselines now receive identical normalized inputs and are evaluated on the same held-out test windows. The impossible thousands-of-km errors are removed, but Persistence remains the lowest mean-error method (`206.714 km` over 6/12/24/48 h). Do not claim learned-model superiority from this small case study.

The horizon-level forecast claim audit is available in:

- `tables/table_forecast_performance_audit.csv`
- `reports/forecast_performance_audit.md`
- `results/module3_baselines/tables/table_forecast_performance_audit.csv`
- `results/module3_baselines/reports/forecast_performance_audit.md`

Supported per-horizon finding: GNO+DynGNN beats Transformer at 6/12/24/48 h and
beats LSTM at 12/24/48 h, but it loses to LSTM at 6 h and loses to Persistence
at every reported case-study horizon.

---

# Latest Physics-Loss Audit Results

The corrected PI-GNO physics-loss rerun is mirrored in:

`results/module2_physics/`

Key regenerated artifacts:

- `results/module2_physics/metrics/full/` — corrected full-physics training and validation logs
- `results/module2_physics/metrics/no_physics/` — matched no-physics ablation logs
- `results/module2_physics/metrics/physics_full_vs_ablation.csv` — final comparison table
- `results/module2_physics/metrics/physics_gradient_diagnostics.csv` — graph/gradient connectivity check
- `results/module2_physics/figures/physics_residuals_full_vs_ablation.png` — regenerated residual plot
- `results/module2_physics/reports/physics_loss_audit.md` — root-cause and validation report

Scientifically supported conclusion from the corrected 20-epoch CPU demo:

The full physics model improves physical consistency versus the no-physics ablation, with final validation residual reductions of 78.0% for diffusion, 50.4% for wind-pressure balance, 30.4% for mass conservation, 17.0% for temporal continuity, 1.3% for kinetic energy, and 0.9% for advection. Predictive track RMSE is better for the no-physics ablation in this short demo (`0.019341` vs `0.022002`), so the current supported claim is physical-consistency improvement, not predictive-accuracy improvement.

**Physics constraint-weight sweep** (`metrics/physics_sweep/sweep_decision_table.csv`, see `reports/physics_loss_audit.md` addendum): swept the global physics-loss weight at scales `{0.1, 0.3, 1.0, 3.0}` plus no-physics. Track RMSE degrades monotonically above scale 0.1, but at **scale 0.1** track RMSE (`0.01894`) is actually *better* than no-physics (`0.01932`) while still keeping physics residuals well below no-physics — a genuine no-accuracy-cost consistency point on this checkpoint, distinct from the current shipped default (scale 1.0), which was tuned for maximum residual reduction rather than this tradeoff-free point.

---

# Latest Humanitarian Metrics Audit Results

The corrected Module 3 humanitarian rerun is mirrored in:

`results/module3_disaster_graph/`

Key regenerated artifacts:

- `results/module3_disaster_graph/checkpoints/disaster_gnn_best.pt` — retrained multitask humanitarian checkpoint
- `results/module3_disaster_graph/metrics/humanitarian_eval_metrics.csv` — corrected held-out humanitarian metrics
- `results/module3_disaster_graph/metrics/humanitarian_label_audit.json` — target-distribution and leakage audit
- `results/module3_disaster_graph/tables/table2_humanitarian_impact.csv` — regenerated manuscript table
- `results/module3_disaster_graph/reports/humanitarian_metrics_audit.md` — root-cause and validation report

Scientifically supported conclusion from the corrected synthetic demo:

The previous humanitarian metrics were invalid because exposed-child MAPE mixed
counts and fractions, school AUC was computed on one-class slices, hospital
targets were nearly constant, and humanitarian heads were not directly
supervised.  The corrected run uses simulator-derived proxy labels with disjoint
train/test seeds and reports finite metrics.  Current results support only a
synthetic-proxy claim: school disruption AUC is `0.8724`, hospital access MAE is
`0.0256` (a `[0,1]` index, not km), and recovery priority Spearman is not
meaningfully positive (`-0.0212`).

**E7 metric reframe:** exposed-child MAPE stayed pathologically high
(`469.3962%`) even after the scenario-level fix above, because it is still
dominated by near-zero true-count denominators. The headline exposed-child
error metric is now **sMAPE** (bounded `[0,200]%`, symmetric denominator),
reported alongside a raw MAE-in-counts figure; MAPE is kept only as a
reference row. Reported plainly, not cherry-picked: on sMAPE and MAE-in-counts,
STORM-CARE-M3 does **not** beat RF/XGBoost (sMAPE `118.66%` vs RF `97.70%` /
XGB `97.28%`; MAE `298.5` vs RF `267.6` / XGB `263.5` children) — the reverse
of the old MAPE ranking, which favored STORM-CARE-M3 only because MAPE
penalizes baseline under-prediction more harshly near zero. This is an open
problem, not a fixed result.

---

# Latest Counterfactual World-Model Audit Results

The corrected Module 5 counterfactual rerun is mirrored in:

`results/module5_counterfactual/`

Key regenerated artifacts:

- `results/module5_counterfactual/metrics/counterfactual_outcomes.csv` — baseline plus nine intervention outcomes
- `results/module5_counterfactual/metrics/counterfactual_mirror_diagnostics.csv` — verifies outcomes do not mirror intervention inputs
- `results/module5_counterfactual/tables/table_counterfactual_outcomes.csv` — regenerated table
- `results/module5_counterfactual/reports/counterfactual_world_model_audit.md` — root-cause and validation report

Scientifically supported conclusion from the corrected 24-sequence held-out demo:

Counterfactual outcomes are now generated by the learned RSSM world model:

`intervened warm-up state -> posterior latent -> learned prior rollout -> decoder`

They are not injected directly into decoded trajectories. Earlier evacuation lowers peak exposure (`0.2915 -> 0.2831`), while delayed evacuation raises peak exposure (`0.2915 -> 0.2969`). A 3-point evacuation-lead dose-response sweep (12h/24h/36h: `0.2831 / 0.2787 / 0.2754`) confirms monotonicity in evacuation lead time (36h < 24h < 12h < baseline < delayed), now with bootstrap uncertainty: **P(fully monotone) = 1.000** over 10,000 resamples of the 24 held-out sequences, and every adjacent-step one-sided Wilcoxon test is significant (p < 2e-5). Other infrastructure/resource interventions have weak or counterintuitive effects under the current short demo checkpoint; raising Monte Carlo samples 12x (5 -> 60) does not change their sign or magnitude, confirming this is a property of the undertrained demo `WorldModel` checkpoint rather than sampling noise or a pipeline bug. These should be treated as model limitations rather than overclaimed planning recommendations.

---

# Latest Ablation Study Audit Results

The regenerated ablation study is mirrored in:

`results/module6_ablations/`

Key regenerated artifacts:

- `results/module6_ablations/tables/table3_ablations.csv` — regenerated ablation table with no blank cells
- `results/module6_ablations/metrics/foundation_ablation_metrics.csv` — selected foundation checkpoint versus random-init no-SSL evaluation
- `results/module6_ablations/metrics/graph_ablation_metrics.csv` — full graph, static graph, and no-transport graph reruns on identical train/test seeds
- `results/module6_ablations/metrics/no_physics_runtime.json` — fresh no-physics runtime measurement
- `results/module6_ablations/metrics/no_world_model_runtime.json` — frozen-latent counterfactual runtime measurement
- `results/module6_ablations/metrics/table3_ablations_sources.json` — source/provenance manifest
- `results/module6_ablations/reports/ablation_study_audit.md` — root-cause, fix, and validity report

Scientifically supported conclusion from the regenerated ablation audit:

Table 3 is now generated by `scripts/run_ablations.py`, not manually filled.
The no-SSL row is evaluated on the same foundation validation windows as the
selected pretrained checkpoint. The graph-edge ablations are retrained with
the corrected multitask humanitarian loss on train seed `123` and evaluated on
the same held-out test seed `999`. Module-specific ablations that do not
legitimately produce a requested metric are marked
`not_applicable_to_changed_component` instead of being left blank or assigned a
fabricated number.

Current graph ablation results are mixed: removing transport edges slightly
improves exposed-child peak MAPE (`526.9260%` versus `533.6304%`) but worsens
recovery-priority ranking correlation (`0.1375` versus `0.2750`). Removing
propagation edges worsens exposure MAPE (`593.6911%`) but improves ranking
correlation (`0.3780`). These results support a nuanced component-sensitivity
claim, not blanket superiority of every full-model component on every metric.

---

# Final Submission Deliverables

The final generated submission package is documented in:

- `reports/validation_report.md` — final validation report with source hashes
- `reports/final_deliverables_manifest.md` — deliverables checklist
- `reports/reproducibility_report.md` — exact regeneration commands
- `reports/experiment_log.md` — source artifact hashes
- `reports/change_log.md` — generated change log
- `manuscript/generated_manuscript.md` — synchronized manuscript draft
- `manuscript/generated_supplement.md` — synchronized supplementary material
- `reports/calibration_consistency_audit.md` — selected-checkpoint calibration audit
- `reports/dataset_integrity_report.md` — split/sample/window-count audit
- `reports/case_study_ian_audit.md` — regenerated Hurricane Ian figure audit

All numeric manuscript claims in the generated manuscript are read from
regenerated CSV/JSON artifacts by `scripts/sync_manuscript.py`.

---

# 🖥 Interactive Demonstration

### 🌐 Streamlit Dashboard

https://stormcare-i9kz6hkvbpsydseiywfpqm.streamlit.app/

---

# 🎥 Visual Demonstrations  
*(Thumbnail links – GitHub friendly, no visible URLs)*

---

### 🌪 Hurricane Irma (2017) – 3D Track Visualization
[![Hurricane Irma 3D](https://img.youtube.com/vi/ZvJ8jOmbHDE/hqdefault.jpg)](https://www.youtube.com/watch?v=ZvJ8jOmbHDE)

---

### 📈 Ensemble Spread & Uncertainty Modeling
[![Uncertainty Spread](https://img.youtube.com/vi/nTIp0jjtJEk/hqdefault.jpg)](https://www.youtube.com/watch?v=nTIp0jjtJEk)

---

### 🔥 Recovery Rays & Impact Heatmaps
[![Recovery Visualization](https://img.youtube.com/vi/TCNdMnLFamw/hqdefault.jpg)](https://www.youtube.com/watch?v=TCNdMnLFamw)

---

# 🚀 Getting Started

```bash
git clone <anonymous-submission-repository-url>
cd STORM_CARE

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
🗺 Visualization Capabilities
Interactive hazard heatmaps

Uncertainty cones (P50 centerline + P90 spread)

Child exposure overlays

Facility vulnerability markers

Recovery prioritization signals

🔐 Governance & Responsible Use
STORM-CARE:

Is a research prototype

Is not a real-time operational forecasting authority

Supports human decision-making

Requires responsible interpretation

Requires ethical handling of vulnerability data

🔬 Research Positioning
This project demonstrates:

Advanced spatio-temporal modeling

Graph neural networks in disaster forecasting

Probabilistic impact translation

Responsible AI for humanitarian systems

Human-centered uncertainty visualization

Suitable for:

AI + Climate research tracks

Humanitarian AI grants

Responsible AI initiatives

Computational sustainability conferences

🛣 Future Directions
Storm surge modeling integration

SAR-based flood segmentation (U-Net)

Real-time ensemble ingestion

Multi-hazard extension (flood + cyclone + heat)

Cross-country vulnerability harmonization
