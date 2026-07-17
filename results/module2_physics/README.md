# Module 2 Physics Results

This folder contains the corrected PI-GNO physics-loss rerun artifacts.

## Main Files

- `metrics/full/pigno_train_log.csv` — corrected full-physics training log
- `metrics/full/pigno_val_metrics.csv` — corrected full-physics validation metrics
- `metrics/no_physics/` — matched no-physics ablation logs
- `metrics/physics_full_vs_ablation.csv` — final full-vs-ablation comparison
- `metrics/physics_gradient_diagnostics.csv` — gradient connectivity diagnostics
- `figures/physics_residuals_full_vs_ablation.png` — regenerated residual curves
- `reports/physics_loss_audit.md` — root-cause analysis and validation report
- `checkpoints/full/pigno_best.pt` — corrected full-physics checkpoint
- `checkpoints/no_physics/pigno_best.pt` — no-physics ablation checkpoint

## Supported Result

The corrected full-physics model improves physical consistency versus the
no-physics ablation in the 20-epoch CPU demo:

| Residual | Reduction vs No-Physics |
| --- | ---: |
| R_diff | 78.0% |
| R_wp | 50.4% |
| R_mass | 30.4% |
| R_cont | 17.0% |
| R_nrg | 1.3% |
| R_adv | 0.9% |

Predictive track RMSE is lower for the no-physics ablation in this short demo
(`0.019341` vs `0.022002`), so the supported claim is improved physical
consistency, not improved predictive accuracy.

## Reproduction Commands

```bash
.venv/bin/python scripts/diagnose_physics_gradients.py
.venv/bin/python -m model.physics.train --demo --metrics-dir metrics/physics/full --checkpoint-dir checkpoints/physics/full
.venv/bin/python -m model.physics.train --demo --no-physics --metrics-dir metrics/physics/no_physics --checkpoint-dir checkpoints/physics/no_physics
.venv/bin/python scripts/plot_physics_residuals.py
```
