"""Diagnose whether each PI-GNO physics residual is connected to optimisation."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.physics.architecture import PIGNOModel
from model.physics.config import PIGNOConfig
from model.physics.graph_builder import GraphDifferentialOps
from model.physics.losses import PhysicsInformedLoss
from model.physics.physics_kernels import PhysicsResiduals
from model.physics.train import HurricaneFieldDataset


def _grad_norm(grads) -> float:
    vals = [g.detach().norm() for g in grads if g is not None]
    if not vals:
        return 0.0
    return float(torch.stack(vals).norm().item())


def main() -> None:
    cfg = PIGNOConfig().apply_demo_overrides()
    cfg.n_synthetic_storms = 4
    cfg.n_steps_per_storm = 2
    device = torch.device("cpu")

    ds = HurricaneFieldDataset(
        n_storms=cfg.n_synthetic_storms,
        grid_size=cfg.grid_size,
        n_steps_per_storm=cfg.n_steps_per_storm,
        domain_radius_deg=cfg.domain_radius_deg,
        seed=cfg.seed,
    )
    s_t, s_tp1, _ = ds[0]
    s_t = s_t.unsqueeze(0).to(device)
    s_tp1 = s_tp1.unsqueeze(0).to(device)

    model = PIGNOModel(cfg, device)
    graph = model.graph
    ops = GraphDifferentialOps(graph, cfg.grid_spacing_m, cfg.grid_size)
    phys = PhysicsResiduals(cfg, ops)
    criterion = PhysicsInformedLoss(cfg, phys).to(device)
    criterion.set_epoch(cfg.physics_warmup_epochs)

    out = model(s_t)
    pred_delta = out["state_pred"]
    target_delta = s_tp1 - s_t
    losses = criterion(pred_delta, target_delta, s_t, s_tp1)

    rows = []
    params = [p for p in model.parameters() if p.requires_grad]
    for key in ["R_adv", "R_diff", "R_mass", "R_wp", "R_cont", "R_nrg", "L_phys", "total"]:
        scalar = losses[key]
        g_pred = torch.autograd.grad(
            scalar,
            pred_delta,
            retain_graph=True,
            allow_unused=True,
        )[0]
        g_params = torch.autograd.grad(
            scalar,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        rows.append({
            "term": key,
            "value": float(scalar.detach().cpu()),
            "grad_norm_pred_delta": 0.0 if g_pred is None else float(g_pred.detach().norm().cpu()),
            "grad_norm_params": _grad_norm(g_params),
            "connected_to_prediction": bool(g_pred is not None and torch.isfinite(g_pred).all() and g_pred.norm() > 0),
            "connected_to_parameters": bool(_grad_norm(g_params) > 0),
        })

    out_dir = Path("metrics/physics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "physics_gradient_diagnostics.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
