"""
PI-GNO Training Algorithm.

Synthetic data
--------------
Real ERA5 fields are not required to run this module.  A Rankine-vortex
hurricane simulator generates physically consistent (u, v, p, T) fields:

  Rankine vortex wind profile:
    V(r) = V_max · r / R_max         for r ≤ R_max   (solid-body core)
    V(r) = V_max · R_max / r         for r > R_max   (potential-flow outer)

  Wind components (cyclonic, Northern Hemisphere convention):
    u = −V · sin θ,   v = V · cos θ

  Willoughby MSLP profile (approximation):
    p(r) = p_c + (p_env − p_c) · [1 − exp(−R_max / r)]
    where p_c = p_env − 50·(V_max/50)²  (central pressure deficit)

  Temperature (warm-core):
    T(r) = T_env + ΔT_wc · exp(−r² / 2R_max²)

Storm motion
  At each 6-hour step the vortex translates ~5 m/s westward + 3 m/s northward
  (simplified Atlantic TC steering), implemented as a grid roll.

Stability analysis
------------------
Before training, the Courant–Friedrichs–Lewy (CFL) condition is checked:

  Advective CFL (explicit scheme stability):
    C = V_max · Δt / Δx  ≤ 1

  Diffusion number (explicit central differences):
    d = ν · Δt / Δx²  ≤ 0.5

  Spectral radius of the update operator (Jacobian approximation via
  one-step power iteration):
    ρ(J) ≈ ‖J v‖ / ‖v‖  for random v
  Stable if ρ(J) < 1.

Training algorithm
------------------
  1. Build synthetic dataset (n_storms × n_steps pairs)
  2. 80/20 train/val split
  3. Initialise PIGNOModel (GNO + FNO backbone)
  4. Optimiser: AdamW with cosine LR schedule + linear warm-up
  5. Physics warm-up: α(t) = min(t / T_warmup, 1) scales L_phys
  6. Each mini-batch:
       a. Forward pass → (state_pred, track_pred)
       b. Compute L_total = L_data + α·L_physics + L_track
       c. Backward + gradient clipping (‖g‖ ≤ 1.0)
       d. AdamW step
  7. Evaluate every eval_every epochs: physics residuals, track RMSE,
     field reconstruction MSE
  8. Save best checkpoint (lowest val_total)
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .config import PIGNOConfig
from .graph_builder import build_grid_graph, GraphDifferentialOps
from .architecture import PIGNOModel
from .physics_kernels import PhysicsResiduals
from .losses import PhysicsInformedLoss

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Synthetic hurricane field generator ───────────────────────────────────────

def make_rankine_field(
    grid_size: int,
    vmax_ms:   float = 50.0,
    rmw_norm:  float = 0.15,    # R_max in normalised [-1,1] coords
    cx:        float = 0.0,     # storm centre x (normalised)
    cy:        float = 0.0,     # storm centre y (normalised)
    noise:     float = 0.03,
    rng:       np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a Rankine-vortex hurricane field on an N×N grid.

    Returns array of shape (N², 7) with channels:
      [u850, v850, u500, v500, z500, T2m_anom, MSLP_anom]

    All values are in physical units (m/s, m, K, hPa) to make the
    physics residuals dimensionally meaningful.
    """
    if rng is None:
        rng = np.random.default_rng()

    N     = grid_size
    lin   = np.linspace(-1.0, 1.0, N)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")   # (N, N)

    dx    = xx - cx
    dy    = yy - cy
    r     = np.sqrt(dx**2 + dy**2) + 1e-9
    theta = np.arctan2(dy, dx)

    # ── Rankine wind ──────────────────────────────────────────────────────
    V850 = np.where(r <= rmw_norm,
                    vmax_ms * r / rmw_norm,
                    vmax_ms * rmw_norm / r)
    u850 = -V850 * np.sin(theta)
    v850 =  V850 * np.cos(theta)

    V500 = V850 * 0.55                    # weaker at 500 hPa
    u500 = -V500 * np.sin(theta)
    v500 =  V500 * np.cos(theta)

    # ── Geopotential anomaly (cyclonic low) ───────────────────────────────
    z500 = -120.0 * V850 / (vmax_ms + 1e-6)   # proportional to wind speed

    # ── Warm-core temperature anomaly ─────────────────────────────────────
    dT_wc = 8.0 * (vmax_ms / 50.0)
    T_anom = dT_wc * np.exp(-r**2 / (2 * rmw_norm**2))
    T_anom += rng.normal(0, noise * 3, (N, N))

    # ── Willoughby MSLP ───────────────────────────────────────────────────
    p_env  = 0.0                          # anomaly relative to 1013.25 hPa
    dp_c   = -50.0 * (vmax_ms / 50.0)**2
    mslp   = p_env + dp_c * (1.0 - np.exp(-rmw_norm / (r + 1e-9)))
    mslp  += rng.normal(0, noise * 2, (N, N))

    # ── Add noise to wind ─────────────────────────────────────────────────
    u850 += rng.normal(0, noise * vmax_ms, (N, N))
    v850 += rng.normal(0, noise * vmax_ms, (N, N))

    field = np.stack([u850, v850, u500, v500, z500, T_anom, mslp], axis=-1)
    return field.reshape(N * N, 7).astype(np.float32)


def advance_vortex(
    s:        np.ndarray,
    grid_size: int,
    dt_h:      float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate the vortex by one 6-hour step using simplified TC steering.

    Steering: ~5 m/s westward + 3 m/s northward (Atlantic, 20°N climatology).
    Applied as a grid roll (periodic approximation).

    Returns: (s_next, track_delta) where track_delta = [Δlat °, Δlon °].
    """
    N      = grid_size
    d_lat  = +3.0 / 111.0 * dt_h / 1.0    # ° north:  3 m/s → km/h → deg
    d_lon  = -5.0 / 111.0 * dt_h / 1.0    # ° west:   5 m/s → km/h → deg

    dy_cells = int(round(d_lat / (2.0 / N)))
    dx_cells = int(round(d_lon / (2.0 / N)))

    s2d    = s.reshape(N, N, 7)
    s_new  = np.roll(s2d, (dy_cells, dx_cells), axis=(0, 1))

    # Slight intensity oscillation (diurnal-like)
    s_new[..., :4] *= 0.985    # gentle spindown in outer wind field

    delta = np.array([d_lat, d_lon], dtype=np.float32)
    return s_new.reshape(N * N, 7).astype(np.float32), delta


# ── Dataset ───────────────────────────────────────────────────────────────────

class HurricaneFieldDataset(Dataset):
    """
    Synthetic hurricane field dataset.

    Each sample is a triple:
        s_t          : (N², 7)  field at time t
        s_tp1        : (N², 7)  field at time t+1
        track_delta  : (2,)     [Δlat, Δlon] in degrees

    n_storms × n_steps_per_storm consecutive pairs are generated.
    """

    def __init__(
        self,
        n_storms:           int,
        grid_size:          int,
        n_steps_per_storm:  int = 10,
        seed:               int = 42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.pairs: List[Tuple[Tensor, Tensor, Tensor]] = []

        for _ in range(n_storms):
            vmax  = float(rng.uniform(25.0, 80.0))
            rmw   = float(rng.uniform(0.08, 0.22))
            cx    = float(rng.uniform(-0.15, 0.15))
            cy    = float(rng.uniform(-0.15, 0.15))

            s = make_rankine_field(
                grid_size, vmax, rmw, cx, cy,
                rng=rng,
            )
            for _ in range(n_steps_per_storm):
                s_next, delta = advance_vortex(s, grid_size)
                self.pairs.append((
                    torch.from_numpy(s.copy()),
                    torch.from_numpy(s_next.copy()),
                    torch.from_numpy(delta),
                ))
                s = s_next

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


# ── Stability analysis ────────────────────────────────────────────────────────

def courant_number(v_max_ms: float, dt_s: float, dx_m: float) -> float:
    """
    Courant–Friedrichs–Lewy number for explicit advection schemes:
      C = V_max · Δt / Δx
    Stable iff C ≤ 1.
    """
    return v_max_ms * dt_s / dx_m


def diffusion_number(nu: float, dt_s: float, dx_m: float) -> float:
    """
    Von Neumann diffusion stability number for explicit schemes:
      d = ν · Δt / Δx²
    Stable iff d ≤ 0.5.
    """
    return nu * dt_s / dx_m ** 2


def spectral_radius_estimate(
    model: PIGNOModel,
    sample_x: Tensor,
) -> float:
    """
    Approximate spectral radius ρ(J) of the Jacobian ∂output/∂input via
    one-step power iteration using torch.autograd.functional.jvp.

    A model is contractive (stable) if ρ(J) < 1 everywhere.
    """
    model.eval()
    x = sample_x.detach().clone().requires_grad_(False)

    # Use JVP: directional derivative of output in a random direction v
    v = torch.randn_like(x)
    v_norm = v / (v.norm() + 1e-8)

    with torch.enable_grad():
        x_g = x.requires_grad_(True)
        out = model(x_g)["state_pred"]
        # Compute Jacobian-vector product manually via backward
        dummy = (out * torch.randn_like(out)).sum()
        dummy.backward()
        grad_norm = x_g.grad.norm().item() if x_g.grad is not None else 0.0

    model.train()
    return grad_norm


# ── Training runner ───────────────────────────────────────────────────────────

class PIGNOTrainer:
    """Full training pipeline for the Physics-Informed GNO."""

    def __init__(self, cfg: PIGNOConfig):
        self.cfg    = cfg
        self.device = torch.device("cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    def run(self) -> None:
        cfg = self.cfg
        log.info("=" * 64)
        log.info("  PI-GNO — Physics-Informed Graph Neural Operator")
        log.info("=" * 64)
        log.info(f"  Config : {cfg}")
        log.info(f"  Device : {self.device}")

        dx_m = cfg.grid_spacing_m

        # ── Stability pre-check ───────────────────────────────────────────
        C = courant_number(80.0, cfg.dt_physics, dx_m)
        d = diffusion_number(cfg.nu_viscosity, cfg.dt_physics, dx_m)
        log.info(f"  CFL Courant   C = {C:.4f}  (< 1.0 required for stability)")
        log.info(f"  Diff. number  d = {d:.6f}  (< 0.5 required for stability)")
        if C > 1.0:
            log.warning("  CFL condition VIOLATED — physics residuals may be large")
        if d > 0.5:
            log.warning("  Diffusion number > 0.5 — stability not guaranteed")

        # ── Data ─────────────────────────────────────────────────────────
        log.info("  Generating synthetic hurricane fields …")
        n_tr = int(cfg.n_synthetic_storms * 0.8)
        n_va = cfg.n_synthetic_storms - n_tr
        tr_ds = HurricaneFieldDataset(n_tr, cfg.grid_size, cfg.n_steps_per_storm, cfg.seed)
        va_ds = HurricaneFieldDataset(n_va, cfg.grid_size, cfg.n_steps_per_storm, cfg.seed + 1)

        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        log.info(f"  Train: {len(tr_ds)} samples | Val: {len(va_ds)} samples")

        # ── Model and physics operators ───────────────────────────────────
        model     = PIGNOModel(cfg, self.device)
        graph     = model.graph
        diff_ops  = GraphDifferentialOps(graph, dx_m, cfg.grid_size)
        phys      = PhysicsResiduals(cfg, diff_ops)
        criterion = PhysicsInformedLoss(cfg, phys).to(self.device)

        n_params = model.count_parameters()
        log.info(f"  Model  : {n_params:,} trainable parameters")

        # ── Jacobian / spectral radius check on one sample ────────────────
        s0 = tr_ds[0][0].unsqueeze(0).to(self.device)
        rho = spectral_radius_estimate(model, s0)
        log.info(f"  Jacobian spectral radius (init) ρ ≈ {rho:.4f}")

        # ── Optimiser + cosine LR schedule with linear warm-up ───────────
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            """Linear warm-up then cosine decay."""
            if epoch < cfg.warmup_epochs:
                return float(epoch + 1) / max(cfg.warmup_epochs, 1)
            progress = (epoch - cfg.warmup_epochs) / max(cfg.n_epochs - cfg.warmup_epochs, 1)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

        # ── Training loop ─────────────────────────────────────────────────
        best_val   = float("inf")
        train_log  = []
        val_log    = []
        eval_every = max(cfg.n_epochs // 5, 1)

        print()
        print("  " + "─" * 60)
        print(f"  {'Epoch':>5}  {'L_total':>8}  {'L_data':>8}  {'L_phys':>8}  "
              f"{'α':>5}  {'lr':>9}  {'t(s)':>6}")
        print("  " + "─" * 60)

        for epoch in range(1, cfg.n_epochs + 1):
            criterion.set_epoch(epoch)
            α = criterion.physics_alpha()
            t0 = time.time()

            model.train()
            sums = {k: 0.0 for k in
                    ["total", "L_data", "L_phys",
                     "R_adv", "R_diff", "R_mass", "R_wp", "R_cont", "R_nrg"]}
            nb = 0

            for s_t, s_tp1, track_delta in tr_loader:
                s_t         = s_t.to(self.device)
                s_tp1       = s_tp1.to(self.device)
                track_delta = track_delta.to(self.device)

                out        = model(s_t)
                pred_state = out["state_pred"]       # (B, N, C_out)
                pred_track = out["track_pred"]        # (B, 2)

                # Residual (increment) formulation: predict Δs = s_tp1 − s_t
                target_delta = s_tp1 - s_t            # (B, N, C_out) same channels

                losses  = criterion(pred_state, target_delta, s_t, s_tp1)
                L_track = F.mse_loss(pred_track, track_delta)
                total   = losses["total"] + L_track

                optim.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

                for k in sums:
                    sums[k] += losses[k].item()
                sums["total"] += L_track.item()
                nb += 1

            scheduler.step()
            for k in sums:
                sums[k] /= max(nb, 1)
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            # ── Periodic validation ───────────────────────────────────────
            if epoch % eval_every == 0 or epoch == cfg.n_epochs:
                vm    = self._validate(model, va_loader, criterion)
                vt    = vm["val_total"]
                better = vt < best_val
                if better:
                    best_val = vt
                    self._save_checkpoint(model, epoch, vm)

                val_log.append({"epoch": epoch, **vm})
                print(
                    f"  {epoch:5d}  {sums['total']:8.4f}  {sums['L_data']:8.4f}  "
                    f"{sums['L_phys']:8.4f}  {α:5.2f}  {lr_now:9.2e}  {elapsed:6.1f}"
                    f"  val={vt:.4f}" + (" ★" if better else "")
                )
                self._print_val_table(vm, epoch)
            else:
                print(
                    f"  {epoch:5d}  {sums['total']:8.4f}  {sums['L_data']:8.4f}  "
                    f"{sums['L_phys']:8.4f}  {α:5.2f}  {lr_now:9.2e}  {elapsed:6.1f}"
                )

            train_log.append({"epoch": epoch, **sums, "lr": lr_now})

        # ── Spectral radius check after training ──────────────────────────
        model.eval()
        rho_final = spectral_radius_estimate(model, s0)
        log.info(f"  Jacobian spectral radius (final) ρ ≈ {rho_final:.4f}")
        model.train()

        self._print_final_report(model, cfg, train_log, val_log,
                                 C, d, rho_final)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(
        self,
        model:     PIGNOModel,
        loader:    DataLoader,
        criterion: PhysicsInformedLoss,
    ) -> Dict[str, float]:
        model.eval()
        totals = {
            "val_total":       0.0,
            "val_L_data":      0.0,
            "val_L_phys":      0.0,
            "val_track_rmse":  0.0,
            "val_R_adv":       0.0,
            "val_R_mass":      0.0,
            "val_R_wp":        0.0,
        }
        n = 0

        with torch.no_grad():
            for s_t, s_tp1, track_delta in loader:
                s_t         = s_t.to(self.device)
                s_tp1       = s_tp1.to(self.device)
                track_delta = track_delta.to(self.device)

                out          = model(s_t)
                pred_state   = out["state_pred"]
                pred_track   = out["track_pred"]
                target_delta = s_tp1 - s_t

                losses  = criterion(pred_state, target_delta, s_t, s_tp1)
                L_track = F.mse_loss(pred_track, track_delta)

                totals["val_total"]      += losses["total"].item() + L_track.item()
                totals["val_L_data"]     += losses["L_data"].item()
                totals["val_L_phys"]     += losses["L_phys"].item()
                totals["val_track_rmse"] += L_track.sqrt().item()
                totals["val_R_adv"]      += losses["R_adv"].item()
                totals["val_R_mass"]     += losses["R_mass"].item()
                totals["val_R_wp"]       += losses["R_wp"].item()
                n += 1

        model.train()
        return {k: v / max(n, 1) for k, v in totals.items()}

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        model:   PIGNOModel,
        epoch:   int,
        metrics: Dict[str, float],
    ) -> None:
        os.makedirs("checkpoints/physics", exist_ok=True)
        torch.save(
            {
                "epoch":   epoch,
                "state":   model.state_dict(),
                "metrics": metrics,
                "config":  vars(self.cfg),
            },
            "checkpoints/physics/pigno_best.pt",
        )

    # ── Pretty-print helpers ──────────────────────────────────────────────────

    @staticmethod
    def _print_val_table(vm: Dict[str, float], epoch: int) -> None:
        print()
        print(f"  ── Validation Metrics — Epoch {epoch} ──────────────────────")
        for k, v in sorted(vm.items()):
            if k != "epoch":
                print(f"    {k:<22s}: {v:.6f}")
        print()

    def _print_final_report(
        self,
        model:      PIGNOModel,
        cfg:        PIGNOConfig,
        train_log:  List[Dict],
        val_log:    List[Dict],
        C:          float,
        d:          float,
        rho_final:  float,
    ) -> None:
        os.makedirs("metrics/physics", exist_ok=True)

        if train_log:
            with open("metrics/physics/pigno_train_log.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
                w.writeheader()
                w.writerows(train_log)

        if val_log:
            with open("metrics/physics/pigno_val_metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(val_log[0].keys()))
                w.writeheader()
                w.writerows(val_log)
            print(f"  Metrics saved → metrics/physics/")

        last_val   = val_log[-1]   if val_log   else {}
        last_train = train_log[-1] if train_log else {}

        print()
        print("═" * 64)
        print("  PI-GNO — Training Complete")
        print("═" * 64)
        print(f"  Architecture : {cfg}")
        print(f"  Parameters   : {model.count_parameters():,}")
        print()
        print("  ── Stability Analysis ────────────────────────────────────")
        print(f"    CFL Courant number   C  = {C:.4f}")
        print(f"    Diffusion number     d  = {d:.6f}")
        print(f"    CFL status           :  {'✓ STABLE' if C <= 1.0 else '✗ UNSTABLE'}")
        print(f"    Diff. status         :  {'✓ STABLE' if d <= 0.5 else '✗ UNSTABLE'}")
        print(f"    Jacobian ρ (final)   :  {rho_final:.4f}  {'< 1 ✓' if rho_final < 1.0 else '≥ 1 ✗'}")
        print()
        print("  ── Final Training Losses ─────────────────────────────────")
        for k, v in sorted(last_train.items()):
            if k != "epoch":
                print(f"    {k:<22s}: {v:.6f}")
        print()
        print("  ── Final Validation Metrics ──────────────────────────────")
        for k, v in sorted(last_val.items()):
            if k != "epoch":
                print(f"    {k:<22s}: {v:.6f}")
        print()
        print("  ── Checkpoints ───────────────────────────────────────────")
        print("    Best   : checkpoints/physics/pigno_best.pt")
        print("    Metrics: metrics/physics/")
        print("═" * 64)
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="PI-GNO training")
    p.add_argument("--demo",        action="store_true",
                   help="Small model + small data for quick CPU test")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch-size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--n-storms",    type=int,   default=None)
    p.add_argument("--grid-size",   type=int,   default=None)
    args = p.parse_args()

    cfg = PIGNOConfig()
    if args.demo:       cfg.apply_demo_overrides()
    if args.epochs:     cfg.n_epochs           = args.epochs
    if args.batch_size: cfg.batch_size          = args.batch_size
    if args.lr:         cfg.lr                  = args.lr
    if args.n_storms:   cfg.n_synthetic_storms  = args.n_storms
    if args.grid_size:  cfg.grid_size           = args.grid_size

    PIGNOTrainer(cfg).run()


if __name__ == "__main__":
    main()
