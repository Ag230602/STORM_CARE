"""
PhysicsInformedLoss — combines data-fitting MSE with all six PDE residuals.

Total loss
----------

  L_total = λ_data · L_data  +  α(t) · L_physics

  L_physics = λ_adv  · R_adv   (momentum / advection)
            + λ_diff · R_diff  (temperature diffusion)
            + λ_mass · R_mass  (mass conservation)
            + λ_wp   · R_wp    (gradient wind balance)
            + λ_cont · R_cont  (temporal continuity)
            + λ_nrg  · R_nrg   (kinetic energy)

Physics warm-up schedule
------------------------
  α(t) = min( t / T_warmup, 1.0 )

Training begins with pure data loss (α=0).  Physics constraints are
gradually switched on over the first T_warmup epochs.  This prevents
the model from getting stuck in a bad local minimum caused by large,
unbalanced physics residuals early in training.

Motivation
----------
Without warm-up, large physics residuals in epoch 1 can overwhelm the
data loss, causing the optimiser to minimise physics at the expense of
prediction accuracy.  The schedule ensures the network first learns a
rough data-driven prior and then refines it to satisfy the PDEs.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import PIGNOConfig
from .physics_kernels import PhysicsResiduals


class PhysicsInformedLoss(nn.Module):
    """
    Computes the combined physics-informed training objective.

    Usage:
        criterion = PhysicsInformedLoss(cfg, phys)
        criterion.set_epoch(epoch)
        losses = criterion(pred_delta, target_delta, s_t, s_tp1)
        losses["total"].backward()

    Args:
        cfg  : PIGNOConfig — holds all λ weights and warm-up schedule.
        phys : PhysicsResiduals — stateless PDE residual computer.
    """

    def __init__(self, cfg: PIGNOConfig, phys: PhysicsResiduals):
        super().__init__()
        self.cfg  = cfg
        self.phys = phys
        # Store epoch as a buffer so it moves with the module to the right device.
        self.register_buffer("_epoch", torch.tensor(0.0))

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch before the training loop."""
        self._epoch.fill_(float(epoch))

    def physics_alpha(self) -> float:
        """Current physics warm-up factor α ∈ [0, 1]."""
        t_warm = max(self.cfg.physics_warmup_epochs, 1)
        return min(float(self._epoch.item()) / t_warm, 1.0)

    def forward(
        self,
        pred_delta:   Tensor,  # (B, N, C_out)  model output (field increments)
        target_delta: Tensor,  # (B, N, C_out)  ground-truth increments
        s_t:          Tensor,  # (B, N, 7)       current state (all channels)
        s_tp1:        Tensor   = None,  # kept for API compat; no longer used
    ) -> Dict[str, Tensor]:
        """
        Returns a dict of scalar tensors:
            total, L_data, L_phys, R_adv, R_diff, R_mass, R_wp, R_cont, R_nrg, alpha
        """
        cfg = self.cfg
        α   = self.physics_alpha()

        # ── Data loss: MSE between predicted and true field increments ────────
        L_data = F.mse_loss(pred_delta, target_delta)

        # ── Physics residuals on the MODEL'S predicted next state ─────────────
        # BUG that was here: all_residuals(s_t, s_tp1) used ground-truth s_tp1,
        # so residuals were constant across epochs regardless of model updates.
        # Fix: evaluate residuals on (s_t, s_pred) where s_pred = s_t + pred_delta.
        s_pred = s_t + pred_delta    # model's predicted next state  (B, N, 7)
        res    = self.phys.all_residuals(s_t, s_pred)
        R_adv  = res["adv"]
        R_diff = res["diff"]
        R_mass = res["mass"]
        R_wp   = res["wp"]
        R_cont = res["cont"]
        R_nrg  = res["nrg"]

        # ── Weighted physics loss ─────────────────────────────────────────────
        L_phys = (
            cfg.lambda_adv    * R_adv
          + cfg.lambda_diff   * R_diff
          + cfg.lambda_mass   * R_mass
          + cfg.lambda_wp     * R_wp
          + cfg.lambda_cont   * R_cont
          + cfg.lambda_energy * R_nrg
        )

        # ── Total ─────────────────────────────────────────────────────────────
        L_total = cfg.lambda_data * L_data + α * L_phys

        return dict(
            total  = L_total,
            L_data = L_data,
            L_phys = L_phys,
            R_adv  = R_adv,
            R_diff = R_diff,
            R_mass = R_mass,
            R_wp   = R_wp,
            R_cont = R_cont,
            R_nrg  = R_nrg,
            alpha  = torch.tensor(α, device=s_t.device),
        )
