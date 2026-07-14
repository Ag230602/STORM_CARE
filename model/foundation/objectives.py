"""
objectives.py — Self-supervised pretraining objectives for STORM-CARE Foundation Model.

Four tasks
----------

Task 1 · Future-State Prediction
   Given the encoded representation of a storm window, predict the
   normalised feature vector at the *next* timestep.
   Loss: Gaussian NLL  →  L_future = -log N(x_{t+1} | μ_t, σ_t)

Task 2 · Masked Graph Reconstruction  (MAE-style)
   Randomly mask mask_ratio of storm observation tokens.  The model must
   reconstruct the original (unmasked) feature values from context.
   Loss: MSE over masked positions  →  L_mask = ‖x - x̂‖² / n_masked

Task 3 · Contrastive Storm Evolution Learning  (SimCLR-style)
   Two independently masked views of the same storm window are encoded.
   The InfoNCE loss pulls their CLS embeddings together while pushing
   away representations from other storms in the batch.
   Loss: InfoNCE  →  L_contrast = -log(exp(z₁·z₂/τ) / Σ exp(z₁·z_k/τ))

Task 4 · Multi-Horizon Forecasting
   From the current state, predict track displacement at
   6 / 12 / 24 / 48 / 72 / 120 h.
   Loss: Gaussian NLL summed over all lead times  →  L_horizon = Σ_k NLL_k

Combined loss
-------------
   L_total = λ_f · L_future + λ_m · L_mask
           + λ_c · L_contrast + λ_h · L_horizon
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FoundationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Gaussian NLL
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood.

    Parameters
    ----------
    mu, sigma : same shape as target
    target    : ground-truth values
    reduce    : if True, return scalar mean; else return per-element loss
    """
    var = sigma ** 2 + 1e-8
    nll = 0.5 * ((target - mu) ** 2 / var + torch.log(var))
    return nll.mean() if reduce else nll


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Future-State Prediction
# ─────────────────────────────────────────────────────────────────────────────

class FutureStateLoss(nn.Module):
    """
    Predict the storm observation features at t+1 from the encoding at t.

    The model's future_mu / future_sigma outputs have shape (B, T, F).
    We use positions 0..T-2 to predict positions 1..T-1 (one-step-ahead).

    The loss is averaged over valid (non-masked) positions.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        future_mu: torch.Tensor,       # (B, T, F)
        future_sigma: torch.Tensor,    # (B, T, F)
        storm_feats: torch.Tensor,     # (B, T, F)  — normalised ground truth
        valid_mask: Optional[torch.Tensor] = None,  # (B, T) True = valid
    ) -> torch.Tensor:
        # Align: predict[0..T-2] → target[1..T-1]
        pred_mu    = future_mu[:, :-1]     # (B, T-1, F)
        pred_sigma = future_sigma[:, :-1]  # (B, T-1, F)
        target     = storm_feats[:, 1:]    # (B, T-1, F)

        nll = _gaussian_nll(pred_mu, pred_sigma, target, reduce=False)  # (B, T-1, F)

        if valid_mask is not None:
            # Only compute loss on non-masked source positions
            vm = valid_mask[:, :-1].unsqueeze(-1).float()  # (B, T-1, 1)
            nll = nll * vm
            denom = vm.sum().clamp(min=1)
            return nll.sum() / denom

        return nll.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Masked Graph Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

class MaskedReconstructionLoss(nn.Module):
    """
    Mean-squared-error reconstruction over the masked token positions.

    recon_pred  : (B, T, F) — model's reconstructed features
    storm_feats : (B, T, F) — original (unmasked) normalised features
    mask        : (B, T)    — True where token was masked
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        recon_pred: torch.Tensor,
        storm_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # mask: (B, T) → (B, T, 1)
        m = mask.unsqueeze(-1).float()
        diff_sq = (recon_pred - storm_feats) ** 2  # (B, T, F)
        masked  = diff_sq * m
        n_masked = m.sum().clamp(min=1)
        return masked.sum() / (n_masked * storm_feats.shape[-1])


def sample_mask(
    B: int,
    T: int,
    mask_ratio: float,
    device: torch.device,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample a random Boolean mask of shape (B, T).
    Each row independently masks approximately mask_ratio tokens.
    Always keeps at least one token unmasked.
    """
    rand = torch.rand(B, T, device=device, generator=rng)
    mask = rand < mask_ratio

    # Guarantee at least one visible token per sequence
    all_masked = mask.all(dim=1)  # (B,)
    if all_masked.any():
        # Randomly uncover one position for those sequences
        rand_idx = torch.randint(0, T, (all_masked.sum(),), device=device, generator=rng)
        rows = all_masked.nonzero(as_tuple=True)[0]
        mask[rows, rand_idx] = False

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Contrastive Storm Evolution Learning (SimCLR / InfoNCE)
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveEvolutionLoss(nn.Module):
    """
    InfoNCE contrastive loss over two augmented views of each storm window.

    Positive pair  : (z1_i, z2_i)  — two independently masked views of storm i
    Negative pairs : all cross-storm pairs within the batch

    Loss (NT-Xent):
        L = -1/(2B) Σᵢ [ log (pos_sim_i / Σ_j neg_sim_j) ]

    where sim(u,v) = exp(u·v / τ) and u,v are L2-normalised.

    Inputs
    ------
    z1 : (B, contrastive_dim)  view-1 projection (already L2-normalised)
    z2 : (B, contrastive_dim)  view-2 projection (already L2-normalised)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        B = z1.size(0)
        z  = torch.cat([z1, z2], dim=0)          # (2B, d)

        # Similarity matrix  (2B, 2B)
        sim = torch.mm(z, z.T) / self.tau

        # Remove self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, float("-inf"))

        # Labels: for row i (0..B-1), the positive is at i+B
        #         for row i (B..2B-1), the positive is at i-B
        labels = torch.cat(
            [torch.arange(B, 2 * B), torch.arange(B)], dim=0
        ).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Multi-Horizon Forecasting
# ─────────────────────────────────────────────────────────────────────────────

class MultiHorizonLoss(nn.Module):
    """
    NLL loss for probabilistic multi-horizon storm track prediction.

    horizon_mu, horizon_sigma : (B, n_leads, 2) — Δlat/Δlon predictions
    horizon_targets           : (B, n_leads, 2) — actual future Δlat/Δlon

    The loss is averaged over leads and batch.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        horizon_mu: torch.Tensor,
        horizon_sigma: torch.Tensor,
        horizon_targets: torch.Tensor,
        valid: Optional[torch.Tensor] = None,  # (B, n_leads) bool
    ) -> torch.Tensor:
        nll = _gaussian_nll(
            horizon_mu, horizon_sigma, horizon_targets, reduce=False
        )  # (B, n_leads, 2)

        if valid is not None:
            v = valid.unsqueeze(-1).float()  # (B, n_leads, 1)
            nll = nll * v
            return nll.sum() / (v.sum() * 2).clamp(min=1)

        return nll.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combined objective
# ─────────────────────────────────────────────────────────────────────────────

class CombinedPretrainingObjective(nn.Module):
    """
    Weighted combination of the four self-supervised pretraining losses.

        L_total = λ_f · L_future
                + λ_m · L_mask
                + λ_c · L_contrast
                + λ_h · L_horizon

    Returns a dict of individual and combined losses for logging.
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        self.future_loss    = FutureStateLoss()
        self.mask_loss      = MaskedReconstructionLoss()
        self.contrast_loss  = ContrastiveEvolutionLoss(cfg.temperature)
        self.horizon_loss   = MultiHorizonLoss()

        self.λ_future   = cfg.lambda_future
        self.λ_mask     = cfg.lambda_mask
        self.λ_contrast = cfg.lambda_contrastive
        self.λ_horizon  = cfg.lambda_horizon

    def forward(
        self,
        # Model outputs for view-1 (with mask applied)
        out1: Dict,
        # Model outputs for view-2 (different random mask — for contrastive)
        out2: Dict,
        # Ground-truth inputs
        storm_feats: torch.Tensor,      # (B, T, F)
        mask: torch.Tensor,             # (B, T)  — mask used for view-1
        horizon_targets: torch.Tensor,  # (B, n_leads, 2)  Δlat/Δlon
        horizon_valid: Optional[torch.Tensor] = None,  # (B, n_leads)
    ) -> Dict[str, torch.Tensor]:

        # Task 1: future-state prediction (from view-1 node embeddings)
        l_future = self.future_loss(
            out1["future_mu"], out1["future_sigma"], storm_feats
        )

        # Task 2: masked graph reconstruction
        l_mask = self.mask_loss(out1["recon_pred"], storm_feats, mask)

        # Task 3: contrastive (view-1 CLS vs view-2 CLS)
        l_contrast = self.contrast_loss(out1["contrast_z"], out2["contrast_z"])

        # Task 4: multi-horizon forecasting
        l_horizon = self.horizon_loss(
            out1["horizon_mu"], out1["horizon_sigma"],
            horizon_targets, horizon_valid,
        )

        l_total = (
            self.λ_future  * l_future
            + self.λ_mask    * l_mask
            + self.λ_contrast * l_contrast
            + self.λ_horizon * l_horizon
        )

        return {
            "loss":       l_total,
            "L_future":   l_future.detach(),
            "L_mask":     l_mask.detach(),
            "L_contrast": l_contrast.detach(),
            "L_horizon":  l_horizon.detach(),
        }
