"""
RSSM WorldModel — Recurrent State Space Model for disaster state forecasting.

Architecture (Module 4)
-----------------------
The model learns a compact latent space z_t that encodes the full disaster
system state.  It uses a Recurrent State Space Model (RSSM) structure:

  Deterministic path:
    h_t = GRU( h_{t-1}, z_{t-1} )         — deterministic carry

  Posterior (during training, has access to true x_t):
    μ_post, σ_post = Encoder( h_t, x_t )
    z_t ~ Normal( μ_post, σ_post )         — posterior sample

  Prior (during rollout, no access to x_t):
    μ_prior, σ_prior = Prior( h_t )
    z_t ~ Normal( μ_prior, σ_prior )       — prior sample

  Decoder:
    x̂_t = Decoder( h_t, z_t )             — reconstructed disaster state

  Four semantic heads decode sub-components of x̂_t:
    x̂_hazard    — storm track and intensity evolution
    x̂_infra     — infrastructure damage and capacity
    x̂_exposure  — population exposure and evacuation status
    x̂_resource  — supply-demand balance at shelters / hospitals

Training objective:
  L = L_recon + β_kl · L_KL + β_pred · L_pred

  L_recon = MSE( x̂_t, x_t )              — reconstruction
  L_KL    = KL( posterior ‖ prior )        — regularisation
  L_pred  = MSE( x̂_{t+k}, x_{t+k} )      — multi-step prediction (teacher-forced)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from .config import WorldModelConfig


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ELU(),
        nn.Linear(hidden, out_dim),
    )


class RSSM(nn.Module):
    """
    Recurrent State Space Model core.

    All methods operate on unbatched single time steps so the caller can
    loop over sequences explicitly (easier to follow and debug).
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        d_s = cfg.d_disaster_state
        d_h = cfg.d_hidden
        d_z = cfg.d_latent
        d_e = cfg.d_enc_hidden

        # Deterministic carry: GRU takes (h_{t-1}, z_{t-1})
        self.gru = nn.GRUCell(d_z, d_h)

        # Posterior encoder: p(z_t | h_t, x_t)
        self.post_net = _mlp(d_h + d_s, d_e, 2 * d_z)   # → (μ, log σ)

        # Prior network: p(z_t | h_t)
        self.prior_net = _mlp(d_h, d_e, 2 * d_z)         # → (μ, log σ)

        # Decoder: (h_t, z_t) → x̂_t
        self.decoder = _mlp(d_h + d_z, cfg.d_dec_hidden, d_s)

    def _split_gaussian(self, raw: Tensor) -> Tuple[Tensor, Tensor]:
        """Split (2·d_z,) into (μ, σ) with σ = softplus(raw_σ)."""
        mu, log_s = raw.chunk(2, dim=-1)
        return mu, F.softplus(log_s) + 1e-4

    def initial_state(self, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Return (h_0, z_0) = zeros."""
        h = torch.zeros(self.cfg.d_hidden, device=device)
        z = torch.zeros(self.cfg.d_latent, device=device)
        return h, z

    def step_posterior(
        self,
        x_t: Tensor,        # (d_disaster_state,)  observed state at t
        h_prev: Tensor,     # (d_hidden,)
        z_prev: Tensor,     # (d_latent,)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        One RSSM step using the posterior (training mode).

        Returns:
            h_t      : new deterministic state
            z_t      : sampled posterior latent
            mu_post  : posterior mean
            sig_post : posterior std
            mu_prior : prior mean
            sig_prior: prior std
        """
        # Deterministic update
        h_t = self.gru(z_prev.unsqueeze(0), h_prev.unsqueeze(0)).squeeze(0)

        # Prior
        mu_prior, sig_prior = self._split_gaussian(self.prior_net(h_t))

        # Posterior
        mu_post, sig_post = self._split_gaussian(
            self.post_net(torch.cat([h_t, x_t], dim=-1))
        )

        # Reparameterisation sample
        eps = torch.randn_like(mu_post)
        z_t = mu_post + sig_post * eps

        return h_t, z_t, mu_post, sig_post, mu_prior, sig_prior

    def step_prior(
        self,
        h_prev: Tensor,
        z_prev: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        One RSSM step using the prior (rollout / inference mode).

        Returns (h_t, z_t).
        """
        h_t = self.gru(z_prev.unsqueeze(0), h_prev.unsqueeze(0)).squeeze(0)
        mu, sig = self._split_gaussian(self.prior_net(h_t))
        z_t = mu + sig * torch.randn_like(mu)
        return h_t, z_t

    def decode(self, h_t: Tensor, z_t: Tensor) -> Tensor:
        """Decode (h_t, z_t) → reconstructed disaster state."""
        return self.decoder(torch.cat([h_t, z_t], dim=-1))


class WorldModel(nn.Module):
    """
    Full World Model wrapping the RSSM.

    Exposes two main methods:
        train_step(seq)  : one gradient step over a sequence
        rollout(x0, k)   : k-step forecast from initial observation x0
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg  = cfg
        self.rssm = RSSM(cfg)

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(self, seq: Tensor) -> dict:
        """
        Process a full T-step sequence in posterior mode.

        seq : (T, d_disaster_state)

        Returns a dict with:
            recon     : (T, d_disaster_state)  reconstructions
            mu_post   : (T, d_latent)
            sig_post  : (T, d_latent)
            mu_prior  : (T, d_latent)
            sig_prior : (T, d_latent)
        """
        T = seq.shape[0]
        device = seq.device
        h, z = self.rssm.initial_state(device)

        recon_list, mu_p_list, sig_p_list, mu_pr_list, sig_pr_list = (
            [], [], [], [], []
        )

        for t in range(T):
            h, z, mp, sp, mpr, spr = self.rssm.step_posterior(seq[t], h, z)
            x_hat = self.rssm.decode(h, z)
            recon_list.append(x_hat)
            mu_p_list.append(mp); sig_p_list.append(sp)
            mu_pr_list.append(mpr); sig_pr_list.append(spr)

        return dict(
            recon     = torch.stack(recon_list),       # (T, d_s)
            mu_post   = torch.stack(mu_p_list),        # (T, d_z)
            sig_post  = torch.stack(sig_p_list),
            mu_prior  = torch.stack(mu_pr_list),
            sig_prior = torch.stack(sig_pr_list),
        )

    # ── Inference / rollout ───────────────────────────────────────────────────

    @torch.no_grad()
    def rollout(
        self,
        warm_up_seq: Tensor,          # (T_warm, d_disaster_state) — observed
        n_steps: int,
        z_override: Optional[Tensor] = None,   # optional latent perturbation
    ) -> Tensor:
        """
        Warm up on observed steps, then roll out n_steps using the prior.

        Returns predicted sequence: (n_steps, d_disaster_state).
        """
        device = warm_up_seq.device
        h, z = self.rssm.initial_state(device)

        # Warm-up (posterior)
        for t in range(warm_up_seq.shape[0]):
            h, z, *_ = self.rssm.step_posterior(warm_up_seq[t], h, z)

        # Optional latent override (counterfactual perturbation)
        if z_override is not None:
            z = z + z_override

        # Rollout (prior) — apply z_override persistently at every step
        # to model a sustained intervention (do-operator), not just a nudge.
        preds = []
        for _ in range(n_steps):
            h, z = self.rssm.step_prior(h, z)
            if z_override is not None:
                z = z + z_override   # sustained causal intervention
            preds.append(self.rssm.decode(h, z))

        return torch.stack(preds)   # (n_steps, d_s)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
