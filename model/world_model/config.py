"""
WorldModelConfig — configuration for the RSSM World Model (Module 4).

The World Model takes sequences of disaster-state vectors produced by the
Dynamic Disaster Graph (Module 3) and learns:
  - A compact latent representation  z_t  of the disaster system
  - Transition dynamics:  z_{t+1} ~ p(z_{t+1} | z_t, h_t)
  - Multi-step forecasting:  z_{t+k}  for k = 1 … n_forecast

The latent state z_t implicitly decomposes into four semantic sub-spaces
(encouraged by the structured decoder heads):
  z_hazard    : storm track / intensity evolution
  z_infra     : infrastructure damage and capacity stress
  z_exposure  : population exposure and evacuation status
  z_resource  : supply-demand balance at shelters and hospitals
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    # ── Interface with Module 3 ────────────────────────────────────────────────
    # Must match DisasterGraphConfig.d_disaster_state
    d_disaster_state: int = 32

    # ── Latent space ───────────────────────────────────────────────────────────
    d_latent:  int = 32    # dimension of z_t
    d_hidden:  int = 64    # GRU hidden size (deterministic state h_t)

    # ── Encoder / decoder ─────────────────────────────────────────────────────
    d_enc_hidden: int = 64   # posterior MLP hidden dim
    d_dec_hidden: int = 64   # decoder MLP hidden dim

    # ── Training ───────────────────────────────────────────────────────────────
    # Sequence length (time steps fed during one training pass)
    n_steps_train: int = 12   # should match DisasterGraphConfig.n_steps
    beta_kl:       float = 0.1   # weight on KL divergence term
    beta_pred:     float = 0.5   # weight on multi-step prediction loss
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    n_epochs:      int   = 40
    batch_size:    int   = 16
    n_sequences:   int   = 400   # synthetic training sequences
    seed:          int   = 42
    demo:          bool  = False

    # ── Forecast horizon ───────────────────────────────────────────────────────
    n_forecast: int = 12   # steps to roll out during evaluation

    def apply_demo_overrides(self) -> "WorldModelConfig":
        self.demo         = True
        self.d_hidden     = 32
        self.d_latent     = 16
        self.d_enc_hidden = 32
        self.d_dec_hidden = 32
        self.n_steps_train = 8
        self.n_sequences   = 120
        self.n_epochs      = 20
        self.batch_size    = 8
        self.n_forecast    = 8
        return self

    def __str__(self) -> str:
        tag = "[DEMO] " if self.demo else ""
        return (f"{tag}WorldModelConfig | "
                f"d_state={self.d_disaster_state} → d_latent={self.d_latent} | "
                f"GRU d_hidden={self.d_hidden} | β_kl={self.beta_kl}")
