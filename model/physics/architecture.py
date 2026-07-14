"""
PIGNOModel — Physics-Informed Graph Neural Operator.

Architecture overview
─────────────────────
Input (B, N, C_in=7)  ──┐
                         │  ← Positional encoding appended → (B, N, C_in+2)
                    Lifting MLP
                         │  → (B, N, d_v)
                    GNOLayer × n_gno_layers
                         │  → (B, N, d_v)   [local, physical-space]
                    reshape (B, d_v, H, W)
                    FNOLayer × n_fno_layers
                         │  → (B, d_v, H, W) [global, spectral-space]
                    reshape (B, N, d_v)
                    ┌────┴──────────────────────┐
               FieldHead                  TrackHead
          (B, N, C_out)              (B, 2) [Δlat, Δlon]
               └────┬──────────────────────┘
                 Outputs

Why GNO first, then FNO?
  GNO operates in physical space and encodes local inter-node interactions
  (momentum, diffusion, pressure gradients).  FNO then operates in spectral
  space and captures long-range correlations (steering flow, wave patterns).
  The dual-scale representation matches the multi-scale physics of hurricanes.

Positional encoding
  Each node i receives its normalised (x, y) coordinate appended to its
  feature vector.  This breaks translational symmetry and allows the model
  to learn location-dependent physics (e.g., Coriolis variation with latitude).

Parameter count (default config, d_v=64, GNO×4, FNO×4, N=33²=1089):
  Lifting:       (7+2) × 64 = 576
  GNO kernel:    4 × [MLP(131→128→128→64)] ≈ 4 × 43,520 = 174,080
  FNO spectral:  4 × [2 × (64×64×12×12)] × 2 ≈ 4 × 1,179,648 = 4,718,592
  FNO skip:      4 × (64×64×1×1) = 16,384
  Field head:    (64→64→7) = 4,672
  Track head:    (64→64→2) = 4,226
  Total ≈ 4.9 M parameters (full) / ~0.3 M (demo)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import PIGNOConfig
from .graph_builder import GraphData, build_grid_graph
from .operators import GNOLayer, FNOLayer


class PIGNOModel(nn.Module):
    """
    Physics-Informed Graph Neural Operator for hurricane field prediction.

    Args:
        cfg    : PIGNOConfig
        device : target device (cpu or cuda)

    Forward input:
        x : (B, N, C_in)   field snapshot at time t

    Forward output (dict):
        state_pred : (B, N, C_out)  predicted increments Δs = s_{t+1} − s_t
        track_pred : (B, 2)         predicted track displacement [Δlat, Δlon] °
    """

    def __init__(self, cfg: PIGNOConfig, device: torch.device):
        super().__init__()
        self.cfg    = cfg
        self.device = device

        # Static graph (shared across all batch elements)
        self.graph: GraphData = build_grid_graph(cfg, device)

        d_v     = cfg.d_v
        C_in    = cfg.n_in_channels
        C_out   = cfg.n_out_channels
        N_modes = (cfg.n_modes_x, cfg.n_modes_y)

        # ── Lifting: C_in + 2 positional dims → d_v ──────────────────────
        self.lift = nn.Sequential(
            nn.Linear(C_in + 2, d_v),
            nn.GELU(),
        )

        # ── GNO layers (physical space) ───────────────────────────────────
        self.gno_layers = nn.ModuleList([
            GNOLayer(d_v, cfg.d_hidden)
            for _ in range(cfg.n_gno_layers)
        ])

        # ── FNO layers (spectral space) ───────────────────────────────────
        self.fno_layers = nn.ModuleList([
            FNOLayer(d_v, *N_modes)
            for _ in range(cfg.n_fno_layers)
        ])

        # ── Field output head: d_v → C_out ────────────────────────────────
        self.field_head = nn.Sequential(
            nn.Linear(d_v, d_v),
            nn.GELU(),
            nn.Linear(d_v, C_out),
        )

        # ── Track head: global mean pool d_v → 2 [Δlat, Δlon] ────────────
        self.track_head = nn.Sequential(
            nn.Linear(d_v, d_v),
            nn.GELU(),
            nn.Linear(d_v, 2),
        )

        self.to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> dict:
        """
        x       : (B, N, C_in)
        Returns dict with keys:
            state_pred : (B, N, C_out)
            track_pred : (B, 2)
        """
        B, N, _ = x.shape
        H = W = self.cfg.grid_size

        # ── Positional encoding ───────────────────────────────────────────
        # Append normalised (x, y) ∈ [−1,1]² coordinates to node features.
        pos = (
            self.graph.x_coords                # (N, 2) on model device
            .unsqueeze(0).expand(B, -1, -1)    # (B, N, 2)
        )
        h = self.lift(torch.cat([x, pos], dim=-1))  # (B, N, d_v)

        # ── GNO message passing (physical space) ──────────────────────────
        for gno in self.gno_layers:
            h = gno(h, self.graph)              # (B, N, d_v)

        # ── Reshape to 2-D grid for FNO: (B, N, d_v) → (B, d_v, H, W) ───
        h_2d = h.permute(0, 2, 1).reshape(B, self.cfg.d_v, H, W)

        # ── FNO spectral layers (Fourier space) ───────────────────────────
        for fno in self.fno_layers:
            h_2d = fno(h_2d)                   # (B, d_v, H, W)

        # ── Reshape back to node format ───────────────────────────────────
        h = h_2d.reshape(B, self.cfg.d_v, N).permute(0, 2, 1)  # (B, N, d_v)

        # ── Output heads ─────────────────────────────────────────────────
        state_pred = self.field_head(h)         # (B, N, C_out)
        track_pred = self.track_head(h.mean(1)) # (B, 2)

        return dict(state_pred=state_pred, track_pred=track_pred)

    # ── Utility ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
