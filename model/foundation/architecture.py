"""
architecture.py — Foundation Model architecture for STORM-CARE.

Architecture overview
---------------------

  ┌──────────────────────────────────────────────────────────────────┐
  │  Multi-modal Encoders                                            │
  │   StormTokenizer    (T, F_storm) → (T, d_model)                 │
  │   ERA5PatchEncoder  (T, 5, G, G) → (T, d_model)                 │
  │   VulnEncoder       (F_vuln,)    → (d_model,)   [optional]      │
  └────────────────────────────┬─────────────────────────────────────┘
                               │  fuse: token = storm + era5 + pos
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Foundation Backbone  (n_layers × interleaved)                   │
  │   GraphAttentionLayer → graph-local aggregation over edges       │
  │   TransformerLayer    → global sequence context                  │
  │   Prepended CLS token for sequence-level representation          │
  └────────────────────────────┬─────────────────────────────────────┘
                               │
              ┌────────────────┼──────────────────┐
              ▼                ▼                  ▼
      cls_embedding      node_embeddings    node_embeddings
      (B, d_model)       (B, T, d_model)   (B, T, d_model)
              │                │                  │
     ┌────────▼──────┐ ┌───────▼───────┐ ┌───────▼────────┐
     │ContrastiveHead│ │FutureStateHead│ │MaskedReconHead │
     │ InfoNCE proj  │ │ NLL (next-t)  │ │ MSE (masked)   │
     └───────────────┘ └───────────────┘ └────────────────┘
                                │
                       ┌────────▼────────┐
                       │MultiHorizonHead │
                       │ NLL (6→120h)    │
                       └─────────────────┘
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FoundationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Positional / time encoding
# ─────────────────────────────────────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(max_len + 1, d_model)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(T, device=device)
        return self.embed(positions)  # (T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Storm Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class StormTokenizer(nn.Module):
    """
    Convert a batch of storm feature sequences into d_model embeddings.

    Inputs
    ------
    storm_feats  : (B, T, F_storm)   normalised continuous features
    basin_ids    : (B, T)            int — basin class index
    status_ids   : (B, T)            int — TC status class index
    mask         : (B, T) bool or None — True = token is masked (replaced with
                                          a learnable MASK token for MAE)
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        d = cfg.d_model
        self.feat_proj = nn.Linear(cfg.n_storm_features, d)
        self.basin_embed  = nn.Embedding(cfg.n_basin_classes  + 1, d)
        self.status_embed = nn.Embedding(cfg.n_status_classes + 1, d)
        self.mask_token   = nn.Parameter(torch.randn(d))
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        storm_feats: torch.Tensor,
        basin_ids: torch.Tensor,
        status_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # storm_feats: (B, T, F)
        x = self.feat_proj(storm_feats)             # (B, T, d)
        x = x + self.basin_embed(basin_ids)         # (B, T, d)
        x = x + self.status_embed(status_ids)       # (B, T, d)

        # Replace masked tokens with learnable MASK vector
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()          # (B, T, 1)
            mask_token    = self.mask_token.view(1, 1, -1)      # (1, 1, d)
            x = x * (1 - mask_expanded) + mask_token * mask_expanded

        return self.norm(x)  # (B, T, d)


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 Patch Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ERA5PatchEncoder(nn.Module):
    """
    Lightweight operator-style convolutional encoder for ERA5 atmospheric patches.

    Input  : (B*T, C, G, G) — batch of atmospheric patches
    Output : (B*T, d_model)
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        C = cfg.era5_in_channels
        d = cfg.d_model
        w = max(32, d // 8)

        self.cnn = nn.Sequential(
            # Local feature extraction
            nn.Conv2d(C, w, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, kernel_size=3, padding=1), nn.GELU(),
            # Coarser scale
            nn.Conv2d(w, w * 2, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(w * 2, w * 2, kernel_size=3, padding=1), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)
        h = self.pool(h)
        return self.norm(self.proj(h))


# ─────────────────────────────────────────────────────────────────────────────
# Vulnerability Encoder (optional side-input)
# ─────────────────────────────────────────────────────────────────────────────

class VulnerabilityEncoder(nn.Module):
    """Simple MLP that maps SVI features → d_model embedding."""

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        d = cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_vuln_features, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Attention Layer  (dense implementation for T ≤ 128)
# ─────────────────────────────────────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """
    Multi-head attention over a sequence with a graph-structured attention bias.

    The edge_index defines which pairs of nodes are directly connected.  An
    additive bias of +0 is applied to connected pairs and −inf to absent pairs,
    so attention is restricted to the graph neighbourhood (like GAT but via MHA).

    For small windows (T ≤ 128) this dense approach is fast on both CPU and GPU.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x         : (B, T, d_model)
        attn_bias : (T, T) additive bias applied before softmax (−inf blocks edges)
        """
        residual = x
        x2, _ = self.attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(residual + self.drop(x2))

        residual = x
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


def build_attn_bias(
    T: int,
    edge_index: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Build a (T, T) attention bias from edge_index.
    Connected pairs → 0.0, absent pairs → -1e9.
    """
    if edge_index is None:
        return None
    bias = torch.full((T, T), -1e9, device=device)
    # Self-connections always allowed
    bias.fill_diagonal_(0.0)
    src, dst = edge_index[0], edge_index[1]
    # Only keep edges within the current window (T nodes)
    mask = (src < T) & (dst < T)
    src, dst = src[mask], dst[mask]
    bias[dst, src] = 0.0
    return bias


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Layer  (standard pre-LN)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN style
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop(x2)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Foundation Backbone
# ─────────────────────────────────────────────────────────────────────────────

class FoundationBackbone(nn.Module):
    """
    Interleaved stack of GraphAttentionLayer and TransformerLayer.

    n_layers specifies the total depth.  Odd layers use graph attention
    (local neighbourhood), even layers use full self-attention (global context).
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            if i % 2 == 0:
                self.layers.append(
                    GraphAttentionLayer(cfg.d_model, cfg.n_heads, cfg.dropout)
                )
            else:
                self.layers.append(
                    TransformerLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                )
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, T+1, d_model)  — includes CLS prepended at index 0."""
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:  # GraphAttention
                x = layer(x, attn_bias)
            else:            # Transformer (full attention, no graph bias)
                x = layer(x)
        return self.out_norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining heads
# ─────────────────────────────────────────────────────────────────────────────

class FutureStateHead(nn.Module):
    """
    Predict the normalised storm features at the next timestep.
    Used for the *future-state prediction* self-supervised task.

    Output: (B, T, n_storm_features) — mean + log-variance per feature.
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        d, F = cfg.d_model, cfg.n_storm_features
        self.mu       = nn.Linear(d, F)
        self.log_sigma = nn.Linear(d, F)

    def forward(self, node_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # node_emb: (B, T, d_model)
        mu        = self.mu(node_emb)
        log_sigma = self.log_sigma(node_emb).clamp(-6, 2)
        return mu, log_sigma.exp()


class MaskedReconstructionHead(nn.Module):
    """
    Reconstruct originally masked storm feature tokens.
    Used for the *masked graph reconstruction* self-supervised task.

    Output: (B, T, n_storm_features) — direct regression of masked values.
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        d, F = cfg.d_model, cfg.n_storm_features
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, F)
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_emb)


class ContrastiveHead(nn.Module):
    """
    Two-layer MLP projection head for SimCLR-style contrastive learning.
    Projects CLS embedding onto the unit hypersphere.

    Output: (B, contrastive_dim) — L2-normalised.
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        d, c = cfg.d_model, cfg.contrastive_dim
        self.proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, c)
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        z = self.proj(cls_emb)
        return F.normalize(z, dim=-1)


class MultiHorizonHead(nn.Module):
    """
    Probabilistic multi-horizon forecasting head.

    Predicts (mu, sigma) for storm lat/lon at each lead time in lead_steps.
    Uses the CLS embedding (global context) + final node embedding (local).

    Output: mu    (B, n_leads, 2)  — Δlat, Δlon
            sigma (B, n_leads, 2)  — uncertainty
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        n_leads = len(cfg.lead_steps)
        d = cfg.d_model
        self.shared = nn.Sequential(
            nn.Linear(d * 2, d * 2), nn.GELU(),
            nn.Linear(d * 2, d),     nn.GELU(),
        )
        self.mu        = nn.Linear(d, n_leads * 2)
        self.log_sigma = nn.Linear(d, n_leads * 2)
        self.n_leads   = n_leads

    def forward(
        self, cls_emb: torch.Tensor, last_node_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # cls_emb, last_node_emb: (B, d_model)
        h = self.shared(torch.cat([cls_emb, last_node_emb], dim=-1))
        mu        = self.mu(h).view(-1, self.n_leads, 2)
        log_sigma = self.log_sigma(h).view(-1, self.n_leads, 2).clamp(-6, 2)
        return mu, log_sigma.exp()


# ─────────────────────────────────────────────────────────────────────────────
# FoundationModel — main model
# ─────────────────────────────────────────────────────────────────────────────

class FoundationModel(nn.Module):
    """
    STORM-CARE Foundation Model for self-supervised pretraining.

    Inputs (one training sample)
    ----------------------------
    storm_feats  : (B, T, F_storm)  normalised scalar features
    basin_ids    : (B, T)           int — basin class
    status_ids   : (B, T)           int — TC status class
    era5_patches : (B, T, C, G, G)  ERA5 crops (may be zero for unavailable)
    era5_valid   : (B, T)           bool — True when patch is real
    edge_index   : (2, E)           temporal graph edges (shared across batch)
    mask         : (B, T) bool      True = token masked for MAE task

    Returns
    -------
    dict with keys:
      cls_emb       (B, d_model)
      node_emb      (B, T, d_model)
      future_mu     (B, T, F)
      future_sigma  (B, T, F)
      recon_pred    (B, T, F)
      contrast_z    (B, contrastive_dim)
      horizon_mu    (B, n_leads, 2)
      horizon_sigma (B, n_leads, 2)
    """

    def __init__(self, cfg: FoundationConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ── Encoders ──────────────────────────────────────────────────────
        self.storm_tok  = StormTokenizer(cfg)
        self.era5_enc   = ERA5PatchEncoder(cfg)
        self.vuln_enc   = VulnerabilityEncoder(cfg)

        # Learnable placeholder for timesteps without ERA5
        self.no_era5_emb = nn.Parameter(torch.zeros(1, d))

        # ── Positional encoding ───────────────────────────────────────────
        self.pos_enc = LearnedPositionalEncoding(cfg.max_seq_len, d)

        # ── CLS token ─────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = FoundationBackbone(cfg)

        # ── Pretraining heads ─────────────────────────────────────────────
        self.future_head  = FutureStateHead(cfg)
        self.recon_head   = MaskedReconstructionHead(cfg)
        self.contrast_head = ContrastiveHead(cfg)
        self.horizon_head  = MultiHorizonHead(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(
        self,
        storm_feats: torch.Tensor,
        basin_ids: torch.Tensor,
        status_ids: torch.Tensor,
        era5_patches: torch.Tensor,
        era5_valid: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shared encoding used by all pretraining heads.

        Returns
        -------
        cls_emb  : (B, d_model)
        node_emb : (B, T, d_model)
        """
        B, T, _ = storm_feats.shape
        device   = storm_feats.device

        # ── Storm token embedding ─────────────────────────────────────────
        x = self.storm_tok(storm_feats, basin_ids, status_ids, mask)  # (B, T, d)

        # ── ERA5 encoding — batched over (B*T) ────────────────────────────
        BT = B * T
        era5_flat = era5_patches.view(BT, self.cfg.era5_in_channels,
                                       self.cfg.grid_size, self.cfg.grid_size)
        era5_emb_flat = self.era5_enc(era5_flat)   # (B*T, d)
        era5_emb = era5_emb_flat.view(B, T, -1)   # (B, T, d)

        # Blank out unavailable patches, replace with learned embedding
        valid_f = era5_valid.unsqueeze(-1).float()  # (B, T, 1)
        era5_emb = (
            era5_emb * valid_f
            + self.no_era5_emb.unsqueeze(0) * (1.0 - valid_f)
        )

        x = x + era5_emb  # fuse modalities

        # ── Positional encoding ───────────────────────────────────────────
        pos = self.pos_enc(T, device).unsqueeze(0)  # (1, T, d)
        x = x + pos

        # ── Prepend CLS token ─────────────────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        x   = torch.cat([cls, x], dim=1)        # (B, T+1, d)

        # ── Build attention bias from edge_index ──────────────────────────
        T1 = T + 1
        attn_bias = None
        if edge_index is not None:
            # Shift node indices by 1 to account for CLS at position 0
            shifted = edge_index + 1
            # Always allow CLS ↔ all nodes
            cls_row_src = torch.zeros(T1, dtype=torch.long, device=device)
            cls_row_dst = torch.arange(T1, dtype=torch.long, device=device)
            shifted = torch.cat([
                shifted,
                torch.stack([cls_row_src, cls_row_dst]),
                torch.stack([cls_row_dst, cls_row_src]),
            ], dim=1)
            shifted = shifted.clamp(0, T1 - 1)
            attn_bias = build_attn_bias(T1, shifted, device)  # (T+1, T+1)

        # ── Backbone ──────────────────────────────────────────────────────
        x = self.backbone(x, attn_bias)  # (B, T+1, d)

        cls_emb  = x[:, 0]    # (B, d)
        node_emb = x[:, 1:]   # (B, T, d)
        return cls_emb, node_emb

    def forward(
        self,
        storm_feats: torch.Tensor,
        basin_ids: torch.Tensor,
        status_ids: torch.Tensor,
        era5_patches: torch.Tensor,
        era5_valid: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass — runs all pretraining heads."""
        cls_emb, node_emb = self.encode(
            storm_feats, basin_ids, status_ids,
            era5_patches, era5_valid, edge_index, mask,
        )

        future_mu, future_sigma = self.future_head(node_emb)
        recon_pred              = self.recon_head(node_emb)
        contrast_z              = self.contrast_head(cls_emb)
        horizon_mu, horizon_sigma = self.horizon_head(cls_emb, node_emb[:, -1])

        return {
            "cls_emb":       cls_emb,
            "node_emb":      node_emb,
            "future_mu":     future_mu,
            "future_sigma":  future_sigma,
            "recon_pred":    recon_pred,
            "contrast_z":    contrast_z,
            "horizon_mu":    horizon_mu,
            "horizon_sigma": horizon_sigma,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Import helper used internally (avoids circular imports if used from objectives)
# ─────────────────────────────────────────────────────────────────────────────

def build_attn_bias(
    T: int,
    edge_index: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    bias = torch.full((T, T), -1e9, device=device)
    bias.fill_diagonal_(0.0)
    src = edge_index[0].clamp(0, T - 1)
    dst = edge_index[1].clamp(0, T - 1)
    bias[dst, src] = 0.0
    return bias
