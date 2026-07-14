"""
config.py — Central configuration for the STORM-CARE Foundation Model.

FoundationConfig is the single source of truth for all hyperparameters,
data paths, architectural choices, and training settings.  Every module
imports this dataclass so that experiments can be varied from one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class FoundationConfig:
    # ── Data sources ──────────────────────────────────────────────────────────
    hurdat2_path: str = "your-repo/data/data/raw/hurdat2/hurdat2_atlantic.txt"

    # Path to IBTrACS CSV (e.g., IBTrACS.ALL.list.v04r00.csv).
    # Set to None to use the built-in synthetic global storm fallback.
    ibtracs_path: Optional[str] = None

    era5_paths: List[str] = field(default_factory=lambda: [
        "your-repo/data/data/raw/era5/irma_2017/era5_pl_irma_2017.nc",
        "your-repo/data/data/raw/era5/ian_2022/era5_pl_ian_2022.nc",
    ])
    # One tag per ERA5 path — used to look up the matching HURDAT2 storm
    era5_storm_tags: List[str] = field(default_factory=lambda: ["IRMA", "IAN"])

    vulnerability_path: str = (
        "your-repo/data/data/raw/data/raw/vulnerability/vulnerability_grid_clean.csv"
    )
    recovery_labels_path: str = "your-repo/data/data/raw/recovery_labels.csv"

    # ── Output paths ──────────────────────────────────────────────────────────
    ckpt_dir: str = "checkpoints/foundation"
    metrics_dir: str = "metrics/foundation"
    era5_cache_dir: str = "data_cache/era5_patches"

    # ── Architecture ──────────────────────────────────────────────────────────
    d_model: int = 256          # token embedding dimension
    n_heads: int = 8            # multi-head attention heads
    n_layers: int = 6           # depth: n_layers GAT + n_layers Transformer interleaved
    d_ff: int = 1024            # feed-forward inner dimension
    dropout: float = 0.1
    max_seq_len: int = 128      # maximum storm sequence length supported

    # ── Storm input features (continuous scalars after normalisation) ──────────
    #  0: lat_norm         (lat / 90)
    #  1: lon_sin          (sin of lon in radians)
    #  2: lon_cos          (cos of lon in radians)
    #  3: vmax_norm        (vmax_kt / 200)
    #  4: mslp_norm        ((mslp_mb - 900) / 120)
    #  5: speed_norm       (speed_kmh / 100)
    #  6: heading_sin      (sin of heading in radians)
    #  7: heading_cos      (cos of heading in radians)
    #  8: dvmax_dt_norm    (intensification rate, clipped to ±1)
    #  9: age_frac         (timestep / total storm timesteps)
    # 10: sin_doy          (sin(2π day-of-year / 365))
    # 11: cos_doy          (cos(2π day-of-year / 365))
    n_storm_features: int = 12

    n_basin_classes: int = 4    # AL, EP, WP, IO
    n_status_classes: int = 7   # TD, TS, HU, EX, SD, SS, LO

    # ── ERA5 patch ────────────────────────────────────────────────────────────
    grid_size: int = 33         # GxG crop around storm centre
    crop_deg: float = 8.0       # half-size in degrees
    era5_in_channels: int = 5   # u850, v850, u500, v500, z500

    # ── Vulnerability features ────────────────────────────────────────────────
    n_vuln_features: int = 5    # RPL_THEME1..4 + RPL_THEMES

    # ── Graph construction ────────────────────────────────────────────────────
    max_inter_storm_dist_km: float = 800.0  # spatial inter-storm edge threshold
    temporal_window_steps: int = 4          # within-storm temporal edge window size

    # ── Pretraining hyperparameters ───────────────────────────────────────────
    window_size: int = 16        # 6-h steps per training window  (96 h)
    stride: int = 4              # sliding window stride
    mask_ratio: float = 0.25     # fraction of storm tokens masked for MAE
    contrastive_dim: int = 128   # projection head output dimension
    temperature: float = 0.07   # InfoNCE temperature (τ)
    # Lead times expressed in 6-h increments → 6, 12, 24, 48, 72, 120 h
    lead_steps: Tuple[int, ...] = (1, 2, 4, 8, 12, 20)

    # ── Training ──────────────────────────────────────────────────────────────
    min_year: int = 1980        # ignore pre-satellite-era storms
    max_storms: Optional[int] = None   # cap for demo runs (None = all storms)
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0
    seed: int = 42

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_future: float = 1.0
    lambda_mask: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_horizon: float = 1.0

    # ── Demo / ablation flags ─────────────────────────────────────────────────
    demo_mode: bool = False      # smaller model + fewer storms for quick runs
    demo_d_model: int = 128
    demo_n_layers: int = 3
    demo_n_heads: int = 4
    demo_epochs: int = 5
    demo_max_storms: int = 400

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def apply_demo_overrides(self) -> "FoundationConfig":
        """Return a copy of the config with demo-sized hyperparameters."""
        import copy
        cfg = copy.deepcopy(self)
        cfg.d_model = self.demo_d_model
        cfg.n_layers = self.demo_n_layers
        cfg.n_heads = self.demo_n_heads
        cfg.d_ff = self.demo_d_model * 4
        cfg.epochs = self.demo_epochs
        cfg.max_storms = self.demo_max_storms
        cfg.demo_mode = True
        return cfg

    def n_parameters_estimate(self) -> int:
        """Rough parameter count estimate (in millions)."""
        emb = self.n_storm_features * self.d_model
        era5_enc = self.era5_in_channels * 48 * 9 + 48 * 48 * 9 + 48 * self.d_model
        backbone = self.n_layers * (
            4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff
        )
        heads = 4 * self.d_model * self.d_model
        return emb + era5_enc + backbone + heads

    def summary(self) -> str:
        mode = "DEMO" if self.demo_mode else "FULL"
        return (
            f"[{mode}] FoundationConfig | d_model={self.d_model} | "
            f"n_layers={self.n_layers} | n_heads={self.n_heads} | "
            f"d_ff={self.d_ff} | window={self.window_size}×6h | "
            f"mask={self.mask_ratio:.0%} | τ={self.temperature} | "
            f"~{self.n_parameters_estimate() / 1e6:.1f}M params"
        )
