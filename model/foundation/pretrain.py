"""
pretrain.py — Pretraining runner for the STORM-CARE Foundation Model.

Quick-start
-----------
  # Full pretraining (GPU recommended)
  python -m model.foundation.pretrain

  # Demo run (CPU-friendly, smaller model, ~5 min)
  python -m model.foundation.pretrain --demo

Architecture diagram (Mermaid — render in any Mermaid viewer)
-------------------------------------------------------------
  graph TD
    subgraph DataSources["Data Sources"]
        H["HURDAT2 · 1700+ Atlantic storms"] 
        I["IBTrACS · Global multi-basin"]
        E["ERA5 · Atmospheric patches (5 vars)"]
        V["Vulnerability · SVI grid (5 themes)"]
        R["Recovery labels"]
    end
    subgraph Pipeline["Data Pipeline"]
        H --> P["MultiSourceDataPipeline"]
        I --> P
        E --> P
        V --> P
        P --> SG["StormGraph\n(temporal + inter-storm edges)"]
        P --> SW["Sliding Windows T=16 × 6h"]
    end
    subgraph Encoding["Multi-modal Encoders"]
        ST["StormTokenizer\n(12 features → d_model)"]
        EE["ERA5PatchEncoder\nConv2D → Pool → d_model"]
        VE["VulnerabilityEncoder\nMLP → d_model"]
    end
    subgraph Backbone["Foundation Backbone"]
        GAT["GraphAttentionLayer × n/2\n(local neighbourhood)"]
        TF["TransformerLayer × n/2\n(global context)"]
        CLS["CLS token → sequence repr"]
    end
    subgraph Heads["Pretraining Heads"]
        FH["FutureStateHead\nNLL(t+1)"]
        MH["MaskedReconHead\nMSE(masked)"]
        CH["ContrastiveHead\nInfoNCE"]
        HH["MultiHorizonHead\nNLL @ 6/12/24/48/72/120h"]
    end
    SW --> ST
    SW --> EE
    ST --> GAT
    EE --> GAT
    VE --> GAT
    GAT --> TF
    TF --> CLS
    CLS --> FH & MH & CH & HH
    FH --> L["L_total = Σ λᵢLᵢ"]
    MH --> L
    CH --> L
    HH --> L
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import FoundationConfig
from .data_pipeline import MultiSourceDataPipeline, StormRecord
from .graph_construction import build_window_graph
from .architecture import FoundationModel
from .objectives import CombinedPretrainingObjective, sample_mask
from .evaluation import FoundationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class StormWindowSample:
    """One sliding window of T steps from a storm track."""
    __slots__ = [
        "storm_feats", "basin_ids", "status_ids",
        "era5_patches", "era5_valid",
        "edge_index", "edge_type",
        "horizon_targets",   # (n_leads, 2)  Δlat, Δlon
        "horizon_valid",     # (n_leads,) bool
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StormSequenceDataset(Dataset):
    """
    Sliding-window dataset built from a list of StormRecord objects.

    Each sample is a window of *window_size* consecutive storm observations.
    The dataset pre-computes all windows at construction time and stores them
    as numpy arrays so DataLoader workers can fetch quickly.

    horizon_targets
    ---------------
    For each lead step l ∈ lead_steps, the target is the Δ(lat, lon) from
    the last observed position in the window to the position l steps ahead.
    If the storm track ends before that lead, the target is marked invalid.
    """

    def __init__(
        self,
        records: List[StormRecord],
        cfg: FoundationConfig,
    ):
        self.cfg = cfg
        T   = cfg.window_size
        stride = cfg.stride
        G   = cfg.grid_size
        C   = cfg.era5_in_channels
        n_leads = len(cfg.lead_steps)

        self.samples: List[StormWindowSample] = []
        ei_base, _ = build_window_graph(T, cfg.temporal_window_steps)

        for rec in records:
            Nobs = rec.T
            if Nobs < T + max(cfg.lead_steps):
                continue  # not enough future data for horizon targets

            for start in range(0, Nobs - T, stride):
                end = start + T  # exclusive

                sf  = rec.features[start:end]           # (T, F)
                bi  = rec.basin_ids[start:end]           # (T,)
                si  = rec.status_ids[start:end]          # (T,)
                ep  = rec.era5_patches[start:end]        # (T, C, G, G)
                ev  = rec.era5_valid[start:end]          # (T,) bool

                # Horizon targets from the last position in the window
                last_lat = float(rec.lat[end - 1])
                last_lon = float(rec.lon[end - 1])
                h_tgt   = np.zeros((n_leads, 2), dtype=np.float32)
                h_valid = np.zeros(n_leads, dtype=bool)
                for ki, ls in enumerate(cfg.lead_steps):
                    fut_idx = end - 1 + ls
                    if fut_idx < Nobs:
                        dlat = float(rec.lat[fut_idx]) - last_lat
                        dlon = float(rec.lon[fut_idx]) - last_lon
                        h_tgt[ki] = [dlat, dlon]
                        h_valid[ki] = True

                self.samples.append(StormWindowSample(
                    storm_feats     = sf.astype(np.float32),
                    basin_ids       = bi.astype(np.int64),
                    status_ids      = si.astype(np.int64),
                    era5_patches    = ep.astype(np.float32),
                    era5_valid      = ev.astype(bool),
                    edge_index      = ei_base.copy(),
                    edge_type       = np.zeros(ei_base.shape[1], dtype=np.int64),
                    horizon_targets = h_tgt,
                    horizon_valid   = h_valid,
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "storm_feats":     torch.from_numpy(s.storm_feats),
            "basin_ids":       torch.from_numpy(s.basin_ids),
            "status_ids":      torch.from_numpy(s.status_ids),
            "era5_patches":    torch.from_numpy(s.era5_patches),
            "era5_valid":      torch.from_numpy(s.era5_valid),
            "edge_index":      torch.from_numpy(s.edge_index),
            "horizon_targets": torch.from_numpy(s.horizon_targets),
            "horizon_valid":   torch.from_numpy(s.horizon_valid),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Standard collate — all tensors have the same shape so we can stack."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule (cosine with warmup)
# ─────────────────────────────────────────────────────────────────────────────

def _lr_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_frac: float = 0.1):
    if step < warmup_steps:
        return float(step) / max(warmup_steps, 1)
    progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(min_lr_frac, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ─────────────────────────────────────────────────────────────────────────────
# PretrainRunner
# ─────────────────────────────────────────────────────────────────────────────

class PretrainRunner:
    """
    Orchestrates the full pretraining loop:
      data build → train/val split → model init → training → evaluation → save
    """

    def __init__(self, cfg: FoundationConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        self.device = torch.device(cfg.device)
        os.makedirs(cfg.ckpt_dir,   exist_ok=True)
        os.makedirs(cfg.metrics_dir, exist_ok=True)

    # ── Public entry-point ─────────────────────────────────────────────────

    def run(self) -> Dict[str, float]:
        """Execute full pretraining and return final evaluation metrics."""
        cfg = self.cfg

        # ── 1. Build dataset ──────────────────────────────────────────────
        print("\n" + "=" * 62)
        print("  STORM-CARE Foundation Model  —  Pretraining")
        print("=" * 62)
        print(f"  Config : {cfg.summary()}")
        print(f"  Device : {self.device}")
        print()

        pipeline = MultiSourceDataPipeline(cfg)
        records  = pipeline.build()

        if not records:
            raise RuntimeError(
                "No StormRecord objects built. Check data paths in FoundationConfig."
            )

        # Train/val split by storm identity group — not by window or source record —
        # to avoid leakage when the same named storm appears in multiple sources.
        train_recs, val_recs, split_audit = self._split_records(records)
        self._write_split_manifest(train_recs, val_recs, split_audit)

        train_ds = StormSequenceDataset(train_recs, cfg)
        val_ds   = StormSequenceDataset(val_recs,   cfg)

        if len(train_ds) == 0:
            raise RuntimeError("Training dataset is empty — storms may be too short.")

        n_workers = min(2, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            shuffle=True, num_workers=0, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size,
            shuffle=False, num_workers=0, collate_fn=collate_fn,
        )

        print(f"  Storms : train={len(train_recs):,}  val={len(val_recs):,}")
        print(f"  Windows: train={len(train_ds):,}  val={len(val_ds):,}")

        # ── 2. Build model ────────────────────────────────────────────────
        model = FoundationModel(cfg).to(self.device)
        n_params = model.n_parameters()
        print(f"  Model  : {n_params:,} trainable parameters")
        print()

        # ── 3. Optimiser + LR schedule ────────────────────────────────────
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        total_steps   = cfg.epochs * len(train_loader)
        warmup_steps  = cfg.warmup_epochs * len(train_loader)
        scheduler     = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps),
        )

        criterion  = CombinedPretrainingObjective(cfg).to(self.device)
        evaluator  = FoundationEvaluator(model, cfg)

        # ── 4. Training loop ──────────────────────────────────────────────
        best_val_score = float("inf")
        best_epoch = None
        epoch_log: List[Dict] = []

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            model.train()
            loss_accum   = {"loss": 0.0, "L_future": 0.0,
                            "L_mask": 0.0, "L_contrast": 0.0, "L_horizon": 0.0}
            n_batches = 0

            for batch in train_loader:
                sf   = batch["storm_feats"].to(self.device)
                bi   = batch["basin_ids"].to(self.device)
                si   = batch["status_ids"].to(self.device)
                era5 = batch["era5_patches"].to(self.device)
                ev   = batch["era5_valid"].to(self.device)
                ei   = batch["edge_index"][0].to(self.device)  # shared edge_index
                h_tgt  = batch["horizon_targets"].to(self.device)
                h_val  = batch["horizon_valid"].to(self.device)

                B, T, _ = sf.shape

                # Sample two independent masks
                mask1 = sample_mask(B, T, cfg.mask_ratio, self.device)
                mask2 = sample_mask(B, T, cfg.mask_ratio, self.device)

                out1 = model(sf, bi, si, era5, ev, ei, mask1)
                out2 = model(sf, bi, si, era5, ev, ei, mask2)

                losses = criterion(out1, out2, sf, mask1, h_tgt, h_val)

                opt.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                opt.step()
                scheduler.step()

                for k, v in losses.items():
                    loss_accum[k] += v.item() if hasattr(v, "item") else float(v)
                n_batches += 1

            # Average epoch losses
            avg = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            print(
                f"  Ep {epoch:>3}/{cfg.epochs}  "
                f"loss={avg['loss']:.4f}  "
                f"L_fut={avg['L_future']:.4f}  "
                f"L_msk={avg['L_mask']:.4f}  "
                f"L_cnt={avg['L_contrast']:.4f}  "
                f"L_hrz={avg['L_horizon']:.4f}  "
                f"lr={lr_now:.2e}  "
                f"t={elapsed:.1f}s"
            )

            # ── Evaluate every epoch using one fixed validation protocol ─
            if len(val_ds) > 0:
                val_metrics = evaluator.evaluate(val_loader)
                val_metrics["epoch"] = epoch
                val_metrics["selection_metric"] = "mean_track_err_km"
                val_metrics["selection_score"] = self._selection_score(val_metrics, cfg)
                val_metrics.update({f"train_{k}": v for k, v in avg.items()})
                epoch_log.append(val_metrics)
                evaluator.print_table(val_metrics, epoch)

                if val_metrics["selection_score"] < best_val_score:
                    best_val_score = val_metrics["selection_score"]
                    best_epoch = epoch
                    ckpt_path = os.path.join(cfg.ckpt_dir, "foundation_best.pt")
                    torch.save({
                        "epoch":           epoch,
                        "state_dict":      model.state_dict(),
                        "cfg":             cfg.__dict__,
                        "selection_metric": val_metrics["selection_metric"],
                        "selection_score":  best_val_score,
                        "metrics":          val_metrics,
                        "split_audit":      split_audit,
                    }, ckpt_path)

        # ── 5. Final save + report ─────────────────────────────────────────
        final_ckpt = os.path.join(cfg.ckpt_dir, "foundation_final.pt")
        torch.save({
            "epoch":      cfg.epochs,
            "state_dict": model.state_dict(),
            "cfg":        cfg.__dict__,
        }, final_ckpt)

        selected_metrics = {}
        for row in epoch_log:
            row["selected_checkpoint"] = bool(row.get("epoch") == best_epoch)
            if row["selected_checkpoint"]:
                selected_metrics = row

        evaluator.save_metrics_csv(
            epoch_log, cfg.metrics_dir, "foundation_eval_metrics.csv"
        )

        # Save training log
        log_df = pd.DataFrame(epoch_log)
        log_df.to_csv(
            os.path.join(cfg.metrics_dir, "foundation_train_log.csv"), index=False
        )

        final_metrics = selected_metrics if selected_metrics else (epoch_log[-1] if epoch_log else {})
        self._print_final_report(cfg, records, train_ds, val_ds, n_params,
                                 best_val_score, final_metrics, split_audit)
        return final_metrics

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _record_group_key(rec) -> str:
        name = (rec.storm_name or rec.storm_id or "UNKNOWN").strip().upper()
        return f"{rec.basin}|{rec.year}|{name}"

    def _split_records(self, records):
        cfg = self.cfg
        groups: Dict[str, List] = {}
        for rec in records:
            groups.setdefault(self._record_group_key(rec), []).append(rec)

        keys = np.array(sorted(groups))
        rng = np.random.RandomState(cfg.seed)
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * 0.20))
        val_keys = set(keys[:n_val])
        train_recs, val_recs = [], []
        for key, recs in groups.items():
            (val_recs if key in val_keys else train_recs).extend(recs)

        train_ids = {r.storm_id for r in train_recs}
        val_ids = {r.storm_id for r in val_recs}
        train_group_keys = {self._record_group_key(r) for r in train_recs}
        val_group_keys = {self._record_group_key(r) for r in val_recs}
        audit = {
            "split_unit": "basin|year|storm_name",
            "seed": cfg.seed,
            "n_records": len(records),
            "n_groups": len(groups),
            "n_train_records": len(train_recs),
            "n_val_records": len(val_recs),
            "n_train_groups": len(train_group_keys),
            "n_val_groups": len(val_group_keys),
            "storm_id_overlap": sorted(train_ids & val_ids),
            "group_key_overlap": sorted(train_group_keys & val_group_keys),
        }
        if audit["storm_id_overlap"] or audit["group_key_overlap"]:
            raise RuntimeError(f"Foundation split leakage detected: {audit}")
        return train_recs, val_recs, audit

    def _write_split_manifest(self, train_recs, val_recs, audit) -> None:
        cfg = self.cfg
        rows = []
        for split, recs in [("train", train_recs), ("val", val_recs)]:
            for rec in recs:
                rows.append({
                    "split": split,
                    "group_key": self._record_group_key(rec),
                    "storm_id": rec.storm_id,
                    "storm_name": rec.storm_name,
                    "year": rec.year,
                    "basin": rec.basin,
                    "source": rec.source,
                    "n_observations": rec.T,
                    "n_era5_observations": int(rec.era5_valid.sum()),
                })
        pd.DataFrame(rows).to_csv(
            os.path.join(cfg.metrics_dir, "foundation_split_manifest.csv"),
            index=False,
        )
        with open(os.path.join(cfg.metrics_dir, "foundation_split_audit.json"), "w") as fh:
            json.dump(audit, fh, indent=2)

    @staticmethod
    def _selection_score(metrics: Dict[str, float], cfg: FoundationConfig) -> float:
        vals = []
        for lead_step in cfg.lead_steps:
            key = f"track_err_km_{lead_step * 6}h"
            value = metrics.get(key)
            if value is not None and math.isfinite(float(value)):
                vals.append(float(value))
        return float(np.mean(vals)) if vals else float("inf")

    @staticmethod
    def _print_final_report(
        cfg, records, train_ds, val_ds, n_params, best_val_score, metrics, split_audit
    ):
        total_obs = sum(r.T for r in records)
        era5_obs  = sum(int(r.era5_valid.sum()) for r in records)

        print("\n" + "═" * 62)
        print("  STORM-CARE Foundation Model  —  Pretraining Complete")
        print("═" * 62)
        print()
        print("  ── Dataset ──────────────────────────────────────────────")
        print(f"    Storms       : {len(records):,}")
        print(f"    Split groups : train={split_audit['n_train_groups']:,}  val={split_audit['n_val_groups']:,}")
        print(f"    Observations : {total_obs:,}")
        print(f"    ERA5-enhanced: {era5_obs:,}  ({100*era5_obs/max(total_obs,1):.1f}%)")
        print(f"    Train windows: {len(train_ds):,}")
        print(f"    Val   windows: {len(val_ds):,}")
        print()
        print("  ── Model ────────────────────────────────────────────────")
        print(f"    Architecture : {cfg.summary()}")
        print(f"    Parameters   : {n_params:,}")
        print(f"    Best val selection score: {best_val_score:.4f}")
        print()
        print("  ── Architecture Diagram (Mermaid) ───────────────────────")
        print("    See module docstring or render pretrain.py header in")
        print("    https://mermaid.live")
        print()
        print("  ── Self-supervised Tasks ────────────────────────────────")
        print("    Task 1  Future-State Prediction   Gaussian NLL (t→t+1)")
        print("    Task 2  Masked Graph Reconstruction  MSE on masked tokens")
        print("    Task 3  Contrastive Evolution    InfoNCE, τ=" + str(cfg.temperature))
        print(f"    Task 4  Multi-Horizon Forecasting   leads=" +
              str([ls * 6 for ls in cfg.lead_steps]) + "h")
        print()
        print("  ── Final Validation Metrics ─────────────────────────────")
        for k, v in sorted(metrics.items()):
            if k not in ("epoch",) and "train_" not in k:
                try:
                    print(f"    {k:<38} {float(v):>8.4f}")
                except (TypeError, ValueError):
                    pass
        print()
        print("  ── Computational Requirements (full-scale) ──────────────")
        print("    Recommended: 1× A100 80 GB  or  4× V100 32 GB")
        print("    Full pretraining (~50 epochs): ≈ 4-8 GPU-hours on A100")
        print("    Demo (5 epochs, d=128, 400 storms): ≈ 5-10 min on CPU")
        print()
        print("  ── Checkpoints ──────────────────────────────────────────")
        print(f"    Best   : {cfg.ckpt_dir}/foundation_best.pt")
        print(f"    Final  : {cfg.ckpt_dir}/foundation_final.pt")
        print(f"    Metrics: {cfg.metrics_dir}/")
        print("═" * 62 + "\n")

    # ── Scalability notes (publication section) ──────────────────────────

    @staticmethod
    def scalability_notes() -> str:
        return """
Scalability
-----------
• HURDAT2 alone provides ~1,700 storms and ~120,000 observations.
  IBTrACS adds ~14,000 global storms (Western Pacific, Indian Ocean, etc.)
  ERA5 reanalysis at 0.25° resolution covers all of these globally.

• The FoundationModel scales linearly with sequence length T (dense MHA
  over window; T ≤ 128), and is designed to be fine-tuned on downstream
  tasks (track forecasting, intensity prediction, recovery planning) with
  minimal additional compute.

• Data parallelism: the model trains efficiently across multiple GPUs via
  torch.nn.DataParallel or DistributedDataParallel (DDP).  For the full
  corpus (HURDAT2 + IBTrACS + ERA5 global), DDP across 4 A100s is
  recommended for 50-epoch pretraining within ≈ 8 hours.

• Mixed-precision (torch.cuda.amp) reduces GPU memory by ~40%.

Physics-informed extensions
---------------------------
The backbone can be augmented with physics-informed losses (Module 2):
  L_physics = α·L_advection + β·L_pressure_wind + γ·L_mass_conservation
added to L_total during a second-stage continued pretraining.
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STORM-CARE Foundation Model Pretraining")
    p.add_argument("--demo",       action="store_true",
                   help="Run in demo mode (small model, subset of data, CPU-friendly)")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--d-model",    type=int,   default=None)
    p.add_argument("--n-layers",   type=int,   default=None)
    p.add_argument("--max-storms", type=int,   default=None)
    p.add_argument("--min-year",   type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = FoundationConfig()

    if args.demo:
        cfg = cfg.apply_demo_overrides()

    # CLI overrides (take precedence over demo)
    if args.epochs     is not None: cfg.epochs     = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr         is not None: cfg.lr         = args.lr
    if args.d_model    is not None: cfg.d_model    = args.d_model
    if args.n_layers   is not None: cfg.n_layers   = args.n_layers
    if args.max_storms is not None: cfg.max_storms = args.max_storms
    if args.min_year   is not None: cfg.min_year   = args.min_year
    if args.seed       is not None: cfg.seed       = args.seed

    runner = PretrainRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
