"""
evaluation.py — Evaluation framework for the STORM-CARE Foundation Model.

Metrics computed after each epoch
----------------------------------
1.  track_err_km     (per-lead)   Haversine distance from horizon predictions
2.  recon_mse                     Masked token reconstruction MSE
3.  contrast_align                Mean cosine similarity of positive pairs
4.  crps_score       (per-lead)   Continuous Ranked Probability Score
5.  linear_probe_acc              Logistic regression on frozen CLS for HU/non-HU
6.  cone_coverage_p50/p90         Forecast cone calibration

All metrics are persisted to a CSV in cfg.metrics_dir.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import FoundationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Haversine
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km_batch(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    R = 6371.0
    φ1, φ2  = np.radians(lat1), np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# CRPS for Gaussian distribution
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_crps(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Analytical CRPS for a univariate Gaussian:
        CRPS(N(μ,σ), y) = σ [ (y-μ)/σ · (2Φ((y-μ)/σ) - 1)
                              + 2φ((y-μ)/σ) - 1/√π ]
    where Φ is the normal CDF and φ is the normal PDF.
    """
    from scipy.special import ndtr  # normal CDF
    z   = (y - mu) / (sigma + 1e-8)
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    cdf = ndtr(z)
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return crps


# ─────────────────────────────────────────────────────────────────────────────
# Ellipse cone coverage
# ─────────────────────────────────────────────────────────────────────────────

_Z_P50 = 1.177   # χ²₂(0.50) → radius for 2-D Gaussian
_Z_P90 = 2.146   # χ²₂(0.90)


def _cone_coverage(
    true_lat: np.ndarray, true_lon: np.ndarray,
    mu_lat: np.ndarray,   mu_lon: np.ndarray,
    sig_lat: np.ndarray,  sig_lon: np.ndarray,
    z: float,
) -> float:
    dx = (true_lat - mu_lat) / (sig_lat + 1e-6)
    dy = (true_lon - mu_lon) / (sig_lon + 1e-6)
    inside = (dx ** 2 + dy ** 2) <= z ** 2
    return float(inside.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe for hurricane intensity classification
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_probe(
    cls_embeds: np.ndarray,   # (N, d_model)
    labels: np.ndarray,       # (N,) int — 1 = hurricane, 0 = non-hurricane
    d_model: int,
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42,
) -> float:
    """Train a linear probe and return test accuracy using scikit-learn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    N = len(cls_embeds)
    if N < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    split = min(N - 1, max(1, int(N * 0.8)))
    tr_idx, te_idx = idx[:split], idx[split:]
    X_tr, y_tr = cls_embeds[tr_idx], labels[tr_idx]
    X_te, y_te = cls_embeds[te_idx], labels[te_idx]
    if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
        return float("nan")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", random_state=seed)
    clf.fit(X_tr, y_tr)
    acc = float((clf.predict(X_te) == y_te).mean())
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# FoundationEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class FoundationEvaluator:
    """
    Evaluates the foundation model on a held-out subset of storm windows.

    Usage
    -----
    evaluator = FoundationEvaluator(model, cfg)
    metrics   = evaluator.evaluate(val_loader)
    evaluator.print_table(metrics, epoch=5)
    """

    def __init__(self, model: nn.Module, cfg: FoundationConfig):
        self.model = model
        self.cfg   = cfg

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        """Run evaluation on a DataLoader and return metric dict."""
        self.model.eval()
        device = next(self.model.parameters()).device

        # Accumulation buffers
        recon_mse_list:  List[float] = []
        contrast_sim_list: List[float] = []

        # Per-lead horizon buffers: list of (B,2) arrays
        n_leads = len(self.cfg.lead_steps)
        horizon_mu_all    = [[] for _ in range(n_leads)]
        horizon_sigma_all = [[] for _ in range(n_leads)]
        horizon_tgt_all   = [[] for _ in range(n_leads)]
        horizon_valid_all = [[] for _ in range(n_leads)]

        # CLS embeddings for linear probe
        cls_list: List[np.ndarray] = []
        label_list: List[np.ndarray] = []

        for batch in loader:
            sf    = batch["storm_feats"].to(device)
            bi    = batch["basin_ids"].to(device)
            si    = batch["status_ids"].to(device)
            era5  = batch["era5_patches"].to(device)
            ev    = batch["era5_valid"].to(device)
            ei_raw = batch.get("edge_index")
            if ei_raw is not None:
                # DataLoader stacks edge_index as (B, 2, E); take first since all windows share the same graph
                ei = (ei_raw[0] if ei_raw.dim() == 3 else ei_raw).to(device)
            else:
                ei = None

            B, T, F = sf.shape

            from .objectives import sample_mask
            mask1 = sample_mask(B, T, self.cfg.mask_ratio, device)
            mask2 = sample_mask(B, T, self.cfg.mask_ratio, device)

            out1 = self.model(sf, bi, si, era5, ev, ei, mask1)
            out2 = self.model(sf, bi, si, era5, ev, ei, mask2)

            # Recon MSE
            recon = out1["recon_pred"]  # (B,T,F)
            diff  = ((recon - sf) ** 2 * mask1.unsqueeze(-1).float()).sum()
            n_m   = mask1.float().sum() * F
            recon_mse_list.append((diff / n_m.clamp(1)).item())

            # Contrastive alignment
            sim = (out1["contrast_z"] * out2["contrast_z"]).sum(dim=-1)  # (B,)
            contrast_sim_list.append(sim.mean().item())

            # Horizon
            h_mu  = out1["horizon_mu"].cpu().numpy()     # (B, n_leads, 2)
            h_sig = out1["horizon_sigma"].cpu().numpy()
            h_tgt = batch["horizon_targets"].numpy()     # (B, n_leads, 2)
            h_val = batch["horizon_valid"].numpy().astype(bool)
            for k in range(n_leads):
                horizon_mu_all[k].append(h_mu[:, k])
                horizon_sigma_all[k].append(h_sig[:, k])
                horizon_tgt_all[k].append(h_tgt[:, k])
                horizon_valid_all[k].append(h_val[:, k])

            # CLS for linear probe  (hurricane = status HU = status_int 2)
            cls_list.append(out1["cls_emb"].cpu().numpy())
            # status_ids: (B, T) → use the middle timestep
            mid   = T // 2
            label = (si[:, mid] == 2).cpu().numpy().astype(np.int32)
            label_list.append(label)

        metrics: Dict[str, float] = {}

        # ── Recon MSE ──────────────────────────────────────────────────────
        metrics["recon_mse"] = float(np.mean(recon_mse_list))

        # ── Contrastive alignment ──────────────────────────────────────────
        metrics["contrast_align"] = float(np.mean(contrast_sim_list))

        # ── Per-lead track error + CRPS + cone coverage ────────────────────
        lead_h = [ls * 6 for ls in self.cfg.lead_steps]

        track_err_per_lead = {}
        crps_per_lead      = {}
        cone50_per_lead    = {}
        cone90_per_lead    = {}

        for k, h in enumerate(lead_h):
            mu_k   = np.concatenate(horizon_mu_all[k], axis=0)     # (N, 2)
            sig_k  = np.concatenate(horizon_sigma_all[k], axis=0)
            tgt_k  = np.concatenate(horizon_tgt_all[k], axis=0)
            valid_k = np.concatenate(horizon_valid_all[k], axis=0)
            n_valid = int(valid_k.sum())

            if n_valid == 0:
                metrics[f"track_err_km_{h}h"] = float("nan")
                metrics[f"crps_{h}h"] = float("nan")
                metrics[f"cone_p50_{h}h"] = float("nan")
                metrics[f"cone_p90_{h}h"] = float("nan")
                metrics[f"n_valid_{h}h"] = 0
                continue

            mu_k = mu_k[valid_k]
            sig_k = sig_k[valid_k]
            tgt_k = tgt_k[valid_k]

            # Track error in km (Δlat, Δlon as degrees → haversine)
            # mu_k[:, 0] = Δlat (degrees),  mu_k[:, 1] = Δlon (degrees)
            # We approximate track error from displacement
            err_km = np.sqrt(
                (mu_k[:, 0] - tgt_k[:, 0]) ** 2 * 111 ** 2
                + (mu_k[:, 1] - tgt_k[:, 1]) ** 2 * (111 * np.cos(np.radians(25))) ** 2
            )
            track_err_per_lead[h] = float(err_km.mean())

            # CRPS over lat and lon separately, then average
            crps_lat = _gaussian_crps(mu_k[:, 0], sig_k[:, 0], tgt_k[:, 0]).mean()
            crps_lon = _gaussian_crps(mu_k[:, 1], sig_k[:, 1], tgt_k[:, 1]).mean()
            crps_per_lead[h] = float((crps_lat + crps_lon) / 2)

            # Cone coverage (using sigma as ellipse radii in degrees)
            cone50_per_lead[h] = _cone_coverage(
                tgt_k[:, 0], tgt_k[:, 1],
                mu_k[:, 0],  mu_k[:, 1],
                sig_k[:, 0], sig_k[:, 1],
                _Z_P50,
            )
            cone90_per_lead[h] = _cone_coverage(
                tgt_k[:, 0], tgt_k[:, 1],
                mu_k[:, 0],  mu_k[:, 1],
                sig_k[:, 0], sig_k[:, 1],
                _Z_P90,
            )

            metrics[f"track_err_km_{h}h"]  = track_err_per_lead[h]
            metrics[f"crps_{h}h"]          = crps_per_lead[h]
            metrics[f"cone_p50_{h}h"]      = cone50_per_lead[h]
            metrics[f"cone_p90_{h}h"]      = cone90_per_lead[h]
            metrics[f"n_valid_{h}h"]       = n_valid

        # ── Linear probe ───────────────────────────────────────────────────
        if len(cls_list) > 0 and len(label_list) > 0:
            all_cls    = np.concatenate(cls_list, axis=0)
            all_labels = np.concatenate(label_list, axis=0)
            if len(np.unique(all_labels)) > 1:
                acc = train_linear_probe(all_cls, all_labels, self.cfg.d_model, seed=self.cfg.seed)
                metrics["linear_probe_acc"] = acc
            else:
                metrics["linear_probe_acc"] = float("nan")

        self.model.train()
        return metrics

    @staticmethod
    def print_table(metrics: Dict[str, float], epoch: int) -> None:
        print(f"\n{'─'*62}")
        print(f"  Validation Metrics — Epoch {epoch}")
        print(f"{'─'*62}")
        print(f"  {'Metric':<35} {'Value':>12}")
        print(f"  {'─'*35} {'─'*12}")
        ordered_keys = sorted(metrics.keys())
        for k in ordered_keys:
            v = metrics[k]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(fv):
                print(f"  {k:<35} {fv:>12.4f}")
        print(f"{'─'*62}")

    @staticmethod
    def save_metrics_csv(
        all_epoch_metrics: List[Dict],
        metrics_dir: str,
        filename: str = "foundation_eval_metrics.csv",
    ) -> None:
        import os, csv
        os.makedirs(metrics_dir, exist_ok=True)
        path = os.path.join(metrics_dir, filename)
        if not all_epoch_metrics:
            return
        keys = ["epoch"] + sorted(k for k in all_epoch_metrics[0] if k != "epoch")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in all_epoch_metrics:
                writer.writerow({k: row.get(k, "") for k in keys})
        print(f"  Metrics saved → {path}")
