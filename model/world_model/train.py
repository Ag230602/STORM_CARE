"""
WorldModelTrainer — training loop for Module 4.

Generates synthetic disaster-state sequences (simulating what Module 3
would produce) and trains the RSSM WorldModel on them.

Run:
    python -m model.world_model.train --demo
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .config import WorldModelConfig
from .architecture import WorldModel

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Synthetic disaster-state sequence generator ────────────────────────────────

def _make_sequences(
    n: int,
    T: int,
    d: int,
    seed: int = 42,
) -> Tensor:
    """
    Generate n synthetic disaster-state sequences of length T.

    Physics:
      - Hazard dims (first d//4): Gaussian bump that peaks at t_peak then decays
      - Infra dims  (next d//4) : monotone increase (cumulative damage)
      - Exposure dims(next d//4): rises then falls (evacuation wave)
      - Resource dims(last d//4): decreases monotonically (supply consumption)

    Returns (n, T, d) float32 tensor.
    """
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, T, d), dtype=np.float32)
    t_arr = np.linspace(0, 1, T)
    dq = max(d // 4, 1)

    for i in range(n):
        t_peak   = rng.uniform(0.3, 0.7)
        vmax     = rng.uniform(0.5, 1.0)
        noise    = rng.normal(0, 0.05, (T, d))

        # Hazard: Gaussian peak
        hazard = vmax * np.exp(-((t_arr - t_peak) ** 2) / 0.08)
        seqs[i, :, :dq] = hazard[:, None] + noise[:, :dq] * 0.1

        # Infra: cumulative damage
        damage = np.cumsum(np.clip(hazard * 0.04, 0, None))
        damage = np.clip(damage, 0, 1)
        seqs[i, :, dq:2*dq] = damage[:, None] + noise[:, dq:2*dq] * 0.05

        # Exposure: rises then falls
        exposure = hazard * (1 - np.linspace(0, 0.8, T))  # evacuation reduces it
        seqs[i, :, 2*dq:3*dq] = np.clip(exposure[:, None], 0, 1) + noise[:, 2*dq:3*dq] * 0.05

        # Resource: depletes
        resource = np.clip(1.0 - np.cumsum(hazard * 0.05), 0, 1)
        seqs[i, :, 3*dq:] = resource[:, None] + noise[:, 3*dq:] * 0.05

    seqs = np.clip(seqs, 0, 1)
    return torch.from_numpy(seqs)


class SequenceDataset(Dataset):
    def __init__(self, seqs: Tensor):
        self.seqs = seqs

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tensor:
        return self.seqs[idx]


# ── Loss function ─────────────────────────────────────────────────────────────

def rssm_loss(
    out:    dict,
    seq:    Tensor,
    beta_kl: float,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    ELBO loss for the RSSM:
      L = L_recon + β_kl · L_KL
    """
    L_recon = F.mse_loss(out["recon"], seq)

    # Analytical KL between two Gaussians:
    # KL(N(μ_post,σ_post) ‖ N(μ_prior,σ_prior))
    var_p = out["sig_post"].pow(2)
    var_q = out["sig_prior"].pow(2)
    kl = 0.5 * (
        (var_p / (var_q + 1e-6))
        + ((out["mu_post"] - out["mu_prior"]).pow(2) / (var_q + 1e-6))
        - 1.0
        + (var_q + 1e-6).log() - (var_p + 1e-6).log()
    ).mean()

    L_total = L_recon + beta_kl * kl
    return L_total, {
        "L_recon": L_recon.item(),
        "L_KL":    kl.item(),
        "L_total": L_total.item(),
    }


# ── Trainer ────────────────────────────────────────────────────────────────────

class WorldModelTrainer:

    def __init__(self, cfg: WorldModelConfig):
        self.cfg    = cfg
        self.device = torch.device("cpu")
        torch.manual_seed(cfg.seed)

    def run(self) -> None:
        cfg = self.cfg
        log.info("=" * 60)
        log.info("  Module 4 — World Model (RSSM)")
        log.info("=" * 60)
        log.info(f"  Config : {cfg}")

        # ── Data ─────────────────────────────────────────────────────────────
        log.info("  Generating synthetic disaster-state sequences …")
        seqs = _make_sequences(cfg.n_sequences, cfg.n_steps_train,
                               cfg.d_disaster_state, cfg.seed)
        n_tr = int(len(seqs) * 0.8)
        tr_seqs, va_seqs = seqs[:n_tr], seqs[n_tr:]

        tr_loader = DataLoader(SequenceDataset(tr_seqs),
                               batch_size=cfg.batch_size, shuffle=True)
        va_loader = DataLoader(SequenceDataset(va_seqs),
                               batch_size=cfg.batch_size, shuffle=False)
        log.info(f"  Train: {len(tr_seqs)} | Val: {len(va_seqs)} sequences")

        # ── Model ─────────────────────────────────────────────────────────────
        model = WorldModel(cfg).to(self.device)
        log.info(f"  Model  : {model.count_parameters():,} trainable parameters")

        optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
        )

        best_val, train_log = float("inf"), []
        print()
        print(f"  {'Ep':>4}  {'L_train':>9}  {'L_recon':>9}  {'L_KL':>8}  {'L_val':>9}  {'t(s)':>6}")
        print("  " + "─" * 56)

        for epoch in range(1, cfg.n_epochs + 1):
            t0 = time.time()
            model.train()
            ep = {"L_total": 0.0, "L_recon": 0.0, "L_KL": 0.0}
            nb = 0
            for batch in tr_loader:
                batch = batch.to(self.device)           # (B, T, d_s)
                B = batch.shape[0]
                batch_loss = torch.tensor(0.0, device=self.device)
                for b in range(B):
                    out = model(batch[b])               # process one sequence
                    l, info = rssm_loss(out, batch[b], cfg.beta_kl)
                    batch_loss = batch_loss + l
                    for k in ep: ep[k] += info[k]
                    nb += 1
                optim.zero_grad()
                (batch_loss / B).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            for k in ep: ep[k] /= max(nb, 1)

            model.eval()
            va_total, va_n = 0.0, 0
            with torch.no_grad():
                for batch in va_loader:
                    batch = batch.to(self.device)
                    for b in range(batch.shape[0]):
                        out = model(batch[b])
                        l, _ = rssm_loss(out, batch[b], cfg.beta_kl)
                        va_total += l.item(); va_n += 1
            va_loss = va_total / max(va_n, 1)

            scheduler.step()
            elapsed = time.time() - t0
            better  = va_loss < best_val
            if better:
                best_val = va_loss
                self._save_checkpoint(model, epoch, va_loss)

            print(
                f"  {epoch:4d}  {ep['L_total']:9.4f}  {ep['L_recon']:9.4f}  "
                f"{ep['L_KL']:8.4f}  {va_loss:9.4f}  {elapsed:6.1f}"
                + (" ★" if better else "")
            )
            train_log.append({
                "epoch":    epoch,
                "L_train":  round(ep["L_total"], 6),
                "L_recon":  round(ep["L_recon"], 6),
                "L_KL":     round(ep["L_KL"], 6),
                "val_loss": round(va_loss, 6),
                "lr":       scheduler.get_last_lr()[0],
            })

        self._save_metrics(train_log)

        print()
        print("═" * 60)
        print("  Module 4 — World Model Training Complete")
        print("═" * 60)
        print(f"  Config      : {cfg}")
        print(f"  Parameters  : {model.count_parameters():,}")
        print(f"  Best val    : {best_val:.6f}")
        print(f"  Checkpoint  : checkpoints/world_model/worldmodel_best.pt")
        print(f"  Metrics     : metrics/world_model/train_log.csv")
        print("═" * 60)

    def _save_checkpoint(
        self, model: WorldModel, epoch: int, val_loss: float
    ) -> None:
        os.makedirs("checkpoints/world_model", exist_ok=True)
        torch.save(
            {"epoch": epoch, "state": model.state_dict(),
             "val_loss": val_loss, "config": vars(self.cfg)},
            "checkpoints/world_model/worldmodel_best.pt",
        )

    def _save_metrics(self, rows: List[Dict]) -> None:
        os.makedirs("metrics/world_model", exist_ok=True)
        path = "metrics/world_model/train_log.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  Metrics saved → {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Module 4 — World Model")
    p.add_argument("--demo",    action="store_true")
    p.add_argument("--epochs",  type=int, default=None)
    args = p.parse_args()

    cfg = WorldModelConfig()
    if args.demo:   cfg.apply_demo_overrides()
    if args.epochs: cfg.n_epochs = args.epochs

    WorldModelTrainer(cfg).run()


if __name__ == "__main__":
    main()
