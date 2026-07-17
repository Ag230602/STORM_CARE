"""
DisasterGraphTrainer — training loop for Module 3.

Run:
    python -m model.disaster_graph.train --demo
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DisasterGraphConfig
from .schema import build_dataset, DisasterScenario, generate_humanitarian_report, humanitarian_targets
from .architecture import DisasterGNN

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)


class DisasterGraphTrainer:

    def __init__(self, cfg: DisasterGraphConfig):
        self.cfg    = cfg
        self.device = torch.device("cpu")
        torch.manual_seed(cfg.seed)

    def run(self) -> None:
        cfg = self.cfg
        log.info("=" * 60)
        log.info("  Module 3 — Dynamic Disaster Graph")
        log.info("=" * 60)
        log.info(f"  Config : {cfg}")

        # ── Data ─────────────────────────────────────────────────────────────
        log.info("  Building synthetic disaster scenarios …")
        all_sc = build_dataset(cfg, seed=cfg.seed)
        split  = int(len(all_sc) * 0.8)
        tr_sc, va_sc = all_sc[:split], all_sc[split:]
        log.info(f"  Train: {len(tr_sc)} scenarios | Val: {len(va_sc)} scenarios")

        # ── Model ─────────────────────────────────────────────────────────────
        model = DisasterGNN(cfg).to(self.device)
        log.info(f"  Model  : {model.count_parameters():,} trainable parameters")

        optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
        )

        # ── Training loop ─────────────────────────────────────────────────────
        best_val  = float("inf")
        train_log = []

        print()
        print(f"  {'Ep':>4}  {'L_train':>9}  {'L_val':>9}  {'t(s)':>6}")
        print("  " + "─" * 36)

        for epoch in range(1, cfg.n_epochs + 1):
            t0 = time.time()
            model.train()
            tr_loss = self._run_epoch(model, tr_sc)

            model.eval()
            with torch.no_grad():
                va_loss = self._run_epoch(model, va_sc)

            scheduler.step()
            elapsed = time.time() - t0
            better  = va_loss < best_val
            if better:
                best_val = va_loss
                self._save_checkpoint(model, epoch, {"val_loss": va_loss})

            print(
                f"  {epoch:4d}  {tr_loss:9.4f}  {va_loss:9.4f}  {elapsed:6.1f}"
                + (" ★" if better else "")
            )
            train_log.append({"epoch": epoch, "train_loss": tr_loss,
                               "val_loss": va_loss, "lr": scheduler.get_last_lr()[0]})

        self._save_metrics(train_log)
        log.info(f"  Best val loss: {best_val:.4f}")
        log.info("  Checkpoints → checkpoints/disaster_graph/")
        log.info("  Metrics     → metrics/disaster_graph/")

    def _run_epoch(
        self,
        model: DisasterGNN,
        scenarios: List,
    ) -> float:
        """Run one full epoch (all scenarios, all time steps). Returns mean loss."""
        total, n = 0.0, 0
        for sc_steps in scenarios:
            for sc in sc_steps:
                sc = self._to_device(sc)
                out  = model(sc)
                loss = F.mse_loss(out["damage_scores"], sc.targets)
                if model.training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # Accumulate gradients across the scenario, step once per scenario step
                    list(model.parameters())[0].grad  # just access to trigger
                total += loss.item()
                n     += 1
        return total / max(n, 1)

    def _to_device(self, sc: DisasterScenario) -> DisasterScenario:
        from dataclasses import replace
        return DisasterScenario(
            node_features = sc.node_features.to(self.device),
            node_types    = sc.node_types.to(self.device),
            edge_index    = sc.edge_index.to(self.device),
            edge_types    = sc.edge_types.to(self.device),
            targets       = sc.targets.to(self.device),
            storm_pos     = sc.storm_pos,
        )

    def _loss(self, model: DisasterGNN, sc: DisasterScenario) -> tuple[torch.Tensor, Dict[str, float]]:
        out = model(sc)
        tgt = humanitarian_targets(self.cfg, sc)
        losses = {
            "damage": F.mse_loss(out["damage_scores"], sc.targets),
            "child": F.mse_loss(out["child_exposure"], tgt["child_exposure_frac"].to(self.device)),
            "school": F.binary_cross_entropy(
                out["school_disruption"].clamp(1e-6, 1.0 - 1e-6),
                tgt["school_disrupted"].to(self.device),
            ),
            "hospital": F.mse_loss(out["hospital_access"], tgt["hospital_access"].to(self.device)),
            "shelter": F.mse_loss(out["shelter_demand"], tgt["shelter_demand"].to(self.device)),
            "priority": F.mse_loss(out["recovery_priority"], tgt["recovery_priority"].to(self.device)),
        }
        total = (
            losses["damage"]
            + 0.5 * losses["child"]
            + 0.25 * losses["school"]
            + 0.5 * losses["hospital"]
            + 0.25 * losses["shelter"]
            + 0.25 * losses["priority"]
        )
        return total, {k: float(v.detach().cpu()) for k, v in losses.items()}

    def _save_checkpoint(
        self, model: DisasterGNN, epoch: int, metrics: Dict
    ) -> None:
        os.makedirs("checkpoints/disaster_graph", exist_ok=True)
        torch.save(
            {"epoch": epoch, "state": model.state_dict(),
             "metrics": metrics, "config": vars(self.cfg)},
            "checkpoints/disaster_graph/disaster_gnn_best.pt",
        )

    def _save_metrics(self, log_rows: List[Dict]) -> None:
        os.makedirs("metrics/disaster_graph", exist_ok=True)
        path = "metrics/disaster_graph/train_log.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader(); w.writerows(log_rows)
        print(f"  Metrics saved → {path}")


# ── Optimiser wrapper so we step per-scenario ─────────────────────────────────
# The _run_epoch above accumulates grads; we need to call optim.step() properly.
# Patch the trainer to do per-step optimisation instead.

class _PatchedTrainer(DisasterGraphTrainer):
    """Corrected trainer: per-sample gradient step."""

    def run(self) -> None:
        cfg = self.cfg
        log.info("=" * 60)
        log.info("  Module 3 — Dynamic Disaster Graph")
        log.info("=" * 60)
        log.info(f"  Config : {cfg}")

        log.info("  Building synthetic disaster scenarios …")
        all_sc = build_dataset(cfg, seed=cfg.seed)
        split  = int(len(all_sc) * 0.8)
        tr_sc, va_sc = all_sc[:split], all_sc[split:]
        log.info(f"  Train: {len(tr_sc)} scenarios ({cfg.n_steps} steps each) | "
                 f"Val: {len(va_sc)} scenarios")

        model = DisasterGNN(cfg).to(self.device)
        log.info(f"  Model  : {model.count_parameters():,} trainable parameters")

        optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
        )

        best_val, train_log = float("inf"), []
        print()
        print(f"  {'Ep':>4}  {'L_train':>9}  {'L_val':>9}  {'lr':>9}  {'t(s)':>6}")
        print("  " + "─" * 44)

        for epoch in range(1, cfg.n_epochs + 1):
            t0 = time.time()
            model.train()
            tr_total, tr_n = 0.0, 0
            for sc_steps in tr_sc:
                for sc in sc_steps:
                    sc_d = self._to_device(sc)
                    loss, _ = self._loss(model, sc_d)
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    tr_total += loss.item(); tr_n += 1

            tr_loss = tr_total / max(tr_n, 1)

            model.eval()
            va_total, va_n = 0.0, 0
            with torch.no_grad():
                for sc_steps in va_sc:
                    for sc in sc_steps:
                        sc_d = self._to_device(sc)
                        loss, _ = self._loss(model, sc_d)
                        va_total += loss.item()
                        va_n += 1
            va_loss = va_total / max(va_n, 1)

            scheduler.step()
            lr_now  = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            better  = va_loss < best_val
            if better:
                best_val = va_loss
                self._save_checkpoint(model, epoch, {"val_loss": va_loss})

            print(
                f"  {epoch:4d}  {tr_loss:9.4f}  {va_loss:9.4f}  {lr_now:9.2e}  {elapsed:6.1f}"
                + (" ★" if better else "")
            )
            train_log.append({"epoch": epoch, "train_loss": round(tr_loss, 6),
                               "val_loss": round(va_loss, 6), "lr": lr_now})

        self._save_metrics(train_log)

        print()
        print("═" * 60)
        print("  Module 3 — Training Complete")
        print("═" * 60)
        print(f"  Config      : {cfg}")
        print(f"  Parameters  : {model.count_parameters():,}")
        print(f"  Best val MSE: {best_val:.6f}")
        print(f"  Checkpoint  : checkpoints/disaster_graph/disaster_gnn_best.pt")
        print(f"  Metrics     : metrics/disaster_graph/train_log.csv")

        # ── Sample humanitarian outputs on one val scenario ───────────────────
        model.eval()
        sample_steps = va_sc[0]
        sc_last = self._to_device(sample_steps[-1])
        with torch.no_grad():
            out = model(sc_last)
        nf = sample_steps[-1].node_features.numpy()
        report = generate_humanitarian_report(cfg, out, nf)

        print()
        print("  ── Sample Humanitarian Outputs (last step, scenario 1) ──────")
        print(f"    Meteorological:")
        print(f"      Peak wind speed (estimated)   : {report['wind_field_max_ms']:.1f} m/s")
        hmap = report['hazard_map']
        print(f"      Hazard map shape              : {hmap.shape[0]}×{hmap.shape[1]} grid")
        print(f"      Hazard map max                : {hmap.max():.3f}")
        print(f"    Humanitarian:")
        print(f"      Exposed children (estimated)  : {report['exposed_children_est']:,}")
        print(f"      School disruption             : {report['school_disruption_pct']:.1f}% of schools disrupted")
        print(f"      Hospital accessibility        : {report['hospital_access_avg']:.3f} (1=fully accessible)")
        print(f"      Shelter demand pressure       : {report['shelter_demand_avg']:.3f} (1=at capacity)")
        print(f"      Shelters at/over capacity     : {report['shelter_at_capacity']}")
        print(f"    Recovery priority zones:")
        for i, label in enumerate(report['top3_priority_labels'], 1):
            print(f"      #{i}: {label}")
        print("═" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Module 3 — Dynamic Disaster Graph")
    p.add_argument("--demo",        action="store_true")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--n-scenarios", type=int, default=None)
    args = p.parse_args()

    cfg = DisasterGraphConfig()
    if args.demo:       cfg.apply_demo_overrides()
    if args.epochs:     cfg.n_epochs    = args.epochs
    if args.n_scenarios: cfg.n_scenarios = args.n_scenarios

    _PatchedTrainer(cfg).run()


if __name__ == "__main__":
    main()
