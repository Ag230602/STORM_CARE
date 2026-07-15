"""
Module 5 — Counterfactual Reasoning Engine runner.

1. Loads (or trains) the WorldModel checkpoint from Module 4.
2. Generates a synthetic warm-up disaster-state sequence.
3. Runs all 5 counterfactual scenarios + baseline.
4. Prints a formatted comparison table.
5. Saves outcome metrics to metrics/counterfactual/.

Run:
    python -m model.counterfactual.run --demo
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import torch

from .config import CounterfactualConfig
from .engine import CounterfactualEngine
from ..world_model.config import WorldModelConfig
from ..world_model.architecture import WorldModel
from ..world_model.train import WorldModelTrainer, _make_sequences

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)

CKPT_PATH = "checkpoints/world_model/worldmodel_best.pt"


def _load_or_train_world_model(
    wm_cfg: WorldModelConfig,
    demo:   bool,
) -> WorldModel:
    """Load checkpoint if available, otherwise train a fresh WorldModel."""
    if os.path.exists(CKPT_PATH):
        log.info(f"  Loading WorldModel from {CKPT_PATH}")
        ckpt  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        # Re-instantiate with saved config
        saved_cfg = WorldModelConfig(**{
            k: v for k, v in ckpt["config"].items()
            if k in WorldModelConfig.__dataclass_fields__
        })
        model = WorldModel(saved_cfg)
        model.load_state_dict(ckpt["state"])
        model.eval()
        return model, saved_cfg
    else:
        log.info("  WorldModel checkpoint not found — training from scratch …")
        wm_cfg_run = WorldModelConfig()
        if demo:
            wm_cfg_run.apply_demo_overrides()
        WorldModelTrainer(wm_cfg_run).run()
        return _load_or_train_world_model(wm_cfg_run, demo)


def main() -> None:
    p = argparse.ArgumentParser(description="Module 5 — Counterfactual Reasoning")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    cf_cfg = CounterfactualConfig()
    wm_cfg = WorldModelConfig()
    if args.demo:
        cf_cfg.apply_demo_overrides()
        wm_cfg.apply_demo_overrides()
        # Keep dims consistent
        cf_cfg.d_disaster_state = wm_cfg.d_disaster_state
        cf_cfg.d_latent         = wm_cfg.d_latent

    log.info("=" * 60)
    log.info("  Module 5 — Counterfactual Reasoning Engine")
    log.info("=" * 60)
    log.info(f"  Config : {cf_cfg}")

    # ── Load WorldModel ───────────────────────────────────────────────────────
    world_model, wm_cfg_loaded = _load_or_train_world_model(wm_cfg, args.demo)

    # ── Generate N warm-up sequences (one per test storm) ────────────────────
    N_test = 5
    T_warm = cf_cfg.n_initial_steps
    d_s    = wm_cfg_loaded.d_disaster_state
    log.info(f"  Generating {N_test} test-storm warm-up sequences …")
    warm_up_seqs = []
    for i in range(N_test):
        seq = _make_sequences(1, T_warm, d_s, seed=cf_cfg.seed + i * 7)
        warm_up_seqs.append(seq[0])          # (T_warm, d_s)

    # Update cf_cfg dims to match loaded model
    cf_cfg.d_disaster_state = d_s
    cf_cfg.d_latent         = wm_cfg_loaded.d_latent

    # ── Run analytic counterfactuals over all test storms ────────────────────
    engine  = CounterfactualEngine(world_model, cf_cfg)
    results = engine.compare_analytic_multi_storm(warm_up_seqs)
    n_storms = N_test

    # ── Print report ──────────────────────────────────────────────────────────
    CounterfactualEngine.print_report(results, n_storms=n_storms)

    # ── Save metrics ──────────────────────────────────────────────────────────
    os.makedirs(cf_cfg.metrics_dir, exist_ok=True)
    rows = []
    for name, res in results.items():
        row = {"scenario": name, "description": res["description"]}
        row.update(res["metrics"])
        rows.append(row)

    out_path = os.path.join(cf_cfg.metrics_dir, "counterfactual_outcomes.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    log.info(f"  Outcomes saved → {out_path}")
    log.info("  Module 5 complete.")


if __name__ == "__main__":
    main()
