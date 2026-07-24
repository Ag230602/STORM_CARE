"""
Module 5 — Counterfactual Reasoning Engine runner.

1. Loads (or trains) the WorldModel checkpoint from Module 4.
2. Generates a synthetic warm-up disaster-state sequence.
3. Runs all counterfactual scenarios + baseline on the held-out test split.
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
    p.add_argument("--n-test", type=int, default=None,
                   help="Optional cap on held-out test sequences; default uses all")
    p.add_argument("--metrics-dir", type=str, default=None)
    args = p.parse_args()

    cf_cfg = CounterfactualConfig()
    wm_cfg = WorldModelConfig()
    if args.demo:
        cf_cfg.apply_demo_overrides()
        wm_cfg.apply_demo_overrides()
        # Keep dims consistent
        cf_cfg.d_disaster_state = wm_cfg.d_disaster_state
        cf_cfg.d_latent         = wm_cfg.d_latent
    if args.n_test is not None:
        cf_cfg.n_test_sequences = args.n_test
    if args.metrics_dir:
        cf_cfg.metrics_dir = args.metrics_dir

    log.info("=" * 60)
    log.info("  Module 5 — Counterfactual Reasoning Engine")
    log.info("=" * 60)

    # ── Load WorldModel ───────────────────────────────────────────────────────
    world_model, wm_cfg_loaded = _load_or_train_world_model(wm_cfg, args.demo)

    # If the available checkpoint is demo-sized, keep counterfactual rollout
    # dimensions aligned even when the user did not pass --demo.
    if wm_cfg_loaded.demo and not cf_cfg.demo:
        cf_cfg.apply_demo_overrides()

    # Update cf_cfg dims to match loaded model
    d_s    = wm_cfg_loaded.d_disaster_state
    cf_cfg.d_disaster_state = d_s
    cf_cfg.d_latent         = wm_cfg_loaded.d_latent
    T_warm = min(cf_cfg.n_initial_steps, wm_cfg_loaded.n_steps_train)
    log.info(f"  Config : {cf_cfg}")

    # ── Generate complete held-out test split ────────────────────────────────
    all_seqs = _make_sequences(
        wm_cfg_loaded.n_sequences,
        wm_cfg_loaded.n_steps_train,
        d_s,
        seed=wm_cfg_loaded.seed,
    )
    split_idx = int(len(all_seqs) * 0.8)
    test_seqs = all_seqs[split_idx:]
    if cf_cfg.n_test_sequences is not None:
        test_seqs = test_seqs[:cf_cfg.n_test_sequences]
    warm_up_seqs = [seq[:T_warm] for seq in test_seqs]
    n_storms = len(warm_up_seqs)
    log.info(f"  Running counterfactuals on {n_storms} held-out test sequences …")

    # ── Run RSSM-mediated counterfactuals over all test storms ───────────────
    engine  = CounterfactualEngine(world_model, cf_cfg)
    results, per_sequence_rows = engine.compare_multi_storm(
        warm_up_seqs, return_per_sequence=True)
    mirror_checks = engine.direct_mirror_diagnostics(results)

    # ── Print report ──────────────────────────────────────────────────────────
    CounterfactualEngine.print_report(results, n_storms=n_storms)

    # ── Save metrics ──────────────────────────────────────────────────────────
    os.makedirs(cf_cfg.metrics_dir, exist_ok=True)
    rows = []
    for name, res in results.items():
        row = {"scenario": name, "description": res["description"]}
        row.update(res["metrics"])
        row["n_test_sequences"] = res.get("n_storms", n_storms)
        rows.append(row)

    out_path = os.path.join(cf_cfg.metrics_dir, "counterfactual_outcomes.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    diag_path = os.path.join(cf_cfg.metrics_dir, "counterfactual_mirror_diagnostics.csv")
    with open(diag_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mirror_checks[0].keys()))
        w.writeheader(); w.writerows(mirror_checks)

    long_path = os.path.join(cf_cfg.metrics_dir, "counterfactual_outcomes_long.csv")
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sequence_rows[0].keys()))
        w.writeheader(); w.writerows(per_sequence_rows)

    log.info(f"  Outcomes saved → {out_path}")
    log.info(f"  Mirror diagnostics saved → {diag_path}")
    log.info(f"  Per-sequence outcomes saved → {long_path}")
    log.info("  Module 5 complete.")


if __name__ == "__main__":
    main()
