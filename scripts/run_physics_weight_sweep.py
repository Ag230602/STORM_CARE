#!/usr/bin/env python
"""E5 — Physics constraint-weight sweep.

Runs model.physics.train at several global physics-weight scales plus the
matched no-physics ablation, then collates final-epoch validation metrics
into one decision table.

MOSTLY FUNCTIONAL: it shells out to the existing trainer exactly like the
audited commands did. One adapter point: the trainer must accept a
--physics-weight-scale argument (a one-line change in model/physics/config.py
+ argparse: multiply every beta_k by the scale). Until that flag exists,
pass --dry-run to print the commands for manual editing.

Decision rule this produces evidence for (see plan E5):
  * if some scale s matches no-physics track RMSE (within the run-to-run
    noise you observe across seeds) while keeping residual improvements
    -> upgraded RQ2 claim ("consistency at no accuracy cost at s=...").
  * else -> keep the reframed tradeoff claim, now backed by a sweep.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")

SCALES = [0.1, 0.3, 1.0, 3.0]
METRIC_COLS = ["val_track_rmse", "val_L_data", "val_R_adv", "val_R_diff",
               "val_R_mass", "val_R_wp", "val_R_cont", "val_R_nrg"]


def run_one(tag: str, extra_args: list[str], epochs: int, dry: bool) -> Path:
    mdir = ROOT / "metrics" / "physics_sweep" / tag
    cdir = ROOT / "checkpoints" / "physics_sweep" / tag
    cmd = [PY, "-m", "model.physics.train", "--demo", "--epochs", str(epochs),
           "--metrics-dir", str(mdir), "--checkpoint-dir", str(cdir),
           *extra_args]
    print(" ".join(cmd))
    if not dry:
        mdir.mkdir(parents=True, exist_ok=True)
        cdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
    return mdir


def final_val_row(mdir: Path) -> dict:
    """Grab the last validation row from pigno_val_metrics.csv."""
    path = mdir / "pigno_val_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df.iloc[-1].to_dict()
    return {k: row.get(k) for k in METRIC_COLS if k in row} | \
           {"n_epochs_logged": len(df)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20,
                    help="demo-scale epoch count (matches the audited "
                         "20-epoch demo protocol used elsewhere in this repo)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="add 2-3 seeds if time allows, to estimate the "
                         "run-to-run noise the decision rule needs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        seed_args = ["--seed", str(seed)]
        m = run_one(f"no_physics_s{seed}", ["--no-physics", *seed_args],
                    args.epochs, args.dry_run)
        rows.append({"variant": "no_physics", "scale": 0.0, "seed": seed,
                     **(final_val_row(m) if not args.dry_run else {})})
        for s in SCALES:
            m = run_one(f"scale{s}_s{seed}",
                        ["--physics-weight-scale", str(s), *seed_args],
                        args.epochs, args.dry_run)
            rows.append({"variant": "physics", "scale": s, "seed": seed,
                         **(final_val_row(m) if not args.dry_run else {})})

    if args.dry_run:
        print("\n(dry run: no collation)")
        return
    out = ROOT / "metrics" / "physics_sweep" / "sweep_decision_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\ndecision table -> {out}")
    print("Disclose the CFL / Jacobian caveat for whichever configuration "
          "is reported in the paper.")


if __name__ == "__main__":
    main()
