#!/usr/bin/env python
"""E1 — Full held-out test-set track benchmark orchestrator.

Drives the corrected pipeline at test-partition scale and emits the
long-format error CSV that scripts/compute_significance.py consumes, plus
the replacement for metrics/inference_test_metrics_summary.csv.

ADAPTERS REQUIRED: train_model / evaluate_model must call the repo's
existing corrected training/eval code. The orchestration, config logging,
fallback mode, and output contract are functional as written.

Protocol locked in by this script (do not weaken):
  * Train on ERA5-complete windows of TRAIN-partition storms only.
  * Checkpoint selection on VAL-partition windows by mean track error only.
  * One final evaluation pass on TEST-partition windows. No test-set
    peeking for tuning; the tuning budget is fixed up front and logged.
  * Same normalized inputs for all learned models (t0 in history, z-scored
    ERA5, lat/90 lon/180, displacement targets decoded to lat/lon).

Modes:
  --mode full      per-model hyperparameter search with --n-trials trials
  --mode frozen    the documented compute-limited fallback: one fixed,
                   pre-registered config per model, zero search
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODELS = ["Persistence", "CLIPER", "LSTM", "Transformer", "GNO+DynGNN",
          "STORM-CARE-FM"]
HORIZONS_H = [6, 12, 24, 48]      # extend to 72/120 iff window builder supports
FROZEN_CONFIGS = {                 # pre-register BEFORE looking at test data
    "LSTM":          {"hidden": 128, "layers": 2, "lr": 1e-3},
    "Transformer":   {"d_model": 128, "heads": 4, "layers": 3, "lr": 3e-4},
    "GNO+DynGNN":    {"width": 64, "modes": 12, "lr": 5e-4},
    "STORM-CARE-FM": {"from_checkpoint":
                      "checkpoints/foundation/foundation_best.pt"},
}  # TODO(Adrija): replace with the configs actually used in the corrected
   # Irma/Ian rerun, so 'frozen' means 'frozen at known-good', not guessed.


# ----------------------- ADAPTERS: wire these ----------------------------
def build_datasets(manifest_csv: Path):
    """Load train/val/test window datasets restricted to ERA5-complete
    windows per the E1 coverage manifest.
    TODO: call the corrected loader in model/track_pipeline_unified_X.py.
    Returns (train_ds, val_ds, test_ds)."""
    raise NotImplementedError


def train_model(name: str, config: dict, train_ds, val_ds) -> dict:
    """Train one learned model with validation-mean-track-error selection.
    Persistence/CLIPER: return {} (no training).
    TODO: call the repo's corrected trainers.
    Returns a run-info dict incl. checkpoint path + selection score."""
    raise NotImplementedError


def evaluate_model(name: str, run_info: dict, test_ds) -> pd.DataFrame:
    """One pass over test windows. Must return long-format rows:
        storm_id, t0, model, horizon, error   (error = great-circle km,
        decoded from normalized displacements)
    TODO: call the repo's corrected inference + metric code."""
    raise NotImplementedError
# -------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="metrics/test_coverage/test_coverage_manifest.csv")
    ap.add_argument("--mode", choices=["full", "frozen"], default="frozen")
    ap.add_argument("--n-trials", type=int, default=10,
                    help="tuning trials per learned model in --mode full; "
                         "logged verbatim for the manuscript's tuning-budget "
                         "promise")
    ap.add_argument("--out-dir", default="metrics/full_test_benchmark")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / args.manifest
    if not manifest.exists():
        raise SystemExit("Run scripts/audit_test_coverage.py first — the "
                         "benchmark must consume the audited manifest, not "
                         "re-derive coverage.")

    run_log = {"mode": args.mode, "n_trials": args.n_trials,
               "seed": args.seed, "horizons_h": HORIZONS_H,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
               "models": {}}

    train_ds, val_ds, test_ds = build_datasets(manifest)

    all_rows = []
    for name in MODELS:
        t0 = time.time()
        if args.mode == "frozen" or name in ("Persistence", "CLIPER"):
            cfg = FROZEN_CONFIGS.get(name, {})
            info = train_model(name, cfg, train_ds, val_ds)
        else:
            # TODO: plug in the sweep (e.g. optuna / grid) selecting purely
            # on validation mean track error; store every trial's config +
            # val score in run_log for the appendix table.
            raise NotImplementedError("full-sweep mode: wire tuner here")
        errs = evaluate_model(name, info, test_ds)
        errs["model"] = name
        all_rows.append(errs)
        run_log["models"][name] = {"config": cfg, "run_info":
                                   {k: str(v) for k, v in (info or {}).items()},
                                   "wallclock_s": round(time.time() - t0, 1)}

    long_df = pd.concat(all_rows, ignore_index=True)
    long_df.to_csv(out / "test_errors_long.csv", index=False)

    summary = (long_df.groupby(["model", "horizon"], as_index=False)["error"]
               .mean().pivot(index="model", columns="horizon", values="error"))
    summary["mean"] = summary.mean(axis=1)
    summary.to_csv(out / "inference_test_metrics_summary.csv")

    with open(out / "benchmark_run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    print(summary.round(3))
    print(f"\nlong-format errors -> {out/'test_errors_long.csv'}")
    print("Next: python scripts/compute_significance.py "
          f"--input {out/'test_errors_long.csv'} --reference Persistence")


if __name__ == "__main__":
    main()
