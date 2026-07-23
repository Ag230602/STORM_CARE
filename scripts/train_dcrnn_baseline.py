"""
Train the DCRNN track baseline and save its checkpoint, matching the exact
paths/config/split convention already used for baseline_lstm.pt,
baseline_transformer.pt, and main_gno_dyngnn.pt.

Run:
    .venv/bin/python scripts/train_dcrnn_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import benchmark
from model import track_pipeline_unified_X as pipeline


def main() -> None:
    metrics_dir = Path(benchmark.PROJECT_ROOT) / "metrics"
    cfg = benchmark.configure_paths(metrics_dir)
    pipeline.seed_all(cfg.seed)
    pipeline.ensure_dirs()

    samples = benchmark.rebuild_samples()
    tr_idx, val_idx, te_idx = pipeline.split_sample_indices(len(samples), seed=cfg.seed)

    tr_ds = pipeline.TrackDataset([samples[i] for i in tr_idx])
    val_ds = pipeline.TrackDataset([samples[i] for i in val_idx])
    te_ds = pipeline.TrackDataset([samples[i] for i in te_idx])
    print(f"Split: train={len(tr_ds)} val={len(val_ds)} test={len(te_ds)}")

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

    dcrnn = pipeline.DCRNNTrackBaseline(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        grid_size=cfg.grid_size,
        use_meta=cfg.include_metadata,
    )
    metrics = pipeline.train_prob_model(
        dcrnn, tr_loader, val_loader, te_loader, cfg.epochs_baseline, "baseline_dcrnn"
    )
    pipeline.save_metrics_row(
        "DCRNN (past + ERA5)", metrics, str(metrics_dir / "track_metrics_dcrnn.csv")
    )
    print("Saved checkpoint:", Path(cfg.ckpt_dir) / "baseline_dcrnn.pt")


if __name__ == "__main__":
    main()
