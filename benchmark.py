from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from model import track_pipeline_unified_X as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_ROOT = PROJECT_ROOT / "your-repo" / "data" / "data"
SUMMARY_SCRIPT = PROJECT_ROOT / "your-repo" / "src" / "evaluation" / "Summary.py"


def configure_paths(metrics_dir: Path) -> pipeline.CFG:
    cfg = pipeline.cfg
    cfg.base = str(LOCAL_DATA_ROOT)
    cfg.irma_track = str(LOCAL_DATA_ROOT / "processed" / "tracks" / "irma_2017_hurdat2.csv")
    cfg.ian_track = str(LOCAL_DATA_ROOT / "processed" / "tracks" / "ian_2022_hurdat2.csv")
    cfg.irma_era5 = str(LOCAL_DATA_ROOT / "raw" / "era5" / "irma_2017" / "era5_pl_irma_2017.nc")
    cfg.ian_era5 = str(LOCAL_DATA_ROOT / "raw" / "era5" / "ian_2022" / "era5_pl_ian_2022.nc")
    cfg.out_root = str(PROJECT_ROOT)
    cfg.ckpt_dir = str(PROJECT_ROOT / "checkpoints")
    cfg.metrics_dir = str(metrics_dir)
    return cfg


def build_train_test_split(samples: List[Dict], train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_train = int(len(samples) * train_ratio)
    return idx[:n_train], idx[n_train:]


def rebuild_samples() -> List[Dict]:
    cfg = pipeline.cfg
    irma_df = pipeline.parse_track(cfg.irma_track, "irma")
    ian_df = pipeline.parse_track(cfg.ian_track, "ian")
    irma_ds = pipeline.open_era5(cfg.irma_era5)
    ian_ds = pipeline.open_era5(cfg.ian_era5)

    print("Building benchmark samples (Irma)...")
    s1 = pipeline.build_samples(irma_df, irma_ds)
    print("Building benchmark samples (Ian)...")
    s2 = pipeline.build_samples(ian_df, ian_ds)

    samples = s1 + s2
    print(f"Total samples available: {len(samples)}")
    return samples


def load_test_dataset(test_ratio: float) -> Tuple[pipeline.TrackDataset, List[Dict]]:
    cfg = pipeline.cfg
    samples = rebuild_samples()
    train_ratio = 1.0 - test_ratio
    _, test_idx = build_train_test_split(samples, train_ratio=train_ratio, seed=cfg.seed)
    test_samples = [samples[i] for i in test_idx]
    print(f"Benchmark split: train={len(samples) - len(test_samples)} test={len(test_samples)}")
    return pipeline.TrackDataset(test_samples), test_samples


def load_model(model_name: str) -> torch.nn.Module:
    cfg = pipeline.cfg
    if model_name == "LSTM":
        model = pipeline.LSTMTrackBaseline(
            feat_ch=len(cfg.features),
            leads=len(cfg.lead_hours),
            use_meta=cfg.include_metadata,
        )
        checkpoint_name = "baseline_lstm.pt"
    elif model_name == "Transformer":
        model = pipeline.TransformerTrackBaseline(
            feat_ch=len(cfg.features),
            leads=len(cfg.lead_hours),
            use_meta=cfg.include_metadata,
        )
        checkpoint_name = "baseline_transformer.pt"
    elif model_name == "GNO+DynGNN":
        model = pipeline.GNO_DynGNN(
            feat_ch=len(cfg.features),
            leads=len(cfg.lead_hours),
            use_meta=cfg.include_metadata,
        )
        checkpoint_name = "main_gno_dyngnn.pt"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = Path(cfg.ckpt_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state["state"])
    model.to(cfg.device)
    model.eval()
    return model


def _sample_key(sample: Dict) -> Tuple[str, str]:
    return str(sample["storm_tag"]), str(sample["t0"])


def _base_prediction_row(sample: Dict) -> Dict[str, object]:
    row: Dict[str, object] = {
        "storm_tag": sample["storm_tag"],
        "t0": sample["t0"],
        "lat0": float(sample["lat0"]),
        "lon0": float(sample["lon0"]),
    }
    for idx, hour in enumerate(pipeline.cfg.lead_hours):
        row[f"true_lat_{hour}h"] = float(sample["y_abs"][idx, 0])
        row[f"true_lon_{hour}h"] = float(sample["y_abs"][idx, 1])
    return row


def predict_probabilistic_rows(model: torch.nn.Module, loader, base_rows: Dict[Tuple[str, str], Dict[str, object]]) -> List[Dict[str, object]]:
    cfg = pipeline.cfg
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for past, X, meta, y, info in loader:
            storm_tag, t0, _, _ = info
            past = past.to(cfg.device)
            X = X.to(cfg.device)
            meta = meta.to(cfg.device)
            mu, sigma = model(past, X, meta)

            mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()

            for batch_index in range(mu_np.shape[0]):
                key = (str(storm_tag[batch_index]), str(t0[batch_index]))
                row = dict(base_rows[key])
                for lead_index, hour in enumerate(cfg.lead_hours):
                    row[f"pred_mu_lat_{hour}h"] = float(mu_np[batch_index, lead_index, 0])
                    row[f"pred_mu_lon_{hour}h"] = float(mu_np[batch_index, lead_index, 1])
                    row[f"pred_sigma_lat_{hour}h"] = float(sigma_np[batch_index, lead_index, 0])
                    row[f"pred_sigma_lon_{hour}h"] = float(sigma_np[batch_index, lead_index, 1])
                rows.append(row)
    return rows


def predict_persistence_rows(test_samples: Iterable[Dict], base_rows: Dict[Tuple[str, str], Dict[str, object]]) -> List[Dict[str, object]]:
    cfg = pipeline.cfg
    baseline = pipeline.PersistenceBaseline()
    lead_steps = [hour // 6 for hour in cfg.lead_hours]
    rows: List[Dict[str, object]] = []

    for sample in test_samples:
        key = _sample_key(sample)
        preds = baseline.predict_np(sample["past"], lead_steps)
        row = dict(base_rows[key])
        for lead_index, hour in enumerate(cfg.lead_hours):
            row[f"pred_mu_lat_{hour}h"] = float(preds[lead_index, 0])
            row[f"pred_mu_lon_{hour}h"] = float(preds[lead_index, 1])
            row[f"pred_sigma_lat_{hour}h"] = np.nan
            row[f"pred_sigma_lon_{hour}h"] = np.nan
        rows.append(row)
    return rows


def load_summary_module():
    if not SUMMARY_SCRIPT.exists():
        raise FileNotFoundError(f"Missing report generator: {SUMMARY_SCRIPT}")

    spec = importlib.util.spec_from_file_location("stormcare_summary", SUMMARY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load report generator from {SUMMARY_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_report(metrics_dir: Path) -> None:
    summary_module = load_summary_module()
    summary_df, preds_df = summary_module.load_inputs(str(metrics_dir))
    plot_dir = summary_module.ensure_plot_dir(str(metrics_dir))
    key_table = summary_module.summarize_from_summary_csv(summary_df)
    pred_err_long = summary_module.compute_track_errors_from_preds(preds_df)
    cov_long = summary_module.compute_cone_coverage_from_preds(preds_df)

    plot_files = {
        "track_curve": summary_module.plot_track_error_curve(summary_df, plot_dir),
    }
    cone50, cone90 = summary_module.plot_cone_coverage(summary_df, plot_dir)
    plot_files["cone50"] = cone50
    plot_files["cone90"] = cone90
    plot_files["landfall"] = summary_module.plot_landfall_error_bar(summary_df, plot_dir)
    summary_module.plot_track_error_boxplot(pred_err_long, plot_dir)

    summary_module.build_report_md(
        metrics_dir=str(metrics_dir),
        summary_csv=summary_df,
        key_table=key_table,
        plot_files=plot_files,
        pred_err_long=pred_err_long,
        cov_long=cov_long,
    )


def run_benchmark(test_ratio: float = 0.20, batch_size: int | None = None) -> None:
    cfg = pipeline.cfg
    pipeline.seed_all(cfg.seed)
    pipeline.ensure_dirs()

    if batch_size is not None:
        cfg.batch_size = batch_size

    test_dataset, test_samples = load_test_dataset(test_ratio=test_ratio)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    base_rows = {_sample_key(sample): _base_prediction_row(sample) for sample in test_samples}

    metrics_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []

    persistence_metrics = pipeline.evaluate_persistence(test_dataset)
    metrics_rows.append({"model": "Persistence", **persistence_metrics})
    for row in predict_persistence_rows(test_samples, base_rows):
        prediction_rows.append({"model": "Persistence", **row})

    for model_name in ["LSTM", "Transformer", "GNO+DynGNN"]:
        print(f"Running benchmark inference for {model_name}...")
        model = load_model(model_name)
        metrics = pipeline.evaluate_prob_model(model, loader)
        metrics_rows.append({"model": model_name, **metrics})
        for row in predict_probabilistic_rows(model, loader, base_rows):
            prediction_rows.append({"model": model_name, **row})

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.DataFrame(prediction_rows)

    metrics_path = Path(cfg.metrics_dir) / "inference_test_metrics_summary.csv"
    predictions_path = Path(cfg.metrics_dir) / "inference_test_predictions_all_models.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(predictions_path, index=False)

    print(f"Saved metrics summary to {metrics_path}")
    print(f"Saved prediction table to {predictions_path}")

    generate_report(Path(cfg.metrics_dir))
    print(f"Report and plots refreshed in {cfg.metrics_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STORM-CARE benchmark evaluation using existing checkpoints.")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "metrics", help="Directory for benchmark outputs.")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Fraction of rebuilt samples reserved for test evaluation.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional evaluation batch size override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir.resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    configure_paths(metrics_dir)
    run_benchmark(test_ratio=args.test_ratio, batch_size=args.batch_size)


if __name__ == "__main__":
    main()