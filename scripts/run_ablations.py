"""
Generate the STORM-CARE ablation table from rerunnable evaluations.

This script deliberately avoids hand-filled placeholders. Numeric cells are
either recomputed here or read from regenerated module outputs. Metrics that do
not apply to a module-specific ablation are filled with an explicit status
string and documented in the audit report.

Outputs:
  tables/table3_ablations.csv
  metrics/ablations/foundation_ablation_metrics.csv
  metrics/ablations/graph_ablation_metrics.csv
  metrics/ablations/table3_ablations_sources.json
  reports/ablation_study_audit.md
  results/module6_ablations/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


OUT_TABLE = Path("tables/table3_ablations.csv")
METRICS_DIR = Path("metrics/ablations")
REPORT_PATH = Path("reports/ablation_study_audit.md")
RESULTS_ROOT = Path("results/module6_ablations")

NA_COMPONENT = "not_applicable_to_changed_component"
NOT_RECORDED = "not_recorded_in_source_run"
NOT_AVAILABLE = "not_available_from_current_evaluator"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _round(x: Any, digits: int = 4) -> Any:
    return round(float(x), digits) if _finite(x) else x


def _mean_values(row: pd.Series, keys: Iterable[str]) -> float:
    vals = [float(row[k]) for k in keys if k in row and _finite(row[k])]
    return float(np.mean(vals)) if vals else float("nan")


def _read_selected_foundation_row() -> Optional[pd.Series]:
    path = Path("metrics/foundation/foundation_eval_metrics.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "selected_checkpoint" in df.columns:
        selected = df[df["selected_checkpoint"].astype(str).str.lower() == "true"]
        if not selected.empty:
            return selected.iloc[-1]
    return df.iloc[-1]


def _cfg_from_checkpoint_dict(cfg_dict: Dict[str, Any]):
    from model.foundation.config import FoundationConfig

    allowed = {f.name for f in fields(FoundationConfig)}
    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in allowed}
    return FoundationConfig(**cfg_kwargs)


def _build_foundation_val_loader(cfg):
    from torch.utils.data import DataLoader
    from model.foundation.data_pipeline import MultiSourceDataPipeline
    from model.foundation.pretrain import PretrainRunner, StormSequenceDataset, collate_fn

    records = MultiSourceDataPipeline(cfg).build()
    runner = PretrainRunner(cfg)
    _, val_records, split_audit = runner._split_records(records)
    val_ds = StormSequenceDataset(val_records, cfg)
    loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return loader, split_audit, len(val_ds)


@torch.no_grad()
def _foundation_intensity_mae_kt(model, loader, cfg) -> float:
    device = next(model.parameters()).device
    errs: List[np.ndarray] = []
    for batch in loader:
        sf = batch["storm_feats"].to(device)
        bi = batch["basin_ids"].to(device)
        si = batch["status_ids"].to(device)
        era5 = batch["era5_patches"].to(device)
        ev = batch["era5_valid"].to(device)
        ei_raw = batch.get("edge_index")
        ei = (ei_raw[0] if ei_raw is not None and ei_raw.dim() == 3 else ei_raw)
        if ei is not None:
            ei = ei.to(device)
        out = model(sf, bi, si, era5, ev, ei, mask=None)
        # Feature index 3 is vmax_norm = vmax_kt / 200.
        pred = out["future_mu"][:, :-1, 3].detach().cpu().numpy()
        true = sf[:, 1:, 3].detach().cpu().numpy()
        errs.append(np.abs(pred - true).reshape(-1) * 200.0)
    return float(np.concatenate(errs).mean()) if errs else float("nan")


def _foundation_eval_pair(skip: bool) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate pretrained and random foundation models on the same validation windows."""
    if skip:
        row = _read_selected_foundation_row()
        if row is None:
            return {}, []
        track_keys = [f"track_err_km_{h}h" for h in (6, 12, 24, 48)]
        p90_keys = [f"cone_p90_{h}h" for h in (6, 12, 24, 48)]
        full = {
            "track_error": _round(_mean_values(row, track_keys)),
            "track_error_units": "mean_km_6_48h_foundation_val",
            "intensity_error": NOT_AVAILABLE,
            "intensity_error_units": "future_vmax_mae_kt",
            "calibration": _round(np.mean([abs(float(row[k]) - 0.9) for k in p90_keys if k in row])),
            "calibration_units": "mean_abs_p90_coverage_error_6_48h",
            "computational_cost_s": NOT_RECORDED,
            "source_files": "metrics/foundation/foundation_eval_metrics.csv",
            "status": "cached_selected_checkpoint",
            "notes": "Foundation row read from selected checkpoint metrics; intensity MAE not in cached evaluator.",
        }
        return {"foundation_full": full}, [{"variant": "foundation_full", **full}]

    ckpt_path = Path("checkpoints/foundation/foundation_best.pt")
    if not ckpt_path.exists():
        return {}, []

    from model.foundation.architecture import FoundationModel
    from model.foundation.evaluation import FoundationEvaluator

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_checkpoint_dict(ckpt.get("cfg", {}))
    torch.manual_seed(cfg.seed)
    loader, split_audit, n_val_windows = _build_foundation_val_loader(cfg)

    out: Dict[str, Dict[str, Any]] = {}
    detail_rows: List[Dict[str, Any]] = []
    for variant, load_weights in [
        ("foundation_full", True),
        ("no_ssl_random_init", False),
    ]:
        t0 = time.time()
        torch.manual_seed(cfg.seed)
        model = FoundationModel(cfg)
        if load_weights:
            model.load_state_dict(ckpt["state_dict"])
        model.eval()
        evaluator = FoundationEvaluator(model, cfg)
        torch.manual_seed(cfg.seed)
        metrics = evaluator.evaluate(loader)
        intensity = _foundation_intensity_mae_kt(model, loader, cfg)
        elapsed = time.time() - t0

        track_keys = [f"track_err_km_{h}h" for h in (6, 12, 24, 48)]
        p90_keys = [f"cone_p90_{h}h" for h in (6, 12, 24, 48)]
        row = {
            "track_error": _round(_mean_values(pd.Series(metrics), track_keys)),
            "track_error_units": "mean_km_6_48h_foundation_val",
            "intensity_error": _round(intensity),
            "intensity_error_units": "future_vmax_mae_kt",
            "calibration": _round(np.mean([abs(float(metrics[k]) - 0.9) for k in p90_keys])),
            "calibration_units": "mean_abs_p90_coverage_error_6_48h",
            "computational_cost_s": _round(elapsed, 2),
            "source_files": "checkpoints/foundation/foundation_best.pt; regenerated in scripts/run_ablations.py",
            "status": "rerun",
            "notes": (
                f"Same validation split for pretrained and random foundation models; "
                f"validation windows={n_val_windows}; group overlap="
                f"{len(split_audit.get('group_key_overlap', []))}."
            ),
        }
        out[variant] = row
        detail_rows.append({"variant": variant, **row})
    return out, detail_rows


def _mask_edge_type(scenarios, edge_type: Optional[int]) -> None:
    if edge_type is None:
        return
    for sc_steps in scenarios:
        for sc in sc_steps:
            keep = sc.edge_types != edge_type
            sc.edge_index = sc.edge_index[:, keep]
            sc.edge_types = sc.edge_types[keep]


def _train_graph_variant(name: str, drop_edge_type: Optional[int], epochs: Optional[int]) -> Dict[str, Any]:
    from model.disaster_graph.architecture import DisasterGNN
    from model.disaster_graph.config import DisasterGraphConfig
    from model.disaster_graph.schema import EDGE_PROPAGATION, EDGE_TRANSPORT, build_dataset
    from model.disaster_graph.train import DisasterGraphTrainer
    from scripts.eval_humanitarian import evaluate_model

    cfg = DisasterGraphConfig()
    cfg.apply_demo_overrides()
    cfg.n_scenarios = 50
    if epochs is not None:
        cfg.n_epochs = epochs
    cfg.seed = 123
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    edge_label = {
        None: "full_graph",
        EDGE_PROPAGATION: "drop_propagation_edges",
        EDGE_TRANSPORT: "drop_transport_edges",
    }[drop_edge_type]

    t0 = time.time()
    train_sc = build_dataset(cfg, seed=123)
    test_sc = build_dataset(cfg, seed=999)
    _mask_edge_type(train_sc, drop_edge_type)
    _mask_edge_type(test_sc, drop_edge_type)

    model = DisasterGNN(cfg)
    trainer = DisasterGraphTrainer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
    )

    best_train_loss = float("inf")
    for _ in range(cfg.n_epochs):
        model.train()
        total, n = 0.0, 0
        for sc_steps in train_sc:
            for sc in sc_steps:
                loss, _ = trainer._loss(model, sc)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.detach().cpu())
                n += 1
        best_train_loss = min(best_train_loss, total / max(n, 1))
        scheduler.step()

    eval_metrics = evaluate_model(model, test_sc, cfg)
    elapsed = time.time() - t0
    return {
        "variant": name,
        "edge_protocol": edge_label,
        "exposure_error": eval_metrics["child_mape"],
        "exposure_error_units": "exposed_children_peak_MAPE_percent",
        "ranking_correlation": eval_metrics["priority_rho"],
        "ranking_units": "spearman_rho_recovery_priority",
        "school_disruption_auc": eval_metrics["school_auc"],
        "hospital_access_mae": eval_metrics["hosp_mae"],
        "computational_cost_s": _round(elapsed, 2),
        "n_train_scenarios": len(train_sc),
        "n_test_scenarios": len(test_sc),
        "n_epochs": cfg.n_epochs,
        "best_train_loss": _round(best_train_loss),
        "source_files": "model/disaster_graph/*; scripts/eval_humanitarian.py; regenerated in scripts/run_ablations.py",
        "status": "rerun",
        "notes": "Same train seed 123 and held-out test seed 999 across graph ablations.",
    }


def _graph_eval_variants(epochs: Optional[int]) -> List[Dict[str, Any]]:
    from model.disaster_graph.schema import EDGE_PROPAGATION, EDGE_TRANSPORT

    return [
        _train_graph_variant("graph_full_multitask", None, epochs),
        _train_graph_variant("static_graph_no_propagation", EDGE_PROPAGATION, epochs),
        _train_graph_variant("no_transport_edges", EDGE_TRANSPORT, epochs),
    ]


def _physics_rows() -> Dict[str, Dict[str, Any]]:
    path = Path("metrics/physics/physics_full_vs_ablation.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        name = str(row["run"])
        rows[name] = {
            "track_error": _round(row.get("final_val_track_rmse")),
            "track_error_units": "module2_validation_track_rmse_normalized",
            "physics_residual": _round(row.get("final_val_L_phys")),
            "physics_residual_units": "weighted_validation_physics_loss",
            "computational_cost_s": NOT_RECORDED,
            "source_files": str(path),
            "status": "cached_regenerated_physics_run",
            "notes": "Physics metrics are from the matched full-vs-no-physics rerun.",
        }
    return rows


def _time_no_physics_runtime(skip: bool) -> Optional[Dict[str, Any]]:
    if skip:
        return None
    from model.physics.config import PIGNOConfig
    from model.physics.train import PIGNOTrainer

    cfg = PIGNOConfig()
    cfg.apply_demo_overrides()
    cfg.metrics_dir = str(METRICS_DIR / "no_physics_runtime")
    cfg.checkpoint_dir = "checkpoints/ablations/no_physics_runtime"
    cfg.lambda_adv = 0.0
    cfg.lambda_diff = 0.0
    cfg.lambda_mass = 0.0
    cfg.lambda_wp = 0.0
    cfg.lambda_cont = 0.0
    cfg.lambda_energy = 0.0

    t0 = time.time()
    PIGNOTrainer(cfg).run()
    elapsed = time.time() - t0
    out = {
        "variant": "no_physics",
        "computational_cost_s": _round(elapsed, 2),
        "runtime_protocol": "fresh 20-epoch demo no-physics PI-GNO training",
        "metrics_dir": cfg.metrics_dir,
        "checkpoint_dir": cfg.checkpoint_dir,
    }
    (METRICS_DIR / "no_physics_runtime.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    return out


def _time_no_world_model_runtime(skip: bool) -> Dict[str, Any]:
    if skip:
        return {
            "computational_cost_s": NOT_RECORDED,
            "source_files": "metrics/counterfactual/counterfactual_outcomes.csv",
            "status": "cached_regenerated_counterfactual_run",
            "notes": "Runtime measurement skipped by CLI flag.",
        }
    from model.counterfactual.config import CounterfactualConfig
    from model.counterfactual.engine import CounterfactualEngine
    from model.world_model.architecture import WorldModel
    from model.world_model.config import WorldModelConfig
    from model.world_model.train import _make_sequences

    ckpt_path = Path("checkpoints/world_model/worldmodel_best.pt")
    if not ckpt_path.exists():
        return {
            "computational_cost_s": NOT_RECORDED,
            "source_files": str(ckpt_path),
            "status": "checkpoint_missing",
            "notes": "World-model checkpoint missing; runtime not measured.",
        }

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = WorldModelConfig(**{
        k: v for k, v in ckpt.get("config", {}).items()
        if k in WorldModelConfig.__dataclass_fields__
    })
    model = WorldModel(saved_cfg)
    model.load_state_dict(ckpt["state"])
    model.eval()

    cf_cfg = CounterfactualConfig()
    if saved_cfg.demo:
        cf_cfg.apply_demo_overrides()
    cf_cfg.d_disaster_state = saved_cfg.d_disaster_state
    cf_cfg.d_latent = saved_cfg.d_latent
    t_warm = min(cf_cfg.n_initial_steps, saved_cfg.n_steps_train)

    all_seqs = _make_sequences(
        saved_cfg.n_sequences,
        saved_cfg.n_steps_train,
        saved_cfg.d_disaster_state,
        seed=saved_cfg.seed,
    )
    warm_up_seqs = [seq[:t_warm] for seq in all_seqs[int(len(all_seqs) * 0.8):]]
    engine = CounterfactualEngine(model, cf_cfg)
    original_rollout = model.rollout

    def frozen_rollout(warm_up, n_steps):
        with torch.no_grad():
            return warm_up[-1].unsqueeze(0).expand(n_steps, -1).contiguous()

    t0 = time.time()
    model.rollout = frozen_rollout
    results = engine.compare_multi_storm(warm_up_seqs)
    model.rollout = original_rollout
    elapsed = time.time() - t0

    out = {
        "variant": "no_world_model",
        "computational_cost_s": _round(elapsed, 2),
        "runtime_protocol": "frozen-latent counterfactual rollout over complete held-out test split",
        "n_test_sequences": len(warm_up_seqs),
        "baseline_peak_exposure": results.get("baseline", {}).get("metrics", {}).get("peak_exposure"),
        "earlier_evacuation_peak_exposure": results.get("earlier_evacuation", {}).get("metrics", {}).get("peak_exposure"),
        "delayed_evacuation_peak_exposure": results.get("delayed_evacuation", {}).get("metrics", {}).get("peak_exposure"),
        "source_files": "checkpoints/world_model/worldmodel_best.pt; model/counterfactual/*",
        "status": "rerun_runtime_only",
        "notes": f"Frozen-rollout runtime measured on {len(warm_up_seqs)} held-out test sequences.",
    }
    (METRICS_DIR / "no_world_model_runtime.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    return out


def _counterfactual_status_row(runtime: Dict[str, Any]) -> Dict[str, Any]:
    path = Path("metrics/counterfactual/counterfactual_outcomes.csv")
    if not path.exists():
        return {
            "source_files": "metrics/counterfactual/counterfactual_outcomes.csv",
            "computational_cost_s": runtime.get("computational_cost_s", NOT_RECORDED),
            "status": "source_missing",
            "notes": "Counterfactual outcomes must be regenerated before this variant can be audited.",
        }
    df = pd.read_csv(path)
    n_seq = int(df["n_test_sequences"].max()) if "n_test_sequences" in df.columns and not df.empty else 0
    return {
        "source_files": f"{path}; {runtime.get('source_files', 'metrics/ablations/no_world_model_runtime.json')}",
        "computational_cost_s": runtime.get("computational_cost_s", NOT_RECORDED),
        "status": runtime.get("status", "cached_regenerated_counterfactual_run"),
        "notes": f"World-model ablation is a causal rollout test, not a forecast/graph/physics metric; source outcomes cover {n_seq} test sequences. {runtime.get('notes', '')}",
    }


def _compose_rows(
    foundation: Dict[str, Dict[str, Any]],
    graph_rows: List[Dict[str, Any]],
    physics: Dict[str, Dict[str, Any]],
    counterfactual_runtime: Dict[str, Any],
) -> List[Dict[str, Any]]:
    graph_by_name = {r["variant"]: r for r in graph_rows}
    f_full = foundation.get("foundation_full", {})
    f_rand = foundation.get("no_ssl_random_init", {})
    p_full = physics.get("full_physics", {})
    p_no = physics.get("no_physics", {})
    g_full = graph_by_name.get("graph_full_multitask", {})
    g_static = graph_by_name.get("static_graph_no_propagation", {})
    g_transport = graph_by_name.get("no_transport_edges", {})
    cf = _counterfactual_status_row(counterfactual_runtime)

    def row(
        variant: str,
        changed_component: str,
        protocol: str,
        track: Any,
        track_units: str,
        intensity: Any,
        intensity_units: str,
        exposure: Any,
        exposure_units: str,
        ranking: Any,
        ranking_units: str,
        calibration: Any,
        calibration_units: str,
        physics_residual: Any,
        physics_units: str,
        cost: Any,
        source_files: str,
        status: str,
        notes: str,
    ) -> Dict[str, Any]:
        return {
            "variant": variant,
            "changed_component": changed_component,
            "evaluation_protocol": protocol,
            "track_error": track,
            "track_error_units": track_units,
            "intensity_error": intensity,
            "intensity_error_units": intensity_units,
            "exposure_error": exposure,
            "exposure_error_units": exposure_units,
            "ranking_correlation": ranking,
            "ranking_units": ranking_units,
            "calibration": calibration,
            "calibration_units": calibration_units,
            "physics_residual": physics_residual,
            "physics_residual_units": physics_units,
            "computational_cost_s": cost,
            "source_files": source_files,
            "status": status,
            "notes": notes,
        }

    full_cost_terms = [
        f_full.get("computational_cost_s"),
        g_full.get("computational_cost_s"),
    ]
    full_cost_numeric = [float(x) for x in full_cost_terms if _finite(x)]
    full_cost = _round(sum(full_cost_numeric), 2) if full_cost_numeric else NOT_RECORDED

    rows = [
        row(
            "STORM-CARE full",
            "none",
            "Composite row: foundation validation, graph held-out synthetic test, physics validation.",
            f_full.get("track_error", NOT_AVAILABLE),
            f_full.get("track_error_units", "mean_km_6_48h_foundation_val"),
            f_full.get("intensity_error", NOT_AVAILABLE),
            f_full.get("intensity_error_units", "future_vmax_mae_kt"),
            g_full.get("exposure_error", NOT_AVAILABLE),
            g_full.get("exposure_error_units", "exposed_children_peak_MAPE_percent"),
            g_full.get("ranking_correlation", NOT_AVAILABLE),
            g_full.get("ranking_units", "spearman_rho_recovery_priority"),
            f_full.get("calibration", NOT_AVAILABLE),
            f_full.get("calibration_units", "mean_abs_p90_coverage_error_6_48h"),
            p_full.get("physics_residual", NOT_AVAILABLE),
            p_full.get("physics_residual_units", "weighted_validation_physics_loss"),
            full_cost,
            "; ".join(sorted(set(filter(None, [
                f_full.get("source_files"),
                g_full.get("source_files"),
                p_full.get("source_files"),
            ])))),
            "rerun_or_cached_regenerated",
            "Full row combines the valid evaluator for each metric; see units and protocol columns.",
        ),
        row(
            "no_physics",
            "Module 2 physics loss weights",
            "Matched physics validation split from metrics/physics/physics_full_vs_ablation.csv.",
            p_no.get("track_error", NOT_AVAILABLE),
            p_no.get("track_error_units", "module2_validation_track_rmse_normalized"),
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            p_no.get("physics_residual", NOT_AVAILABLE),
            p_no.get("physics_residual_units", "weighted_validation_physics_loss"),
            p_no.get("computational_cost_s", NOT_RECORDED),
            p_no.get("source_files", "metrics/physics/physics_full_vs_ablation.csv"),
            p_no.get("status", "source_missing"),
            p_no.get("notes", "No-physics ablation affects Module 2 only."),
        ),
        row(
            "no_ssl_random_init",
            "Module 1 self-supervised pretraining",
            "Same foundation validation windows as pretrained checkpoint.",
            f_rand.get("track_error", NOT_AVAILABLE),
            f_rand.get("track_error_units", "mean_km_6_48h_foundation_val"),
            f_rand.get("intensity_error", NOT_AVAILABLE),
            f_rand.get("intensity_error_units", "future_vmax_mae_kt"),
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            f_rand.get("calibration", NOT_AVAILABLE),
            f_rand.get("calibration_units", "mean_abs_p90_coverage_error_6_48h"),
            NA_COMPONENT,
            "not_applicable",
            f_rand.get("computational_cost_s", NOT_RECORDED),
            f_rand.get("source_files", "checkpoints/foundation/foundation_best.pt"),
            f_rand.get("status", "source_missing"),
            f_rand.get("notes", "Random initialization baseline for foundation model."),
        ),
        row(
            "static_graph_no_propagation",
            "Module 3 storm propagation edges",
            "Same graph train/test seeds as graph_full_multitask.",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            g_static.get("exposure_error", NOT_AVAILABLE),
            g_static.get("exposure_error_units", "exposed_children_peak_MAPE_percent"),
            g_static.get("ranking_correlation", NOT_AVAILABLE),
            g_static.get("ranking_units", "spearman_rho_recovery_priority"),
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            g_static.get("computational_cost_s", NOT_RECORDED),
            g_static.get("source_files", "model/disaster_graph/*"),
            g_static.get("status", "source_missing"),
            g_static.get("notes", "Graph ablation affects humanitarian heads only."),
        ),
        row(
            "no_transport_edges",
            "Module 3 transport edges",
            "Same graph train/test seeds as graph_full_multitask.",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            g_transport.get("exposure_error", NOT_AVAILABLE),
            g_transport.get("exposure_error_units", "exposed_children_peak_MAPE_percent"),
            g_transport.get("ranking_correlation", NOT_AVAILABLE),
            g_transport.get("ranking_units", "spearman_rho_recovery_priority"),
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            g_transport.get("computational_cost_s", NOT_RECORDED),
            g_transport.get("source_files", "model/disaster_graph/*"),
            g_transport.get("status", "source_missing"),
            g_transport.get("notes", "Graph ablation affects humanitarian heads only."),
        ),
        row(
            "no_world_model",
            "Module 4 latent dynamics",
            "Counterfactual causal rollout audit; not a track or humanitarian-head evaluator.",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            NA_COMPONENT,
            "not_applicable",
            cf.get("computational_cost_s", NOT_RECORDED),
            cf.get("source_files", "metrics/counterfactual/counterfactual_outcomes.csv"),
            cf.get("status", "source_missing"),
            cf.get("notes", "World model ablation is evaluated in the counterfactual audit."),
        ),
    ]
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: List[Dict[str, Any]], graph_rows: List[Dict[str, Any]], foundation_rows: List[Dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    graph_df = pd.DataFrame(graph_rows)
    foundation_df = pd.DataFrame(foundation_rows)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Ablation Study Audit\n\n")
        f.write("## Root Causes\n")
        f.write("- The previous ablation script wrote dash placeholders and hardcoded full-model values.\n")
        f.write("- Static-graph and no-transport variants were trained with damage-only MSE, so they did not evaluate the humanitarian heads requested in the table.\n")
        f.write("- Runtime was not recorded in several earlier module logs, so targeted runtime reruns were added for the no-physics and no-world-model cost cells.\n\n")
        f.write("## Fixes\n")
        f.write("- `scripts/run_ablations.py` now regenerates Table 3 from sourced metrics and reruns Module 3 graph ablations on the same train/test seeds.\n")
        f.write("- The no-SSL foundation row is evaluated as a random-initialization model on the same validation windows as the selected pretrained checkpoint.\n")
        f.write("- No-physics and no-world-model computational costs are measured by fresh runtime-only reruns and saved under `metrics/ablations/`.\n")
        f.write("- Non-applicable cells are filled with explicit status values so the table has no hidden blanks or fabricated numbers.\n\n")
        f.write("## Regenerated Table 3\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        if not foundation_df.empty:
            f.write("## Foundation Detail\n")
            f.write(foundation_df.to_markdown(index=False))
            f.write("\n\n")
        if not graph_df.empty:
            f.write("## Graph Ablation Detail\n")
            f.write(graph_df.to_markdown(index=False))
            f.write("\n\n")
        f.write("## Validity Notes\n")
        f.write("- Metrics are only compared within the evaluator named by the units/protocol columns.\n")
        f.write("- Physics ablation supports physical-consistency comparisons; it does not supply humanitarian-head metrics.\n")
        f.write("- Graph-edge ablations support humanitarian-head comparisons; they do not supply foundation forecast calibration.\n")
        f.write("- Counterfactual world-model ablation remains documented by the counterfactual audit outputs, not by forced forecast-table metrics.\n")


def _mirror_outputs() -> None:
    for sub in ["metrics", "tables", "reports"]:
        (RESULTS_ROOT / sub).mkdir(parents=True, exist_ok=True)
    copy_pairs = [
        (OUT_TABLE, RESULTS_ROOT / "tables" / OUT_TABLE.name),
        (REPORT_PATH, RESULTS_ROOT / "reports" / REPORT_PATH.name),
        (METRICS_DIR / "foundation_ablation_metrics.csv", RESULTS_ROOT / "metrics" / "foundation_ablation_metrics.csv"),
        (METRICS_DIR / "graph_ablation_metrics.csv", RESULTS_ROOT / "metrics" / "graph_ablation_metrics.csv"),
        (METRICS_DIR / "table3_ablations_sources.json", RESULTS_ROOT / "metrics" / "table3_ablations_sources.json"),
        (METRICS_DIR / "no_physics_runtime.json", RESULTS_ROOT / "metrics" / "no_physics_runtime.json"),
        (METRICS_DIR / "no_world_model_runtime.json", RESULTS_ROOT / "metrics" / "no_world_model_runtime.json"),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            shutil.copy2(src, dst)
    readme = RESULTS_ROOT / "README.md"
    readme.write_text(
        "# Module 6 Ablation Study Results\n\n"
        "This folder mirrors the regenerated ablation study outputs.\n\n"
        "- `tables/table3_ablations.csv` contains the submission Table 3 ablation audit with no blank cells.\n"
        "- `metrics/foundation_ablation_metrics.csv` contains the same-split foundation full vs random-init evaluation.\n"
        "- `metrics/graph_ablation_metrics.csv` contains the rerun graph-edge ablations on train seed 123 and test seed 999.\n"
        "- `metrics/no_physics_runtime.json` and `metrics/no_world_model_runtime.json` contain targeted runtime measurements.\n"
        "- `reports/ablation_study_audit.md` documents root causes, fixes, and validity limits.\n\n"
        "Cells marked `not_applicable_to_changed_component` are intentional scientific status values, not missing data.\n",
        encoding="utf-8",
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate STORM-CARE ablation outputs.")
    parser.add_argument("--skip-foundation-eval", action="store_true",
                        help="Use cached selected foundation metrics instead of evaluating the checkpoint and random-init model.")
    parser.add_argument("--graph-epochs", type=int, default=None,
                        help="Override demo graph ablation epochs. Default uses the demo config value.")
    parser.add_argument("--skip-runtime-reruns", action="store_true",
                        help="Skip extra runtime-only reruns for cached no-physics/no-world-model cost cells.")
    args = parser.parse_args(argv)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Evaluating foundation ablation rows...")
    foundation, foundation_detail = _foundation_eval_pair(args.skip_foundation_eval)
    if foundation_detail:
        _write_csv(METRICS_DIR / "foundation_ablation_metrics.csv", foundation_detail)

    print("Rerunning graph ablations on identical train/test seeds...")
    graph_detail = _graph_eval_variants(args.graph_epochs)
    _write_csv(METRICS_DIR / "graph_ablation_metrics.csv", graph_detail)

    print("Reading regenerated physics comparison...")
    physics = _physics_rows()
    print("Measuring no-physics runtime...")
    no_physics_runtime = _time_no_physics_runtime(args.skip_runtime_reruns)
    if no_physics_runtime and "no_physics" in physics:
        physics["no_physics"]["computational_cost_s"] = no_physics_runtime["computational_cost_s"]
        physics["no_physics"]["source_files"] = (
            physics["no_physics"]["source_files"] + "; metrics/ablations/no_physics_runtime.json"
        )
        physics["no_physics"]["status"] = "cached_metrics_plus_rerun_runtime"
        physics["no_physics"]["notes"] = (
            physics["no_physics"]["notes"] + " Computational cost measured by a fresh no-physics runtime rerun."
        )
    print("Measuring no-world-model runtime...")
    counterfactual_runtime = _time_no_world_model_runtime(args.skip_runtime_reruns)

    rows = _compose_rows(foundation, graph_detail, physics, counterfactual_runtime)
    _write_csv(OUT_TABLE, rows)
    _write_report(rows, graph_detail, foundation_detail)

    source_audit = {
        "generated_at_unix": time.time(),
        "table": str(OUT_TABLE),
        "metrics": {
            "foundation": str(METRICS_DIR / "foundation_ablation_metrics.csv"),
            "graph": str(METRICS_DIR / "graph_ablation_metrics.csv"),
            "physics": "metrics/physics/physics_full_vs_ablation.csv",
            "counterfactual": "metrics/counterfactual/counterfactual_outcomes.csv",
            "no_physics_runtime": "metrics/ablations/no_physics_runtime.json",
            "no_world_model_runtime": "metrics/ablations/no_world_model_runtime.json",
        },
        "constraints": [
            "No manual CSV edits",
            "No fabricated numbers",
            "Identical train/test data within each ablation evaluator",
            "Explicit status values for non-applicable module metrics",
        ],
    }
    (METRICS_DIR / "table3_ablations_sources.json").write_text(
        json.dumps(source_audit, indent=2),
        encoding="utf-8",
    )
    _mirror_outputs()
    print(f"Saved {OUT_TABLE}")
    print(f"Saved {REPORT_PATH}")
    print(f"Mirrored outputs to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
