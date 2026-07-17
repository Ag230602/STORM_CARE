"""Verify split, sample, and window-count integrity for regenerated outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "metrics/dataset_integrity"
REPORT = ROOT / "reports/dataset_integrity_report.md"


def _foundation_window_count(n_obs: int, window_size: int = 16, stride: int = 4, max_lead: int = 20) -> int:
    if n_obs < window_size + max_lead:
        return 0
    return len(range(0, n_obs - window_size, stride))


def _counts(series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def main() -> None:
    issues = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_map_path = ROOT / "splits/storm_splits.json"
    split_summary_path = ROOT / "splits/split_summary.csv"
    baseline_manifest_path = ROOT / "metrics/baseline_split_manifest.csv"
    foundation_manifest_path = ROOT / "metrics/foundation/foundation_split_manifest.csv"
    foundation_audit_path = ROOT / "metrics/foundation/foundation_split_audit.json"
    predictions_path = ROOT / "metrics/inference_test_predictions_all_models.csv"

    split_map = json.loads(split_map_path.read_text())
    split_summary = pd.read_csv(split_summary_path)
    summary_counts = dict(zip(split_summary["partition"], split_summary["n_storms"].astype(int)))
    map_counts = {k: list(split_map.values()).count(k) for k in sorted(set(split_map.values()))}
    if summary_counts != map_counts:
        issues.append(f"Split summary counts {summary_counts} do not match split map counts {map_counts}")

    baseline = pd.read_csv(baseline_manifest_path)
    duplicate_baseline = int(baseline.duplicated(["storm_tag", "t0"]).sum())
    if duplicate_baseline:
        issues.append(f"Baseline split manifest has {duplicate_baseline} duplicate storm_tag/t0 rows")
    baseline_counts = _counts(baseline["split"])
    era5_by_split = baseline.groupby("split")["era5_available"].agg(["sum", "count"]).reset_index()
    if not bool(baseline["era5_available"].all()):
        issues.append("Baseline case-study manifest contains rows without ERA5 availability")

    preds = pd.read_csv(predictions_path)
    pred_models = sorted(preds["model"].unique().tolist())
    pred_windows = preds.groupby(["storm_tag", "t0"]).size().reset_index(name="n_models")
    expected_models = len(pred_models)
    bad_windows = pred_windows[pred_windows["n_models"] != expected_models]
    if not bad_windows.empty:
        issues.append(f"{len(bad_windows)} prediction windows do not contain all {expected_models} models")

    foundation = pd.read_csv(foundation_manifest_path)
    foundation_counts = _counts(foundation["split"])
    foundation["window_count"] = foundation["n_observations"].astype(int).apply(_foundation_window_count)
    foundation_windows = foundation.groupby("split")["window_count"].sum().astype(int).to_dict()
    duplicate_groups = foundation.groupby(["split", "group_key"]).size().reset_index(name="n")
    train_groups = set(foundation[foundation["split"] == "train"]["group_key"])
    val_groups = set(foundation[foundation["split"] == "val"]["group_key"])
    group_overlap = sorted(train_groups & val_groups)
    if group_overlap:
        issues.append(f"Foundation train/val group overlap: {group_overlap[:5]}")
    foundation_audit = json.loads(foundation_audit_path.read_text())
    if foundation_audit.get("storm_id_overlap") or foundation_audit.get("group_key_overlap"):
        issues.append("Foundation split audit reports overlap")

    tables = {
        "split_map_counts": pd.DataFrame([{"split": k, "n_storms": v} for k, v in map_counts.items()]),
        "baseline_case_study_counts": pd.DataFrame([{"split": k, "n_samples": v} for k, v in baseline_counts.items()]),
        "baseline_era5_counts": era5_by_split,
        "foundation_record_counts": pd.DataFrame([{"split": k, "n_records": v} for k, v in foundation_counts.items()]),
        "foundation_window_counts": pd.DataFrame([{"split": k, "n_windows": v} for k, v in foundation_windows.items()]),
        "prediction_window_counts": pd.DataFrame([{
            "n_prediction_rows": len(preds),
            "n_models": expected_models,
            "n_unique_storm_t0_windows": len(pred_windows),
            "n_bad_windows": len(bad_windows),
        }]),
    }
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    audit = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "split_map_counts": map_counts,
        "baseline_case_study_counts": baseline_counts,
        "foundation_record_counts": foundation_counts,
        "foundation_window_counts": {str(k): int(v) for k, v in foundation_windows.items()},
        "prediction_rows": int(len(preds)),
        "prediction_models": pred_models,
        "prediction_unique_storm_t0_windows": int(len(pred_windows)),
        "foundation_group_overlap_count": len(group_overlap),
    }
    (OUT_DIR / "dataset_integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Dataset Integrity Report\n\n"
        f"- Status: **{audit['status']}**\n"
        f"- Storm split counts: `{map_counts}`\n"
        f"- Baseline case-study sample counts: `{baseline_counts}`\n"
        f"- Foundation record counts: `{foundation_counts}`\n"
        f"- Foundation window counts: `{audit['foundation_window_counts']}`\n"
        f"- Prediction rows: `{audit['prediction_rows']}` across `{len(pred_models)}` models and `{len(pred_windows)}` storm/t0 windows.\n"
        f"- Foundation train/val group overlap count: `{len(group_overlap)}`\n\n"
        "## Issues\n"
        + ("\n".join(f"- {x}" for x in issues) if issues else "- None.\n")
        + "\n\n## Generated CSV Audits\n"
        + "\n".join(f"- `metrics/dataset_integrity/{name}.csv`" for name in tables)
        + "\n",
        encoding="utf-8",
    )
    if issues:
        raise SystemExit("Dataset integrity audit failed")
    print("Dataset integrity PASS")


if __name__ == "__main__":
    main()
