"""Package real-data AOTS2Action deliverables for manuscript verification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_AOTS2Action"
CSV_DIR = RESULTS / "csv"
TABLE_DIR = RESULTS / "tables"
FIG_DIR = RESULTS / "figures"
CONFIG_DIR = RESULTS / "config"
MARKER = "REAL_HUMANITARIAN_GEOSPATIAL_DATA"


ESTIMATOR_LABELS = {
    "Deterministic mean-track": "Deterministic",
    "P90 envelope": "P90 envelope",
    "Ensemble probability-weighted": "Ensemble",
}

ESTIMATOR_COLORS = {
    "Deterministic mean-track": "#1f77b4",
    "P90 envelope": "#d62728",
    "Ensemble probability-weighted": "#2ca02c",
}


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    (ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def save_figure(fig, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIG_DIR / f"{stem}.png", FIG_DIR / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths


def draw_rq2_absolute_error(plt) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq2_estimator_summary_REAL.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for estimator, part in df.groupby("estimator", sort=False):
        part = part.sort_values("horizon_h")
        yerr = [
            part["mean_absolute_error"] - part["ci95_low"],
            part["ci95_high"] - part["mean_absolute_error"],
        ]
        ax.errorbar(
            part["horizon_h"],
            part["mean_absolute_error"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color=ESTIMATOR_COLORS.get(estimator),
            label=ESTIMATOR_LABELS.get(estimator, estimator),
        )
    ax.set_title("RQ2 real-data exposure absolute error")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Mean absolute error")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, "rq2_absolute_error_REAL")
    plt.close(fig)
    return paths


def draw_rq2_signed_error(plt) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq2_bias_all_horizons_REAL.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for estimator, part in df.groupby("estimator", sort=False):
        part = part.sort_values("horizon_h")
        yerr = [
            part["mean_signed_error"] - part["signed_error_ci95_low"],
            part["signed_error_ci95_high"] - part["mean_signed_error"],
        ]
        ax.errorbar(
            part["horizon_h"],
            part["mean_signed_error"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color=ESTIMATOR_COLORS.get(estimator),
            label=ESTIMATOR_LABELS.get(estimator, estimator),
        )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("RQ2 real-data signed exposure error")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Mean signed error")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, "rq2_signed_error_REAL")
    plt.close(fig)
    return paths


def draw_rq2_exposure_ratio(plt) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq2_bias_all_horizons_REAL.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for estimator, part in df.groupby("estimator", sort=False):
        part = part.sort_values("horizon_h")
        yerr = [
            part["mean_exposure_ratio"] - part["ratio_ci95_low"],
            part["ratio_ci95_high"] - part["mean_exposure_ratio"],
        ]
        ax.errorbar(
            part["horizon_h"],
            part["mean_exposure_ratio"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color=ESTIMATOR_COLORS.get(estimator),
            label=ESTIMATOR_LABELS.get(estimator, estimator),
        )
    ax.axhline(1, color="#333333", linewidth=1)
    ax.set_title("RQ2 real-data exposure ratio")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Mean predicted / realized exposure")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, "rq2_exposure_ratio_REAL")
    plt.close(fig)
    return paths


def draw_rq2_brier(plt) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq2_brier_scores_REAL.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for estimator, part in df.groupby("estimator", sort=False):
        part = part.sort_values("horizon_h")
        yerr = [
            part["brier_score"] - part["ci95_low"],
            part["ci95_high"] - part["brier_score"],
        ]
        ax.errorbar(
            part["horizon_h"],
            part["brier_score"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color=ESTIMATOR_COLORS.get(estimator),
            label=ESTIMATOR_LABELS.get(estimator, estimator),
        )
    ax.set_title("RQ2 real-data Brier score")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Brier score")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, "rq2_brier_score_REAL")
    plt.close(fig)
    return paths


def draw_rq3_metric(plt, metric_key: str, title: str, stem: str) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq3_regional_ranking_REAL.csv")
    df = df[df["metric_key"] == metric_key].copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for estimator, part in df.groupby("estimator", sort=False):
        part = part.sort_values("horizon_h")
        yerr = [
            part["storm_level_mean"] - part["ci95_low"],
            part["ci95_high"] - part["storm_level_mean"],
        ]
        ax.errorbar(
            part["horizon_h"],
            part["storm_level_mean"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color=ESTIMATOR_COLORS.get(estimator),
            label=ESTIMATOR_LABELS.get(estimator, estimator),
        )
    ax.set_title(title)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel(title.split(":", 1)[-1].strip())
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, stem)
    plt.close(fig)
    return paths


def draw_rq3_48h_headline(plt) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq3_regional_ranking_REAL.csv")
    metrics = ["ndcg_at_5", "ndcg_at_10", "recall_at_5", "recall_at_10", "spearman"]
    part = df[(df["horizon_h"] == 48) & (df["metric_key"].isin(metrics))].copy()
    part["metric_order"] = part["metric_key"].map({m: i for i, m in enumerate(metrics)})
    part = part.sort_values(["metric_order", "estimator"])
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x_positions = range(len(metrics))
    width = 0.24
    offsets = {
        "Deterministic mean-track": -width,
        "P90 envelope": 0.0,
        "Ensemble probability-weighted": width,
    }
    labels = {
        "ndcg_at_5": "nDCG@5",
        "ndcg_at_10": "nDCG@10",
        "recall_at_5": "Recall@5",
        "recall_at_10": "Recall@10",
        "spearman": "Spearman",
    }
    for estimator, group in part.groupby("estimator", sort=False):
        xs = []
        ys = []
        lows = []
        highs = []
        for metric in metrics:
            row = group[group["metric_key"] == metric]
            if row.empty or pd.isna(row.iloc[0]["storm_level_mean"]):
                continue
            row = row.iloc[0]
            xs.append(metrics.index(metric) + offsets.get(estimator, 0))
            ys.append(row["storm_level_mean"])
            lows.append(row["storm_level_mean"] - row["ci95_low"])
            highs.append(row["ci95_high"] - row["storm_level_mean"])
        ax.bar(
            xs,
            ys,
            width=width,
            color=ESTIMATOR_COLORS.get(estimator),
            label=ESTIMATOR_LABELS.get(estimator, estimator),
        )
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt="none", ecolor="#333333", capsize=2, linewidth=0.9)
    ax.set_xticks(list(x_positions), [labels[m] for m in metrics])
    ax.set_title("RQ3 real-data regional ranking: 48 h headline")
    ax.set_ylabel("Storm-level mean")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    paths = save_figure(fig, "rq3_48h_headline_REAL")
    plt.close(fig)
    return paths


def draw_rq4_runtime(plt) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq4_scalability_results_REAL.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for label, part in df.groupby("m_label", sort=False):
        part = part.sort_values("x_size")
        ax.plot(part["x_size"], part["mean_runtime_s"], marker="o", linewidth=2, label=f"M={label}")
    ax.set_title("RQ4 real-data scalability runtime")
    ax.set_xlabel("Grid cells |X|")
    ax.set_ylabel("Mean runtime per forecast case (s)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    paths = save_figure(fig, "rq4_runtime_REAL")
    plt.close(fig)
    return paths


def draw_rq4_memory_throughput(plt) -> list[Path]:
    df = pd.read_csv(TABLE_DIR / "rq4_scalability_results_REAL.csv")
    full = df[df["x_fraction"] == 1.0].copy().sort_values("m_size")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    axes[0].plot(full["m_size"], full["peak_memory_gb"], marker="o", linewidth=2, color="#9467bd")
    axes[0].set_title("Peak memory at full grid")
    axes[0].set_xlabel("Ensemble size M")
    axes[0].set_ylabel("Peak memory (GB)")
    axes[1].plot(
        full["m_size"],
        full["mean_throughput_items_per_s"],
        marker="o",
        linewidth=2,
        color="#ff7f0e",
    )
    axes[1].set_title("Throughput at full grid")
    axes[1].set_xlabel("Ensemble size M")
    axes[1].set_ylabel("Items/s")
    fig.suptitle("RQ4 real-data memory and throughput", y=1.02)
    fig.tight_layout()
    paths = save_figure(fig, "rq4_memory_throughput_REAL")
    plt.close(fig)
    return paths


def write_settings() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    metadata_paths = [
        CSV_DIR / "humanitarian_grid_REAL.metadata.json",
        CSV_DIR / "rq2_metadata_REAL.json",
        CSV_DIR / "rq2_bias_metadata_REAL.json",
        CSV_DIR / "rq2_brier_metadata_REAL.json",
        CSV_DIR / "rq3_metadata_REAL.json",
        CSV_DIR / "rq4_scalability_metadata_REAL.json",
    ]
    metadata = {}
    for path in metadata_paths:
        if path.exists():
            metadata[path.name] = json.loads(path.read_text())

    settings = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "marker": MARKER,
        "results_directory": str(RESULTS.relative_to(ROOT)),
        "priority_scope": "Real-data RQ2/RQ3 rerun, with RQ4 scalability and dataset details.",
        "forecast_source": "../UNICEF_DATA/AOTS_DATA_SHARE (5).csv",
        "corpus": "results_AOTS2Action/csv/table1_evaluation_corpus.csv",
        "real_grid": "results_AOTS2Action/csv/humanitarian_grid_REAL.csv",
        "real_grid_metadata": "results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json",
        "rq2_horizons_h": [6, 12, 24, 48, 72, 96],
        "rq3_horizons_h": [24, 48, 72],
        "rq4_horizons_h": [24, 48, 72],
        "estimators": [
            "Deterministic mean-track",
            "P90 envelope",
            "Ensemble probability-weighted",
        ],
        "impact_radius_km": 25.0,
        "p90_cone_buffer_km": 25.0,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260817,
        "paired_test": "Two-sided Wilcoxon signed-rank on cyclone-level paired differences, Holm-adjusted.",
        "scalability_repeats": 10,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "source_metadata": metadata,
        "commands": [
            "PYTHONPATH=scripts python3 scripts/build_aots2action_rq2.py --forecasts '../UNICEF_DATA/AOTS_DATA_SHARE (5).csv' --corpus results_AOTS2Action/csv/table1_evaluation_corpus.csv --grid results_AOTS2Action/csv/humanitarian_grid_REAL.csv --grid-kind real --grid-metadata results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json --horizons 6,12,24,48,72,96",
            "PYTHONPATH=scripts python3 scripts/build_aots2action_rq2_bias.py --forecasts '../UNICEF_DATA/AOTS_DATA_SHARE (5).csv' --corpus results_AOTS2Action/csv/table1_evaluation_corpus.csv --grid results_AOTS2Action/csv/humanitarian_grid_REAL.csv --grid-kind real --grid-metadata results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json",
            "PYTHONPATH=scripts python3 scripts/build_aots2action_rq2_brier.py --forecasts '../UNICEF_DATA/AOTS_DATA_SHARE (5).csv' --corpus results_AOTS2Action/csv/table1_evaluation_corpus.csv --grid results_AOTS2Action/csv/humanitarian_grid_REAL.csv --grid-kind real --grid-metadata results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json",
            "PYTHONPATH=scripts python3 scripts/build_aots2action_rq3.py --forecasts '../UNICEF_DATA/AOTS_DATA_SHARE (5).csv' --corpus results_AOTS2Action/csv/table1_evaluation_corpus.csv --grid results_AOTS2Action/csv/humanitarian_grid_REAL.csv --grid-kind real --grid-metadata results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json",
            "PYTHONPATH=scripts python3 scripts/run_aots2action_scalability.py --forecasts '../UNICEF_DATA/AOTS_DATA_SHARE (5).csv' --corpus results_AOTS2Action/csv/table1_evaluation_corpus.csv --grid results_AOTS2Action/csv/humanitarian_grid_REAL.csv --grid-kind real --grid-metadata results_AOTS2Action/csv/humanitarian_grid_REAL.metadata.json --output-suffix REAL --out-dir results_AOTS2Action --repeats 10 --horizons 24,48,72",
        ],
    }
    out = CONFIG_DIR / "real_rerun_settings.json"
    out.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def deliverable_category(path: Path) -> str:
    name = path.name
    if path.parent.name == "figures":
        return "plot"
    if path.parent.name == "config" or name.endswith(".json"):
        return "configuration/settings"
    if path.parent.name == "tables":
        return "summary table"
    if path.parent.name == "csv":
        return "case-level/source CSV"
    if name.endswith(".md"):
        return "result report"
    return "supporting file"


def write_manifest() -> tuple[Path, Path]:
    include_roots = [RESULTS]
    rows = []
    for root in include_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if "_PROXY" in rel:
                continue
            if "PLACEHOLDER" in rel.upper():
                continue
            if path.name == "manifest_real_deliverables.csv":
                continue
            if path.name == "MANIFEST_REAL_DELIVERABLES.md":
                continue
            if path.suffix.lower() not in {".csv", ".json", ".md", ".png", ".pdf"}:
                continue
            is_real = "_REAL" in path.name or path.name in {
                "README.md",
                "REAL_DATA_WORKFLOW.md",
                "DATASET_EXPERIMENT_DETAILS_REAL.md",
                "real_rerun_settings.json",
                "humanitarian_grid_REAL.csv",
                "humanitarian_grid_REAL.metadata.json",
            }
            if not is_real:
                continue
            rows.append(
                {
                    "category": deliverable_category(path),
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    csv_out = CSV_DIR / "manifest_real_deliverables.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(row)

    md = [
        "# AOTS2Action Real-Data Deliverables Manifest",
        "",
        f"Marker: **{MARKER}**",
        "",
        "This manifest lists the final real-data result files, CSVs, plots, configuration/settings, and summary tables staged under `results_AOTS2Action` for manuscript verification.",
        "",
        f"Machine-readable manifest: `results_AOTS2Action/csv/{csv_out.name}`",
        "",
    ]
    for category in sorted(by_category):
        md.append(f"## {category.title()}")
        md.append("")
        for row in by_category[category]:
            md.append(f"- `{row['path']}` ({row['size_bytes']} bytes)")
        md.append("")

    md_out = RESULTS / "MANIFEST_REAL_DELIVERABLES.md"
    md_out.write_text("\n".join(md).rstrip() + "\n")
    return md_out, csv_out


def write_archive(md_manifest: Path, csv_manifest: Path) -> Path:
    archive = RESULTS / "AOTS2Action_REAL_DELIVERABLES.zip"
    manifest_df = pd.read_csv(csv_manifest)
    paths = [ROOT / path for path in manifest_df["path"].tolist()]
    paths.extend([md_manifest, csv_manifest])
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or path == archive:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in unique_paths:
            if path.exists():
                zf.write(path, path.relative_to(ROOT).as_posix())
    return archive


def main() -> None:
    plt = setup_matplotlib()
    generated = []
    generated.extend(draw_rq2_absolute_error(plt))
    generated.extend(draw_rq2_signed_error(plt))
    generated.extend(draw_rq2_exposure_ratio(plt))
    generated.extend(draw_rq2_brier(plt))
    generated.extend(draw_rq3_metric(plt, "ndcg_at_5", "RQ3 real-data ranking: nDCG@5", "rq3_ndcg_at_5_REAL"))
    generated.extend(draw_rq3_metric(plt, "ndcg_at_10", "RQ3 real-data ranking: nDCG@10", "rq3_ndcg_at_10_REAL"))
    generated.extend(draw_rq3_metric(plt, "recall_at_5", "RQ3 real-data ranking: Recall@5", "rq3_recall_at_5_REAL"))
    generated.extend(draw_rq3_metric(plt, "recall_at_10", "RQ3 real-data ranking: Recall@10", "rq3_recall_at_10_REAL"))
    generated.extend(draw_rq3_metric(plt, "spearman", "RQ3 real-data ranking: Spearman", "rq3_spearman_REAL"))
    generated.extend(draw_rq3_48h_headline(plt))
    generated.extend(draw_rq4_runtime(plt))
    generated.extend(draw_rq4_memory_throughput(plt))
    settings = write_settings()
    md_manifest, csv_manifest = write_manifest()
    archive = write_archive(md_manifest, csv_manifest)
    print("Generated figures:")
    for path in generated:
        print(f"  {path.relative_to(ROOT)}")
    print(f"Generated settings: {settings.relative_to(ROOT)}")
    print(f"Generated manifests: {md_manifest.relative_to(ROOT)}, {csv_manifest.relative_to(ROOT)}")
    print(f"Generated archive: {archive.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
