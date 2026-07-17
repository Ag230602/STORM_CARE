"""
Regenerate publication-style Hurricane Ian case-study figures.

The figures are generated from current model outputs, not hand-edited artwork:
  - metrics/inference_test_predictions_all_models.csv
  - metrics/humanitarian/humanitarian_eval_metrics.csv
  - metrics/counterfactual/counterfactual_outcomes.csv
  - checkpoints/disaster_graph/disaster_gnn_best.pt

Outputs are written to figures/case_study/ and mirrored to legacy filenames
used by the README/results bundle.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PREDS_CSV = Path("metrics/inference_test_predictions_all_models.csv")
COUNTERFACTUAL_CSV = Path("metrics/counterfactual/counterfactual_outcomes.csv")
OUT_DIR = Path("figures/case_study")
LEGACY_DIR = Path("figures")
RESULTS_FIG_DIR = Path("results/module3_baselines/figures")
METRICS_DIR = Path("metrics/case_study")
REPORT_PATH = Path("reports/case_study_ian_audit.md")
IAN_ID = "ian"
LEADS = [6, 12, 24, 48]
COLORS = {
    "Persistence": "#6b7280",
    "LSTM": "#c2410c",
    "Transformer": "#b45309",
    "GNO+DynGNN": "#075985",
}
MARKERS = {"Persistence": "s", "LSTM": "o", "Transformer": "^", "GNO+DynGNN": "D"}


def _require_inputs() -> None:
    missing = [str(p) for p in [PREDS_CSV, COUNTERFACTUAL_CSV] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required case-study inputs: {missing}")


def _setup_matplotlib():
    cache_dir = Path(".cache/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon
    from matplotlib.colors import LinearSegmentedColormap
    return plt, Ellipse, Polygon, LinearSegmentedColormap


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_ian_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDS_CSV)
    ian = df[df["storm_tag"].astype(str).str.lower() == IAN_ID].copy()
    if ian.empty:
        raise ValueError(f"No Hurricane Ian rows found in {PREDS_CSV}")
    ian["t0_dt"] = pd.to_datetime(ian["t0"])
    return ian.sort_values(["model", "t0_dt"]).reset_index(drop=True)


def true_track(ian: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = ian[ian["model"] == "Persistence"].sort_values("t0_dt")
    for _, row in base.iterrows():
        rows.append({"time": row["t0"], "lat": row["lat0"], "lon": row["lon0"]})
        for h in LEADS:
            lat_key = f"true_lat_{h}h"
            lon_key = f"true_lon_{h}h"
            if pd.notna(row.get(lat_key)) and pd.notna(row.get(lon_key)):
                rows.append({"time": f"{row['t0']}+{h}h", "lat": row[lat_key], "lon": row[lon_key]})
    out = pd.DataFrame(rows)
    out["_key"] = list(zip(out["lat"].round(2), out["lon"].round(2)))
    out = out.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    return out


def forecast_track(ian: pd.DataFrame, model: str, t0=None) -> pd.DataFrame:
    mdf = ian[ian["model"] == model].sort_values("t0_dt")
    if t0 is not None:
        mdf = mdf[mdf["t0"] == t0]
    if mdf.empty:
        return pd.DataFrame(columns=["lead_h", "lat", "lon", "sig_lat", "sig_lon"])
    row = mdf.iloc[0]
    rows = [{"lead_h": 0, "lat": row["lat0"], "lon": row["lon0"], "sig_lat": np.nan, "sig_lon": np.nan}]
    for h in LEADS:
        rows.append({
            "lead_h": h,
            "lat": row.get(f"pred_mu_lat_{h}h", np.nan),
            "lon": row.get(f"pred_mu_lon_{h}h", np.nan),
            "sig_lat": row.get(f"pred_sigma_lat_{h}h", np.nan),
            "sig_lon": row.get(f"pred_sigma_lon_{h}h", np.nan),
        })
    return pd.DataFrame(rows).dropna(subset=["lat", "lon"]).reset_index(drop=True)


def map_extent(ian: pd.DataFrame) -> Tuple[float, float, float, float]:
    vals_lat, vals_lon = [], []
    for col in ian.columns:
        if col.startswith(("true_lat", "pred_mu_lat")) or col == "lat0":
            vals_lat.extend(pd.to_numeric(ian[col], errors="coerce").dropna().tolist())
        if col.startswith(("true_lon", "pred_mu_lon")) or col == "lon0":
            vals_lon.extend(pd.to_numeric(ian[col], errors="coerce").dropna().tolist())
    return min(vals_lon) - 4, max(vals_lon) + 4, min(vals_lat) - 3, max(vals_lat) + 3


def draw_noaa_basemap(ax, extent, Polygon) -> None:
    min_lon, max_lon, min_lat, max_lat = extent
    ax.set_facecolor("#d9edf7")

    # Hand-coded coarse coastline polygons keep the workflow offline and
    # deterministic. They are contextual backdrops, not data layers.
    land_polys = [
        [(-88, 30.2), (-86, 30.4), (-84, 30.4), (-82, 29.8), (-81.2, 28.2),
         (-80.6, 26.0), (-80.1, 25.2), (-81.2, 24.7), (-82.4, 26.0),
         (-83.0, 28.0), (-84.6, 29.6), (-87.0, 30.1)],
        [(-91, 29.2), (-89, 30.4), (-86.5, 30.4), (-86.5, 33.5), (-91, 33.5)],
        [(-84, 31.0), (-80, 31.0), (-78, 35.5), (-84, 35.5)],
        [(-85.5, 21.5), (-74.0, 21.5), (-74.0, 23.5), (-79.0, 23.3), (-85.5, 22.6)],
        [(-90.5, 18.5), (-86.2, 18.5), (-86.2, 21.6), (-90.5, 21.6)],
    ]
    for poly in land_polys:
        ax.add_patch(Polygon(poly, closed=True, facecolor="#f3efe2", edgecolor="#5b6472", linewidth=0.8, zorder=0))

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    mean_lat = (min_lat + max_lat) / 2
    ax.set_aspect(1 / max(np.cos(np.radians(mean_lat)), 0.2), adjustable="box")
    ax.grid(True, color="#ffffff", linewidth=0.8)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Longitude (degrees east; western longitudes are negative)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.text(
        0.01,
        0.01,
        "NOAA-style research visualization; coastline schematic",
        transform=ax.transAxes,
        fontsize=7,
        color="#334155",
        ha="left",
        va="bottom",
    )


def add_ellipse(ax, Ellipse, x, y, sig_lon, sig_lat, probability: str, color: str, alpha: float) -> None:
    z = 1.177 if probability == "P50" else 2.146
    if not all(math.isfinite(float(v)) for v in [x, y, sig_lon, sig_lat]):
        return
    width = 2 * z * max(float(sig_lon), 1e-6)
    height = 2 * z * max(float(sig_lat), 1e-6)
    ax.add_patch(Ellipse((x, y), width=width, height=height, facecolor=color, edgecolor=color, alpha=alpha, lw=1.0, zorder=2))


def figure_track_map(ian, true_df, plt, Ellipse, Polygon):
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    extent = map_extent(ian)
    draw_noaa_basemap(ax, extent, Polygon)
    ax.plot(true_df["lon"], true_df["lat"], color="#111827", lw=2.6, marker="o", ms=4.5, label="Best track", zorder=5)
    ax.scatter(true_df["lon"].iloc[0], true_df["lat"].iloc[0], marker="^", s=90, color="#111827", zorder=6, label="Case-study start")
    ax.scatter(true_df["lon"].iloc[-1], true_df["lat"].iloc[-1], marker="*", s=150, color="#111827", zorder=6, label="Last observed point")

    first_t0 = ian["t0"].sort_values().iloc[0]
    for model, color in COLORS.items():
        ft = forecast_track(ian, model, first_t0)
        if ft.empty:
            continue
        ax.plot(ft["lon"], ft["lat"], "--", color=color, lw=1.6, marker=MARKERS.get(model, "o"), ms=4.5, label=model, zorder=4)
        last = ft[ft["lead_h"] == max(ft["lead_h"])]
        if not last.empty:
            r = last.iloc[0]
            add_ellipse(ax, Ellipse, r["lon"], r["lat"], r["sig_lon"], r["sig_lat"], "P90", color, 0.13)
    ax.set_title("Hurricane Ian Track Forecast Case Study\nOpen-loop forecasts from earliest Ian test window", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
    return fig


def figure_uncertainty_cones(ian, true_df, plt, Ellipse, Polygon):
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    draw_noaa_basemap(ax, map_extent(ian), Polygon)
    ax.plot(true_df["lon"], true_df["lat"], color="#111827", lw=2.2, marker="o", ms=4.2, label="Best track", zorder=5)
    first_t0 = ian["t0"].sort_values().iloc[0]
    for model in ["LSTM", "Transformer", "GNO+DynGNN"]:
        color = COLORS[model]
        ft = forecast_track(ian, model, first_t0)
        if ft.empty:
            continue
        is_primary = model == "GNO+DynGNN"
        ax.plot(
            ft["lon"], ft["lat"],
            color=color,
            lw=2.4 if is_primary else 1.2,
            marker=MARKERS[model],
            ms=5 if is_primary else 3.5,
            alpha=1.0 if is_primary else 0.55,
            label=f"{model} mean" + (" + P50/P90 cone" if is_primary else ""),
            zorder=4 if is_primary else 3,
        )
        if is_primary:
            for _, r in ft[ft["lead_h"] > 0].iterrows():
                add_ellipse(ax, Ellipse, r["lon"], r["lat"], r["sig_lon"], r["sig_lat"], "P90", color, 0.10)
                add_ellipse(ax, Ellipse, r["lon"], r["lat"], r["sig_lon"], r["sig_lat"], "P50", color, 0.22)
                ax.text(r["lon"], r["lat"] + 0.35, f"{int(r['lead_h'])}h", fontsize=8, color=color, ha="center")
    ax.set_title("Probabilistic Forecast Cone\nGNO+DynGNN P50 darker fill; P90 lighter fill", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    return fig


def figure_trajectory_errors(ian, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    rng = np.random.default_rng(42)
    rows = []
    for model, color in COLORS.items():
        mdf = ian[ian["model"] == model]
        means, lows, highs = [], [], []
        for h in LEADS:
            cols = [f"true_lat_{h}h", f"true_lon_{h}h", f"pred_mu_lat_{h}h", f"pred_mu_lon_{h}h"]
            v = mdf[cols].dropna()
            errs = haversine_km(v[cols[0]], v[cols[1]], v[cols[2]], v[cols[3]]) if not v.empty else np.array([])
            if len(errs) == 0:
                means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            boot = np.array([rng.choice(errs, len(errs), replace=True).mean() for _ in range(1000)])
            means.append(float(errs.mean()))
            lows.append(float(np.percentile(boot, 2.5)))
            highs.append(float(np.percentile(boot, 97.5)))
            rows.append({"model": model, "lead_h": h, "mean_track_error_km": float(errs.mean()), "n_windows": int(len(errs))})
        ax.plot(LEADS, means, color=color, marker=MARKERS.get(model, "o"), lw=2.2, ms=6, label=model)
        ax.fill_between(LEADS, lows, highs, color=color, alpha=0.14)
    ax.set_title("Ian Case Study: Track Error by Lead Time", fontweight="bold")
    ax.set_xlabel("Forecast lead time (hours)")
    ax.set_ylabel("Mean great-circle track error (km)")
    ax.set_xticks(LEADS)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_DIR / "ian_track_error_by_lead.csv", index=False)
    return fig


def _node_layout(cfg) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(202207)
    layout = {}
    n_atm_side = int(round(math.sqrt(cfg.n_atm)))
    xs = np.linspace(-83.8, -80.0, n_atm_side)
    ys = np.linspace(25.0, 29.6, n_atm_side)
    xx, yy = np.meshgrid(xs, ys)
    layout["atm"] = np.c_[xx.ravel()[:cfg.n_atm], yy.ravel()[:cfg.n_atm]]
    layout["region"] = np.c_[rng.uniform(-83.6, -80.3, cfg.n_regions), rng.uniform(25.2, 29.4, cfg.n_regions)]
    layout["school"] = np.c_[rng.uniform(-83.4, -80.2, cfg.n_schools), rng.uniform(25.4, 29.2, cfg.n_schools)]
    layout["hospital"] = np.c_[rng.uniform(-83.3, -80.4, cfg.n_hospitals), rng.uniform(25.6, 29.0, cfg.n_hospitals)]
    layout["shelter"] = np.c_[rng.uniform(-83.5, -80.3, cfg.n_shelters), rng.uniform(25.5, 29.1, cfg.n_shelters)]
    layout["pop"] = np.c_[rng.uniform(-83.7, -80.1, cfg.n_pop), rng.uniform(25.3, 29.5, cfg.n_pop)]
    return layout


def _humanitarian_snapshot():
    import torch
    from model.disaster_graph.architecture import DisasterGNN
    from model.disaster_graph.config import DisasterGraphConfig
    from model.disaster_graph.schema import build_dataset, generate_humanitarian_report

    cfg = DisasterGraphConfig()
    cfg.apply_demo_overrides()
    model = DisasterGNN(cfg)
    ckpt = Path("checkpoints/disaster_graph/disaster_gnn_best.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state"], strict=False)
    model.eval()
    scenarios = build_dataset(cfg, seed=2022)
    best = max((sc for steps in scenarios for sc in steps), key=lambda s: float(s.node_features[:cfg.n_atm, 4].max()))
    with torch.no_grad():
        out = model(best)
    report = generate_humanitarian_report(cfg, out, best.node_features.numpy())
    return cfg, best, out, report


def figure_impact_map(plt, Polygon, LinearSegmentedColormap):
    cfg, sc, out, report = _humanitarian_snapshot()
    layout = _node_layout(cfg)
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    draw_noaa_basemap(ax, (-85, -79, 24.5, 30.5), Polygon)
    damage = out["damage_scores"].detach().cpu().numpy()
    wind = sc.node_features[:cfg.n_atm, 4].detach().cpu().numpy()
    cmap = LinearSegmentedColormap.from_list("noaa_damage", ["#fefce8", "#f97316", "#991b1b"])
    atm = layout["atm"]
    s = ax.scatter(atm[:, 0], atm[:, 1], c=wind, s=180, cmap="Blues", alpha=0.55, edgecolor="#075985", label="Wind proxy")

    offsets = {
        "region": cfg.n_atm,
        "school": cfg.n_atm + cfg.n_regions,
        "hospital": cfg.n_atm + cfg.n_regions + cfg.n_schools,
        "shelter": cfg.n_atm + cfg.n_regions + cfg.n_schools + cfg.n_hospitals,
        "pop": cfg.n_atm + cfg.n_regions + cfg.n_schools + cfg.n_hospitals + cfg.n_shelters,
    }
    markers = {"school": "s", "hospital": "P", "shelter": "^", "pop": "o"}
    labels = {"school": "Schools", "hospital": "Hospitals", "shelter": "Shelters", "pop": "Population"}
    for key in ["school", "hospital", "shelter", "pop"]:
        coords = layout[key]
        vals = damage[offsets[key]:offsets[key] + len(coords)]
        ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=95, marker=markers[key], cmap=cmap, vmin=0, vmax=1, edgecolor="#111827", linewidth=0.6, label=labels[key], zorder=4)
    cb = fig.colorbar(s, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Normalized wind proxy")
    ax.set_title("Synthetic Ian-Like Humanitarian Impact Map\nModule 3 proxy outputs at peak wind step", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "ian_impact_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2)
    return fig


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def figure_intervention_map(plt, Polygon, LinearSegmentedColormap):
    df = pd.read_csv(COUNTERFACTUAL_CSV)
    base = df[df["scenario"] == "baseline"].iloc[0]
    rows = df[df["scenario"] != "baseline"].copy()
    rows["delta_peak_exposure"] = rows["peak_exposure"].astype(float) - float(base["peak_exposure"])
    coords = {
        "earlier_evacuation": (-82.7, 27.5),
        "delayed_evacuation": (-81.2, 26.5),
        "shelter_failure": (-82.4, 28.7),
        "hospital_failure": (-80.8, 27.7),
        "road_blockage": (-81.7, 29.0),
        "intensity_increase": (-83.6, 26.2),
        "intensity_decrease": (-83.4, 25.4),
        "additional_emergency_resources": (-80.5, 28.5),
    }
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    draw_noaa_basemap(ax, (-85, -79, 24.5, 30.5), Polygon)
    vmax = max(abs(rows["delta_peak_exposure"]).max(), 1e-6)
    cmap = LinearSegmentedColormap.from_list("delta", ["#0f766e", "#f8fafc", "#b91c1c"])
    for _, r in rows.iterrows():
        x, y = coords.get(r["scenario"], (-82, 27))
        ax.scatter(x, y, c=[r["delta_peak_exposure"]], cmap=cmap, vmin=-vmax, vmax=vmax, s=420, edgecolor="#111827", linewidth=0.8, zorder=4)
        label = r["scenario"].replace("_", "\n")
        ax.text(x, y - 0.28, label, ha="center", va="top", fontsize=7.5, color="#111827")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(-vmax, vmax)
    cb = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Change in peak exposure vs baseline")
    ax.set_title("Counterfactual Intervention Impact Map\nScenario deltas from learned world-model rollout", fontweight="bold")
    rows.to_csv(METRICS_DIR / "ian_intervention_deltas.csv", index=False)
    return fig


def save_figure(fig, stem: str) -> Tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return png, pdf


def mirror_outputs(paths: Iterable[Path]) -> None:
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, RESULTS_FIG_DIR / path.name)


def write_manifest(paths: List[Path], ian: pd.DataFrame, true_df: pd.DataFrame) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{
        "artifact": str(p),
        "source_predictions": str(PREDS_CSV),
        "n_ian_prediction_rows": len(ian),
        "n_ian_t0_windows": int(ian["t0"].nunique()),
        "n_true_track_points": len(true_df),
        "protocol": "Hurricane Ian test-case figure regenerated from current metrics",
    } for p in paths]
    with (METRICS_DIR / "ian_case_study_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(paths: List[Path], ian: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Hurricane Ian Case Study Figure Audit\n\n"
        "## Root Causes Addressed\n"
        "- The previous case-study plot inverted longitude, which reverses west-east geography.\n"
        "- Uncertainty cones were only shown as a final-lead diagnostic instead of explicit P50/P90 probability regions.\n"
        "- Impact and intervention visualizations were not separated into publication figures.\n\n"
        "## Regenerated Outputs\n"
        + "\n".join(f"- `{p}`" for p in paths)
        + "\n\n## Protocol\n"
        f"- Source predictions: `{PREDS_CSV}`\n"
        f"- Ian prediction rows: {len(ian)}\n"
        f"- Ian forecast windows: {ian['t0'].nunique()}\n"
        "- Axes use true latitude and longitude; western longitudes remain negative and are not inverted.\n"
        "- Coastlines are schematic offline context layers, not operational NOAA GIS products.\n"
        "- Humanitarian and intervention maps are synthetic/proxy research visualizations and must not be interpreted as observed damage maps.\n",
        encoding="utf-8",
    )


def main() -> None:
    _require_inputs()
    plt, Ellipse, Polygon, LinearSegmentedColormap = _setup_matplotlib()
    ian = load_ian_predictions()
    true_df = true_track(ian)
    generated: List[Path] = []

    specs = [
        ("ian_noaa_track_map", lambda: figure_track_map(ian, true_df, plt, Ellipse, Polygon)),
        ("ian_uncertainty_cones", lambda: figure_uncertainty_cones(ian, true_df, plt, Ellipse, Polygon)),
        ("ian_trajectory_errors", lambda: figure_trajectory_errors(ian, plt)),
        ("ian_impact_map", lambda: figure_impact_map(plt, Polygon, LinearSegmentedColormap)),
        ("ian_intervention_map", lambda: figure_intervention_map(plt, Polygon, LinearSegmentedColormap)),
    ]
    figs = []
    for stem, fn in specs:
        fig = fn()
        png, pdf = save_figure(fig, stem)
        generated.extend([png, pdf])
        figs.append((stem, fig))
        plt.close(fig)
        print(f"Saved {png}")

    # Combined publication panel.
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    panel_fns = [
        lambda ax: (draw_noaa_basemap(ax, map_extent(ian), Polygon), ax.plot(true_df["lon"], true_df["lat"], color="#111827", lw=2.4, marker="o", ms=4), ax.set_title("Best Track Map")),
        lambda ax: (draw_noaa_basemap(ax, map_extent(ian), Polygon), [ax.plot(forecast_track(ian, m)["lon"], forecast_track(ian, m)["lat"], marker=MARKERS.get(m, "o"), color=c, lw=1.5, label=m) for m, c in COLORS.items()], ax.legend(fontsize=7), ax.set_title("Forecast Trajectories")),
        lambda ax: _mini_error_panel(ax, ian),
        lambda ax: _mini_cone_panel(ax, ian),
        lambda ax: _mini_counterfactual_panel(ax),
        lambda ax: _mini_artifact_panel(ax),
    ]
    for ax, fn in zip(axes.ravel(), panel_fns):
        fn(ax)
    fig.suptitle("Hurricane Ian Case Study: Forecast, Uncertainty, Impact, and Interventions", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png, pdf = save_figure(fig, "ian_publication_multipanel")
    generated.extend([png, pdf])
    plt.close(fig)

    # Legacy compatibility filenames.
    legacy_map = {
        OUT_DIR / "ian_publication_multipanel.png": LEGACY_DIR / "case_study_ian_combined.png",
        OUT_DIR / "ian_publication_multipanel.pdf": LEGACY_DIR / "case_study_ian_combined.pdf",
        OUT_DIR / "ian_noaa_track_map.png": LEGACY_DIR / "case_study_ian_panel1_track.png",
        OUT_DIR / "ian_trajectory_errors.png": LEGACY_DIR / "case_study_ian_panel2_error.png",
        OUT_DIR / "ian_uncertainty_cones.png": LEGACY_DIR / "case_study_ian_panel3_cone.png",
        OUT_DIR / "ian_impact_map.png": LEGACY_DIR / "case_study_ian_panel4_humanitarian.png",
    }
    for src, dst in legacy_map.items():
        if src.exists():
            shutil.copy2(src, dst)
            generated.append(dst)

    mirror_outputs(generated)
    write_manifest(generated, ian, true_df)
    write_report(generated, ian)
    print(f"Saved {REPORT_PATH}")


def _mini_error_panel(ax, ian):
    for model, color in COLORS.items():
        vals = []
        for h in LEADS:
            v = ian[ian["model"] == model][[f"true_lat_{h}h", f"true_lon_{h}h", f"pred_mu_lat_{h}h", f"pred_mu_lon_{h}h"]].dropna()
            vals.append(haversine_km(v.iloc[:, 0], v.iloc[:, 1], v.iloc[:, 2], v.iloc[:, 3]).mean() if not v.empty else np.nan)
        ax.plot(LEADS, vals, marker=MARKERS.get(model, "o"), color=color, label=model)
    ax.set_title("Track Error")
    ax.set_xlabel("Lead h")
    ax.set_ylabel("km")
    ax.grid(True, alpha=0.3)


def _mini_cone_panel(ax, ian):
    ax.axhline(0.9, color="#64748b", ls="--", lw=1, label="Ideal P90")
    for model, color in COLORS.items():
        vals = []
        for h in LEADS:
            cols = [f"true_lat_{h}h", f"true_lon_{h}h", f"pred_mu_lat_{h}h", f"pred_mu_lon_{h}h", f"pred_sigma_lat_{h}h", f"pred_sigma_lon_{h}h"]
            if cols[-1] not in ian.columns:
                vals.append(np.nan)
                continue
            v = ian[ian["model"] == model][cols].dropna()
            if v.empty:
                vals.append(np.nan)
                continue
            dx = (v[cols[1]] - v[cols[3]]) / v[cols[5]].clip(lower=1e-6)
            dy = (v[cols[0]] - v[cols[2]]) / v[cols[4]].clip(lower=1e-6)
            vals.append(float(((dx ** 2 + dy ** 2) <= 2.146 ** 2).mean()))
        ax.plot(LEADS, vals, marker=MARKERS.get(model, "o"), color=color, label=model)
    ax.set_ylim(0, 1.05)
    ax.set_title("P90 Coverage")
    ax.set_xlabel("Lead h")
    ax.grid(True, alpha=0.3)


def _mini_counterfactual_panel(ax):
    df = pd.read_csv(COUNTERFACTUAL_CSV)
    base = float(df[df["scenario"] == "baseline"].iloc[0]["peak_exposure"])
    rows = df[df["scenario"] != "baseline"].copy()
    rows["delta"] = rows["peak_exposure"].astype(float) - base
    rows = rows.sort_values("delta")
    ax.barh(rows["scenario"].str.replace("_", " "), rows["delta"], color=np.where(rows["delta"] < 0, "#0f766e", "#b91c1c"))
    ax.axvline(0, color="#111827", lw=1)
    ax.set_title("Intervention Delta")
    ax.set_xlabel("Peak exposure delta")
    ax.tick_params(axis="y", labelsize=7)


def _mini_artifact_panel(ax):
    ax.axis("off")
    lines = [
        "Generated artifacts",
        "Track map",
        "P50/P90 cones",
        "Trajectory error",
        "Impact proxy map",
        "Intervention delta map",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12, fontweight="bold")


if __name__ == "__main__":
    main()
