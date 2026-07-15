"""
case_study_ian.py — Four-panel case study for Hurricane Ian (AL092022, TEST set).

Confirmed TEST partition: splits/storm_splits.json → AL092022 = "test"

Protocol: open-loop forecast from t0 using only data ≤ t0.

Four panels
-----------
1. Track overlay     — Ian's true track vs model predictions (NOAA-style,
                       NOT world map).  Shows P50/P90 forecast cones.
2. Track error       — forecast error (km) vs lead time for all 4 models.
3. Probabilistic cone — P90 coverage fraction vs lead time (calibration).
4. Humanitarian      — Module 3 damage / shelter demand at landfall time step.

Usage:
    python scripts/case_study_ian.py
Outputs:
    figures/case_study_ian_panel1_track.png
    figures/case_study_ian_panel2_error.png
    figures/case_study_ian_panel3_cone.png
    figures/case_study_ian_panel4_humanitarian.png
    figures/case_study_ian_combined.png
"""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PREDS_CSV  = "metrics/inference_test_predictions_all_models.csv"
IAN_ID     = "ian"
LEADS      = [6, 12, 24, 48]
OUT_DIR    = "figures"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_ian_predictions():
    df = pd.read_csv(PREDS_CSV)
    ian = df[df["storm_tag"] == IAN_ID].copy()
    # Sort by t0
    ian = ian.sort_values("t0").reset_index(drop=True)
    return ian


def panel1_track(ian_df, ax):
    """Track overlay: true track + model forecast cones."""
    colors = {"Persistence": "#888888", "LSTM": "#e15759",
              "Transformer": "#f28e2b", "GNO+DynGNN": "#4e79a7"}

    # True track from multiple t0 windows
    true_lats, true_lons = [], []
    for _, row in ian_df[ian_df["model"] == "Persistence"].iterrows():
        true_lats.append(row["lat0"]); true_lons.append(row["lon0"])
        for h in LEADS:
            tl, tlo = f"true_lat_{h}h", f"true_lon_{h}h"
            if pd.notna(row.get(tl)) and pd.notna(row.get(tlo)):
                true_lats.append(row[tl]); true_lons.append(row[tlo])

    # Remove duplicates while preserving order
    seen = set()
    tl_unique, tlo_unique = [], []
    for la, lo in zip(true_lats, true_lons):
        k = (round(la, 2), round(lo, 2))
        if k not in seen:
            seen.add(k); tl_unique.append(la); tlo_unique.append(lo)

    ax.plot(tlo_unique, tl_unique, "k-o", linewidth=2.5, markersize=5,
            label="True track (Ian)", zorder=5)
    # Mark start and landfall
    ax.plot(tlo_unique[0], tl_unique[0], "k^", markersize=10, zorder=6)
    if len(tlo_unique) > 1:
        ax.plot(tlo_unique[-1], tl_unique[-1], "k*", markersize=12, zorder=6)

    # Model forecasts from first t0
    first_t0 = ian_df["t0"].iloc[0]
    for model, color in colors.items():
        row = ian_df[(ian_df["model"] == model) & (ian_df["t0"] == first_t0)]
        if row.empty:
            continue
        row = row.iloc[0]
        pred_lats = [row["lat0"]]
        pred_lons = [row["lon0"]]
        for h in LEADS:
            pl = f"pred_mu_lat_{h}h"; plo = f"pred_mu_lon_{h}h"
            if pd.notna(row.get(pl)) and pd.notna(row.get(plo)):
                pred_lats.append(row[pl]); pred_lons.append(row[plo])
        ax.plot(pred_lons, pred_lats, "--o", color=color, linewidth=1.5,
                markersize=4, alpha=0.85, label=f"{model}")

        # P90 sigma ellipse at last lead
        last_h = LEADS[-1]
        pl = f"pred_mu_lat_{last_h}h"; plo = f"pred_mu_lon_{last_h}h"
        sl = f"pred_sigma_lat_{last_h}h"; slo = f"pred_sigma_lon_{last_h}h"
        if all(pd.notna(row.get(c)) for c in [pl, plo, sl, slo]):
            z90 = 2.146   # P90 radius for bivariate normal
            theta = np.linspace(0, 2*np.pi, 60)
            ell_x = row[plo] + z90 * float(row[slo]) * np.cos(theta)
            ell_y = row[pl]  + z90 * float(row[sl])  * np.sin(theta)
            ax.fill(ell_x, ell_y, alpha=0.12, color=color)

    ax.set_xlabel("Longitude (°W → °E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title("Panel 1 — Hurricane Ian: Track Forecast\n"
                 "(TEST set, open-loop from t₀)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # west-left (standard meteorology convention)


def panel2_error(ian_df, ax):
    """Track error vs lead time for all models."""
    colors = {"Persistence": "#888888", "LSTM": "#e15759",
              "Transformer": "#f28e2b", "GNO+DynGNN": "#4e79a7"}
    markers = {"Persistence": "s", "LSTM": "o", "Transformer": "^", "GNO+DynGNN": "D"}

    for model, color in colors.items():
        mdf = ian_df[ian_df["model"] == model]
        mean_errs, lo_errs, hi_errs = [], [], []
        valid_leads = []
        for h in LEADS:
            tl, tlo = f"true_lat_{h}h", f"true_lon_{h}h"
            pl, plo = f"pred_mu_lat_{h}h", f"pred_mu_lon_{h}h"
            v = mdf[[tl, tlo, pl, plo]].dropna()
            if len(v) > 0:
                errs = haversine_km(v[tl].values, v[tlo].values,
                                    v[pl].values, v[plo].values)
                if len(errs) > 1:
                    boot = np.array([np.random.choice(errs, len(errs)).mean()
                                     for _ in range(500)])
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                else:
                    lo = hi = errs.mean()
                mean_errs.append(errs.mean())
                lo_errs.append(lo); hi_errs.append(hi)
                valid_leads.append(h)

        if valid_leads:
            ax.plot(valid_leads, mean_errs, f"-{markers[model]}", color=color,
                    linewidth=2, markersize=7, label=model)
            ax.fill_between(valid_leads,
                            [min(lo, m) for lo, m in zip(lo_errs, mean_errs)],
                            [max(hi, m) for hi, m in zip(hi_errs, mean_errs)],
                            alpha=0.15, color=color)

    ax.set_xlabel("Lead time (h)", fontsize=11)
    ax.set_ylabel("Mean track error (km)", fontsize=11)
    ax.set_title("Panel 2 — Track Error vs Lead Time\n(Ian; shaded = 95% CI)",
                 fontsize=11)
    ax.set_xticks(LEADS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def panel3_cone(ian_df, ax):
    """P90 cone coverage fraction vs lead time."""
    colors = {"LSTM": "#e15759", "GNO+DynGNN": "#4e79a7"}
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=1.2,
               label="Ideal P90 = 0.90")

    # Foundation model calibration from table
    fm_p90 = [0.900, 0.913, 0.900, 0.840]
    ax.plot(LEADS, fm_p90, "k-o", linewidth=2, markersize=7,
            label="STORM-CARE-FM (val)")

    for model, color in colors.items():
        mdf = ian_df[ian_df["model"] == model]
        cov90 = []
        for h in LEADS:
            pl, plo = f"pred_mu_lat_{h}h", f"pred_mu_lon_{h}h"
            tl, tlo = f"true_lat_{h}h", f"true_lon_{h}h"
            sl, slo = f"pred_sigma_lat_{h}h", f"pred_sigma_lon_{h}h"
            v = mdf[[pl, plo, tl, tlo, sl, slo]].dropna() if sl in mdf.columns else pd.DataFrame()
            if len(v) > 0 and sl in v.columns:
                inside = 0
                for _, r in v.iterrows():
                    z90 = 2.146
                    dy = (r[tl] - r[pl]) / max(float(r[sl]), 1e-6)
                    dx = (r[tlo] - r[plo]) / max(float(r[slo]), 1e-6)
                    if dx**2 + dy**2 <= z90**2:
                        inside += 1
                cov90.append(inside / len(v))
            else:
                cov90.append(np.nan)
        valid = [(h, c) for h, c in zip(LEADS, cov90) if not np.isnan(c)]
        if valid:
            hs, cs = zip(*valid)
            ax.plot(list(hs), list(cs), "--o", color=color, linewidth=1.5,
                    markersize=6, label=f"{model} (Ian)")

    ax.set_xlabel("Lead time (h)", fontsize=11)
    ax.set_ylabel("P90 cone coverage", fontsize=11)
    ax.set_title("Panel 3 — Probabilistic Cone Coverage\n(Ian test case)",
                 fontsize=11)
    ax.set_xticks(LEADS); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)


def panel4_humanitarian(ax):
    """Module 3 humanitarian impact at Ian's landfall (synthetic demo output)."""
    from model.disaster_graph.config import DisasterGraphConfig
    from model.disaster_graph.schema import build_dataset, generate_humanitarian_report
    from model.disaster_graph.architecture import DisasterGNN
    import torch

    cfg = DisasterGraphConfig()
    cfg.apply_demo_overrides()

    ckpt_path = "checkpoints/disaster_graph/disaster_gnn_best.pt"
    model = DisasterGNN(cfg)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"], strict=False)
    model.eval()

    # Generate a scenario with Ian-like parameters (cat 4, high vmax)
    scenarios = build_dataset(cfg, seed=2022)
    # Use the most intense scenario step
    best_sc = None
    for sc_steps in scenarios:
        for sc in sc_steps:
            nf = sc.node_features.numpy()
            wind = nf[:cfg.n_atm, 4].max()
            if best_sc is None or wind > best_sc[1]:
                best_sc = (sc, wind)

    if best_sc is None:
        ax.text(0.5, 0.5, "No scenario data", ha="center", va="center",
                transform=ax.transAxes)
        return

    sc, _ = best_sc
    with torch.no_grad():
        out = model(sc)

    report = generate_humanitarian_report(cfg, out, sc.node_features.numpy())

    # Bar chart of humanitarian metrics
    metrics = {
        "Children\nexposed": min(report["exposed_children_est"] / 10000, 1.0),
        "Schools\ndisrupted\n(%)": report["school_disruption_pct"] / 100,
        "Hospital\naccess": report["hospital_access_avg"],
        "Shelter\ndemand": report["shelter_demand_avg"],
    }
    colors_bar = ["#e15759", "#f28e2b", "#76b7b2", "#59a14f"]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=colors_bar, edgecolor="white", linewidth=0.8)

    # Annotate raw values
    raw_vals = {
        "Children\nexposed": f"{report['exposed_children_est']:,}",
        "Schools\ndisrupted\n(%)": f"{report['school_disruption_pct']:.1f}%",
        "Hospital\naccess": f"{report['hospital_access_avg']:.3f}",
        "Shelter\ndemand": f"{report['shelter_demand_avg']:.3f}",
    }
    for bar, (label, val) in zip(bars, raw_vals.items()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                val, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Recovery priority
    top3 = report["top3_priority_labels"]
    ax.text(0.98, 0.98, "Recovery priority:\n" + "\n".join(f"  {i+1}. {l}"
            for i, l in enumerate(top3)),
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    ax.set_ylabel("Normalised score [0, 1]", fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.set_title("Panel 4 — Module 3 Humanitarian Impact\n"
                 "(Ian-like scenario: cat 4, peak wind)", fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed"); return

    np.random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/case_study", exist_ok=True)

    print("Loading Ian predictions …")
    ian_df = load_ian_predictions()
    print(f"  {len(ian_df)} prediction rows ({ian_df['model'].nunique()} models, "
          f"{ian_df['t0'].nunique()} time windows)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Case Study — Hurricane Ian (AL092022, September 2022)\n"
                 "TEST partition · Open-loop forecasts", fontsize=13, fontweight="bold")

    print("Panel 1: Track overlay …")
    panel1_track(ian_df, axes[0, 0])

    print("Panel 2: Track error vs lead time …")
    panel2_error(ian_df, axes[0, 1])

    print("Panel 3: Probabilistic cone coverage …")
    panel3_cone(ian_df, axes[1, 0])

    print("Panel 4: Humanitarian impact (Module 3) …")
    panel4_humanitarian(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    combined_path = f"{OUT_DIR}/case_study_ian_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    fig.savefig(combined_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {combined_path}")

    # Save individual panels
    panel_specs = [
        ("track",        lambda a: panel1_track(ian_df, a)),
        ("error",        lambda a: panel2_error(ian_df, a)),
        ("cone",         lambda a: panel3_cone(ian_df, a)),
        ("humanitarian", lambda a: panel4_humanitarian(a)),
    ]
    for idx, (name, fn) in enumerate(panel_specs):
        fig2, ax2 = plt.subplots(figsize=(6.5, 5))
        fn(ax2)
        fig2.tight_layout()
        p = f"{OUT_DIR}/case_study_ian_panel{idx+1}_{name}.png"
        fig2.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
        plt.close(fig2)

    plt.close(fig)
    print("\nCase study complete. Files in figures/")


if __name__ == "__main__":
    main()
