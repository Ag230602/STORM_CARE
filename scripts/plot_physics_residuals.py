"""Plot full-vs-ablation PI-GNO residual curves after retraining."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def main() -> None:
    cache_dir = Path(".cache/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_dir = Path("metrics/physics/full")
    ablation_dir = Path("metrics/physics/no_physics")
    full_train = pd.read_csv(full_dir / "pigno_train_log.csv")
    abl_train = pd.read_csv(ablation_dir / "pigno_train_log.csv")
    full_val = pd.read_csv(full_dir / "pigno_val_metrics.csv")
    abl_val = pd.read_csv(ablation_dir / "pigno_val_metrics.csv")

    residual_cols = ["R_adv", "R_diff", "R_mass", "R_wp", "R_cont", "R_nrg"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, col in zip(axes.ravel(), residual_cols):
        ax.plot(full_train["epoch"], full_train[col], label="full physics", linewidth=2)
        ax.plot(abl_train["epoch"], abl_train[col], label="no physics", linestyle="--")
        ax.set_title(col)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    fig.suptitle("PI-GNO normalized physics residuals")
    fig.tight_layout()

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    png = out_dir / "physics_residuals_full_vs_ablation.png"
    pdf = out_dir / "physics_residuals_full_vs_ablation.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for name, train, val in [
        ("full_physics", full_train, full_val),
        ("no_physics", abl_train, abl_val),
    ]:
        last_train = train.iloc[-1]
        final_val = val.iloc[-1]
        best_val = val.sort_values("val_total").iloc[0]
        row = {
            "run": name,
            "final_L_data": last_train["L_data"],
            "final_L_phys": last_train["L_phys"],
            "final_total": last_train["total"],
            "final_val_epoch": final_val["epoch"],
            "final_val_total": final_val["val_total"],
            "final_val_L_data": final_val["val_L_data"],
            "final_val_L_phys": final_val["val_L_phys"],
            "final_val_track_rmse": final_val["val_track_rmse"],
            "best_val_total": best_val["val_total"],
            "best_val_track_rmse": best_val["val_track_rmse"],
        }
        for col in residual_cols:
            row[f"final_{col}"] = last_train[col]
        for col in ["val_R_adv", "val_R_diff", "val_R_mass", "val_R_wp", "val_R_cont", "val_R_nrg"]:
            if col in final_val:
                row[f"final_{col}"] = final_val[col]
        for col in ["val_R_adv", "val_R_diff", "val_R_mass", "val_R_wp", "val_R_cont", "val_R_nrg"]:
            if col in best_val:
                row[f"best_{col}"] = best_val[col]
        summary_rows.append(row)

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv("metrics/physics/physics_full_vs_ablation.csv", index=False)
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    print("Saved metrics/physics/physics_full_vs_ablation.csv")


if __name__ == "__main__":
    main()
