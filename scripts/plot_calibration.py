"""
plot_calibration.py — Cone coverage (P50 / P90) vs lead time figure.

Reads tables/table_calibration_cone_coverage.csv and produces a
publication-quality matplotlib figure saved to figures/calibration.png.

The ideal P50 = 0.50 (dashed line) and P90 = 0.90 (dashed line)
are shown as horizontal references.

Usage:
    python scripts/plot_calibration.py
Outputs:
    figures/calibration.png
    figures/calibration.pdf
"""
import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

CSV_PATH = "tables/table_calibration_cone_coverage.csv"


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    leads = [int(r["lead_time_h"]) for r in rows]
    p50_ep8  = [float(r["cone_p50_ep8"])  for r in rows]
    p90_ep8  = [float(r["cone_p90_ep8"])  for r in rows]
    p50_ep4  = [float(r["cone_p50_ep4"])  for r in rows]
    p90_ep4  = [float(r["cone_p90_ep4"])  for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Ideal references
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Ideal P90 = 0.90")
    ax.axhline(0.50, color="gray", linestyle=":",  linewidth=1.2, alpha=0.7, label="Ideal P50 = 0.50")

    # Model results (ep 8 = best short-range checkpoint)
    ax.plot(leads, p90_ep8, "o-", color="#1f77b4", linewidth=2,
            markersize=7, label="STORM-CARE P90 (ep 8)")
    ax.plot(leads, p50_ep8, "s--", color="#1f77b4", linewidth=1.5,
            markersize=6, alpha=0.8, label="STORM-CARE P50 (ep 8)")

    # ep 4 for comparison (dashed, lighter)
    ax.plot(leads, p90_ep4, "^-", color="#aec7e8", linewidth=1.2,
            markersize=5, label="STORM-CARE P90 (ep 4)", alpha=0.6)

    # Annotate P90 values at each lead
    for x, y in zip(leads, p90_ep8):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")

    ax.set_xlabel("Lead time (hours)", fontsize=12)
    ax.set_ylabel("Cone coverage fraction", fontsize=12)
    ax.set_title("Forecast Cone Coverage vs Lead Time\n"
                 "(STORM-CARE Foundation Model, validation set)", fontsize=11)
    ax.set_xticks(leads)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/calibration.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/calibration.pdf", bbox_inches="tight")
    print("Saved figures/calibration.png and figures/calibration.pdf")

    plt.close()


if __name__ == "__main__":
    main()
