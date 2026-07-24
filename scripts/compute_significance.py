#!/usr/bin/env python
"""E4 — Storm-level bootstrap CIs and Holm-corrected paired tests.

Self-contained: needs only pandas / numpy / scipy.

Input: a long-format CSV of per-window errors with (at least) the columns
    storm_id, model, horizon, error
Column names are configurable via CLI flags, so the same script serves the
track benchmark (error = track error km), humanitarian metrics
(horizon can be a constant, error = per-scenario metric), scenario deltas
(storm_id -> sequence_id), and ablation deltas.

Statistical design (matches the manuscript's promises):
  * Windows within a storm are correlated, so ALL resampling and pairing is
    done at the storm level: per-storm mean error first, then statistics.
  * Bootstrap 95% CI: resample storms with replacement (default B=10000).
  * Paired tests: Wilcoxon signed-rank on per-storm paired differences
    between each model and the reference model, per horizon.
  * Holm correction is applied across the whole headline family
    (all model-vs-reference comparisons at all horizons), which is the
    conservative reading of "Holm-corrected for headline comparisons".

Outputs (written to --out-dir):
  significance_summary.csv   per model x horizon: mean, storm-bootstrap CI
  significance_pairwise.csv  per model-pair x horizon: mean diff, CI of the
                             diff, raw p, Holm-adjusted p, claimable flag

Example:
  python scripts/compute_significance.py \
      --input metrics/inference_test_errors_long.csv \
      --reference Persistence --out-dir metrics/significance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def per_storm_means(df: pd.DataFrame, storm_col: str, model_col: str,
                    horizon_col: str, error_col: str) -> pd.DataFrame:
    """Collapse windows to one mean error per storm x model x horizon."""
    g = (df.groupby([storm_col, model_col, horizon_col], as_index=False)
           [error_col].mean())
    return g


def bootstrap_ci(values_by_storm: pd.Series, n_boot: int, seed: int,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """values_by_storm: index = storm_id, one mean value per storm."""
    vals = values_by_storm.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(vals.mean()), float(lo), float(hi)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="long-format CSV of errors")
    ap.add_argument("--out-dir", default="metrics/significance")
    ap.add_argument("--storm-col", default="storm_id")
    ap.add_argument("--model-col", default="model")
    ap.add_argument("--horizon-col", default="horizon")
    ap.add_argument("--error-col", default="error")
    ap.add_argument("--reference", default="Persistence",
                    help="reference model for the headline paired family")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-pairs", action="store_true",
                    help="test every model pair, not just vs the reference "
                         "(Holm family grows accordingly)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    for c in (args.storm_col, args.model_col, args.horizon_col, args.error_col):
        if c not in df.columns:
            raise SystemExit(f"column '{c}' not in {args.input}; "
                             f"have {list(df.columns)}")

    sm = per_storm_means(df, args.storm_col, args.model_col,
                         args.horizon_col, args.error_col)
    models = sorted(sm[args.model_col].unique())
    horizons = sorted(sm[args.horizon_col].unique())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per model x horizon summary with storm-bootstrap CI ----
    rows = []
    for m in models:
        for h in horizons:
            sub = sm[(sm[args.model_col] == m) & (sm[args.horizon_col] == h)]
            series = sub.set_index(args.storm_col)[args.error_col]
            mean, lo, hi = bootstrap_ci(series, args.n_boot, args.seed)
            rows.append({"model": m, "horizon": h, "n_storms": len(series),
                         "mean": mean, "ci95_lo": lo, "ci95_hi": hi})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "significance_summary.csv", index=False)

    # ---- paired family ----
    if args.all_pairs:
        pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    else:
        if args.reference not in models:
            raise SystemExit(f"reference '{args.reference}' not among models "
                             f"{models}")
        pairs = [(m, args.reference) for m in models if m != args.reference]

    pair_rows = []
    for (a, b) in pairs:
        for h in horizons:
            wa = sm[(sm[args.model_col] == a) & (sm[args.horizon_col] == h)] \
                .set_index(args.storm_col)[args.error_col]
            wb = sm[(sm[args.model_col] == b) & (sm[args.horizon_col] == h)] \
                .set_index(args.storm_col)[args.error_col]
            common = wa.index.intersection(wb.index)
            da, db = wa.loc[common], wb.loc[common]
            diff = (da - db).to_numpy(dtype=float)
            n = len(diff)
            if n < 3 or np.allclose(diff, 0):
                p = np.nan
            else:
                try:
                    p = float(stats.wilcoxon(diff, zero_method="wilcox",
                                             alternative="two-sided").pvalue)
                except ValueError:
                    p = np.nan
            mean_d, lo_d, hi_d = bootstrap_ci(pd.Series(diff, index=common),
                                              args.n_boot, args.seed)
            pair_rows.append({"model_a": a, "model_b": b, "horizon": h,
                              "n_paired_storms": n, "mean_diff_a_minus_b": mean_d,
                              "diff_ci95_lo": lo_d, "diff_ci95_hi": hi_d,
                              "p_raw": p})
    pw = pd.DataFrame(pair_rows)
    mask = pw["p_raw"].notna()
    pw["p_holm"] = np.nan
    if mask.any():
        pw.loc[mask, "p_holm"] = holm_adjust(pw.loc[mask, "p_raw"].to_numpy())
    pw["claimable_at_0.05"] = (pw["p_holm"] < 0.05) & \
                              ((pw["diff_ci95_lo"] > 0) | (pw["diff_ci95_hi"] < 0))
    pw.to_csv(out_dir / "significance_pairwise.csv", index=False)

    print(f"wrote {out_dir/'significance_summary.csv'} "
          f"({len(summary)} rows) and {out_dir/'significance_pairwise.csv'} "
          f"({len(pw)} rows)")
    print("Manuscript rule: a comparison may be phrased as a win ONLY if "
          "claimable_at_0.05 is True; otherwise report as 'comparable'.")


if __name__ == "__main__":
    main()
