#!/usr/bin/env python
"""E3 — Dose-response monotonicity check with bootstrap uncertainty.

Fully functional given the sweep outputs. Accepts either:

  (a) PREFERRED: a per-sequence long CSV with columns
        sequence_id, scenario, peak_exposure
      -> enables a bootstrap over the 24 held-out sequences and a
         sequence-paired sign test of each adjacent ordering.

  (b) FALLBACK: the aggregated counterfactual_outcomes.csv with columns
        scenario, peak_exposure  (one row per scenario)
      -> point-estimate ordering check only (no uncertainty). The script
         will nag you to emit the per-sequence file from
         model/counterfactual/run.py (a ~3-line change: write per-sequence
         outcomes before averaging).

Verdicts written to <out-dir>/dose_response_verdict.csv and printed:
  * point-estimate monotone?           baseline > 12h > 24h > 36h exposure
  * bootstrap P(monotone)              fraction of sequence-resamples in
                                       which the full ordering holds
  * adjacent-step paired Wilcoxon      12h vs baseline, 24h vs 12h, 36h vs 24h

Manuscript rule this enforces: the abstract may claim a monotone
dose-response ONLY if the point ordering holds AND bootstrap P(monotone)
is high (report the number itself; >=0.95 is a comfortable bar). Otherwise
the claim is cut and the sweep is reported as a limitation finding.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ORDER = ["baseline", "earlier_evacuation_12h",
                 "earlier_evacuation_24h", "earlier_evacuation_36h"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="metrics/counterfactual")
    ap.add_argument("--metric-col", default="peak_exposure")
    ap.add_argument("--scenario-col", default="scenario")
    ap.add_argument("--sequence-col", default="sequence_id")
    ap.add_argument("--order", nargs="+", default=DEFAULT_ORDER,
                    help="scenario names from least to most intervention; "
                         "metric must strictly DECREASE along this order")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    missing = [s for s in args.order if s not in set(df[args.scenario_col])]
    if missing:
        raise SystemExit(f"scenarios missing from {args.input}: {missing}\n"
                         f"present: {sorted(df[args.scenario_col].unique())}")

    per_sequence = args.sequence_col in df.columns
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if not per_sequence:
        means = (df.set_index(args.scenario_col)[args.metric_col]
                   .loc[args.order].to_numpy(dtype=float))
        monotone = bool(np.all(np.diff(means) < 0))
        for s, m in zip(args.order, means):
            rows.append({"scenario": s, "mean": m})
        verdict = pd.DataFrame(rows)
        verdict.to_csv(out_dir / "dose_response_verdict.csv", index=False)
        print("means along order:", dict(zip(args.order, means.round(4))))
        print(f"point-estimate monotone decrease: {monotone}")
        print("WARNING: aggregated input only — no uncertainty possible. "
              "Emit per-sequence outcomes from model/counterfactual/run.py "
              "and rerun for a defensible claim.")
        return

    # ---- per-sequence path: pivot to [sequence x scenario] ----
    pivot = df.pivot_table(index=args.sequence_col, columns=args.scenario_col,
                           values=args.metric_col, aggfunc="mean")[args.order]
    pivot = pivot.dropna()
    mat = pivot.to_numpy(dtype=float)          # [n_seq, n_scenarios]
    n_seq = mat.shape[0]
    means = mat.mean(axis=0)
    point_monotone = bool(np.all(np.diff(means) < 0))

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_seq, size=(args.n_boot, n_seq))
    boot_means = mat[idx].mean(axis=1)         # [n_boot, n_scenarios]
    p_monotone = float(np.mean(np.all(np.diff(boot_means, axis=1) < 0, axis=1)))

    from scipy import stats
    print(f"n sequences: {n_seq}")
    for s, m in zip(args.order, means):
        lo, hi = np.quantile(boot_means[:, args.order.index(s)], [0.025, 0.975])
        rows.append({"scenario": s, "mean": float(m),
                     "ci95_lo": float(lo), "ci95_hi": float(hi)})
        print(f"  {s:32s} mean={m:.4f}  CI95=[{lo:.4f}, {hi:.4f}]")
    print(f"point-estimate monotone decrease: {point_monotone}")
    print(f"bootstrap P(fully monotone):      {p_monotone:.3f}")

    adj = []
    for a, b in zip(args.order[:-1], args.order[1:]):
        diff = pivot[b] - pivot[a]             # want negative
        try:
            p = float(stats.wilcoxon(diff, alternative="less").pvalue)
        except ValueError:
            p = np.nan
        adj.append({"step": f"{b} vs {a}", "mean_diff": float(diff.mean()),
                    "wilcoxon_p_one_sided_less": p})
        print(f"  {b} vs {a}: mean diff {diff.mean():+.4f}, "
              f"one-sided p={p:.4g}")

    pd.DataFrame(rows).to_csv(out_dir / "dose_response_verdict.csv", index=False)
    pd.DataFrame(adj).to_csv(out_dir / "dose_response_adjacent_tests.csv",
                             index=False)
    claim = point_monotone and p_monotone >= 0.95
    print(f"\nABSTRACT CLAIM SUPPORTED (monotone + P>=0.95): {claim}")
    if not claim:
        print("-> cut the dose-response sentence from abstract/intro/"
              "conclusion; report the sweep as a limitation finding.")


if __name__ == "__main__":
    main()
