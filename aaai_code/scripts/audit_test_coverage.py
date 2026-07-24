#!/usr/bin/env python
"""E1 step 1 — ERA5 coverage manifest for the full test partition.

Generalizes the audit that produced metrics/baseline_input_audit.csv (which
covered only Irma + Ian) to every storm in the test partition, so you know
the true size of the ERA5-complete test benchmark BEFORE committing compute.

ADAPTER REQUIRED: two functions below must be wired to the repo's existing
loaders. Everything else is functional. The window-completeness logic should
be IDENTICAL to what the corrected benchmark uses (t0 included in history,
same crop size, same required channels) — import it rather than reimplement
if possible, so the manifest and the benchmark can never disagree.

Outputs:
  metrics/test_coverage/test_coverage_manifest.csv   one row per test storm
  metrics/test_coverage/test_coverage_summary.json   totals for the paper's
                                                     coverage footnote
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "metrics" / "test_coverage"


# ----------------------- ADAPTERS: wire these two -------------------------
def load_test_storm_ids() -> list[str]:
    """Return the 107 test-partition storm ids.

    TODO(Adrija): read from the frozen split artifact the rest of the repo
    uses — splits/storm_splits.json or metrics/dataset_integrity/
    split_map_counts.csv — NOT by re-deriving years, so this can never
    drift from the audited split.
    """
    split_file = ROOT / "splits" / "storm_splits.json"
    with open(split_file) as f:
        splits = json.load(f)
    return list(splits["test"])  # TODO: confirm key name


def enumerate_windows(storm_id: str) -> pd.DataFrame:
    """Return per-candidate-window completeness for one storm.

    Must return a DataFrame with columns:
        t0, era5_complete (bool), missing_reason (str or "")
    TODO(Adrija): call the SAME window builder + ERA5 crop/time checker used
    by the corrected pipeline (model/track_pipeline_unified_X.py). The Irma
    audit found 21/48 complete via crop/time coverage — reuse that exact
    predicate. Suggested: refactor the predicate into
    model/track_windows.py::window_is_era5_complete(...) and import it here
    and in the benchmark.
    """
    raise NotImplementedError("wire to the repo's window builder")
# --------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in load_test_storm_ids():
        try:
            w = enumerate_windows(sid)
            n_cand = len(w)
            n_ok = int(w["era5_complete"].sum())
            reasons = (w.loc[~w["era5_complete"], "missing_reason"]
                        .value_counts().to_dict())
            rows.append({"storm_id": sid, "candidate_windows": n_cand,
                         "era5_complete_windows": n_ok,
                         "usable": n_ok > 0,
                         "exclusion_reasons": json.dumps(reasons)})
        except Exception as e:  # keep sweeping; a broken storm is a finding
            rows.append({"storm_id": sid, "candidate_windows": -1,
                         "era5_complete_windows": 0, "usable": False,
                         "exclusion_reasons": json.dumps(
                             {"loader_error": str(e)})})

    m = pd.DataFrame(rows)
    m.to_csv(OUT / "test_coverage_manifest.csv", index=False)
    summary = {
        "n_test_storms": len(m),
        "n_usable_storms": int(m["usable"].sum()),
        "n_era5_complete_windows": int(
            m["era5_complete_windows"].clip(lower=0).sum()),
        "note": ("Report n_usable_storms / n_test_storms and the dominant "
                 "exclusion reasons as the ERA5-coverage footnote the "
                 "manuscript margin note asks for."),
    }
    with open(OUT / "test_coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"manifest -> {OUT/'test_coverage_manifest.csv'}")


if __name__ == "__main__":
    main()
