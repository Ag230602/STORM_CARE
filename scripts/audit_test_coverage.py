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
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "metrics" / "test_coverage"

sys.path.insert(0, str(ROOT))

from model import track_pipeline_unified_X as pipeline  # noqa: E402
from model.foundation.data_pipeline import parse_hurdat2_full  # noqa: E402
import benchmark as bench  # noqa: E402  (reuses _candidate_windows so the
                            # manifest and the actual benchmark can't disagree)

HURDAT2_PATH = ROOT / "your-repo" / "data" / "data" / "raw" / "hurdat2" / "hurdat2_atlantic.txt"

# The only storms with a downloaded ERA5 file in this environment (both
# pre-extracted single-storm track CSVs + NetCDF crops). Everything else in
# the test partition has no ERA5 on disk at all -- that is the finding this
# audit exists to surface, not a bug in the adapter.
ERA5_AVAILABLE = {
    "AL112017": ROOT / "your-repo" / "data" / "data" / "raw" / "era5" / "irma_2017" / "era5_pl_irma_2017.nc",
    "AL092022": ROOT / "your-repo" / "data" / "data" / "raw" / "era5" / "ian_2022" / "era5_pl_ian_2022.nc",
}

_HURDAT2_BY_ID: dict[str, pd.DataFrame] | None = None


def _hurdat2_by_id() -> dict[str, pd.DataFrame]:
    global _HURDAT2_BY_ID
    if _HURDAT2_BY_ID is None:
        storms = parse_hurdat2_full(str(HURDAT2_PATH), min_year=1995)
        _HURDAT2_BY_ID = {df["storm_id"].iloc[0]: df for df in storms}
    return _HURDAT2_BY_ID


# ----------------------- ADAPTERS: wired -----------------------------------
def load_test_storm_ids() -> list[str]:
    """Return the test-partition storm ids from the frozen split artifact."""
    split_file = ROOT / "splits" / "storm_splits.json"
    with open(split_file) as f:
        splits = json.load(f)
    return sorted(sid for sid, partition in splits.items() if partition == "test")


def enumerate_windows(storm_id: str) -> pd.DataFrame:
    """Per-candidate-window completeness for one storm.

    Candidate-window count comes from benchmark._candidate_windows (the same
    function the corrected benchmark protocol uses). ERA5 completeness comes
    from pipeline.build_samples (the same crop/time predicate the corrected
    benchmark uses) when an ERA5 file exists locally; otherwise every
    candidate window is reported incomplete with reason "no_era5_file".
    """
    by_id = _hurdat2_by_id()
    track_df = by_id.get(storm_id)
    if track_df is None:
        return pd.DataFrame([{"t0": None, "era5_complete": False,
                               "missing_reason": "storm_not_in_hurdat2"}])

    n_cand = bench._candidate_windows(track_df)
    if n_cand <= 0:
        return pd.DataFrame([{"t0": None, "era5_complete": False,
                               "missing_reason": "track_too_short_for_any_window"}])

    era5_path = ERA5_AVAILABLE.get(storm_id)
    if era5_path is None or not era5_path.exists():
        return pd.DataFrame([
            {"t0": None, "era5_complete": False, "missing_reason": "no_era5_file"}
            for _ in range(n_cand)
        ])

    track_df = track_df.copy()
    track_df["storm_tag"] = storm_id
    era5_ds = pipeline.open_era5(str(era5_path))
    samples = pipeline.build_samples(track_df, era5_ds)
    ok_t0 = {s["t0"] for s in samples}
    rows = [{"t0": t0, "era5_complete": True, "missing_reason": ""} for t0 in ok_t0]
    n_missing = n_cand - len(ok_t0)
    rows += [
        {"t0": None, "era5_complete": False, "missing_reason": "crop_incomplete_or_out_of_domain"}
        for _ in range(max(n_missing, 0))
    ]
    return pd.DataFrame(rows)
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
