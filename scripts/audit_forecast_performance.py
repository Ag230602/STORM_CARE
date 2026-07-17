"""Audit forecast-performance claims against regenerated metric tables.

The goal is not to create a new metric.  It records which outperformance claims
are supported under each existing protocol and flags protocol mismatches.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_TABLE = ROOT / "tables" / "table_case_study_track_error.csv"
STORM_TABLE = ROOT / "tables" / "table1_track_error_vs_baselines.csv"
FOUNDATION_TABLE = ROOT / "tables" / "table_foundation_model_training.csv"
OUT_TABLE = ROOT / "tables" / "table_forecast_performance_audit.csv"
OUT_REPORT = ROOT / "reports" / "forecast_performance_audit.md"
RESULTS_TABLE = ROOT / "results" / "module3_baselines" / "tables" / OUT_TABLE.name
RESULTS_REPORT = ROOT / "results" / "module3_baselines" / "reports" / OUT_REPORT.name
LEADS_CASE = [6, 12, 24, 48]
LEADS_STORM = [6, 12, 24, 48, 72, 120]


def _case_study_rows() -> list[dict[str, object]]:
    df = pd.read_csv(CASE_TABLE)
    gno = df[df["model"] == "GNO+DynGNN"].iloc[0]
    rows = []
    for baseline in ["Persistence", "LSTM", "Transformer"]:
        brow = df[df["model"] == baseline].iloc[0]
        for h in LEADS_CASE:
            stormcare = float(gno[f"track_km_{h}h"])
            base = float(brow[f"track_km_{h}h"])
            rows.append({
                "protocol": "Irma/Ian ERA5-complete case study",
                "stormcare_model": "GNO+DynGNN",
                "baseline": baseline,
                "lead_h": h,
                "stormcare_track_km": stormcare,
                "baseline_track_km": base,
                "delta_km": stormcare - base,
                "outperforms_baseline": stormcare < base,
                "verdict": "supported_win" if stormcare < base else "not_supported",
                "source_csv": "tables/table_case_study_track_error.csv",
                "note": "Comparable within this small window-level case study only.",
            })
    rows.append({
        "protocol": "Irma/Ian ERA5-complete case study",
        "stormcare_model": "GNO+DynGNN",
        "baseline": "CLIPER",
        "lead_h": "all",
        "stormcare_track_km": pd.NA,
        "baseline_track_km": pd.NA,
        "delta_km": pd.NA,
        "outperforms_baseline": pd.NA,
        "verdict": "not_evaluated",
        "source_csv": "tables/table_case_study_track_error.csv",
        "note": "CLIPER was not generated for the Irma/Ian ERA5-complete case-study split.",
    })
    return rows


def _storm_level_rows() -> list[dict[str, object]]:
    df = pd.read_csv(STORM_TABLE)
    rows = []
    p = df[df["model"] == "Persistence"].iloc[0]
    c = df[df["model"].str.startswith("CLIPER")].iloc[0]
    for h in LEADS_STORM:
        p_err = float(p[f"track_km_{h}h"])
        c_err = float(c[f"track_km_{h}h"])
        winner = "Persistence" if p_err < c_err else "CLIPER"
        rows.append({
            "protocol": "Storm-level HURDAT2 2020-2024 test split",
            "stormcare_model": "not_evaluated",
            "baseline": "Persistence_vs_CLIPER",
            "lead_h": h,
            "stormcare_track_km": pd.NA,
            "baseline_track_km": min(p_err, c_err),
            "delta_km": pd.NA,
            "outperforms_baseline": pd.NA,
            "verdict": "no_stormcare_row",
            "source_csv": "tables/table1_track_error_vs_baselines.csv",
            "note": f"No STORM-CARE neural forecast is evaluated under this protocol; best baseline is {winner}.",
        })
    return rows


def _foundation_context_rows() -> list[dict[str, object]]:
    if not FOUNDATION_TABLE.exists() or not STORM_TABLE.exists():
        return []
    fm = pd.read_csv(FOUNDATION_TABLE).iloc[0]
    storm = pd.read_csv(STORM_TABLE)
    baselines = [
        ("Persistence", storm[storm["model"] == "Persistence"].iloc[0]),
        ("CLIPER", storm[storm["model"].str.startswith("CLIPER")].iloc[0]),
    ]
    rows = []
    for baseline, brow in baselines:
        for h in LEADS_STORM:
            rows.append({
                "protocol": "Cross-protocol context only",
                "stormcare_model": "Foundation demo checkpoint",
                "baseline": baseline,
                "lead_h": h,
                "stormcare_track_km": float(fm[f"track_km_{h}h"]),
                "baseline_track_km": float(brow[f"track_km_{h}h"]),
                "delta_km": float(fm[f"track_km_{h}h"]) - float(brow[f"track_km_{h}h"]),
                "outperforms_baseline": False,
                "verdict": "not_comparable",
                "source_csv": "tables/table_foundation_model_training.csv; tables/table1_track_error_vs_baselines.csv",
                "note": "Foundation metrics are a short validation-demo rerun; Persistence/CLIPER are storm-level test metrics.",
            })
    return rows


def _write_report(df: pd.DataFrame) -> None:
    OUT_REPORT.parent.mkdir(exist_ok=True)
    case = df[df["protocol"] == "Irma/Ian ERA5-complete case study"].copy()
    pivot = case[case["lead_h"] != "all"].pivot_table(
        index=["baseline"],
        columns="lead_h",
        values="verdict",
        aggfunc="first",
    )
    with OUT_REPORT.open("w", encoding="utf-8") as fh:
        fh.write("# Forecast Performance Claim Audit\n\n")
        fh.write("## Verdict\n")
        fh.write("- STORM-CARE does not currently have a supported all-horizon forecast-superiority claim against Persistence.\n")
        fh.write("- On the Irma/Ian ERA5-complete case study, GNO+DynGNN loses to Persistence at 6/12/24/48 h.\n")
        fh.write("- On the same case study, GNO+DynGNN beats Transformer at 6/12/24/48 h and beats LSTM at 12/24/48 h, but loses to LSTM at 6 h.\n")
        fh.write("- CLIPER is only available in the storm-level HURDAT2 baseline table; no STORM-CARE neural model is evaluated under that exact protocol.\n")
        fh.write("- The foundation checkpoint metrics are validation-demo numbers and must not be compared as a test-set superiority claim.\n\n")
        fh.write("## Case-Study Horizon Verdicts\n")
        fh.write(pivot.to_markdown())
        fh.write("\n\n")
        fh.write("## Required Manuscript Claim\n")
        fh.write("Use: \"The corrected benchmark reports calibrated/probabilistic forecasts and a reproducible neural baseline study. Persistence remains the strongest short-horizon and overall case-study baseline; learned-model superiority is not claimed.\"\n\n")
        fh.write("Avoid: \"STORM-CARE outperforms Persistence/CLIPER/LSTM/Transformer/GNO at all horizons.\"\n\n")
        fh.write(f"Source table: `{OUT_TABLE.relative_to(ROOT)}`\n")


def main() -> None:
    OUT_TABLE.parent.mkdir(exist_ok=True)
    rows = []
    rows.extend(_case_study_rows())
    rows.extend(_storm_level_rows())
    rows.extend(_foundation_context_rows())
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE, index=False)
    _write_report(df)
    RESULTS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_TABLE, RESULTS_TABLE)
    shutil.copy2(OUT_REPORT, RESULTS_REPORT)
    print(f"Wrote {OUT_TABLE}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Mirrored {RESULTS_TABLE}")
    print(f"Mirrored {RESULTS_REPORT}")


if __name__ == "__main__":
    main()
