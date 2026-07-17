"""Audit that calibration tables/figures use the selected foundation checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics/foundation/foundation_eval_metrics.csv"
TABLE = ROOT / "tables/table_calibration_cone_coverage.csv"
TRAIN_TABLE = ROOT / "tables/table_foundation_model_training.csv"
FIGS = [ROOT / "figures/calibration.png", ROOT / "figures/calibration.pdf"]
OUT_JSON = ROOT / "metrics/foundation/calibration_consistency_audit.json"
OUT_REPORT = ROOT / "reports/calibration_consistency_audit.md"


def _selected_row(df: pd.DataFrame) -> pd.Series:
    if "selected_checkpoint" in df.columns:
        sel = df[df["selected_checkpoint"].astype(str).str.lower().isin(["true", "1"])]
        if len(sel) == 1:
            return sel.iloc[0]
        raise ValueError(f"Expected exactly one selected checkpoint row; found {len(sel)}")
    if "selection_score" in df.columns:
        return df.sort_values("selection_score").iloc[0]
    return df.iloc[-1]


def main() -> None:
    eval_df = pd.read_csv(METRICS)
    cal = pd.read_csv(TABLE)
    train = pd.read_csv(TRAIN_TABLE)
    selected = _selected_row(eval_df)
    epoch = int(selected["epoch"])
    issues = []

    if len(train) != 1 or int(train.iloc[0]["epoch"]) != epoch:
        issues.append("table_foundation_model_training.csv does not contain exactly the selected epoch")

    for lead in cal["lead_time_h"].astype(int).tolist():
        for prob in ["p50", "p90"]:
            table_col = f"cone_{prob}_ep{epoch}"
            eval_col = f"cone_{prob}_{lead}h"
            if table_col not in cal.columns:
                issues.append(f"Missing {table_col} in calibration table")
                continue
            table_val = float(cal[cal["lead_time_h"] == lead].iloc[0][table_col])
            eval_val = float(selected[eval_col])
            if abs(table_val - eval_val) > 1e-4:
                issues.append(f"Mismatch {eval_col}: eval={eval_val} table={table_val}")

    missing_figs = [str(p.relative_to(ROOT)) for p in FIGS if not p.exists() or p.stat().st_size == 0]
    if missing_figs:
        issues.append(f"Missing/empty calibration figures: {missing_figs}")

    audit = {
        "selected_epoch": epoch,
        "selection_metric": selected.get("selection_metric", ""),
        "selection_score": float(selected.get("selection_score", float("nan"))),
        "source_metrics": str(METRICS.relative_to(ROOT)),
        "table": str(TABLE.relative_to(ROOT)),
        "figures": [str(p.relative_to(ROOT)) for p in FIGS],
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(
        "# Calibration Consistency Audit\n\n"
        f"- Status: **{audit['status']}**\n"
        f"- Selected checkpoint epoch: `{epoch}`\n"
        f"- Selection metric: `{audit['selection_metric']}`\n"
        f"- Selection score: `{audit['selection_score']}`\n"
        f"- Source metrics: `{audit['source_metrics']}`\n"
        f"- Calibration table: `{audit['table']}`\n"
        f"- Figures: {', '.join(f'`{x}`' for x in audit['figures'])}\n\n"
        "## Issues\n"
        + ("\n".join(f"- {x}" for x in issues) if issues else "- None.\n"),
        encoding="utf-8",
    )
    if issues:
        raise SystemExit("Calibration consistency audit failed")
    print(f"Calibration consistency PASS for selected epoch {epoch}")


if __name__ == "__main__":
    main()
