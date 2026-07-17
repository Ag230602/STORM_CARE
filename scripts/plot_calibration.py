"""Regenerate calibration table and figure from foundation evaluation metrics."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.generate_submission_outputs import (  # noqa: E402
    ROOT,
    regenerate_calibration_figure,
    regenerate_foundation_tables,
)


def main():
    metrics_dir = ROOT / "metrics"
    tables_dir = ROOT / "tables"
    figures_dir = ROOT / "figures"
    regenerate_foundation_tables(metrics_dir, tables_dir)
    written = regenerate_calibration_figure(tables_dir, figures_dir)
    for path in written:
        print(f"Saved {Path(path)}")


if __name__ == "__main__":
    main()
