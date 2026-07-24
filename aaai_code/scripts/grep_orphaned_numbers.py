#!/usr/bin/env python
"""Manuscript hygiene — hunt down orphaned numbers from dead runs.

Fully functional. Scans manuscript/ (and optionally the LaTeX source) for
numbers that came from superseded runs and must not survive the sync:
the old mirror-operator counterfactual deltas, the 8-epoch calibration
figure, the pre-audit probe accuracy, the stale ablation percentage, the
stale split sizes, and the dead Table 1/Table 4 values.

Exit code 1 if any orphan is found (so it can gate a submission make
target). Add new patterns as claims are retired.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# pattern -> why it is dead
ORPHANS = {
    r"97\.7\s*%?":            "old linear-probe accuracy (regenerated: 0.875 "
                              "from the 2-epoch demo; replace with E2 run)",
    r"\bep\s*8\b|\(ep 8\)":   "calibration figure cites epoch 8; corrected "
                              "selected checkpoint is epoch 2 (E2 replaces)",
    r"[+\u2212-]40\.3|[+\u2212-]54\.5|[+\u2212-]70\.1":
                              "old mirror-operator evacuation deltas (E3 "
                              "sweep replaces)",
    r"\+29\.9|\+24\.7|\+15\.6":
                              "old adverse-scenario deltas from injected "
                              "outputs (rebuild Table 4 from "
                              "counterfactual_outcomes.csv)",
    r"\b4\.3\s*%":            "old physics validation-loss claim; corrected "
                              "run shows the opposite tradeoff (E5/RQ2)",
    r"\bn\s*=\s*201\b|\bn\s*=\s*45\b|\bn\s*=\s*273\b":
                              "stale split sizes; integrity report says "
                              "342/70/107",
    r"1995\u20132006|2007\u20132009|1995-2006|2007-2009":
                              "stale split year boundaries; integrity report "
                              "says 1995\u20132015 / 2016\u20132019 / 2020+",
    r"\b1466(\.\d+)?\b.{0,40}\b1750(\.\d+)?\b":
                              "120h overtake claim built on validation-demo "
                              "foundation numbers (E1/E2 must re-establish "
                              "or delete)",
    r"\b0\.90,?\s*0\.91,?\s*(and\s*)?0\.90\b":
                              "near-nominal coverage triplet with no "
                              "reproducible source (E2 replaces Figure 2)",
    r"shelter[- ]shortfall.{0,80}(fixed|no longer degenerate)":
                              "no artifact substantiates the shelter-"
                              "shortfall fix; keep the limitation sentence "
                              "or produce the artifact",
}

EXTS = {".md", ".tex", ".txt"}


def main() -> None:
    targets = [p for d in ("manuscript", "reports") if (ROOT / d).exists()
               for p in (ROOT / d).rglob("*") if p.suffix in EXTS]
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:]]
    found = 0
    for path in targets:
        text = path.read_text(errors="replace")
        for pat, why in ORPHANS.items():
            for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
                line_no = text.count("\n", 0, m.start()) + 1
                line = text.splitlines()[line_no - 1].strip()[:100]
                print(f"{path}:{line_no}: [{m.group(0)!r}] {why}\n    > {line}")
                found += 1
    if found:
        print(f"\n{found} orphaned number(s) found — fix before submission.")
        sys.exit(1)
    print("clean: no orphaned numbers found.")


if __name__ == "__main__":
    main()
