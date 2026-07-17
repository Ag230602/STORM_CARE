"""
create_splits.py — Freeze storm-level train / val / test partitions.

Protocol (non-negotiable per advisor):
  - Storm-level split: no storm appears in two partitions
  - Time-based: prevents any future-data leakage
  - All downstream scripts must import SPLIT_FILE and respect these assignments

Split boundaries for the bundled HURDAT2 file:
  train : 1995 – 2015  (~66%)
  val   : 2016 – 2019  (~14%)
  test  : 2020 – 2024  (~21%)

Usage:
    python scripts/create_splits.py
Outputs:
    splits/storm_splits.json   — {storm_id: "train"|"val"|"test"}
    splits/split_summary.csv   — counts and year ranges per partition
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.foundation.data_pipeline import parse_hurdat2_full

HURDAT2_PATH = "your-repo/data/data/raw/hurdat2/hurdat2_atlantic.txt"
SPLIT_FILE   = "splits/storm_splits.json"
SUMMARY_FILE = "splits/split_summary.csv"

TRAIN_END = 2015
VAL_END   = 2019


def main():
    print("Parsing HURDAT2 for split assignment …")
    storm_dfs = parse_hurdat2_full(HURDAT2_PATH, min_year=1995, min_track_len=4)

    if not storm_dfs:
        print(f"ERROR: HURDAT2 not found at {HURDAT2_PATH}. Check path.")
        return

    splits = {}
    counts = {"train": 0, "val": 0, "test": 0}
    years  = {"train": [], "val": [], "test": []}

    for df in storm_dfs:
        storm_id = df["storm_id"].iloc[0]
        year     = int(df["year"].iloc[0])

        if year <= TRAIN_END:
            partition = "train"
        elif year <= VAL_END:
            partition = "val"
        else:
            partition = "test"

        splits[storm_id] = partition
        counts[partition] += 1
        years[partition].append(year)

    os.makedirs("splits", exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Saved {SPLIT_FILE}")

    with open(SUMMARY_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["partition", "n_storms", "year_min", "year_max", "boundary"])
        for p in ["train", "val", "test"]:
            yr = years[p]
            b  = (f"1995–{TRAIN_END}" if p == "train"
                  else f"{TRAIN_END+1}–{VAL_END}" if p == "val"
                  else f"{VAL_END+1}+")
            w.writerow([p, counts[p], min(yr) if yr else "—", max(yr) if yr else "—", b])
    print(f"  Saved {SUMMARY_FILE}")

    total = sum(counts.values())
    for p, c in counts.items():
        yr = years[p]
        print(f"  {p:6s}: {c:4d} storms ({100*c/total:.1f}%)  "
              f"years {min(yr) if yr else '—'}–{max(yr) if yr else '—'}")

    print(f"\nTotal: {total} storms split. "
          "All reported numbers MUST come from the TEST partition only.")


if __name__ == "__main__":
    main()
