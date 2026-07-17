"""
cliper_baseline.py — Persistence and CLIPER climatology baseline forecasts.

CLIPER (CLImatology and PERsistence) is a standard null-hypothesis baseline
for tropical cyclone track forecasting.  It requires no ML training.

  Persistence:
    forecast(t+n) = last_position + n * last_displacement_6h

  CLIPER (simplified):
    forecast(t+n) = last_position + n * (α*persist + (1−α)*climatology)
    where climatology = mean 6h displacement for storms in same
    month + 10° latitude band, and α = 0.7 (standard CLIPER weighting).

Usage:
    python scripts/cliper_baseline.py
Outputs:
    metrics/cliper_baseline_metrics.csv
    tables/table1_track_error_vs_baselines.csv
"""
import sys, os, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.foundation.data_pipeline import parse_hurdat2_full

HURDAT2_PATH = "your-repo/data/data/raw/hurdat2/hurdat2_atlantic.txt"
SPLIT_FILE   = "splits/storm_splits.json"
METRICS_FILE = "metrics/cliper_baseline_metrics.csv"
LEADS        = [1, 2, 4, 8, 12, 20]   # steps × 6h = 6, 12, 24, 48, 72, 120 h
LEAD_H       = [l * 6 for l in LEADS]
DEG_TO_KM    = 111.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def persistence_forecast(obs_lat, obs_lon, n_steps):
    """Persist last observed 6h displacement."""
    if len(obs_lat) < 2:
        return obs_lat[-1], obs_lon[-1]
    dlat = obs_lat[-1] - obs_lat[-2]
    dlon = obs_lon[-1] - obs_lon[-2]
    return obs_lat[-1] + n_steps * dlat, obs_lon[-1] + n_steps * dlon


def cliper_forecast(obs_lat, obs_lon, month, clim_lut, n_steps, alpha=0.7):
    """CLIPER: blend persistence with climatological displacement."""
    if len(obs_lat) < 2:
        return obs_lat[-1], obs_lon[-1]
    dlat_p = obs_lat[-1] - obs_lat[-2]
    dlon_p = obs_lon[-1] - obs_lon[-2]

    lat_band = int(obs_lat[-1] / 10) * 10
    key = (month, lat_band)
    if key in clim_lut:
        dlat_c, dlon_c = clim_lut[key]
    else:
        dlat_c, dlon_c = dlat_p, dlon_p   # fallback to persistence

    dlat = alpha * dlat_p + (1 - alpha) * dlat_c
    dlon = alpha * dlon_p + (1 - alpha) * dlon_c
    return obs_lat[-1] + n_steps * dlat, obs_lon[-1] + n_steps * dlon


def build_climatology(storm_dfs):
    """Build mean 6h displacement LUT keyed by (month, lat_band_10deg)."""
    lut = {}
    for df in storm_dfs:
        lats = df["lat"].values
        lons = df["lon"].values
        months = df["datetime_utc"].dt.month.values if hasattr(df["datetime_utc"].dtype, 'tzinfo') else \
                 [int(str(d)[5:7]) for d in df["datetime_utc"].values]
        for i in range(1, len(lats)):
            m = months[i]
            lb = int(lats[i-1] / 10) * 10
            key = (m, lb)
            dlat = lats[i] - lats[i-1]
            dlon = lons[i] - lons[i-1]
            if key not in lut:
                lut[key] = []
            lut[key].append((dlat, dlon))
    return {k: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]))
            for k, v in lut.items()}


def evaluate(storm_dfs, splits, partition, clim_lut):
    errors_persist = {h: [] for h in LEAD_H}
    errors_cliper  = {h: [] for h in LEAD_H}

    for df in storm_dfs:
        sid = df["storm_id"].iloc[0]
        if splits.get(sid) != partition:
            continue
        lats  = df["lat"].values
        lons  = df["lon"].values
        months = [int(str(d)[5:7]) for d in df["datetime_utc"].astype(str).values]

        for t in range(1, len(lats)):
            for s_idx, (n, h) in enumerate(zip(LEADS, LEAD_H)):
                t_fut = t + n
                if t_fut >= len(lats):
                    continue
                true_lat, true_lon = lats[t_fut], lons[t_fut]
                obs_lat = lats[:t+1]
                obs_lon = lons[:t+1]

                p_lat, p_lon = persistence_forecast(obs_lat, obs_lon, n)
                c_lat, c_lon = cliper_forecast(obs_lat, obs_lon, months[t], clim_lut, n)

                errors_persist[h].append(haversine_km(p_lat, p_lon, true_lat, true_lon))
                errors_cliper[h].append(haversine_km(c_lat, c_lon, true_lat, true_lon))

    return errors_persist, errors_cliper


def ci95(values):
    """Bootstrap 95% CI for mean."""
    if not values:
        return None, None, None
    arr = np.array(values)
    mean = arr.mean()
    boot = np.array([np.random.choice(arr, len(arr)).mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return round(float(mean), 1), round(float(lo), 1), round(float(hi), 1)


def main():
    np.random.seed(42)

    print("Parsing HURDAT2 …")
    storm_dfs = parse_hurdat2_full(HURDAT2_PATH, min_year=1995, min_track_len=4)
    if not storm_dfs:
        print("HURDAT2 not found. Exiting."); return

    # Load splits
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            splits = json.load(f)
    else:
        print("Splits not found. Run create_splits.py first."); return

    print(f"Loaded {len(storm_dfs)} storms. Building climatology LUT …")
    # Build climatology from TRAIN storms only
    train_dfs = [df for df in storm_dfs if splits.get(df["storm_id"].iloc[0]) == "train"]
    clim_lut  = build_climatology(train_dfs)
    print(f"  Climatology LUT: {len(clim_lut)} (month, lat_band) buckets")

    print("Evaluating on TEST partition …")
    ep, ec = evaluate(storm_dfs, splits, "test", clim_lut)

    # Print and build rows
    rows_persist = []
    rows_cliper  = []
    print(f"\n{'Lead':>6}  {'Persist mean':>14} {'95% CI':>16}  {'CLIPER mean':>12} {'95% CI':>16}")
    for h in LEAD_H:
        pm, plo, phi = ci95(ep[h])
        cm, clo, chi = ci95(ec[h])
        n = len(ep[h])
        if pm is None:
            continue
        print(f"{h:>4}h   {pm:>8.1f} km  [{plo:.1f}, {phi:.1f}]   "
              f"{cm:>8.1f} km  [{clo:.1f}, {chi:.1f}]   n={n}")
        rows_persist.append({"lead_h": h, "mean_km": pm, "ci95_lo": plo, "ci95_hi": phi, "n": n})
        rows_cliper.append( {"lead_h": h, "mean_km": cm, "ci95_lo": clo, "ci95_hi": chi, "n": n})

    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    metric_rows = []
    for model_name, data_rows in [
        ("Persistence", rows_persist),
        ("CLIPER (climatology+persistence)", rows_cliper),
    ]:
        for item in data_rows:
            metric_rows.append({"model": model_name, **item, "partition": "test"})
    with open(METRICS_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        w.writeheader(); w.writerows(metric_rows)
    print(f"  Saved {METRICS_FILE}")

    os.makedirs("tables", exist_ok=True)
    _update_table1(rows_persist, rows_cliper)
    print("\nDone. Table 1 updated.")


def _update_table1(persist_rows, cliper_rows):
    path = "tables/table1_track_error_vs_baselines.csv"
    fieldnames = [
        "model",
        "track_km_6h", "track_km_12h", "track_km_24h", "track_km_48h",
        "track_km_72h", "track_km_120h",
        "ci95_lo_6h", "ci95_hi_6h", "ci95_lo_12h", "ci95_hi_12h",
        "ci95_lo_24h", "ci95_hi_24h", "ci95_lo_48h", "ci95_hi_48h",
        "ci95_lo_72h", "ci95_hi_72h", "ci95_lo_120h", "ci95_hi_120h",
        "partition", "protocol", "source_csv",
    ]

    # Map lead to column name
    lead_col = {6: "track_km_6h", 12: "track_km_12h", 24: "track_km_24h",
                48: "track_km_48h", 72: "track_km_72h", 120: "track_km_120h"}

    def make_row(name, data_rows):
        r = {
            "model": name,
            "partition": "test",
            "protocol": "Storm-level HURDAT2 time split",
            "source_csv": METRICS_FILE,
        }
        for item in data_rows:
            h = item["lead_h"]
            col = lead_col.get(h)
            if col:
                r[col] = item["mean_km"]
                r[f"ci95_lo_{h}h"] = item["ci95_lo"]
                r[f"ci95_hi_{h}h"] = item["ci95_hi"]
        return r

    rows = [
        make_row("Persistence", persist_rows),
        make_row("CLIPER (climatology+persistence)", cliper_rows),
    ]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Updated {path}")


if __name__ == "__main__":
    main()
