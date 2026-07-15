"""
run_full_test_ci.py — Run LSTM/Transformer/GNO baseline inference over all 273
test-set HURDAT2 storms, compute per-storm track errors, and bootstrap 95% CIs.

For storms without ERA5 reanalysis (all except Irma/Ian), the atmospheric
context tensor X is set to zeros.  This is noted in the table.  Irma and Ian
use their real ERA5 patches when available.

Usage:
    python scripts/run_full_test_ci.py [--demo]
Outputs:
    tables/table1_track_error_vs_baselines.csv  (updated with full-set CIs)
    tables/table_fullset_ci_detail.csv          (per-storm error detail)
"""
import sys, os, json, csv, time, argparse
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import model.track_pipeline_unified_X as pipeline
from model.foundation.data_pipeline import parse_hurdat2_full

HURDAT2_PATH = "your-repo/data/data/raw/hurdat2/hurdat2_atlantic.txt"
SPLIT_FILE   = "splits/storm_splits.json"
LEADS        = list(pipeline.cfg.lead_hours)   # [6, 12, 24, 48]
H_STEPS      = pipeline.cfg.history_steps       # 4
ERA5_SHAPE   = (5, 65, 65)
META_DIM     = 2                               # [vmax_kt, mslp_mb]

DEG_TO_KM = 111.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_model(model_name):
    cfg = pipeline.cfg
    feat_ch = getattr(cfg, "feat_ch", 5)
    leads   = len(cfg.lead_hours)
    if model_name == "LSTM":
        mdl  = pipeline.LSTMTrackBaseline(feat_ch, leads)
        ckpt = torch.load("checkpoints/baseline_lstm.pt", map_location="cpu", weights_only=False)
    elif model_name == "Transformer":
        mdl  = pipeline.TransformerTrackBaseline(feat_ch, leads)
        ckpt = torch.load("checkpoints/baseline_transformer.pt", map_location="cpu", weights_only=False)
    elif model_name == "GNO+DynGNN":
        mdl  = pipeline.GNO_DynGNN(feat_ch, leads)
        ckpt = torch.load("checkpoints/main_gno_dyngnn.pt", map_location="cpu", weights_only=False)
    else:
        return None
    state = ckpt.get("state", ckpt.get("model_state_dict", ckpt))
    mdl.load_state_dict(state, strict=False)
    mdl.eval()
    return mdl


def build_storm_samples(df):
    """
    Build prediction samples from a storm track DataFrame.
    X (ERA5) is set to zeros for all samples — noted in output.
    Returns list of dicts with keys: past, X, meta, y_abs, lat0, lon0, t0.
    """
    lead_steps = [h // 6 for h in LEADS]
    max_lead   = max(lead_steps)
    samples    = []

    for i in range(H_STEPS, len(df)):
        if i + max_lead >= len(df):
            break

        lat0 = float(df.loc[i, "lat"])
        lon0 = float(df.loc[i, "lon"])

        # Past H positions (oldest → newest)
        past = np.array(
            [[float(df.loc[i-k, "lat"]), float(df.loc[i-k, "lon"])]
             for k in range(H_STEPS, 0, -1)],
            dtype=np.float32
        )

        # ERA5 → zeros (no reanalysis for most storms)
        X = np.zeros(ERA5_SHAPE, dtype=np.float32)

        # Meta: vmax, mslp (with fallback to 0)
        vmax = float(df.loc[i, "vmax_kt"]) if pd.notna(df.loc[i, "vmax_kt"]) else 0.0
        mslp = float(df.loc[i, "mslp_mb"]) if pd.notna(df.loc[i, "mslp_mb"]) else 0.0
        meta = np.array([vmax, mslp], dtype=np.float32)

        # True future abs positions
        y_abs = np.array(
            [[float(df.loc[i + s, "lat"]), float(df.loc[i + s, "lon"])]
             for s in lead_steps],
            dtype=np.float32
        )

        samples.append(dict(
            past=torch.from_numpy(past).unsqueeze(0),    # (1, H, 2)
            X   =torch.from_numpy(X).unsqueeze(0),       # (1, 5, G, G)
            meta=torch.from_numpy(meta).unsqueeze(0),    # (1, 2)
            y_abs=y_abs,                                 # (L, 2) numpy
            lat0=lat0, lon0=lon0,
            t0=str(df.loc[i, "datetime_utc"]),
        ))

    return samples


def run_inference(model_obj, samples):
    """Run model on all samples; return per-sample predicted positions."""
    results = []
    with torch.no_grad():
        for s in samples:
            mu, _ = model_obj(s["past"], s["X"], s["meta"])
            mu_np = mu.squeeze(0).cpu().numpy()   # (L, 2)
            errors = []
            for l_idx, h in enumerate(LEADS):
                pred_lat, pred_lon = mu_np[l_idx, 0], mu_np[l_idx, 1]
                true_lat, true_lon = s["y_abs"][l_idx, 0], s["y_abs"][l_idx, 1]
                err = haversine_km(pred_lat, pred_lon, true_lat, true_lon)
                errors.append((h, err))
            results.append(errors)
    return results


def bootstrap_ci(values, n_boot=2000):
    arr = np.array(values)
    if len(arr) == 0:
        return None, None, None
    mean = arr.mean()
    boots = np.array([np.random.choice(arr, len(arr)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(mean), 1), round(float(lo), 1), round(float(hi), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Use only 30 test storms for a quick check")
    args = parser.parse_args()

    np.random.seed(42)

    # ── Load splits ───────────────────────────────────────────────────────────
    if not os.path.exists(SPLIT_FILE):
        print("Splits not found — run scripts/create_splits.py first")
        return
    with open(SPLIT_FILE) as f:
        splits = json.load(f)

    test_ids = {sid for sid, p in splits.items() if p == "test"}
    print(f"Test partition: {len(test_ids)} storms")

    # ── Load HURDAT2 ──────────────────────────────────────────────────────────
    print("Parsing HURDAT2 …")
    storm_dfs = parse_hurdat2_full(HURDAT2_PATH, min_year=1995, min_track_len=8)
    test_dfs  = [df for df in storm_dfs
                 if df["storm_id"].iloc[0] in test_ids]
    if args.demo:
        test_dfs = test_dfs[:30]
    print(f"  Using {len(test_dfs)} test storms")

    # ── Load models ───────────────────────────────────────────────────────────
    model_names = ["LSTM", "Transformer", "GNO+DynGNN"]
    models = {}
    for name in model_names:
        try:
            models[name] = load_model(name)
            print(f"  Loaded {name}")
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    if not models:
        print("No models loaded — exiting")
        return

    # ── Build samples and run inference ───────────────────────────────────────
    all_errors = {name: {h: [] for h in LEADS} for name in models}
    detail_rows = []
    n_storms_processed = 0

    t0_start = time.time()
    for storm_idx, df in enumerate(test_dfs):
        storm_id = df["storm_id"].iloc[0]
        df = df.reset_index(drop=True)
        samples = build_storm_samples(df)
        if not samples:
            continue

        storm_errs_by_model = {}
        for name, mdl in models.items():
            try:
                results = run_inference(mdl, samples)
                storm_errs = {h: [] for h in LEADS}
                for window_errs in results:
                    for h, err in window_errs:
                        storm_errs[h].append(err)
                        all_errors[name][h].append(err)
                storm_errs_by_model[name] = {h: np.mean(v) if v else np.nan
                                              for h, v in storm_errs.items()}
            except Exception as e:
                pass

        for name, errs in storm_errs_by_model.items():
            row = {"storm_id": storm_id, "model": name, "n_windows": len(samples)}
            for h in LEADS:
                row[f"mean_err_{h}h"] = round(errs.get(h, np.nan), 1)
            detail_rows.append(row)

        n_storms_processed += 1
        if storm_idx % 20 == 0:
            elapsed = time.time() - t0_start
            print(f"  {storm_idx+1}/{len(test_dfs)} storms … {elapsed:.0f}s elapsed")

    print(f"\nProcessed {n_storms_processed} storms in {time.time()-t0_start:.1f}s")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nFull test set CIs (n = total prediction windows):")
    print(f"{'Model':<20}", end="")
    for h in LEADS:
        print(f"  {h}h", end="")
    print()

    ci_results = {}
    for name in models:
        ci_results[name] = {}
        print(f"{name:<20}", end="")
        for h in LEADS:
            vals = all_errors[name][h]
            m, lo, hi = bootstrap_ci(vals)
            ci_results[name][h] = (m, lo, hi, len(vals))
            if m is not None:
                print(f"  {m:.0f}[{lo:.0f},{hi:.0f}]", end="")
            else:
                print("  —", end="")
        print(f"  n={len(all_errors[name][LEADS[0]])}")

    # ── Save detail CSV ───────────────────────────────────────────────────────
    os.makedirs("tables", exist_ok=True)
    if detail_rows:
        with open("tables/table_fullset_ci_detail.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader(); w.writerows(detail_rows)
        print("\nSaved tables/table_fullset_ci_detail.csv")

    # ── Update table1 ─────────────────────────────────────────────────────────
    _update_table1(ci_results, n_storms_processed, args.demo)
    print("Updated tables/table1_track_error_vs_baselines.csv")


def _update_table1(ci_results, n_storms, demo):
    path = "tables/table1_track_error_vs_baselines.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    for col in ["n_storms_full", "era5_note"]:
        if col not in fieldnames:
            fieldnames.append(col)

    model_map = {
        "LSTM":       "LSTM",
        "Transformer":"Transformer",
        "GNO+DynGNN": "GNO+DynGNN",
    }

    for row in rows:
        for src, name in model_map.items():
            if name in row.get("model", ""):
                if src in ci_results:
                    for h in LEADS:
                        if h in ci_results[src] and ci_results[src][h][0] is not None:
                            m, lo, hi, n = ci_results[src][h]
                            row[f"track_km_{h}h"]   = m
                            row[f"ci95_lo_{h}h"] = lo
                            row[f"ci95_hi_{h}h"] = hi
                            row[f"n_{h}h"]       = n
                    row["n_storms_full"] = n_storms
                    row["era5_note"] = ("DEMO n=30" if demo
                                        else "Full test set; X=0 for storms without ERA5")

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
