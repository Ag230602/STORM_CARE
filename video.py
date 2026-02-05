# # ============================================
# 3D Hurricane Video Generator (FINAL)
# Works on Windows without system ffmpeg
# ============================================

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import imageio


# ======================
# USER CONFIG
# ======================
CONFIGS = [
    {
        "name": "irma",
        "title": "Hurricane Irma (3D Wind Surface)",
        "era5_nc": r"data\irma_era5_u10v10.nc",
        "track_csv": r"data\irma_track.csv",
        "frames_dir": r"out_frames_irma",
        "out_mp4": r"irma_3d.mp4",
    }
]

FPS = 30
DURATION_SEC = 30
PAD_DEG = 8
STRIDE = 2
Z_SCALE = 0.22


# ======================
# HELPERS
# ======================
def resolve_path(p):
    p = Path(p)
    if not p.is_absolute():
        p = (Path(__file__).parent / p).resolve()
    return p


def check_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Missing {label}:\n  {path}\n"
            f"Fix: put the file there OR update the path in CONFIGS.\n"
        )


def load_track(csv_path):
    df = pd.read_csv(csv_path)
    if not {"time", "lat", "lon"}.issubset(df.columns):
        raise ValueError("Track CSV must contain columns: time, lat, lon")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def infer_lat_lon(ds):
    for lat in ["latitude", "lat"]:
        for lon in ["longitude", "lon"]:
            if lat in ds.coords and lon in ds.coords:
                return lat, lon
    raise ValueError("Latitude/Longitude not found in NetCDF")


def infer_uv(ds):
    for u, v in [("u10", "v10"), ("10u", "10v")]:
        if u in ds.variables and v in ds.variables:
            return u, v
    raise ValueError("Wind variables (u10/v10 or 10u/10v) not found")


def subset_ds(ds, track, lat, lon):
    lat_min, lat_max = track.lat.min() - PAD_DEG, track.lat.max() + PAD_DEG
    lon_min, lon_max = track.lon.min() - PAD_DEG, track.lon.max() + PAD_DEG

    lat_vals = ds[lat].values
    if lat_vals[0] > lat_vals[-1]:
        return ds.sel({lat: slice(lat_max, lat_min), lon: slice(lon_min, lon_max)})
    return ds.sel({lat: slice(lat_min, lat_max), lon: slice(lon_min, lon_max)})


def nearest_track(track, times):
    t = track.time.values.astype("datetime64[ns]")
    idx = np.searchsorted(t, times)
    idx = np.clip(idx, 0, len(track) - 1)
    return track.iloc[idx].reset_index(drop=True)


# ======================
# VIDEO STITCH (FIXED)
# ======================
def stitch_frames(frames_dir, out_mp4, fps):
    files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda x: int(re.findall(r"\d+", x)[-1])
    )

    with imageio.get_writer(out_mp4, fps=fps, codec="libx264") as writer:
        for f in files:
            frame = imageio.imread(Path(frames_dir) / f)
            writer.append_data(frame)

    print(f"[OK] Video written → {out_mp4}")


# ======================
# MAIN RENDER
# ======================
def make_video(cfg):
    era5_nc = resolve_path(cfg["era5_nc"])
    track_csv = resolve_path(cfg["track_csv"])
    frames_dir = resolve_path(cfg["frames_dir"])
    out_mp4 = resolve_path(cfg["out_mp4"])

    check_exists(era5_nc, "ERA5 NetCDF")
    check_exists(track_csv, "Track CSV")
    frames_dir.mkdir(parents=True, exist_ok=True)

    track = load_track(track_csv)
    ds = xr.open_dataset(era5_nc)

    lat, lon = infer_lat_lon(ds)
    u, v = infer_uv(ds)

    ds = subset_ds(ds, track, lat, lon)
    wspd = np.sqrt(ds[u] ** 2 + ds[v] ** 2)

    times = pd.to_datetime(ds.time.values)
    n_frames = FPS * DURATION_SEC
    times = times if len(times) <= n_frames else times[np.linspace(0, len(times)-1, n_frames).astype(int)]

    track_near = nearest_track(track, times)

    LON, LAT = np.meshgrid(
        ds[lon].values[::STRIDE],
        ds[lat].values[::STRIDE]
    )

    print(f"[INFO] Rendering {len(times)} frames → {frames_dir}")

    for i, t in enumerate(times):
        W = wspd.sel(time=t).isel(
            {lat: slice(None, None, STRIDE), lon: slice(None, None, STRIDE)}
        ).values

        Z = W * Z_SCALE

        fig = go.Figure()

        fig.add_surface(x=LON, y=LAT, z=Z, colorbar=dict(title="Wind m/s"))

        fig.add_scatter3d(
            x=track.lon, y=track.lat, z=np.zeros(len(track)),
            mode="lines", line=dict(width=6), name="Track"
        )

        fig.add_scatter3d(
            x=[track_near.lon.iloc[i]],
            y=[track_near.lat.iloc[i]],
            z=[np.nanmax(Z) * 1.1],
            mode="markers+text",
            text=["Eye"],
            marker=dict(size=6)
        )

        fig.update_layout(
            title=f"{cfg['title']} — {pd.to_datetime(t).strftime('%Y-%m-%d %H:%M UTC')}",
            scene=dict(
                xaxis_title="Lon",
                yaxis_title="Lat",
                zaxis_title="Wind Height",
                camera=dict(eye=dict(x=1.6, y=1.3, z=0.9))
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )

        fig.write_image(frames_dir / f"frame_{i:04d}.png", width=1280, height=720)

        if (i + 1) % 30 == 0:
            print(f"  rendered {i+1}/{len(times)} frames")

    stitch_frames(frames_dir, out_mp4, FPS)


# ======================
# ENTRY
# ======================
if __name__ == "__main__":
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name'].upper()} ===")
        make_video(cfg)

    print("\n[DONE] 3D hurricane video created.")
