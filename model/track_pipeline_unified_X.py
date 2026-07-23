# =============================================================================
# Save as (recommended):
#   C:\Users\Adrija\Downloads\DFGCN\track_pipeline_unified_X.py
#
# "Paper-ready" PATH-ONLY pipeline with:
#   ✅ Unified representation: X(t,node,modality,features)
#       node     = storm-centered grid points (GxG)
#       modality = ERA5 atmospheric
#       features = [u850,v850,u500,v500,z500]  (pressure fields can be added later)
#
#   ✅ Primary model:
#       Graph-Neural-Operator-style Encoder + Dynamic GNN + Probabilistic Head
#
#   ✅ Strong baselines for benchmarking (publication-connected):
#       - Persistence (constant velocity)
#       - LSTM baseline (past track sequence + ERA5 patch)
#       - Transformer baseline (past track sequence + ERA5 patch)
#       - HAFS hook (NOAA operational)   -> loader stub + evaluation interface
#       - Pangu/FourCastNet hook         -> loader stub + evaluation interface
#
#   ✅ Metrics (path-only, publication standard):
#       - Track error (km) at 6/12/24/48h
#       - Along-track / cross-track error (diagnostic approximation)
#       - Cone coverage P50/P90 (calibration; Gaussian ellipse approx)
#       - Landfall time error (proxy using Florida bounding box)
#
# Data expected:
#   Tracks (processed):
#     data\processed\tracks\irma_2017_hurdat2.csv
#     data\processed\tracks\ian_2022_hurdat2.csv
#   ERA5 NetCDF (raw):
#     data\raw\era5\irma_2017\era5_pl_irma_2017.nc
#     data\raw\era5\ian_2022\era5_pl_ian_2022.nc
#
# Install:
#   pip install numpy pandas xarray netCDF4 torch tqdm scikit-learn
#
# Run:
#   python "C:\Users\Adrija\Downloads\DFGCN\track_pipeline_unified_X.py"
# =============================================================================

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    base: str = r"C:\Users\Adrija\Downloads\DFGCN\data"

    irma_track: str = r"C:\Users\Adrija\Downloads\DFGCN\data\processed\tracks\irma_2017_hurdat2.csv"
    ian_track:  str = r"C:\Users\Adrija\Downloads\DFGCN\data\processed\tracks\ian_2022_hurdat2.csv"

    irma_era5: str = r"C:\Users\Adrija\Downloads\DFGCN\data\raw\era5\irma_2017\era5_pl_irma_2017.nc"
    ian_era5:  str = r"C:\Users\Adrija\Downloads\DFGCN\data\raw\era5\ian_2022\era5_pl_ian_2022.nc"

    # Unified representation X(t,node,modality,features)
    grid_size: int = 33          # node grid size: GxG
    crop_deg: float = 8.0        # storm-centered crop half-size in degrees
    features: Tuple[str, ...] = ("u850","v850","u500","v500","z500")

    history_steps: int = 4       # last N steps (each 6h)
    lead_hours: Tuple[int, ...] = (6, 12, 24, 48)

    include_metadata: bool = True  # vmax, mslp
    target_scale_deg: float = 10.0  # normalize future displacements for stable Gaussian NLL

    # Training (keep small for fast iteration; you can increase later)
    seed: int = 42
    batch_size: int = 16
    epochs_main: int = 20
    epochs_baseline: int = 12
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    lr: float = 2e-4
    wd: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_root: str = r"C:\Users\Adrija\Downloads\DFGCN"
    ckpt_dir: str = r"C:\Users\Adrija\Downloads\DFGCN\checkpoints"
    metrics_dir: str = r"C:\Users\Adrija\Downloads\DFGCN\metrics"

    # Florida landfall proxy (simple bounding box)
    florida_bbox: Tuple[float, float, float, float] = (24.0, 32.0, -88.0, -79.0)  # (lat_min, lat_max, lon_min, lon_max)

    # If prediction/label never enters bbox within leads, landfall error is undefined
    landfall_missing_policy: str = "ignore"  # "ignore" or "max"

cfg = CFG()


# ----------------------------
# Repro helpers
# ----------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)


def split_sample_indices(n_samples: int, seed: Optional[int] = None, val_ratio: Optional[float] = None, test_ratio: Optional[float] = None):
    if n_samples < 3:
        raise ValueError("At least three samples are required for train/val/test splitting.")
    seed = cfg.seed if seed is None else seed
    val_ratio = cfg.val_ratio if val_ratio is None else val_ratio
    test_ratio = cfg.test_ratio if test_ratio is None else test_ratio
    rng = np.random.RandomState(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_test = max(1, int(n_samples * test_ratio))
    n_val = max(1, int(n_samples * val_ratio))
    if n_test + n_val >= n_samples:
        n_val = max(1, n_samples - n_test - 1)
    n_train = n_samples - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Invalid split sizes for n={n_samples}: train={n_train}, val={n_val}, test={n_test}")
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def wrap_lon_delta(delta):
    return ((delta + 180.0) % 360.0) - 180.0


def wrap_lon_abs(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def normalize_era5_patch(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    return ((X - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def normalize_past_track(past: np.ndarray) -> np.ndarray:
    past = np.asarray(past, dtype=np.float32).copy()
    past[:, 0] = past[:, 0] / 90.0
    past[:, 1] = past[:, 1] / 180.0
    return past.astype(np.float32)


def normalize_meta(meta: Optional[np.ndarray]) -> np.ndarray:
    if meta is None:
        return np.zeros(2, dtype=np.float32)
    vmax = float(meta[0]) if np.isfinite(meta[0]) and meta[0] > 0 else 0.0
    mslp = float(meta[1]) if np.isfinite(meta[1]) and meta[1] > 800.0 else 950.0
    return np.array([vmax / 150.0, (mslp - 950.0) / 80.0], dtype=np.float32)


def normalize_target_delta(y_abs: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    y_abs = np.asarray(y_abs, dtype=np.float32)
    out = np.empty_like(y_abs, dtype=np.float32)
    out[:, 0] = (y_abs[:, 0] - lat0) / cfg.target_scale_deg
    out[:, 1] = wrap_lon_delta(y_abs[:, 1] - lon0) / cfg.target_scale_deg
    return out


def _batch_tensor(values, device):
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.float32)
    return torch.as_tensor(values, device=device, dtype=torch.float32)


def decode_track_delta(mu_delta: torch.Tensor, sigma_delta: Optional[torch.Tensor], lat0, lon0):
    lat0_t = _batch_tensor(lat0, mu_delta.device).view(-1, 1)
    lon0_t = _batch_tensor(lon0, mu_delta.device).view(-1, 1)
    lat = lat0_t + mu_delta[..., 0] * cfg.target_scale_deg
    lon = torch.remainder(lon0_t + mu_delta[..., 1] * cfg.target_scale_deg + 180.0, 360.0) - 180.0
    if sigma_delta is None:
        return lat, lon, None, None
    sig_lat = sigma_delta[..., 0] * cfg.target_scale_deg
    sig_lon = sigma_delta[..., 1] * cfg.target_scale_deg
    return lat, lon, sig_lat, sig_lon


# ----------------------------
# Geodesy / metrics helpers
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


def bearing_rad(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    return math.atan2(y, x)


def along_cross_track_errors(lat0, lon0, lat_true, lon_true, lat_pred, lon_pred):
    """
    Approx diagnostic decomposition: along-track and cross-track.
    Uses bearing origin->true as the reference direction.
    """
    ref = bearing_rad(lat0, lon0, lat_true, lon_true)
    d_op = haversine_km(lat0, lon0, lat_pred, lon_pred)
    b_op = bearing_rad(lat0, lon0, lat_pred, lon_pred)
    dtheta = b_op - ref
    along = d_op * math.cos(dtheta)
    cross = d_op * math.sin(dtheta)
    d_ot = haversine_km(lat0, lon0, lat_true, lon_true)
    along_err = along - d_ot
    cross_err = cross
    return along_err, cross_err


def in_bbox(lat, lon, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


def first_entry_lead_index(lat_seq, lon_seq, bbox):
    """
    Returns index in [0..L-1] of first entry into bbox, else None.
    lat_seq/lon_seq are sequences over lead steps only.
    """
    for i, (la, lo) in enumerate(zip(lat_seq, lon_seq)):
        if in_bbox(float(la), float(lo), bbox):
            return i
    return None


# ----------------------------
# ERA5 IO + robust crop (matches your data)
# ----------------------------
def open_era5(nc_path: str) -> xr.Dataset:
    ds = xr.open_dataset(nc_path)

    # vars
    var_u = "u" if "u" in ds.variables else "u_component_of_wind"
    var_v = "v" if "v" in ds.variables else "v_component_of_wind"
    var_z = "z" if "z" in ds.variables else "geopotential"
    if var_u not in ds.variables or var_v not in ds.variables or var_z not in ds.variables:
        raise ValueError(f"ERA5 vars not found. ds.variables={list(ds.variables)}")

    # pressure coord
    if "level" in ds.coords:
        plev = "level"
    elif "pressure_level" in ds.coords:
        plev = "pressure_level"
    else:
        raise ValueError(f"Pressure level coord not found. coords={list(ds.coords)}")

    # time coord (your files can be valid_time)
    if "time" in ds.coords:
        tcoord = "time"
    elif "valid_time" in ds.coords:
        tcoord = "valid_time"
    else:
        raise ValueError(f"Time coord not found. coords={list(ds.coords)}")

    # Normalize longitude to [-180, 180] and sort (prevents empty crops)
    if "longitude" not in ds.coords:
        raise ValueError("ERA5 missing longitude coordinate")
    lon = ds["longitude"]
    if float(lon.max()) > 180:
        lon_new = ((lon + 180) % 360) - 180
        ds = ds.assign_coords(longitude=lon_new)
    ds = ds.sortby("longitude")

    ds.attrs["_u"] = var_u
    ds.attrs["_v"] = var_v
    ds.attrs["_z"] = var_z
    ds.attrs["_plev"] = plev
    ds.attrs["_tcoord"] = tcoord
    return ds


def parse_track(csv_path: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df["storm_tag"] = tag
    for col in ["vmax_kt", "mslp_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def crop_era5_to_X(ds: xr.Dataset, tstamp: pd.Timestamp, lat0: float, lon0: float) -> np.ndarray:
    """
    Returns X(t,node,atmo,features) as (F,G,G) where
      F = [u850,v850,u500,v500,z500]
    """
    tcoord = ds.attrs["_tcoord"]
    uvar, vvar, zvar = ds.attrs["_u"], ds.attrs["_v"], ds.attrs["_z"]
    plev = ds.attrs["_plev"]

    dsel = ds.sel({tcoord: np.datetime64(tstamp.to_datetime64())}, method="nearest")

    lat_min, lat_max = lat0 - cfg.crop_deg, lat0 + cfg.crop_deg
    lon_min, lon_max = lon0 - cfg.crop_deg, lon0 + cfg.crop_deg

    # latitude slice direction-aware
    lat_vals = dsel["latitude"].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    lon_slice = slice(lon_min, lon_max)  # longitude normalized to [-180,180] and sorted

    box = dsel.sel(latitude=lat_slice, longitude=lon_slice)

    if box.sizes.get("longitude", 0) == 0 or box.sizes.get("latitude", 0) == 0:
        raise RuntimeError(
            f"Empty crop at t={tstamp} lat0={lat0:.2f} lon0={lon0:.2f} "
            f"lat[{lat_min:.2f},{lat_max:.2f}] lon[{lon_min:.2f},{lon_max:.2f}] "
            f"got lat={box.sizes.get('latitude',0)} lon={box.sizes.get('longitude',0)}"
        )

    def pl(varname: str, level: int):
        return box[varname].sel({plev: level}).values.astype(np.float32)

    u850 = pl(uvar, 850); v850 = pl(vvar, 850)
    u500 = pl(uvar, 500); v500 = pl(vvar, 500)
    z500 = pl(zvar, 500)

    X = np.stack([u850, v850, u500, v500, z500], axis=0)  # (F,H,W)
    Xt = torch.from_numpy(X).unsqueeze(0)  # (1,F,H,W)
    Xt = F.interpolate(Xt, size=(cfg.grid_size, cfg.grid_size), mode="bilinear", align_corners=False)
    return Xt.squeeze(0).numpy()  # (F,G,G)


# ----------------------------
# Sample builder (Input/Output definition)
# ----------------------------
def build_samples(track_df: pd.DataFrame, era5_ds: xr.Dataset) -> List[Dict]:
    lead_steps = [h // 6 for h in cfg.lead_hours]
    samples = []
    skipped = 0

    for i in range(cfg.history_steps - 1, len(track_df)):
        if i + max(lead_steps) >= len(track_df):
            break

        t0 = track_df.loc[i, "datetime_utc"]
        lat0 = float(track_df.loc[i, "lat"])
        lon0 = float(track_df.loc[i, "lon"])

        # past positions (H,2) oldest->newest, including the current t0 state
        past = []
        for k in range(cfg.history_steps - 1, -1, -1):
            past.append([float(track_df.loc[i-k, "lat"]), float(track_df.loc[i-k, "lon"])])
        past = np.array(past, dtype=np.float32)

        # meta
        meta = None
        if cfg.include_metadata:
            vmax = float(track_df.loc[i, "vmax_kt"]) if pd.notna(track_df.loc[i, "vmax_kt"]) else 0.0
            mslp = float(track_df.loc[i, "mslp_mb"]) if pd.notna(track_df.loc[i, "mslp_mb"]) else 0.0
            meta = np.array([vmax, mslp], dtype=np.float32)

        # unified X(t,node,atmo,features)
        try:
            X = crop_era5_to_X(era5_ds, t0, lat0, lon0)
        except Exception:
            skipped += 1
            continue

        # labels: future absolute positions at leads
        y_abs = []
        for step in lead_steps:
            y_abs.append([float(track_df.loc[i+step, "lat"]), float(track_df.loc[i+step, "lon"])])
        y_abs = np.array(y_abs, dtype=np.float32)

        samples.append({
            "storm_tag": track_df.loc[i, "storm_tag"],
            "t0": t0.isoformat(),
            "past": past,
            "X": X,
            "meta": meta,
            "y_abs": y_abs,
            "lat0": lat0,
            "lon0": lon0
        })

    print(f"[build_samples:{track_df['storm_tag'].iloc[0]}] kept={len(samples)} skipped={skipped}")
    return samples


class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        lat0 = float(s["lat0"])
        lon0 = float(s["lon0"])
        past = torch.from_numpy(normalize_past_track(s["past"]))       # (H,2)
        X = torch.from_numpy(normalize_era5_patch(s["X"]))             # (F,G,G)
        y = torch.from_numpy(normalize_target_delta(s["y_abs"], lat0, lon0))  # (L,2)
        meta = torch.from_numpy(normalize_meta(s["meta"]))
        info = (s["storm_tag"], s["t0"], s["lat0"], s["lon0"])
        return past, X, meta, y, info


# ----------------------------
# Baseline HOOKS (HAFS / Pangu / FourCastNet)
# ----------------------------
def load_hafs_track(storm_tag: str) -> pd.DataFrame:
    """
    Hook for NOAA HAFS operational tracks.

    Expected return format (example columns):
      datetime_utc (timestamp), lead_hours (int), lat (float), lon (float)

    Implement later by downloading HAFS track products for Irma/Ian and parsing.
    For now, raise NotImplementedError so the pipeline remains "paper-ready"
    with a clear plug-in point.
    """
    raise NotImplementedError("HAFS loader not implemented yet. Add parser for NOAA HAFS track products.")


def load_pangu_or_fourcastnet_track(storm_tag: str) -> pd.DataFrame:
    """
    Hook for extracted tracks from Pangu-Weather/FourCastNet reanalysis-style outputs.

    Expected return format:
      datetime_utc (timestamp), lead_hours (int), lat (float), lon (float)

    Implement later: run/download the model output fields and extract a track
    (e.g., using minimum sea-level pressure center or vorticity center).
    """
    raise NotImplementedError("Pangu/FourCastNet track extraction not implemented yet.")


# ----------------------------
# Models
# ----------------------------
class PersistenceBaseline:
    """Constant velocity extrapolation using last two points in history."""
    def predict_np(self, past: np.ndarray, lead_steps: List[int]) -> np.ndarray:
        p1 = past[-2]
        p2 = past[-1]
        v = p2 - p1  # degrees per 6h
        preds = [p2 + v * s for s in lead_steps]
        return np.array(preds, dtype=np.float32)  # (L,2)


class OperatorEncoder(nn.Module):
    """
    Lightweight operator-style encoder (fast). For a true FNO later,
    swap this with an FNO/AFNO module without changing the pipeline.
    """
    def __init__(self, in_ch: int, width: int = 48, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, out_dim),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.head(self.net(X))


class DynamicGNN(nn.Module):
    """Dynamic message passing over history nodes (positions)."""
    def __init__(self, node_dim=32, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Linear(2, node_dim)
        self.mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, node_dim))
            for _ in range(layers)
        ])
        self.readout = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, past):
        # past: (B,H,2)
        h = self.embed(past)  # (B,H,node_dim)
        tau = 2.0
        d2 = ((past[:, :, None, :] - past[:, None, :, :]) ** 2).sum(-1)  # (B,H,H)
        A = torch.softmax(-d2 / tau, dim=-1)
        for layer in self.mlp:
            m = torch.einsum("bij,bjn->bin", A, h)
            h = h + layer(m)
        g = h.mean(dim=1)
        return self.readout(g)  # (B,hidden)


class ProbTrackHead(nn.Module):
    """Outputs Gaussian (mu, sigma) for each lead time (lat,lon)."""
    def __init__(self, in_dim: int, leads: int):
        super().__init__()
        self.leads = leads
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, leads * 2)
        self.log_sigma = nn.Linear(128, leads * 2)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h).view(-1, self.leads, 2)
        log_sigma = self.log_sigma(h).view(-1, self.leads, 2).clamp(-6, 3)
        sigma = torch.exp(log_sigma)
        return mu, sigma


class GNO_DynGNN(nn.Module):
    """
    Primary model:
      Operator-style encoder over ERA5 patch + DynamicGNN over history + meta -> probabilistic head
    """
    def __init__(self, feat_ch: int, leads: int, use_meta: bool = True):
        super().__init__()
        self.use_meta = use_meta
        self.op = OperatorEncoder(in_ch=feat_ch, width=48, out_dim=128)
        self.gnn = DynamicGNN(node_dim=32, hidden=64, layers=2)
        self.past_mlp = nn.Sequential(nn.Flatten(), nn.Linear(cfg.history_steps * 2, 64), nn.ReLU())
        meta_dim = 2 if use_meta else 0
        self.head = ProbTrackHead(in_dim=128 + 64 + 64 + meta_dim, leads=leads)

    def forward(self, past, X, meta):
        op = self.op(X)          # (B,128)
        g  = self.gnn(past)      # (B,64)
        p  = self.past_mlp(past) # (B,64)
        parts = [op, g, p]
        if self.use_meta:
            parts.append(meta)
        h = torch.cat(parts, dim=-1)
        return self.head(h)


class LSTMTrackBaseline(nn.Module):
    """
    Baseline: LSTM over past positions + ERA5 operator encoder + meta -> probabilistic head
    """
    def __init__(self, feat_ch: int, leads: int, use_meta: bool = True, hidden: int = 64):
        super().__init__()
        self.use_meta = use_meta
        self.op = OperatorEncoder(in_ch=feat_ch, width=32, out_dim=96)
        self.pos_embed = nn.Linear(2, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden, num_layers=1, batch_first=True)
        meta_dim = 2 if use_meta else 0
        self.head = ProbTrackHead(in_dim=96 + hidden + meta_dim, leads=leads)

    def forward(self, past, X, meta):
        op = self.op(X)  # (B,96)
        seq = self.pos_embed(past)  # (B,H,32)
        out, (hn, cn) = self.lstm(seq)
        h_last = hn[-1]  # (B,hidden)
        parts = [op, h_last]
        if self.use_meta:
            parts.append(meta)
        h = torch.cat(parts, dim=-1)
        return self.head(h)


class TransformerTrackBaseline(nn.Module):
    """
    Baseline: Transformer encoder over past positions + ERA5 operator encoder + meta -> probabilistic head
    """
    def __init__(self, feat_ch: int, leads: int, use_meta: bool = True, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.use_meta = use_meta
        self.op = OperatorEncoder(in_ch=feat_ch, width=32, out_dim=96)
        self.pos_embed = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        meta_dim = 2 if use_meta else 0
        self.head = ProbTrackHead(in_dim=96 + d_model + meta_dim, leads=leads)

    def forward(self, past, X, meta):
        op = self.op(X)  # (B,96)
        seq = self.pos_embed(past)          # (B,H,d)
        z = self.tr(seq)                    # (B,H,d)
        pooled = z.mean(dim=1)              # (B,d)
        parts = [op, pooled]
        if self.use_meta:
            parts.append(meta)
        h = torch.cat(parts, dim=-1)
        return self.head(h)


def build_grid_diffusion_matrices(grid_size: int, k_neighbors: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward/backward random-walk diffusion transition matrices for a regular
    grid_size x grid_size grid, using a k-nearest-neighbor graph over
    normalized grid coordinates. Used by DCRNN's diffusion convolution.

    Returns (P_forward, P_backward), each (N, N) with N = grid_size**2.
      P_forward  = D^-1 A     (row-normalized adjacency: forward random walk)
      P_backward = D^-1 A^T   (row-normalized transpose: backward random walk)
    """
    N = grid_size * grid_size
    lin = np.linspace(-1.0, 1.0, grid_size)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    coords = np.stack([yy.ravel(), xx.ravel()], axis=-1)  # (N,2)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    knn_idx = np.argsort(d2, axis=1)[:, :k_neighbors]
    A = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), k_neighbors)
    A[rows, knn_idx.ravel()] = 1.0
    A = np.maximum(A, A.T)  # symmetrize into an undirected k-NN graph
    D = A.sum(axis=1, keepdims=True)
    D[D == 0] = 1.0
    P_f = torch.tensor(A / D, dtype=torch.float32)
    P_b = torch.tensor((A / D).T.copy(), dtype=torch.float32)
    return P_f, P_b


class DiffusionConv(nn.Module):
    """
    K-hop bidirectional diffusion graph convolution
    (Li, Yi, Shahabi & Liu, "Diffusion Convolutional Recurrent Neural
    Network", ICLR 2018):

        y = sum_{k=0}^{K} theta_f_k (P_f^k x) + theta_b_k (P_b^k x)

    P_f / P_b are the fixed forward/backward random-walk transition
    matrices of a k-NN graph over the ERA5 grid nodes (build_grid_diffusion_
    matrices). This replaces the plain CNN used by OperatorEncoder in the
    other baselines with a genuine graph diffusion operator.
    """

    def __init__(self, in_ch: int, out_ch: int, grid_size: int, k_hops: int = 2, k_neighbors: int = 8):
        super().__init__()
        self.k_hops = k_hops
        P_f, P_b = build_grid_diffusion_matrices(grid_size, k_neighbors)
        self.register_buffer("P_f", P_f)
        self.register_buffer("P_b", P_b)
        self.lin = nn.Linear(in_ch * 2 * (k_hops + 1), out_ch)

    def forward(self, x):  # x: (B, N, in_ch)
        feats = []
        for P in (self.P_f, self.P_b):
            Tx = x
            feats.append(Tx)
            for _ in range(self.k_hops):
                Tx = torch.einsum("nm,bmc->bnc", P, Tx)
                feats.append(Tx)
        h = torch.cat(feats, dim=-1)  # (B, N, in_ch * 2 * (k_hops+1))
        return self.lin(h)


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU cell: standard GRU gating with the linear
    projections replaced by DiffusionConv layers, so each recurrent update
    mixes information across the graph (not just across channels).
    """

    def __init__(self, in_ch: int, hidden_ch: int, grid_size: int, k_hops: int = 2):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gate_conv = DiffusionConv(in_ch + hidden_ch, 2 * hidden_ch, grid_size, k_hops)
        self.cand_conv = DiffusionConv(in_ch + hidden_ch, hidden_ch, grid_size, k_hops)

    def forward(self, x, h):  # x: (B,N,in_ch), h: (B,N,hidden_ch)
        xh = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(xh))
        r, u = gates.chunk(2, dim=-1)
        xh_r = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.cand_conv(xh_r))
        return u * h + (1 - u) * c


class DCRNNTrackBaseline(nn.Module):
    """
    Baseline: Diffusion Convolutional Recurrent Neural Network
    (Li, Yi, Shahabi & Liu, ICLR 2018), adapted to this pipeline's inputs.

    The ERA5 patch is treated as a signal on a fixed k-NN graph over the
    storm-centered grid; a DCGRU cell recurrently updates a per-node hidden
    state using bidirectional diffusion convolution. Only one ERA5 snapshot
    is available per sample in this pipeline (unlike the original DCRNN's
    multi-step traffic-sensor sequences), so the DCGRU is unrolled over
    `history_steps`, broadcasting the corresponding past-track position onto
    the shared ERA5 signal at each step -- the recurrence still carries a
    genuine hidden state h_t forward exactly like the DCRNN encoder in the
    original paper, it is simply driven by (ERA5, past-position_t) pairs
    instead of (traffic-speed_t) snapshots.
    """

    def __init__(self, feat_ch: int, leads: int, grid_size: int, use_meta: bool = True,
                 hidden_ch: int = 8, k_hops: int = 2):
        super().__init__()
        self.use_meta = use_meta
        self.hidden_ch = hidden_ch
        self.pos_proj = nn.Linear(2, feat_ch)  # broadcast past position onto every graph node
        self.cell = DCGRUCell(feat_ch, hidden_ch, grid_size, k_hops)
        meta_dim = 2 if use_meta else 0
        self.head = ProbTrackHead(in_dim=hidden_ch + meta_dim, leads=leads)

    def forward(self, past, X, meta):
        B, C, H, W = X.shape
        N = H * W
        x_nodes = X.reshape(B, C, N).permute(0, 2, 1)  # (B,N,C)
        h = torch.zeros(B, N, self.hidden_ch, device=X.device, dtype=X.dtype)
        for t in range(past.shape[1]):
            pos_feat = self.pos_proj(past[:, t, :]).unsqueeze(1)  # (B,1,C)
            h = self.cell(x_nodes + pos_feat, h)
        pooled = h.mean(dim=1)  # (B, hidden_ch)  graph readout
        parts = [pooled]
        if self.use_meta:
            parts.append(meta)
        return self.head(torch.cat(parts, dim=-1))


# ----------------------------
# Loss + cone coverage
# ----------------------------
def gaussian_nll(mu, sigma, y):
    eps = 1e-6
    var = sigma**2 + eps
    return 0.5 * torch.mean(((y - mu) ** 2) / var + torch.log(var))


def ellipse_inclusion(lat_true, lon_true, mu_lat, mu_lon, sigma_lat, sigma_lon, z):
    """
    Axis-aligned Gaussian ellipse inclusion:
      ((x-mu)/sigma)^2 sum <= z^2
    """
    dx = (lat_true - mu_lat) / (sigma_lat + 1e-6)
    dy = (lon_true - mu_lon) / (sigma_lon + 1e-6)
    return (dx*dx + dy*dy) <= (z*z)


# For 2D Gaussian: use chi-square cutoffs (approx)
Z_P50 = 1.177  # sqrt(chi2_2(0.50))
Z_P90 = 2.146  # sqrt(chi2_2(0.90))


# ----------------------------
# Evaluation (includes landfall time error)
# ----------------------------
@torch.no_grad()
def evaluate_prob_model(model: nn.Module, loader) -> Dict[str, float]:
    model.eval()

    track_err = [[] for _ in cfg.lead_hours]
    along_err = [[] for _ in cfg.lead_hours]
    cross_err = [[] for _ in cfg.lead_hours]
    cov50 = [[] for _ in cfg.lead_hours]
    cov90 = [[] for _ in cfg.lead_hours]

    landfall_err_hours = []

    for past, X, meta, y, info in loader:
        _, _, lat0_info, lon0_info = info
        past = past.to(cfg.device)
        X = X.to(cfg.device)
        meta = meta.to(cfg.device)
        y = y.to(cfg.device)

        mu, sigma = model(past, X, meta)  # (B,L,2)
        mu_lat, mu_lon, sig_lat, sig_lon = decode_track_delta(mu, sigma, lat0_info, lon0_info)
        y_lat, y_lon, _, _ = decode_track_delta(y, None, lat0_info, lon0_info)

        for b in range(mu.size(0)):
            lat0 = float(_batch_tensor(lat0_info, cfg.device)[b].cpu())
            lon0 = float(_batch_tensor(lon0_info, cfg.device)[b].cpu())

            # landfall proxy: among lead points only
            true_lat_seq = y_lat[b, :].cpu().numpy()
            true_lon_seq = y_lon[b, :].cpu().numpy()
            pred_lat_seq = mu_lat[b, :].cpu().numpy()
            pred_lon_seq = mu_lon[b, :].cpu().numpy()

            t_idx = first_entry_lead_index(true_lat_seq, true_lon_seq, cfg.florida_bbox)
            p_idx = first_entry_lead_index(pred_lat_seq, pred_lon_seq, cfg.florida_bbox)

            if t_idx is None and p_idx is None:
                pass
            elif t_idx is None or p_idx is None:
                if cfg.landfall_missing_policy == "max":
                    landfall_err_hours.append(float(max(cfg.lead_hours)))
            else:
                landfall_err_hours.append(abs(cfg.lead_hours[p_idx] - cfg.lead_hours[t_idx]))

            for li, h in enumerate(cfg.lead_hours):
                lat_t = float(y_lat[b, li].cpu())
                lon_t = float(y_lon[b, li].cpu())
                lat_p = float(mu_lat[b, li].cpu())
                lon_p = float(mu_lon[b, li].cpu())

                km = haversine_km(lat_t, lon_t, lat_p, lon_p)
                at, ct = along_cross_track_errors(lat0, lon0, lat_t, lon_t, lat_p, lon_p)

                track_err[li].append(km)
                along_err[li].append(at)
                cross_err[li].append(ct)

                cov50[li].append(bool(ellipse_inclusion(
                    lat_t, lon_t,
                    float(mu_lat[b, li].cpu()),
                    float(mu_lon[b, li].cpu()),
                    float(sig_lat[b, li].cpu()),
                    float(sig_lon[b, li].cpu()),
                    Z_P50
                )))
                cov90[li].append(bool(ellipse_inclusion(
                    lat_t, lon_t,
                    float(mu_lat[b, li].cpu()),
                    float(mu_lon[b, li].cpu()),
                    float(sig_lat[b, li].cpu()),
                    float(sig_lon[b, li].cpu()),
                    Z_P90
                )))

    metrics: Dict[str, float] = {}
    for i, h in enumerate(cfg.lead_hours):
        metrics[f"track_km_{h}h"] = float(np.mean(track_err[i])) if track_err[i] else float("nan")
        metrics[f"along_err_km_{h}h"] = float(np.mean(along_err[i])) if along_err[i] else float("nan")
        metrics[f"cross_err_km_{h}h"] = float(np.mean(cross_err[i])) if cross_err[i] else float("nan")
        metrics[f"cone_cov50_{h}h"] = float(np.mean(cov50[i])) if cov50[i] else float("nan")
        metrics[f"cone_cov90_{h}h"] = float(np.mean(cov90[i])) if cov90[i] else float("nan")

    metrics["landfall_time_err_hours"] = float(np.mean(landfall_err_hours)) if landfall_err_hours else float("nan")
    return metrics


def evaluate_persistence(te_ds: TrackDataset) -> Dict[str, float]:
    pers = PersistenceBaseline()
    lead_steps = [h // 6 for h in cfg.lead_hours]
    out: Dict[str, List[float]] = {f"track_km_{h}h": [] for h in cfg.lead_hours}

    landfall_err_hours = []

    for s in te_ds.samples:
        past = s["past"]
        y = s["y_abs"]
        preds = pers.predict_np(past, lead_steps)

        # landfall proxy among leads
        t_idx = first_entry_lead_index(y[:, 0], y[:, 1], cfg.florida_bbox)
        p_idx = first_entry_lead_index(preds[:, 0], preds[:, 1], cfg.florida_bbox)

        if t_idx is None and p_idx is None:
            pass
        elif t_idx is None or p_idx is None:
            if cfg.landfall_missing_policy == "max":
                landfall_err_hours.append(float(max(cfg.lead_hours)))
        else:
            landfall_err_hours.append(abs(cfg.lead_hours[p_idx] - cfg.lead_hours[t_idx]))

        for i, h in enumerate(cfg.lead_hours):
            out[f"track_km_{h}h"].append(haversine_km(float(y[i,0]), float(y[i,1]), float(preds[i,0]), float(preds[i,1])))

    metrics = {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}
    metrics["landfall_time_err_hours"] = float(np.mean(landfall_err_hours)) if landfall_err_hours else float("nan")
    # cone metrics are not defined for deterministic persistence
    for h in cfg.lead_hours:
        metrics[f"cone_cov50_{h}h"] = float("nan")
        metrics[f"cone_cov90_{h}h"] = float("nan")
        metrics[f"along_err_km_{h}h"] = float("nan")
        metrics[f"cross_err_km_{h}h"] = float("nan")
    return metrics


# ----------------------------
# Training utilities
# ----------------------------
def train_prob_model(model: nn.Module, tr_loader, val_loader, test_loader, epochs: int, name: str) -> Dict[str, float]:
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best = float("inf")
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{name}.pt")
    best_epoch = None

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for past, X, meta, y, info in tqdm(tr_loader, desc=f"{name} Ep {ep}/{epochs}", leave=False):
            past = past.to(cfg.device)
            X = X.to(cfg.device)
            meta = meta.to(cfg.device)
            y = y.to(cfg.device)

            mu, sigma = model(past, X, meta)
            loss = gaussian_nll(mu, sigma, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate_prob_model(model, val_loader)
        mean_km = float(np.mean([val_metrics[f"track_km_{h}h"] for h in cfg.lead_hours]))

        print(f"{name} | Ep {ep:02d} | train_nll={np.mean(losses):.4f} | val_mean_track_km={mean_km:.2f} | val_landfall_err_h={val_metrics['landfall_time_err_hours']:.2f}")
        if mean_km < best:
            best = mean_km
            best_epoch = ep
            torch.save({
                "state": model.state_dict(),
                "cfg": cfg.__dict__,
                "selection_metric": "val_mean_track_km_6_48h",
                "selection_score": best,
                "selected_epoch": best_epoch,
                "target_convention": "normalized_future_displacement_from_current_t0",
            }, ckpt_path)

    # load best and return metrics
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["state"])
    final_metrics = evaluate_prob_model(model, test_loader)
    print(f"[{name}] best checkpoint saved:", ckpt_path)
    return final_metrics


def save_metrics_row(model_name: str, metrics: Dict[str, float], out_csv: str):
    row = {"model": model_name, **metrics}
    df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dirs()
    seed_all(cfg.seed)

    # Load data
    irma_df = parse_track(cfg.irma_track, "irma")
    ian_df  = parse_track(cfg.ian_track,  "ian")
    irma_ds = open_era5(cfg.irma_era5)
    ian_ds  = open_era5(cfg.ian_era5)

    print("Building samples (Irma)...")
    s1 = build_samples(irma_df, irma_ds)
    print("Building samples (Ian)...")
    s2 = build_samples(ian_df, ian_ds)

    samples = s1 + s2
    print(f"Total samples: {len(samples)}")
    if len(samples) < 30:
        print("WARNING: few samples. If needed, expand your ERA5 time window and rebuild.")

    tr_idx, val_idx, te_idx = split_sample_indices(len(samples), seed=cfg.seed)

    tr_ds = TrackDataset([samples[i] for i in tr_idx])
    val_ds = TrackDataset([samples[i] for i in val_idx])
    te_ds = TrackDataset([samples[i] for i in te_idx])
    print(f"Split: train={len(tr_ds)} val={len(val_ds)} test={len(te_ds)}")

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

    # ----------------------------
    # Baseline: Persistence (must-have)
    # ----------------------------
    pers_metrics = evaluate_persistence(te_ds)
    save_metrics_row(
        "Persistence",
        pers_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_persistence.csv")
    )

    # ----------------------------
    # Baseline: LSTM
    # ----------------------------
    lstm = LSTMTrackBaseline(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        use_meta=cfg.include_metadata
    )
    lstm_metrics = train_prob_model(lstm, tr_loader, val_loader, te_loader, cfg.epochs_baseline, "baseline_lstm")
    save_metrics_row(
        "LSTM (past + ERA5)",
        lstm_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_lstm.csv")
    )

    # ----------------------------
    # Baseline: Transformer
    # ----------------------------
    trm = TransformerTrackBaseline(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        use_meta=cfg.include_metadata
    )
    trm_metrics = train_prob_model(trm, tr_loader, val_loader, te_loader, cfg.epochs_baseline, "baseline_transformer")
    save_metrics_row(
        "Transformer (past + ERA5)",
        trm_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_transformer.csv")
    )

    # ----------------------------
    # Baseline: DCRNN
    # ----------------------------
    dcrnn = DCRNNTrackBaseline(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        grid_size=cfg.grid_size,
        use_meta=cfg.include_metadata
    )
    dcrnn_metrics = train_prob_model(dcrnn, tr_loader, val_loader, te_loader, cfg.epochs_baseline, "baseline_dcrnn")
    save_metrics_row(
        "DCRNN (past + ERA5)",
        dcrnn_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_dcrnn.csv")
    )

    # ----------------------------
    # Primary model: GNO + DynGNN
    # ----------------------------
    main_model = GNO_DynGNN(
        feat_ch=len(cfg.features),
        leads=len(cfg.lead_hours),
        use_meta=cfg.include_metadata
    )
    main_metrics = train_prob_model(main_model, tr_loader, val_loader, te_loader, cfg.epochs_main, "main_gno_dyngnn")
    save_metrics_row(
        "GNO+DynGNN (prob)",
        main_metrics,
        os.path.join(cfg.metrics_dir, "track_metrics_gno_dyngnn.csv")
    )

    # ----------------------------
    # Hooks (not executed): HAFS / Pangu / FourCastNet
    # ----------------------------
    print("\n[INFO] HAFS / Pangu / FourCastNet hooks are included as loader stubs.")
    print("       Implement loaders later to benchmark operational + SOTA reference tracks.\n")


if __name__ == "__main__":
    main()
