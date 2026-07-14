"""
data_pipeline.py — Multi-source data pipeline for the STORM-CARE Foundation Model.

Sources
-------
1. HURDAT2        — full Atlantic basin (1851–present), 57 K+ lines
2. IBTrACS        — global multi-basin stub (plug real CSV in when available)
3. ERA5           — atmospheric patch extraction + on-disk cache (NumPy)
4. Vulnerability  — CDC SVI grid (RPL themes)
5. Recovery       — post-event recovery proxy labels

Primary output
--------------
MultiSourceDataPipeline.build() → List[StormRecord]

Each StormRecord holds:
  • track_df        — pandas DataFrame (datetime, lat, lon, vmax, mslp, …)
  • features        — np.ndarray (T, n_storm_features)  – normalised
  • basin_ids       — np.ndarray (T,)  int
  • status_ids      — np.ndarray (T,)  int
  • era5_patches    — np.ndarray (T, C, G, G) or None per step
  • era5_valid      — np.ndarray (T,)  bool
  • storm_id        — str
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_coord(tok: str) -> float:
    tok = tok.strip()
    sign = -1.0 if tok[-1] in ("S", "W") else 1.0
    return sign * float(tok[:-1])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


# ─────────────────────────────────────────────────────────────────────────────
# HURDAT2 full parser
# ─────────────────────────────────────────────────────────────────────────────

_H2_HEADER = re.compile(r"^[A-Z]{2}\d{6}")
_STATUS_MAP = {
    "TD": 0, "TS": 1, "HU": 2, "EX": 3, "SD": 4, "SS": 5, "LO": 6,
    "DB": 6, "WV": 6, "ET": 3, "PT": 6, "ST": 6,
}
_BASIN_MAP = {"AL": 0, "EP": 1, "WP": 2, "IO": 3, "CP": 1, "SH": 3, "SI": 3, "SP": 3}


def parse_hurdat2_full(
    path: str,
    min_year: int = 1980,
    min_track_len: int = 4,
) -> List[pd.DataFrame]:
    """
    Parse the full HURDAT2 Atlantic (or Pacific) text file.

    Returns a list of track DataFrames sorted by datetime, one per storm.
    Storms with fewer than *min_track_len* valid observations or earlier than
    *min_year* are discarded.

    Column schema
    -------------
    datetime_utc, lat, lon, vmax_kt, mslp_mb, status, status_int,
    storm_id, storm_name, year, basin, basin_int
    """
    path = str(path)
    if not os.path.exists(path):
        log.warning("HURDAT2 not found at %s — returning empty list.", path)
        return []

    storms: List[pd.DataFrame] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        parts = [p.strip() for p in raw.split(",")]

        if len(parts) >= 3 and _H2_HEADER.match(parts[0]):
            storm_id = parts[0]
            storm_name = parts[1] if len(parts) > 1 else "UNNAMED"
            try:
                n_records = int(parts[2])
            except ValueError:
                i += 1
                continue

            year = int(storm_id[4:8])
            basin = storm_id[:2]

            records: List[Dict] = []
            for j in range(1, n_records + 1):
                if i + j >= len(lines):
                    break
                dp = [p.strip() for p in lines[i + j].split(",")]
                if len(dp) < 8:
                    continue
                try:
                    dt = datetime.strptime(
                        dp[0].replace(" ", "") + dp[1].zfill(4), "%Y%m%d%H%M"
                    ).replace(tzinfo=timezone.utc)
                    status = dp[3]
                    lat = _parse_coord(dp[4])
                    lon = _parse_coord(dp[5])
                    vmax = float(dp[6]) if float(dp[6]) > 0 else np.nan
                    mslp = float(dp[7]) if float(dp[7]) > 0 else np.nan
                    records.append(
                        dict(
                            datetime_utc=dt,
                            lat=lat,
                            lon=lon,
                            vmax_kt=vmax,
                            mslp_mb=mslp,
                            status=status,
                            status_int=_STATUS_MAP.get(status, 6),
                            storm_id=storm_id,
                            storm_name=storm_name,
                            year=year,
                            basin=basin,
                            basin_int=_BASIN_MAP.get(basin, 0),
                        )
                    )
                except (ValueError, IndexError):
                    continue

            i += n_records + 1

            if year < min_year or len(records) < min_track_len:
                continue

            df = (
                pd.DataFrame(records)
                .sort_values("datetime_utc")
                .reset_index(drop=True)
            )
            storms.append(df)
        else:
            i += 1

    log.info("Parsed %d storms from HURDAT2 (year >= %d).", len(storms), min_year)
    return storms


# ─────────────────────────────────────────────────────────────────────────────
# IBTrACS stub  (plug real CSV in when available)
# ─────────────────────────────────────────────────────────────────────────────

def parse_ibtracs(path: Optional[str], min_year: int = 1980) -> List[pd.DataFrame]:
    """
    Load IBTrACS CSV (WMO columns).

    When *path* is None or the file is absent, return a small set of
    synthetic global storms so the pipeline can still demonstrate
    multi-basin pretraining without requiring the download (~100 MB).

    Real file: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/IBTrACS.ALL.list.v04r00.csv
    """
    if path and os.path.exists(path):
        df_all = pd.read_csv(
            path,
            skiprows=[1],  # IBTrACS has a unit row on line 2
            low_memory=False,
            parse_dates=["ISO_TIME"],
        )
        # Normalise column names
        df_all.columns = [c.strip().lower() for c in df_all.columns]
        required = {"sid", "iso_time", "lat", "lon", "usa_wind", "usa_pres", "basin"}
        missing = required - set(df_all.columns)
        if missing:
            log.warning("IBTrACS missing columns %s; falling back to synthetic.", missing)
            return _synthetic_global_storms(min_year)

        storms: List[pd.DataFrame] = []
        for sid, group in df_all.groupby("sid"):
            g = group.dropna(subset=["lat", "lon"]).sort_values("iso_time").copy()
            g["year"] = pd.to_datetime(g["iso_time"]).dt.year
            if int(g["year"].iloc[0]) < min_year or len(g) < 4:
                continue
            g = g.rename(
                columns={
                    "iso_time": "datetime_utc",
                    "usa_wind": "vmax_kt",
                    "usa_pres": "mslp_mb",
                }
            )
            g["storm_id"] = str(sid)
            g["storm_name"] = str(g.get("name", sid).iloc[0]) if "name" in g else str(sid)
            g["basin_int"] = _BASIN_MAP.get(str(g["basin"].iloc[0])[:2], 0)
            g["status"] = g.get("nature", "HU").fillna("HU")
            g["status_int"] = g["status"].map(_STATUS_MAP).fillna(6).astype(int)
            storms.append(g[["datetime_utc", "lat", "lon", "vmax_kt", "mslp_mb",
                               "status", "status_int", "storm_id", "storm_name",
                               "year", "basin", "basin_int"]].reset_index(drop=True))
        log.info("Loaded %d storms from IBTrACS.", len(storms))
        return storms

    log.info("IBTrACS not found — using synthetic global storm supplement.")
    return _synthetic_global_storms(min_year)


def _synthetic_global_storms(min_year: int) -> List[pd.DataFrame]:
    """
    Generate ~50 synthetic plausible global storm tracks for demonstration.

    These are NOT real historical events; they serve as stand-ins when the
    real IBTrACS file is unavailable so that the pipeline can exercise all
    data paths without requiring the download.
    """
    rng = np.random.RandomState(seed=2024)
    basin_specs = [
        # (basin, lat0_range, lon0_range, track_len, heading_deg, n_storms)
        ("WP", (8, 20),  (130, 155), (15, 40), 315, 15),
        ("IO", (8, 18),  (75, 95),   (10, 25), 280, 8),
        ("EP", (8, 18),  (-120, -90),(10, 30), 315, 10),
        ("SH", (-8, -20),(50, 80),   (10, 25), 225, 7),
        ("SP", (-8, -20),(160, 175), (10, 25), 210, 5),
    ]
    all_storms: List[pd.DataFrame] = []
    year = min_year

    for basin, lat_r, lon_r, len_r, hdg, n in basin_specs:
        for _ in range(n):
            n_steps = rng.randint(len_r[0], len_r[1])
            lat0 = rng.uniform(lat_r[0], lat_r[1])
            lon0 = rng.uniform(lon_r[0], lon_r[1])
            year_s = year + rng.randint(0, 5)
            month_s = rng.randint(6, 11)
            start = datetime(year_s, month_s, 1, 0, tzinfo=timezone.utc)

            rows = []
            lat, lon = lat0, lon0
            vmax = float(rng.uniform(30, 50))
            mslp = float(rng.uniform(990, 1010))
            heading = float(hdg + rng.uniform(-30, 30))

            for t in range(n_steps):
                speed_kmh = rng.uniform(15, 35)
                dist = speed_kmh * 6
                dlat = (dist / 111.0) * math.cos(math.radians(heading))
                dlon = (dist / (111.0 * max(0.01, math.cos(math.radians(lat))))) * math.sin(math.radians(heading))
                lat += dlat
                lon += dlon
                heading = (heading + rng.uniform(-10, 10)) % 360
                vmax += rng.uniform(-5, 8)
                vmax = float(np.clip(vmax, 20, 160))
                mslp -= rng.uniform(-3, 5)
                mslp = float(np.clip(mslp, 900, 1015))
                dt = datetime(year_s, month_s, 1, tzinfo=timezone.utc)
                import datetime as dt_mod
                dt = start + dt_mod.timedelta(hours=6 * t)
                cat = "HU" if vmax >= 64 else ("TS" if vmax >= 34 else "TD")
                rows.append(dict(
                    datetime_utc=dt,
                    lat=round(lat, 2),
                    lon=round(lon, 2),
                    vmax_kt=round(vmax, 1),
                    mslp_mb=round(mslp, 1),
                    status=cat,
                    status_int=_STATUS_MAP.get(cat, 6),
                    storm_id=f"{basin}SYN{year_s:04d}{len(all_storms):03d}",
                    storm_name=f"SYN-{basin}-{len(all_storms):03d}",
                    year=year_s,
                    basin=basin,
                    basin_int=_BASIN_MAP.get(basin, 0),
                ))

            if len(rows) >= 4:
                all_storms.append(
                    pd.DataFrame(rows).sort_values("datetime_utc").reset_index(drop=True)
                )

    log.info("Generated %d synthetic global storms.", len(all_storms))
    return all_storms


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_storm_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 12-dimensional normalised feature vector for each track point.

    Returns
    -------
    feats       : (T, 12)  float32 — normalised scalar features
    basin_ids   : (T,)     int32
    status_ids  : (T,)     int32
    """
    T = len(df)
    feats = np.zeros((T, 12), dtype=np.float32)
    basin_ids  = df["basin_int"].to_numpy(dtype=np.int32)
    status_ids = df["status_int"].to_numpy(dtype=np.int32)

    lats = df["lat"].to_numpy(dtype=np.float64)
    lons = df["lon"].to_numpy(dtype=np.float64)
    vmax = df["vmax_kt"].to_numpy(dtype=np.float64)
    mslp = df["mslp_mb"].to_numpy(dtype=np.float64)

    # Fill NaN with reasonable defaults before normalising
    vmax = np.where(np.isnan(vmax), 30.0, vmax)
    mslp = np.where(np.isnan(mslp), 1005.0, mslp)

    # 0 lat_norm
    feats[:, 0] = (lats / 90.0).astype(np.float32)
    # 1,2 lon sin/cos
    lon_rad = np.radians(lons)
    feats[:, 1] = np.sin(lon_rad).astype(np.float32)
    feats[:, 2] = np.cos(lon_rad).astype(np.float32)
    # 3 vmax_norm
    feats[:, 3] = (vmax / 200.0).astype(np.float32)
    # 4 mslp_norm
    feats[:, 4] = ((mslp - 900.0) / 120.0).astype(np.float32)

    # Speed and heading computed from successive positions
    speeds = np.zeros(T, dtype=np.float64)
    headings = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        d = _haversine_km(lats[t - 1], lons[t - 1], lats[t], lons[t])
        dt_h = 6.0  # assume 6-h intervals
        speeds[t] = d / dt_h  # km/h
        headings[t] = _bearing_deg(lats[t - 1], lons[t - 1], lats[t], lons[t])
    speeds[0] = speeds[1] if T > 1 else 0.0
    headings[0] = headings[1] if T > 1 else 0.0

    # 5 speed_norm
    feats[:, 5] = (np.clip(speeds, 0, 100) / 100.0).astype(np.float32)
    # 6,7 heading sin/cos
    hdg_rad = np.radians(headings)
    feats[:, 6] = np.sin(hdg_rad).astype(np.float32)
    feats[:, 7] = np.cos(hdg_rad).astype(np.float32)

    # 8 intensification rate Δvmax / 6h, clipped to ±20 kt
    dvmax = np.gradient(vmax)
    feats[:, 8] = (np.clip(dvmax, -20, 20) / 20.0).astype(np.float32)

    # 9 normalised storm age (lifecycle position)
    feats[:, 9] = (np.arange(T) / max(T - 1, 1)).astype(np.float32)

    # 10,11 day-of-year circular encoding
    if hasattr(df["datetime_utc"].iloc[0], "timetuple"):
        doys = np.array([
            r.timetuple().tm_yday if hasattr(r, "timetuple")
            else r.dayofyear
            for r in df["datetime_utc"]
        ], dtype=np.float64)
    else:
        doys = pd.DatetimeIndex(df["datetime_utc"]).dayofyear.to_numpy(dtype=np.float64)

    doy_rad = 2 * np.pi * doys / 365.25
    feats[:, 10] = np.sin(doy_rad).astype(np.float32)
    feats[:, 11] = np.cos(doy_rad).astype(np.float32)

    return feats, basin_ids, status_ids


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 patch cache
# ─────────────────────────────────────────────────────────────────────────────

def _crop_era5_patch(ds, tstamp, lat0: float, lon0: float, cfg) -> Optional[np.ndarray]:
    """
    Extract a (5, grid_size, grid_size) patch centred on (lat0, lon0) from ds.
    Returns None on failure.
    """
    import torch
    import torch.nn.functional as F

    try:
        # Resolve coordinate names robustly
        tcoord = next(
            (c for c in ["time", "valid_time"] if c in ds.coords), None
        )
        plev = next(
            (c for c in ["level", "pressure_level"] if c in ds.coords), None
        )
        uvar = next(
            (v for v in ["u", "u_component_of_wind"] if v in ds.variables), None
        )
        vvar = next(
            (v for v in ["v", "v_component_of_wind"] if v in ds.variables), None
        )
        zvar = next(
            (v for v in ["z", "geopotential"] if v in ds.variables), None
        )
        if None in (tcoord, plev, uvar, vvar, zvar):
            return None

        dsel = ds.sel({tcoord: np.datetime64(tstamp.to_datetime64())}, method="nearest")
        lat_vals = dsel["latitude"].values
        lat_slice = (
            slice(lat0 + cfg.crop_deg, lat0 - cfg.crop_deg)
            if lat_vals[0] > lat_vals[-1]
            else slice(lat0 - cfg.crop_deg, lat0 + cfg.crop_deg)
        )
        box = dsel.sel(latitude=lat_slice, longitude=slice(lon0 - cfg.crop_deg, lon0 + cfg.crop_deg))

        if box.sizes.get("longitude", 0) == 0 or box.sizes.get("latitude", 0) == 0:
            return None

        def pl(varname, level):
            return box[varname].sel({plev: level}).values.astype(np.float32)

        u850, v850 = pl(uvar, 850), pl(vvar, 850)
        u500, v500 = pl(uvar, 500), pl(vvar, 500)
        z500 = pl(zvar, 500)
        X = np.stack([u850, v850, u500, v500, z500], axis=0)  # (5, H, W)

        Xt = torch.from_numpy(X).unsqueeze(0)
        Xt = F.interpolate(
            Xt, size=(cfg.grid_size, cfg.grid_size),
            mode="bilinear", align_corners=False,
        )
        return Xt.squeeze(0).numpy()  # (5, G, G)
    except Exception:
        return None


def build_era5_cache(
    era5_paths: List[str],
    era5_storm_ids: List[str],
    all_storms: List[pd.DataFrame],
    cfg,
) -> Dict[str, np.ndarray]:
    """
    Pre-extract ERA5 patches for all storms that have a corresponding NetCDF.

    The resulting dict maps  "<storm_id>_<timestamp_iso>" → (5, G, G) array.
    The cache is also persisted to disk (cfg.era5_cache_dir) as individual .npy
    files so subsequent runs skip the expensive extraction.
    """
    cache_dir = Path(cfg.era5_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build a lookup from storm_id → DataFrame
    storm_lookup: Dict[str, pd.DataFrame] = {}
    for df in all_storms:
        sid = df["storm_id"].iloc[0]
        storm_lookup[sid] = df

    patch_cache: Dict[str, np.ndarray] = {}

    for era5_path, storm_id_key in zip(era5_paths, era5_storm_ids):
        if not os.path.exists(era5_path):
            log.warning("ERA5 file not found: %s", era5_path)
            continue

        # Find matching storm(s) in our database
        matches = [
            sid for sid in storm_lookup
            if storm_id_key.upper() in sid.upper()
        ]
        if not matches:
            # Try by name for tagged versions
            tag = storm_id_key.lower()
            matches = [
                sid for sid, df in storm_lookup.items()
                if tag in df["storm_name"].iloc[0].lower()
            ]
        if not matches:
            log.warning("No storm match for ERA5 tag '%s'", storm_id_key)
            continue

        log.info("Building ERA5 patch cache for tag '%s' → storm(s) %s",
                 storm_id_key, matches)

        try:
            import xarray as xr
            ds = xr.open_dataset(era5_path)
            # Normalise longitude to [-180, 180]
            lon = ds["longitude"]
            if float(lon.max()) > 180:
                ds = ds.assign_coords(longitude=((lon + 180) % 360) - 180)
            ds = ds.sortby("longitude")
        except Exception as exc:
            log.warning("Could not open ERA5 %s: %s", era5_path, exc)
            continue

        for storm_id in matches:
            df = storm_lookup[storm_id]
            for _, row in df.iterrows():
                ts = row["datetime_utc"]
                key = f"{storm_id}_{ts.isoformat()}"
                npy_path = cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"

                if npy_path.exists():
                    patch_cache[key] = np.load(str(npy_path))
                    continue

                patch = _crop_era5_patch(
                    ds, pd.Timestamp(ts), float(row["lat"]), float(row["lon"]), cfg
                )
                if patch is not None:
                    np.save(str(npy_path), patch)
                    patch_cache[key] = patch

        ds.close()

    log.info("ERA5 patch cache: %d patches extracted.", len(patch_cache))
    return patch_cache


# ─────────────────────────────────────────────────────────────────────────────
# Vulnerability loader
# ─────────────────────────────────────────────────────────────────────────────

def load_vulnerability_grid(path: str) -> Optional[np.ndarray]:
    """
    Load CDC SVI vulnerability grid.

    Returns (N, 5) float32 array: [RPL_THEME1, RPL_THEME2, RPL_THEME3, RPL_THEME4, RPL_THEMES]
    or None if the file is unavailable.
    """
    if not os.path.exists(path):
        log.warning("Vulnerability grid not found: %s", path)
        return None
    df = pd.read_csv(path)
    cols = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4", "RPL_THEMES"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None
    arr = df[cols].fillna(0.0).to_numpy(dtype=np.float32)
    log.info("Loaded vulnerability grid: %d nodes × %d features.", *arr.shape)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# StormRecord dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StormRecord:
    """All processed data for one storm, ready for graph/model consumption."""
    storm_id: str
    storm_name: str
    year: int
    basin: str
    source: str                        # "hurdat2" | "ibtracs" | "synthetic"

    track_df: pd.DataFrame             # raw track with datetime, lat, lon, vmax, mslp
    features: np.ndarray               # (T, n_storm_features)  float32
    basin_ids: np.ndarray              # (T,)  int32
    status_ids: np.ndarray             # (T,)  int32

    era5_patches: np.ndarray           # (T, 5, G, G) float32  (zeros when unavailable)
    era5_valid: np.ndarray             # (T,) bool

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def T(self) -> int:
        return len(self.track_df)

    @property
    def lat(self) -> np.ndarray:
        return self.track_df["lat"].to_numpy(dtype=np.float32)

    @property
    def lon(self) -> np.ndarray:
        return self.track_df["lon"].to_numpy(dtype=np.float32)

    @property
    def vmax(self) -> np.ndarray:
        return self.track_df["vmax_kt"].to_numpy(dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-source pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MultiSourceDataPipeline:
    """
    Orchestrates loading, harmonisation, temporal & spatial alignment, and
    feature engineering across all data sources.

    Usage
    -----
    pipeline = MultiSourceDataPipeline(cfg)
    records  = pipeline.build()   # List[StormRecord]
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> List[StormRecord]:
        """
        Full pipeline: parse → harmonise → feature-engineer → ERA5 cache.
        Returns a list of StormRecord objects.
        """
        cfg = self.cfg
        log.info("=" * 60)
        log.info("STORM-CARE Foundation — Data Pipeline")
        log.info("=" * 60)

        # 1. Parse HURDAT2
        log.info("Step 1/5: Loading HURDAT2 …")
        h2_storms = parse_hurdat2_full(
            cfg.hurdat2_path, min_year=cfg.min_year, min_track_len=4
        )

        # 2. Parse IBTrACS (or synthetic supplement)
        log.info("Step 2/5: Loading IBTrACS …")
        ibt_storms = parse_ibtracs(cfg.ibtracs_path, min_year=cfg.min_year)

        # 3. Merge + deduplicate
        log.info("Step 3/5: Merging and deduplicating …")
        all_track_dfs = self._merge_and_deduplicate(h2_storms, ibt_storms)

        # Respect max_storms cap for demo mode
        if cfg.max_storms and len(all_track_dfs) > cfg.max_storms:
            all_track_dfs = all_track_dfs[: cfg.max_storms]
            log.info("Capped to %d storms (max_storms).", cfg.max_storms)

        log.info("Total storms after merge: %d", len(all_track_dfs))

        # 4. ERA5 patch cache
        log.info("Step 4/5: Building ERA5 patch cache …")
        patch_cache = build_era5_cache(
            cfg.era5_paths,
            cfg.era5_storm_tags,
            all_track_dfs,
            cfg,
        )

        # 5. Build StormRecord objects
        log.info("Step 5/5: Computing storm features …")
        records: List[StormRecord] = []
        for df in all_track_dfs:
            rec = self._make_record(df, patch_cache)
            if rec is not None:
                records.append(rec)

        log.info("Pipeline complete: %d StormRecord objects.", len(records))
        self._print_dataset_summary(records)
        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    def _merge_and_deduplicate(
        self,
        h2_storms: List[pd.DataFrame],
        ibt_storms: List[pd.DataFrame],
    ) -> List[pd.DataFrame]:
        """
        Tag sources and remove obvious duplicates (same storm_id appearing
        in both datasets — IBTrACS often references HURDAT2 IDs directly).
        """
        seen_ids = set()
        merged: List[pd.DataFrame] = []

        for df in h2_storms:
            sid = df["storm_id"].iloc[0]
            if sid not in seen_ids:
                seen_ids.add(sid)
                df = df.copy()
                df["source"] = "hurdat2"
                merged.append(df)

        for df in ibt_storms:
            sid = df["storm_id"].iloc[0]
            if sid not in seen_ids:
                seen_ids.add(sid)
                df = df.copy()
                df["source"] = df.get("source", pd.Series(["ibtracs"] * len(df))).fillna("ibtracs")
                if "source" not in df.columns:
                    df["source"] = "ibtracs"
                merged.append(df)

        return merged

    def _make_record(
        self,
        df: pd.DataFrame,
        patch_cache: Dict[str, np.ndarray],
    ) -> Optional["StormRecord"]:
        """Convert one track DataFrame into a StormRecord."""
        cfg = self.cfg
        try:
            feats, basin_ids, status_ids = compute_storm_features(df)
            T = len(df)
            G = cfg.grid_size
            C = cfg.era5_in_channels

            era5_patches = np.zeros((T, C, G, G), dtype=np.float32)
            era5_valid = np.zeros(T, dtype=bool)

            storm_id = df["storm_id"].iloc[0]
            for t, row in df.iterrows():
                ts = row["datetime_utc"]
                key = f"{storm_id}_{ts.isoformat()}"
                if key in patch_cache:
                    era5_patches[t] = patch_cache[key]
                    era5_valid[t] = True

            return StormRecord(
                storm_id=storm_id,
                storm_name=df["storm_name"].iloc[0],
                year=int(df["year"].iloc[0]),
                basin=df["basin"].iloc[0],
                source=df.get("source", pd.Series(["unknown"])).iloc[0],
                track_df=df.reset_index(drop=True),
                features=feats,
                basin_ids=basin_ids,
                status_ids=status_ids,
                era5_patches=era5_patches,
                era5_valid=era5_valid,
            )
        except Exception as exc:
            log.debug("Skipped storm %s: %s", df["storm_id"].iloc[0], exc)
            return None

    @staticmethod
    def _print_dataset_summary(records: List["StormRecord"]) -> None:
        if not records:
            print("No records built.")
            return
        total_obs = sum(r.T for r in records)
        era5_obs  = sum(r.era5_valid.sum() for r in records)
        basins = pd.Series([r.basin for r in records]).value_counts().to_dict()
        years  = [r.year for r in records]
        print("\n" + "=" * 62)
        print("  STORM-CARE FOUNDATION  —  Dataset Summary")
        print("=" * 62)
        print(f"  Storms              : {len(records):,}")
        print(f"  Total observations  : {total_obs:,}")
        print(f"  ERA5-enhanced obs   : {era5_obs:,}  ({100*era5_obs/max(total_obs,1):.1f}%)")
        print(f"  Year range          : {min(years)}–{max(years)}")
        print(f"  Basin breakdown     : {basins}")
        print("=" * 62 + "\n")
