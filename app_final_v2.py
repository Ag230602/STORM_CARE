# app.py
# -----------------------------------------------------------------------------
# UNICEF-style Interactive Decision Dashboard (Streamlit)
#
# Views
#   • Hurricane path + uncertainty cone (P50/P90)
#   • Flood + evacuation overlay (optional)
#   • Children & hospital risk heatmaps (optional)
#   • Ranked recommended actions (confidence + KG-grounded reasons)
#
# Run:
#   pip install streamlit pandas numpy pydeck folium streamlit-folium shapely networkx
#   streamlit run app.py
# -----------------------------------------------------------------------------

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import folium
from streamlit_folium import st_folium
from shapely.geometry import LineString, Point, shape
import networkx as nx


# -----------------------------
# Config
# -----------------------------
# =========================
# DROP-IN FIX BLOCK (PASTE INTO app_final_v2.py)
# Fixes: AttributeError: 'float' object has no attribute 'fillna'
# =========================

import pandas as pd
import numpy as np

def safe_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """
    Always returns a numeric Series of length len(df).
    - If col exists: numeric + fillna(default)
    - If col missing: returns default for every row (prevents df.get(...)=float issues)
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype="float64")
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def aggregate_to_grid(
    grid_df: pd.DataFrame,
    shelters_df: pd.DataFrame,
    hospitals_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    to_balltree_fn,
    R_km: float = 20.0,
    SHELTER_CAP_COL: str = "capacity",
    HOSPITAL_BEDS_COL: str = "beds",
    SCHOOL_CAP_COL: str = "capacity",
):
    """
    Drop-in replacement for your aggregate_to_grid() that avoids the .get(...).fillna crash.
    Requires:
      - grid_df has columns: lat, lon
      - shelters_df/hospitals_df/schools_df have lat/lon (any extra columns OK)
      - to_balltree_fn(df) -> BallTree built from df[['lat','lon']] in radians haversine
    """

    g = grid_df.copy()

    # initialize outputs
    g["shelter_count_nearby"] = 0
    g["shelter_capacity_nearby"] = 0.0
    g["hospital_count_nearby"] = 0
    g["hospital_beds_nearby"] = 0.0
    g["school_count_nearby"] = 0
    g["school_capacity_nearby"] = 0.0

    # grid coords
    g_coords = np.deg2rad(g[["lat", "lon"]].to_numpy())
    R = float(R_km) / 6371.0  # km -> radians (Earth radius)

    # --- Shelters ---
    if shelters_df is not None and len(shelters_df) > 0:
        s = shelters_df.copy()
        # ✅ FIX HERE
        s[SHELTER_CAP_COL] = safe_numeric_series(s, SHELTER_CAP_COL, default=0.0)

        tree = to_balltree_fn(s)
        ind = tree.query_radius(g_coords, r=R)

        for i, nbrs in enumerate(ind):
            nbrs = list(nbrs)
            g.at[i, "shelter_count_nearby"] = int(len(nbrs))
            if len(nbrs) > 0:
                g.at[i, "shelter_capacity_nearby"] = float(s.iloc[nbrs][SHELTER_CAP_COL].sum())

    # --- Hospitals ---
    if hospitals_df is not None and len(hospitals_df) > 0:
        h = hospitals_df.copy()
        # ✅ FIX HERE
        h[HOSPITAL_BEDS_COL] = safe_numeric_series(h, HOSPITAL_BEDS_COL, default=0.0)

        tree = to_balltree_fn(h)
        ind = tree.query_radius(g_coords, r=R)

        for i, nbrs in enumerate(ind):
            nbrs = list(nbrs)
            g.at[i, "hospital_count_nearby"] = int(len(nbrs))
            if len(nbrs) > 0:
                g.at[i, "hospital_beds_nearby"] = float(h.iloc[nbrs][HOSPITAL_BEDS_COL].sum())

    # --- Schools ---
    if schools_df is not None and len(schools_df) > 0:
        c = schools_df.copy()

        # choose capacity-like column safely
        cap_col = None
        if SCHOOL_CAP_COL in c.columns:
            cap_col = SCHOOL_CAP_COL
        elif "students" in c.columns:
            cap_col = "students"

        if cap_col is None:
            c["__cap__"] = 0.0
        else:
            # ✅ FIX HERE
            c["__cap__"] = safe_numeric_series(c, cap_col, default=0.0)

        tree = to_balltree_fn(c)
        ind = tree.query_radius(g_coords, r=R)

        for i, nbrs in enumerate(ind):
            nbrs = list(nbrs)
            g.at[i, "school_count_nearby"] = int(len(nbrs))
            if len(nbrs) > 0:
                g.at[i, "school_capacity_nearby"] = float(c.iloc[nbrs]["__cap__"].sum())

    return g

# ================================
# COMMON HELPERS (PASTE ONCE)
# ================================
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------- SAFE NUMERIC ----------
def safe_numeric_series(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)

# ---------- LAT/LON NORMALIZATION ----------
def normalize_latlon(df: pd.DataFrame, kind: str = "data") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    cols = {c.lower(): c for c in out.columns}

    def col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    lat_col = col("lat", "latitude")
    lon_col = col("lon", "lng", "longitude")
    x_col = col("x")
    y_col = col("y")
    ll_col = col("longlat")

    if lat_col and lon_col:
        out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
        out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    elif x_col and y_col:
        out["lat"] = pd.to_numeric(out[y_col], errors="coerce")
        out["lon"] = pd.to_numeric(out[x_col], errors="coerce")
    elif ll_col:
        s = out[ll_col].astype(str).str.replace(r"[()]", "", regex=True)
        parts = s.str.split(",", expand=True)
        if parts.shape[1] == 2:
            out["lon"] = pd.to_numeric(parts[0], errors="coerce")
            out["lat"] = pd.to_numeric(parts[1], errors="coerce")
        else:
            out["lat"] = np.nan
            out["lon"] = np.nan
    else:
        out["lat"] = np.nan
        out["lon"] = np.nan

    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out

# ---------- NODE ID ----------
def ensure_node_id(df):
    if "node_id" not in df.columns:
        if "FIPS" in df.columns:
            df["node_id"] = df["FIPS"].astype(str)
        elif "cell_id" in df.columns:
            df["node_id"] = df["cell_id"].astype(str)
        else:
            df["node_id"] = [f"node_{i}" for i in range(len(df))]
    return df

# ---------- JOIN PRED TO NODES ----------
def join_pred_to_nodes(nodes, pred, t):
    p = pred[pred["t"] == t]
    out = nodes.merge(p, on="node_id", how="left")
    return out

# ---------- LOAD RECOVERY ----------
def load_recovery_pred(file):
    df = pd.read_csv(file)
    df["node_id"] = df["node_id"].astype(str)
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["pred_recovery"] = pd.to_numeric(df["pred_recovery"], errors="coerce")
    return df.dropna(subset=["t", "node_id"])


# ================================
# SIDEBAR LOADERS (INSIDE main app)
# ================================
st.sidebar.header("📂 Data Uploads")

data = {}

shelters_file = st.sidebar.file_uploader("Shelters CSV", type=["csv"])
hospitals_file = st.sidebar.file_uploader("Hospitals CSV", type=["csv"])
schools_file = st.sidebar.file_uploader("Schools CSV", type=["csv"])
nodes_file = st.sidebar.file_uploader("Nodes (nodes.csv)", type=["csv"])
weather_file = st.sidebar.file_uploader("Weather Ensemble", type=["csv"])
predrec_file = st.sidebar.file_uploader("Pred Recovery", type=["csv"])

if shelters_file:
    data["shelters"] = normalize_latlon(pd.read_csv(shelters_file), "shelter")

if hospitals_file:
    data["hospitals"] = normalize_latlon(pd.read_csv(hospitals_file), "hospital")

if schools_file:
    data["schools"] = normalize_latlon(pd.read_csv(schools_file), "school")

if nodes_file:
    data["nodes"] = ensure_node_id(
        normalize_latlon(pd.read_csv(nodes_file), "nodes")
    )

if weather_file:
    data["weather_ens"] = pd.read_csv(weather_file)

if predrec_file:
    data["pred_recovery"] = load_recovery_pred(predrec_file)


# ================================
# TABS (ADD RECOVERY TAB)
# ================================
tabs = st.tabs([
    "🌀 Path/Cone",
    "🌊 Flood/Evac",
    "🔥 Risk",
    "🧭 Actions",
    "🛠 Recovery"
])

# ================================
# RECOVERY TAB
# ================================
with tabs[4]:
    st.subheader("🛠 Recovery (ST-GNN) – Predicted Recovery Index")

    nodes = data.get("nodes")
    st.write("nodes loaded:", nodes is not None)
if nodes is not None:
    st.write("nodes columns:", list(nodes.columns)[:20])

    predrec = data.get("pred_recovery")

    if nodes is None or predrec is None:
        st.info("Upload nodes.csv and pred_recovery.csv in the sidebar.")
    else:
        tmin = int(predrec["t"].min())
        tmax = int(predrec["t"].max())
        t = st.slider("Time step (t)", tmin, tmax, tmin, 1)

        joined = join_pred_to_nodes(nodes, predrec, t)
        joined = joined.dropna(subset=["lat", "lon"])

        st.write("Rows plotted:", len(joined))

        center = [joined["lat"].mean(), joined["lon"].mean()]
        m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

        def color(v):
            if pd.isna(v): return "#999999"
            if v < 0.25: return "#d73027"
            if v < 0.50: return "#fc8d59"
            if v < 0.75: return "#fee08b"
            return "#1a9850"

        for _, r in joined.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=3,
                color=color(r["pred_recovery"]),
                fill=True,
                fill_opacity=0.85,
                popup=f"node_id={r['node_id']}<br>recovery={r['pred_recovery']:.3f}"
            ).add_to(m)

        st_folium(m, width=1000, height=650)


@dataclass
class CFG:
    default_center: Tuple[float, float] = (27.5, -82.5)  # Florida-ish
    default_zoom: int = 5
    refresh_seconds: int = 60

    # Scales consistent with chi-square sqrt for 2D confidence regions (approx)
    p50_scale: float = 1.177
    p90_scale: float = 2.146


cfg = CFG()


# -----------------------------
# IO helpers
# -----------------------------
def _try_read_csv(file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file)
    except Exception:
        return None


def _try_read_geojson(file) -> Optional[dict]:
    try:
        if hasattr(file, "read"):
            return json.load(file)
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_local_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_local_geojson(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_data_sources() -> Dict[str, Any]:
    """
    Supports either:
      - Upload via sidebar
      - Or reading local ./data/* if present
    """
    sources: Dict[str, Any] = {}

    st.sidebar.markdown("## Data Inputs (optional)")

    track_file = st.sidebar.file_uploader(
        "Track predictions CSV",
        type=["csv"],
        key="track_csv",
        help="Expected columns include: storm_tag, t0, lead_hours, pred_lat, pred_lon, sigma_lat, sigma_lon, model "
             "(and optionally gt_lat, gt_lon).",
    )
    flood_file = st.sidebar.file_uploader("Flood polygons GeoJSON", type=["geojson", "json"], key="flood_geojson")
    roads_file = st.sidebar.file_uploader("Roads GeoJSON", type=["geojson", "json"], key="roads_geojson")
    shelters_file = st.sidebar.file_uploader("Shelters CSV", type=["csv"], key="shelters_csv")
    hospitals_file = st.sidebar.file_uploader("Hospitals CSV", type=["csv"], key="hospitals_csv")
    schools_file = st.sidebar.file_uploader("Schools CSV", type=["csv"], key="schools_csv")
    vuln_file = st.sidebar.file_uploader("Vulnerability grid CSV", type=["csv"], key="vuln_csv")

    sources["track"] = _try_read_csv(track_file) if track_file else _load_local_csv("data/track_predictions.csv")
    sources["flood_geojson"] = _try_read_geojson(flood_file) if flood_file else _load_local_geojson("data/flood_polygons.geojson")
    sources["roads_geojson"] = _try_read_geojson(roads_file) if roads_file else _load_local_geojson("data/roads.geojson")
    sources["shelters"] = _try_read_csv(shelters_file) if shelters_file else _load_local_csv("data/shelters.csv")
    sources["hospitals"] = _try_read_csv(hospitals_file) if hospitals_file else _load_local_csv("data/hospitals.csv")
    sources["schools"] = _try_read_csv(schools_file) if schools_file else _load_local_csv("data/schools.csv")
    sources["vuln"] = _try_read_csv(vuln_file) if vuln_file else _load_local_csv("data/vulnerability_grid.csv")

    return sources


# -----------------------------
# Track normalization
# -----------------------------
def normalize_track_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Normalizes various track CSV formats into a consistent schema.

    Output columns (at least):
      - storm_tag (optional)
      - t0 (optional)
      - model (optional)
      - lead_hours
      - mu_lat, mu_lon
      - sigma_lat, sigma_lon (optional; NaN => cone disabled)
      - gt_lat, gt_lon (optional)

    Supports common naming variants including:
      - pred_lat/pred_lon (your CSV)
      - lat/lon
      - mu_lat/mu_lon
      - lead_h / lead / lead_time
    """
    if df is None or len(df) == 0:
        return None

    d = df.copy()
    cols = {c.lower(): c for c in d.columns}

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    # Required-ish
    lead_col = pick(["lead_hours", "lead_h", "lead", "lead_time", "horizon_hours"])
    if lead_col is None:
        d["lead_hours"] = np.arange(len(d)) * 6
    else:
        if lead_col != "lead_hours":
            d = d.rename(columns={lead_col: "lead_hours"})

    # Mean / predicted location
    mu_lat_col = pick(["mu_lat", "pred_lat", "lat_pred", "lat", "latitude"])
    mu_lon_col = pick(["mu_lon", "pred_lon", "lon_pred", "lon", "longitude"])
    if mu_lat_col is None or mu_lon_col is None:
        return None
    if mu_lat_col != "mu_lat":
        d = d.rename(columns={mu_lat_col: "mu_lat"})
    if mu_lon_col != "mu_lon":
        d = d.rename(columns={mu_lon_col: "mu_lon"})

    # Optional sigma
    sig_lat_col = pick(["sigma_lat", "std_lat", "lat_sigma"])
    sig_lon_col = pick(["sigma_lon", "std_lon", "lon_sigma"])
    if sig_lat_col is None:
        d["sigma_lat"] = np.nan
    elif sig_lat_col != "sigma_lat":
        d = d.rename(columns={sig_lat_col: "sigma_lat"})
    if sig_lon_col is None:
        d["sigma_lon"] = np.nan
    elif sig_lon_col != "sigma_lon":
        d = d.rename(columns={sig_lon_col: "sigma_lon"})

    # Optional metadata
    storm_col = pick(["storm_tag", "storm", "storm_name"])
    if storm_col is not None and storm_col != "storm_tag":
        d = d.rename(columns={storm_col: "storm_tag"})
    if "storm_tag" not in d.columns:
        d["storm_tag"] = "UnknownStorm"

    model_col = pick(["model", "model_name"])
    if model_col is not None and model_col != "model":
        d = d.rename(columns={model_col: "model"})
    if "model" not in d.columns:
        d["model"] = "UnknownModel"

    t0_col = pick(["t0", "init_time", "datetime_utc", "start_time"])
    if t0_col is not None and t0_col != "t0":
        d = d.rename(columns={t0_col: "t0"})

    gt_lat_col = pick(["gt_lat", "true_lat", "obs_lat"])
    gt_lon_col = pick(["gt_lon", "true_lon", "obs_lon"])
    if gt_lat_col is not None and gt_lat_col != "gt_lat":
        d = d.rename(columns={gt_lat_col: "gt_lat"})
    if gt_lon_col is not None and gt_lon_col != "gt_lon":
        d = d.rename(columns={gt_lon_col: "gt_lon"})

    # Numeric coercion
    for c in ["lead_hours", "mu_lat", "mu_lon", "sigma_lat", "sigma_lon"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "gt_lat" in d.columns:
        d["gt_lat"] = pd.to_numeric(d["gt_lat"], errors="coerce")
    if "gt_lon" in d.columns:
        d["gt_lon"] = pd.to_numeric(d["gt_lon"], errors="coerce")

    d = d.dropna(subset=["mu_lat", "mu_lon"]).copy()
    d["lead_hours"] = d["lead_hours"].fillna(0)

    return d


# -----------------------------
# Geospatial helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def geojson_polygons(geo: Optional[dict]) -> List[Any]:
    if not geo:
        return []
    polys = []
    for feat in geo.get("features", []):
        try:
            geom = shape(feat["geometry"])
            if geom.is_valid:
                polys.append((geom, feat.get("properties", {})))
        except Exception:
            continue
    return polys


def point_in_flood(lat: float, lon: float, flood_polys: List[Any]) -> bool:
    if not flood_polys:
        return False
    p = Point(float(lon), float(lat))
    for geom, _props in flood_polys:
        if geom.contains(p):
            return True
    return False


# -----------------------------
# Lightweight KG + action scoring
# -----------------------------
def build_light_kg(
    shelters: Optional[pd.DataFrame],
    hospitals: Optional[pd.DataFrame],
    schools: Optional[pd.DataFrame],
) -> nx.DiGraph:
    G = nx.DiGraph()

    def add_nodes(df: pd.DataFrame, kind: str, name_col: str = "name"):
        for i, r in df.iterrows():
            nid = f"{kind}:{i}"
            try:
                lat, lon = float(r["lat"]), float(r["lon"])
            except Exception:
                continue
            G.add_node(
                nid,
                kind=kind,
                name=str(r.get(name_col, nid)),
                lat=lat,
                lon=lon,
                **{k: r.get(k) for k in df.columns if k not in ["lat", "lon"]},
            )

    if shelters is not None and {"lat", "lon"}.issubset(set(shelters.columns)):
        add_nodes(shelters, "shelter")
    if hospitals is not None and {"lat", "lon"}.issubset(set(hospitals.columns)):
        add_nodes(hospitals, "hospital")
    if schools is not None and {"lat", "lon"}.issubset(set(schools.columns)):
        add_nodes(schools, "school")

    # proximity edges (for explanations)
    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        ni, ai = nodes[i]
        for j in range(i + 1, len(nodes)):
            nj, aj = nodes[j]
            d_km = haversine_km(ai["lat"], ai["lon"], aj["lat"], aj["lon"])
            if d_km <= 25:
                G.add_edge(ni, nj, rel="near", km=d_km)
                G.add_edge(nj, ni, rel="near", km=d_km)

    return G


def _missingness_score(track, flood_geo, shelters, hospitals, schools, vuln) -> float:
    parts = [
        track is not None,
        flood_geo is not None,
        shelters is not None,
        hospitals is not None,
        schools is not None,
        vuln is not None,
    ]
    return 1.0 - (sum(parts) / len(parts))


def score_actions(
    track: Optional[pd.DataFrame],
    flood_geo: Optional[dict],
    shelters: Optional[pd.DataFrame],
    hospitals: Optional[pd.DataFrame],
    schools: Optional[pd.DataFrame],
    vuln: Optional[pd.DataFrame],
) -> pd.DataFrame:
    flood_polys = geojson_polygons(flood_geo) if flood_geo else []
    kg = build_light_kg(shelters, hospitals, schools)

    # Hazard proxy from track: closer to Florida center & higher uncertainty => higher risk
    hazard = 0.5
    if track is not None and len(track) > 0:
        # Use ~24h lead if available, else last row
        lead = 24
        pick = track.iloc[(track["lead_hours"] - lead).abs().argsort()[:1]]
        mu_lat = float(pick["mu_lat"].iloc[0])
        mu_lon = float(pick["mu_lon"].iloc[0])
        sig = float(np.nanmean([pick["sigma_lat"].iloc[0], pick["sigma_lon"].iloc[0]]))
        d_to_center = haversine_km(mu_lat, mu_lon, cfg.default_center[0], cfg.default_center[1])
        hazard = float(np.clip(1.0 - (d_to_center / 1200.0), 0.0, 1.0))
        hazard = float(np.clip(hazard + np.clip(sig / 2.0, 0, 0.3), 0, 1))

    coverage = 0.0
    if vuln is not None and {"lat", "lon"}.issubset(set(vuln.columns)):
        coverage = float(np.clip(len(vuln) / 5000.0, 0, 1))

    actions: List[Dict[str, Any]] = []

    actions.append({
        "action": "Monitor and update evacuation routes (bottlenecks/passability)",
        "confidence": 0.55 + 0.35 * hazard,
        "reason": "Track-based hazard increased; prioritize route monitoring. (KG: roads ↔ shelters ↔ vulnerable zones)",
        "category": "Evacuation"
    })

    facilities = []
    if hospitals is not None and {"lat", "lon"}.issubset(set(hospitals.columns)):
        facilities.append("hospitals")
    if schools is not None and {"lat", "lon"}.issubset(set(schools.columns)):
        facilities.append("schools")

    conf_fac = 0.45 + 0.45 * hazard + 0.1 * (1 if facilities else 0)
    reason_fac = "Facilities available; prioritize children + critical care."
    any_edge = next(iter(kg.edges(data=True)), None)
    if any_edge:
        u, v, a = any_edge
        reason_fac += f" (KG: {kg.nodes[u]['kind']} near {kg.nodes[v]['kind']} ~{a['km']:.1f}km)"
    actions.append({
        "action": "Pre-position supplies for children and hospitals (WASH, meds, power backup)",
        "confidence": float(np.clip(conf_fac, 0, 0.98)),
        "reason": reason_fac,
        "category": "Resource Allocation"
    })

    if shelters is not None and {"capacity", "current_load"}.issubset(set(shelters.columns)):
        load = pd.to_numeric(shelters["current_load"], errors="coerce")
        cap = pd.to_numeric(shelters["capacity"], errors="coerce").replace(0, np.nan)
        util = float(np.nanmean(load / cap))
        conf_sh = 0.40 + 0.45 * hazard + 0.15 * np.clip(util, 0, 1)
        actions.append({
            "action": "Check shelter capacity and prepare overflow re-routing",
            "confidence": float(np.clip(conf_sh, 0, 0.98)),
            "reason": f"Shelter utilization ~{util:.2f}. (KG: shelter capacity constraint → routing)",
            "category": "Shelters"
        })
    else:
        actions.append({
            "action": "Validate shelter inventory (locations, capacity, accessibility)",
            "confidence": float(np.clip(0.50 + 0.30 * hazard, 0, 0.95)),
            "reason": "Shelter metadata incomplete; fill gaps to enable routing decisions. (KG: missingness reduces decision certainty)",
            "category": "Shelters"
        })

    if flood_polys:
        actions.append({
            "action": "Issue flood risk advisory for low-lying neighborhoods; prioritize high-SVI zones",
            "confidence": float(np.clip(0.55 + 0.35 * hazard, 0, 0.98)),
            "reason": "Flood layer present; combine with vulnerability to prioritize. (KG: floodplain ↔ population ↔ access)",
            "category": "Flood"
        })
    else:
        actions.append({
            "action": "Fetch/compute flood extent forecast (SAR/terrain) for overlay and routing",
            "confidence": float(np.clip(0.45 + 0.25 * hazard, 0, 0.90)),
            "reason": "Flood layer missing → uncertainty in evacuation passability. (KG: flood extent constrains roads)",
            "category": "Flood"
        })

    df = pd.DataFrame(actions).sort_values("confidence", ascending=False).reset_index(drop=True)
    df["confidence"] = df["confidence"].round(3)
    df.attrs["kpis"] = {
        "hazard_proxy": hazard,
        "coverage_proxy": coverage,
        "actionability": float(len(df)),
        "missingness": float(_missingness_score(track, flood_geo, shelters, hospitals, schools, vuln)),
    }
    return df


# -----------------------------
# Map rendering
# -----------------------------
def make_track_layers(track: pd.DataFrame, show_cone: bool = True, show_gt: bool = True) -> List[pdk.Layer]:
    # Mean/predicted track line
    pts = track.sort_values("lead_hours")[["mu_lat", "mu_lon", "lead_hours", "sigma_lat", "sigma_lon"]].copy()
    pts = pts.rename(columns={"mu_lat": "lat", "mu_lon": "lon"})

    layers: List[pdk.Layer] = []

    layers.append(
        pdk.Layer(
            "PathLayer",
            data=[{"path": pts[["lon", "lat"]].values.tolist(), "name": "Predicted Track"}],
            get_path="path",
            get_width=5,
            pickable=True,
        )
    )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position=["lon", "lat"],
            get_radius=6000,
            pickable=True,
        )
    )

    # Optional uncertainty cone (visual proxy: circles)
    if show_cone and np.isfinite(pts["sigma_lat"]).any() and np.isfinite(pts["sigma_lon"]).any():
        cone_pts = []
        for _, r in pts.iterrows():
            sig = float(np.nanmean([r["sigma_lat"], r["sigma_lon"]]))
            if not np.isfinite(sig):
                continue
            p50_m = cfg.p50_scale * sig * 111_000
            p90_m = cfg.p90_scale * sig * 111_000
            cone_pts.append({"lat": r["lat"], "lon": r["lon"], "r50": p50_m, "r90": p90_m, "lead": r["lead_hours"]})

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=cone_pts,
                get_position=["lon", "lat"],
                get_radius="r90",
                opacity=0.06,
                pickable=False,
            )
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=cone_pts,
                get_position=["lon", "lat"],
                get_radius="r50",
                opacity=0.10,
                pickable=False,
            )
        )

    # Optional ground-truth overlay
    if show_gt and {"gt_lat", "gt_lon"}.issubset(track.columns):
        gt = track.sort_values("lead_hours")[["gt_lat", "gt_lon", "lead_hours"]].dropna().copy()
        if len(gt) >= 2:
            gt = gt.rename(columns={"gt_lat": "lat", "gt_lon": "lon"})
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": gt[["lon", "lat"]].values.tolist(), "name": "Ground Truth"}],
                    get_path="path",
                    get_width=4,
                    pickable=True,
                )
            )
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=gt,
                    get_position=["lon", "lat"],
                    get_radius=4500,
                    pickable=True,
                )
            )

    return layers


def pydeck_map(layers: List[pdk.Layer], center: Tuple[float, float], zoom: int):
    view = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom, pitch=0)
    tooltip = {"text": "Lead: {lead_hours}h"} if layers else None
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip), width='stretch')


def folium_overlay_map(
    center: Tuple[float, float],
    flood_geo: Optional[dict],
    roads_geo: Optional[dict],
    shelters: Optional[pd.DataFrame],
):
    m = folium.Map(location=[center[0], center[1]], zoom_start=cfg.default_zoom, tiles="cartodbpositron")

    if flood_geo:
        folium.GeoJson(
            flood_geo,
            name="Flood Extent",
            style_function=lambda x: {"fillOpacity": 0.25, "weight": 1},
        ).add_to(m)

    if roads_geo:
        folium.GeoJson(
            roads_geo,
            name="Roads",
            style_function=lambda x: {"weight": 2},
        ).add_to(m)

    if shelters is not None and {"lat", "lon"}.issubset(set(shelters.columns)):
        for _, r in shelters.iterrows():
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=5,
                popup=str(r.get("name", "shelter")),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, height=520, width=None)


# -----------------------------
# Risk layers
# -----------------------------
def compute_risk_points(
    hospitals: Optional[pd.DataFrame],
    schools: Optional[pd.DataFrame],
    flood_geo: Optional[dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flood_polys = geojson_polygons(flood_geo) if flood_geo else []

    hosp_risk = pd.DataFrame(columns=["lat", "lon", "risk", "name"])
    school_risk = pd.DataFrame(columns=["lat", "lon", "risk", "name"])

    if hospitals is not None and {"lat", "lon"}.issubset(set(hospitals.columns)):
        tmp = hospitals.copy()
        tmp["flood"] = tmp.apply(lambda r: point_in_flood(r["lat"], r["lon"], flood_polys), axis=1)
        beds = pd.to_numeric(tmp.get("beds", 100), errors="coerce").fillna(100)
        tmp["risk"] = (tmp["flood"].astype(float) * 0.6 + (1.0 / np.sqrt(beds)) * 0.4).clip(0, 1)
        tmp["name"] = tmp.get("name", "hospital")
        hosp_risk = tmp[["lat", "lon", "risk", "name"]]

    if schools is not None and {"lat", "lon"}.issubset(set(schools.columns)):
        tmp = schools.copy()
        tmp["flood"] = tmp.apply(lambda r: point_in_flood(r["lat"], r["lon"], flood_polys), axis=1)
        kids = pd.to_numeric(tmp.get("children_est", 300), errors="coerce").fillna(300)
        tmp["risk"] = (tmp["flood"].astype(float) * 0.55 + np.clip(kids / 1500.0, 0, 1) * 0.45).clip(0, 1)
        tmp["name"] = tmp.get("name", "school")
        school_risk = tmp[["lat", "lon", "risk", "name"]]

    return hosp_risk, school_risk


def heatmap_layer(df: pd.DataFrame, value_col: str = "risk") -> Optional[pdk.Layer]:
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0.0)
    return pdk.Layer(
        "HeatmapLayer",
        data=d,
        get_position=["lon", "lat"],
        get_weight=value_col,
        radiusPixels=70,
        aggregation="MEAN",
    )


# -----------------------------
# KPI panel
# -----------------------------
def kpi_panel(kpis: Dict[str, float], refresh_seconds: int):
    c1, c2, c3, c4, c5 = st.columns(5)

    latency_min = refresh_seconds / 60.0
    c1.metric("Update Latency", f"{latency_min:.1f} min", delta="target < 5 min" if latency_min <= 5 else "over target")

    coverage = kpis.get("coverage_proxy", 0.0)
    c2.metric("Coverage", f"{coverage*100:.0f}%")

    robustness = 1.0 - kpis.get("missingness", 0.0)
    c3.metric("Robustness", f"{robustness*100:.0f}%")

    actionability = kpis.get("actionability", 0.0)
    c4.metric("Actionability", f"{actionability:.0f} actions")

    hazard = kpis.get("hazard_proxy", 0.0)
    c5.metric("Hazard Proxy", f"{hazard*100:.0f}%")


# -----------------------------
# App
# -----------------------------
def main():
    st.set_page_config(page_title="UNICEF Hurricane Decision Dashboard", layout="wide")

    st.title("🌪️ UNICEF-Style Hurricane Decision Dashboard (Demo)")
    st.caption("Decision-native demo: track + cone, optional flood/evac overlays, child & hospital risk, and ranked actions with KG-grounded reasons.")

    sources = load_data_sources()

    # Sidebar controls
    st.sidebar.markdown("## Controls")
    if "storm_name_display" not in st.session_state:
        st.session_state["storm_name_display"] = "Hurricane IAN"
    storm_name_display = st.sidebar.text_input("Storm name (display)", value=st.session_state["storm_name_display"])
    center_lat = st.sidebar.number_input("Map center lat", value=float(cfg.default_center[0]), format="%.4f")
    center_lon = st.sidebar.number_input("Map center lon", value=float(cfg.default_center[1]), format="%.4f")
    zoom = st.sidebar.slider("Zoom", min_value=3, max_value=10, value=cfg.default_zoom)

    show_cone = st.sidebar.toggle("Show uncertainty cone (P50/P90)", value=True)
    show_gt = st.sidebar.toggle("Show ground-truth path (if available)", value=True)
    show_flood = st.sidebar.toggle("Show flood overlay (if provided)", value=True)
    show_roads = st.sidebar.toggle("Show roads overlay (if provided)", value=True)
    show_heat = st.sidebar.toggle("Show risk heatmaps (if facilities provided)", value=True)

    refresh_seconds = st.sidebar.slider("Refresh interval (sec)", 10, 300, cfg.refresh_seconds, step=10)

    # Normalize and filter track
    track_all = normalize_track_df(sources.get("track"))
    track = None

    # --- Limit storms to IAN/IRMA if present (keeps UI clean) ---
    if track_all is not None and "storm_tag" in track_all.columns:
        _storm_up = track_all["storm_tag"].astype(str).str.upper()
        _allowed = ["IAN", "IRMA"]
        if _storm_up.isin(_allowed).any():
            track_all = track_all[_storm_up.isin(_allowed)].copy()

    if track_all is not None and len(track_all) > 0:
        st.sidebar.markdown("### Track Filters")

        storms = sorted(track_all["storm_tag"].astype(str).str.upper().unique().tolist())
        default_storm = "IAN" if "IAN" in storms else storms[0]
        storm_sel = st.sidebar.selectbox("Storm", storms, index=storms.index(default_storm))

        # Keep display label in sync unless user edits it
        st.session_state["storm_name_display"] = f"Hurricane {storm_sel.title()}"
        storm_name_display = st.session_state["storm_name_display"]

        df_storm = track_all[track_all["storm_tag"].astype(str).str.upper() == storm_sel].copy()
        models = sorted(df_storm["model"].astype(str).unique().tolist())
        model_sel = st.sidebar.selectbox("Model", models, index=0)

        df2 = df_storm[df_storm["model"].astype(str) == model_sel].copy()

        if "t0" in df2.columns:
            t0s = sorted(df2["t0"].astype(str).unique().tolist())
            t0_sel = st.sidebar.selectbox("Init time (t0)", t0s, index=0)
            df2 = df2[df2["t0"].astype(str) == t0_sel].copy()
        else:
            t0_sel = None

        track = df2.sort_values("lead_hours").reset_index(drop=True)

        with st.expander("Loaded track CSV (preview)"):
            st.write(f"Rows: {len(track_all):,} • Columns: {len(track_all.columns)}")
            st.dataframe(track_all.head(30), width='stretch')
    else:
        st.info("Upload a track predictions CSV to enable the hurricane path + uncertainty cone.")

    flood_geo = sources.get("flood_geojson") if show_flood else None
    roads_geo = sources.get("roads_geojson") if show_roads else None
    shelters = sources.get("shelters")
    hospitals = sources.get("hospitals")
    schools = sources.get("schools")
    vuln = sources.get("vuln")

    # Actions + KPIs
    actions_df = score_actions(track, flood_geo, shelters, hospitals, schools, vuln)
    kpis = actions_df.attrs.get("kpis", {})
    kpi_panel(kpis, refresh_seconds)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌀 Path + Cone",
        "🌊 Flood + Evacuation",
        "🏥 Children & Hospitals Risk",
        "✅ Recommended Actions",
    ])

    with tab1:
        st.subheader(f"Track Forecast: {storm_name_display}")
        if track is None:
            st.warning("No track loaded.")
        else:
            layers = make_track_layers(track, show_cone=show_cone, show_gt=show_gt)
            # Center map on first predicted point if available
            c = (float(track["mu_lat"].iloc[0]), float(track["mu_lon"].iloc[0]))
            pydeck_map(layers, center=c, zoom=zoom)
            with st.expander("Track (filtered)"):
                st.dataframe(track, width='stretch')

    # with tab2:
    #     st.subheader("Flood + Evacuation Overlay (optional)")
    #     folium_overlay_map(
    #         center=(center_lat, center_lon),
    #         flood_geo=flood_geo,
    #         roads_geo=roads_geo,
    #         shelters=shelters,
    #     )
    # with tabs[2]:
    #  st.subheader("🔥 Risk (fallback if no hazard layers)")

    # nodes = data.get("nodes")
    # predrec = data.get("pred_recovery")

    # if nodes is None:
    #     st.info("Upload nodes.csv to view risk.")
    # else:
    #     nodes = nodes.copy()

    #     # --- Risk proxy ---
    #     # vulnerability in [0,1]
    #     if "RPL_THEMES" in nodes.columns:
    #         vuln = pd.to_numeric(nodes["RPL_THEMES"], errors="coerce").fillna(nodes["RPL_THEMES"].median())
    #     else:
    #         vuln = pd.Series(0.5, index=nodes.index)

    #     vuln = vuln.clip(0, 1)

    #     # facility scarcity proxy (lower capacity -> higher risk)
    #     shelter_cap = pd.to_numeric(nodes.get("shelter_capacity_nearby", 0.0), errors="coerce").fillna(0.0)
    #     hosp_beds   = pd.to_numeric(nodes.get("hospital_beds_nearby", 0.0), errors="coerce").fillna(0.0)

    #     # Normalize (avoid /0)
    #     shelter_norm = (shelter_cap - shelter_cap.min()) / (shelter_cap.max() - shelter_cap.min() + 1e-9)
    #     hosp_norm    = (hosp_beds - hosp_beds.min()) / (hosp_beds.max() - hosp_beds.min() + 1e-9)

    #     scarcity = 1.0 - 0.5*(shelter_norm + hosp_norm)  # higher = fewer resources
    #     scarcity = scarcity.clip(0, 1)

    #     # If recovery exists, use worst recovery at current t as extra risk
    #     if predrec is not None:
    #         tmin = int(predrec["t"].min())
    #         tmax = int(predrec["t"].max())
    #         t = st.slider("Time step (t) for risk overlay", tmin, tmax, tmin, 1)

    #         p = predrec[predrec["t"] == t][["node_id", "pred_recovery"]]
    #         nodes["node_id"] = nodes["node_id"].astype(str)
    #         p["node_id"] = p["node_id"].astype(str)
    #         nodes = nodes.merge(p, on="node_id", how="left")
    #         rec = pd.to_numeric(nodes["pred_recovery"], errors="coerce").fillna(nodes["pred_recovery"].median())
    #         rec = rec.clip(0, 1)
    #         rec_risk = (1.0 - rec)
    #     else:
    #         rec_risk = pd.Series(0.0, index=nodes.index)

    #     # Final risk score
    #     nodes["risk_score"] = (0.6*vuln + 0.3*scarcity + 0.1*rec_risk).clip(0, 1)

    #     nodes = nodes.dropna(subset=["lat", "lon"])
    #     center = [nodes["lat"].mean(), nodes["lon"].mean()]
    #     m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

    #     def color(v):
    #         if pd.isna(v): return "#999999"
    #         if v < 0.25: return "#1a9850"
    #         if v < 0.50: return "#fee08b"
    #         if v < 0.75: return "#fc8d59"
    #         return "#d73027"

    #     for _, r in nodes.iterrows():
    #         folium.CircleMarker(
    #             location=[float(r["lat"]), float(r["lon"])],
    #             radius=3,
    #             color=color(r["risk_score"]),
    #             fill=True,
    #             fill_opacity=0.85,
    #             popup=f"node_id={r['node_id']}<br>risk={r['risk_score']:.3f}"
    #         ).add_to(m)

    #     st_folium(m, width=1000, height=650)

    # # with tab3:
    # #     st.subheader("Children & Hospitals Risk Heatmaps (optional)")
    # #     hosp_risk, school_risk = compute_risk_points(hospitals, schools, flood_geo)
    # #     layers = []
    # #     if show_heat:
    # #         h1 = heatmap_layer(hosp_risk) if hosp_risk is not None else None
    # #         h2 = heatmap_layer(school_risk) if school_risk is not None else None
    # #         if h1 is not None:
    # #             layers.append(h1)
    # #         if h2 is not None:
    # #             layers.append(h2)

    # #     if len(layers) == 0:
    # #         st.info("Upload hospitals/schools CSV (with lat/lon) to enable risk heatmaps.")
    # #     else:
    # #         pydeck_map(layers, center=(center_lat, center_lon), zoom=zoom)

    # #     c1, c2 = st.columns(2)
    # #     with c1:
    # #         st.caption("Hospitals risk points (preview)")
    # #         st.dataframe(hosp_risk.head(20), width='stretch')
    # #     with c2:
    # #         st.caption("Schools risk points (preview)")
    # #         st.dataframe(school_risk.head(20), width='stretch')

    # # with tab4:
    # #     st.subheader("Ranked Recommended Actions")
    # #     st.dataframe(actions_df, width='stretch')
    # #     st.caption("These are demo heuristics; replace the scorer with your planner / temporal-KG constraints without changing the UI.")

    # # st.caption("Tip: Place optional files under ./data/ to auto-load without uploading (track_predictions.csv, flood_polygons.geojson, roads.geojson, etc.).")
with tabs[2]:
    st.subheader("🔥 Risk (fallback if no hazard layers)")

    import folium
    from streamlit_folium import st_folium

    nodes = data.get("nodes")
    predrec = data.get("pred_recovery")

    st.write("nodes loaded:", nodes is not None)

    if nodes is None:
        st.info("Upload nodes.csv to view risk.")
    else:
        nodes = nodes.copy()
        st.write("nodes columns:", list(nodes.columns)[:20])

        nodes["lat"] = pd.to_numeric(nodes["lat"], errors="coerce")
        nodes["lon"] = pd.to_numeric(nodes["lon"], errors="coerce")
        nodes = nodes.dropna(subset=["lat", "lon"])

        st.write("rows after lat/lon clean:", len(nodes))
        if len(nodes) == 0:
            st.error("No valid lat/lon rows found in nodes.csv.")
        else:
            # vulnerability in [0,1]
            if "RPL_THEMES" in nodes.columns:
                vuln = pd.to_numeric(nodes["RPL_THEMES"], errors="coerce").fillna(nodes["RPL_THEMES"].median())
            else:
                vuln = pd.Series(0.5, index=nodes.index)
            vuln = vuln.clip(0, 1)

            # facility scarcity proxy
            shelter_cap = pd.to_numeric(nodes.get("shelter_capacity_nearby", 0.0), errors="coerce").fillna(0.0)
            hosp_beds   = pd.to_numeric(nodes.get("hospital_beds_nearby", 0.0), errors="coerce").fillna(0.0)

            shelter_norm = (shelter_cap - shelter_cap.min()) / (shelter_cap.max() - shelter_cap.min() + 1e-9)
            hosp_norm    = (hosp_beds - hosp_beds.min()) / (hosp_beds.max() - hosp_beds.min() + 1e-9)
            scarcity = (1.0 - 0.5*(shelter_norm + hosp_norm)).clip(0, 1)

            # recovery overlay (optional)
            if predrec is not None and "t" in predrec.columns:
                predrec = predrec.copy()
                predrec["t"] = pd.to_numeric(predrec["t"], errors="coerce")
                predrec = predrec.dropna(subset=["t"])
                tmin = int(predrec["t"].min())
                tmax = int(predrec["t"].max())
                t = st.slider("Time step (t) for risk overlay", tmin, tmax, tmin, 1)

                p = predrec[predrec["t"] == t][["node_id", "pred_recovery"]].copy()
                nodes["node_id"] = nodes["node_id"].astype(str)
                p["node_id"] = p["node_id"].astype(str)
                nodes = nodes.merge(p, on="node_id", how="left")
                rec = pd.to_numeric(nodes["pred_recovery"], errors="coerce").fillna(nodes["pred_recovery"].median()).clip(0, 1)
                rec_risk = (1.0 - rec)
            else:
                rec_risk = pd.Series(0.0, index=nodes.index)

            nodes["risk_score"] = (0.6*vuln + 0.3*scarcity + 0.1*rec_risk).clip(0, 1)

            center = [float(nodes["lat"].mean()), float(nodes["lon"].mean())]
            m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

            def color(v):
                if pd.isna(v): return "#999999"
                if v < 0.25: return "#1a9850"
                if v < 0.50: return "#fee08b"
                if v < 0.75: return "#fc8d59"
                return "#d73027"

            for _, r in nodes.iterrows():
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=3,
                    color=color(float(r["risk_score"])),
                    fill=True,
                    fill_opacity=0.85,
                    popup=f"node_id={r.get('node_id','')}<br>risk={float(r['risk_score']):.3f}"
                ).add_to(m)

            st_folium(m, width=1000, height=650)
    
with tabs[3]:
    st.subheader("🧭 Recommended Actions (rule-based demo)")

    nodes = data.get("nodes")
    predrec = data.get("pred_recovery")

    if nodes is None:
        st.info("Upload nodes.csv to generate actions.")
    else:
        nodes = nodes.copy()

        # vulnerability
        vuln = pd.to_numeric(nodes.get("RPL_THEMES", 0.5), errors="coerce")
        if isinstance(vuln, pd.Series):
            vuln = vuln.fillna(vuln.median()).clip(0, 1)
        else:
            vuln = pd.Series(0.5, index=nodes.index)

        # facility scarcity
        shelter_cap = pd.to_numeric(nodes.get("shelter_capacity_nearby", 0.0), errors="coerce").fillna(0.0)
        hosp_beds   = pd.to_numeric(nodes.get("hospital_beds_nearby", 0.0), errors="coerce").fillna(0.0)

        shelter_norm = (shelter_cap - shelter_cap.min()) / (shelter_cap.max() - shelter_cap.min() + 1e-9)
        hosp_norm    = (hosp_beds - hosp_beds.min()) / (hosp_beds.max() - hosp_beds.min() + 1e-9)

        scarcity = 1.0 - 0.5*(shelter_norm + hosp_norm)
        scarcity = scarcity.clip(0, 1)

        # recovery at time t (optional)
        if predrec is not None:
            tmin = int(predrec["t"].min())
            tmax = int(predrec["t"].max())
            t = st.slider("Time step (t) for actions", tmin, tmax, tmin, 1)

            p = predrec[predrec["t"] == t][["node_id", "pred_recovery"]]
            nodes["node_id"] = nodes["node_id"].astype(str)
            p["node_id"] = p["node_id"].astype(str)
            nodes = nodes.merge(p, on="node_id", how="left")
            rec = pd.to_numeric(nodes["pred_recovery"], errors="coerce").fillna(nodes["pred_recovery"].median()).clip(0, 1)
        else:
            rec = pd.Series(0.5, index=nodes.index)

        # priority score: high vuln + low recovery + scarcity
        nodes["priority"] = (0.45*vuln + 0.35*(1.0-rec) + 0.20*scarcity).clip(0, 1)

        # assign action category
        def action_row(r):
            if r["priority"] >= 0.75:
                return "Immediate resource dispatch + shelter support"
            if r["priority"] >= 0.50:
                return "Targeted aid + monitor recovery"
            return "Monitor (low urgency)"

        nodes["action"] = nodes.apply(action_row, axis=1)

        # show top 25 table
        show = nodes[["node_id", "priority", "action"]].sort_values("priority", ascending=False).head(25)
        st.dataframe(show, use_container_width=True)

        # map
        nodes = nodes.dropna(subset=["lat", "lon"])
        center = [nodes["lat"].mean(), nodes["lon"].mean()]
        m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

        def color(v):
            if v < 0.25: return "#1a9850"
            if v < 0.50: return "#fee08b"
            if v < 0.75: return "#fc8d59"
            return "#d73027"

        for _, r in nodes.iterrows():
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=3,
                color=color(float(r["priority"])),
                fill=True,
                fill_opacity=0.85,
                popup=f"node_id={r['node_id']}<br>priority={r['priority']:.3f}<br>{r['action']}"
            ).add_to(m)

        st_folium(m, width=1000, height=650)


if __name__ == "__main__":
    main()