"""
graph_construction.py — Storm graph builder for the STORM-CARE Foundation Model.

Graph topology
--------------
Nodes: each (storm_id, timestep) observation is a node.
       Two node types: STORM (primary), ERA5-CELL (auxiliary, future extension).

Edge types
----------
0  TEMPORAL_NEXT    (t → t+1)   within-storm consecutive steps
1  TEMPORAL_SKIP    (t → t+k)   within-storm skip connection (k ≤ temporal_window_steps)
2  INTER_STORM      (spatial)   concurrent storms within max_inter_storm_dist_km
3  SELF_LOOP        (t → t)     for residual message passing

Output per sequence window
--------------------------
edge_index : (2, E)  int64
edge_type  : (E,)    int64  — edge type indices from above
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Edge type constants
# ─────────────────────────────────────────────────────────────────────────────

EDGE_TEMPORAL_NEXT  = 0
EDGE_TEMPORAL_SKIP  = 1
EDGE_INTER_STORM    = 2
EDGE_SELF_LOOP      = 3
N_EDGE_TYPES        = 4


# ─────────────────────────────────────────────────────────────────────────────
# Haversine
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


# ─────────────────────────────────────────────────────────────────────────────
# Within-window graph (used per training sample)
# ─────────────────────────────────────────────────────────────────────────────

def build_window_graph(
    T: int,
    temporal_window_steps: int = 4,
    include_self_loops: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build temporal graph edges for a single storm window of length T.

    Returns
    -------
    edge_index : (2, E) int64
    edge_type  : (E,)   int64
    """
    src_list, dst_list, typ_list = [], [], []

    # Self-loops
    if include_self_loops:
        for t in range(T):
            src_list.append(t)
            dst_list.append(t)
            typ_list.append(EDGE_SELF_LOOP)

    # Temporal next (directed: past → future)
    for t in range(T - 1):
        src_list.append(t)
        dst_list.append(t + 1)
        typ_list.append(EDGE_TEMPORAL_NEXT)

    # Temporal skip connections (bidirectional within window)
    for k in range(2, min(temporal_window_steps + 1, T)):
        for t in range(T - k):
            src_list.append(t)
            dst_list.append(t + k)
            typ_list.append(EDGE_TEMPORAL_SKIP)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_type  = np.array(typ_list, dtype=np.int64)
    return edge_index, edge_type


# ─────────────────────────────────────────────────────────────────────────────
# Global storm graph (all storms → used for inter-storm edge pre-computation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalStormGraph:
    """
    Compact representation of the full multi-storm graph.

    Attributes
    ----------
    node_storm_ids   : List[str]   storm_id per node
    node_timesteps   : np.ndarray  (N,)  step index within storm
    node_lats        : np.ndarray  (N,)  float32
    node_lons        : np.ndarray  (N,)  float32
    edge_index       : np.ndarray  (2, E)
    edge_type        : np.ndarray  (E,)
    storm_node_slices: dict  storm_id → (start, end) indices in node arrays
    """
    node_storm_ids: List[str]
    node_timesteps: np.ndarray
    node_lats: np.ndarray
    node_lons: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    storm_node_slices: dict

    @property
    def N(self) -> int:
        return len(self.node_storm_ids)

    @property
    def E(self) -> int:
        return self.edge_index.shape[1]

    def summary(self) -> str:
        n_storms = len(self.storm_node_slices)
        type_counts = {
            "temporal_next": int((self.edge_type == EDGE_TEMPORAL_NEXT).sum()),
            "temporal_skip": int((self.edge_type == EDGE_TEMPORAL_SKIP).sum()),
            "inter_storm":   int((self.edge_type == EDGE_INTER_STORM).sum()),
            "self_loops":    int((self.edge_type == EDGE_SELF_LOOP).sum()),
        }
        return (
            f"GlobalStormGraph | storms={n_storms:,} | nodes={self.N:,} | "
            f"edges={self.E:,} | {type_counts}"
        )


def build_global_storm_graph(
    records,         # List[StormRecord]
    temporal_window_steps: int = 4,
    max_inter_storm_dist_km: float = 800.0,
    include_self_loops: bool = True,
) -> GlobalStormGraph:
    """
    Build the global multi-storm graph from a list of StormRecord objects.

    Algorithm
    ---------
    1.  Assign a global node index to every (storm, timestep) observation.
    2.  Add TEMPORAL_NEXT and TEMPORAL_SKIP edges within each storm.
    3.  Add INTER_STORM edges between concurrent storms within the distance
        threshold (O(N²) — acceptable for ≤ 5 000 concurrent node pairs).
    4.  Optionally add self-loops.
    """
    # ── 1. Build node arrays ─────────────────────────────────────────────────
    node_storm_ids  : List[str]   = []
    node_timesteps  : List[int]   = []
    node_lats       : List[float] = []
    node_lons       : List[float] = []
    storm_node_slices: dict       = {}

    for rec in records:
        start = len(node_storm_ids)
        for t in range(rec.T):
            node_storm_ids.append(rec.storm_id)
            node_timesteps.append(t)
            node_lats.append(float(rec.lat[t]))
            node_lons.append(float(rec.lon[t]))
        storm_node_slices[rec.storm_id] = (start, len(node_storm_ids))

    N = len(node_storm_ids)
    node_lats_np = np.array(node_lats, dtype=np.float32)
    node_lons_np = np.array(node_lons, dtype=np.float32)

    src_list: List[int] = []
    dst_list: List[int] = []
    typ_list: List[int] = []

    # ── 2. Within-storm temporal edges ───────────────────────────────────────
    for rec in records:
        s, e = storm_node_slices[rec.storm_id]
        T = e - s
        for t in range(T):
            i = s + t
            # Self-loops
            if include_self_loops:
                src_list.append(i); dst_list.append(i); typ_list.append(EDGE_SELF_LOOP)
            # NEXT
            if t < T - 1:
                src_list.append(i); dst_list.append(i + 1); typ_list.append(EDGE_TEMPORAL_NEXT)
            # SKIP
            for k in range(2, min(temporal_window_steps + 1, T - t)):
                src_list.append(i); dst_list.append(i + k); typ_list.append(EDGE_TEMPORAL_SKIP)

    # ── 3. Inter-storm spatial edges (concurrent storms) ─────────────────────
    # Group nodes by their storm's datetime to find concurrent observations
    # For efficiency, use a time-bucketed approach (6-h buckets)
    if max_inter_storm_dist_km > 0:
        # Build time → list-of-node-indices mapping
        import datetime as dt_mod
        time_to_nodes: dict = {}
        for rec in records:
            for t, row in rec.track_df.iterrows():
                ts = row["datetime_utc"]
                # Bucket to nearest 6h
                bucket = ts.replace(minute=0, second=0, microsecond=0)
                g_idx  = storm_node_slices[rec.storm_id][0] + t
                time_to_nodes.setdefault(bucket, []).append(g_idx)

        for bucket, node_ids in time_to_nodes.items():
            if len(node_ids) < 2:
                continue
            # O(k²) within each time bucket (k = concurrent storms, typically <20)
            for a_i, a in enumerate(node_ids):
                for b in node_ids[a_i + 1:]:
                    dist = _haversine_km(
                        float(node_lats_np[a]), float(node_lons_np[a]),
                        float(node_lats_np[b]), float(node_lons_np[b]),
                    )
                    if dist <= max_inter_storm_dist_km:
                        src_list.append(a); dst_list.append(b); typ_list.append(EDGE_INTER_STORM)
                        src_list.append(b); dst_list.append(a); typ_list.append(EDGE_INTER_STORM)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_type  = np.array(typ_list,             dtype=np.int64)

    g = GlobalStormGraph(
        node_storm_ids   = node_storm_ids,
        node_timesteps   = np.array(node_timesteps, dtype=np.int64),
        node_lats        = node_lats_np,
        node_lons        = node_lons_np,
        edge_index       = edge_index,
        edge_type        = edge_type,
        storm_node_slices= storm_node_slices,
    )
    return g
