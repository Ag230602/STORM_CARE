"""
DisasterGraphConfig — configuration for the Dynamic Disaster Graph (Module 3).

Graph topology
--------------
Six node types form a heterogeneous graph:
  atm_cell    — atmospheric grid cells carrying storm state
  region      — administrative regions (counties / parishes)
  school      — school buildings used as evacuation staging areas
  hospital    — medical facilities
  shelter     — designated emergency shelters
  pop_cluster — population clusters (census tracts)

Four edge types encode physical and social connectivity:
  0  storm_propagation   atm_cell → atm_cell   (spatial adjacency on the grid)
  1  exposure            atm_cell → infra node  (storm directly affects infra)
  2  transportation      region   → region       (road network connectivity)
  3  infrastructure_dep  region   → any node     (region governs/contains node)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DisasterGraphConfig:
    # ── Node counts ────────────────────────────────────────────────────────────
    n_atm:      int = 25    # 5×5 atmospheric grid
    n_regions:  int = 8     # administrative regions
    n_schools:  int = 12    # school buildings
    n_hospitals: int = 4    # hospitals
    n_shelters: int = 6     # designated shelters
    n_pop:      int = 10    # population clusters

    # ── Node feature dimensions ────────────────────────────────────────────────
    # atm_cell  : [u850, v850, mslp_anom, temp_anom, wind_speed, humidity, precip]
    # region    : [pop_density, vulnerability, income_idx, elevation, coast_dist]
    # school    : [capacity, condition, elevation, evac_route_ok]
    # hospital  : [capacity, condition, elevation, has_backup_power]
    # shelter   : [max_capacity, curr_occupancy, supply_level, elevation]
    # pop_cluster: [pop_count, vulnerability, mobility]
    d_feat_max: int = 7     # unified (zero-padded) feature vector length

    # ── GNN architecture ───────────────────────────────────────────────────────
    d_hidden:    int = 64
    d_type_emb:  int = 16   # learnable node-type embedding dimension
    d_edge_emb:  int = 16   # learnable edge-type embedding dimension
    n_gnn_layers: int = 3
    n_node_types: int = 6   # atm / region / school / hospital / shelter / pop
    n_edge_types: int = 4   # propagation / exposure / transport / dep

    # ── Output ─────────────────────────────────────────────────────────────────
    # Global disaster-state vector fed as input to the World Model (Module 4).
    d_disaster_state: int = 32

    # ── Scenario generation ────────────────────────────────────────────────────
    n_scenarios: int = 100   # synthetic scenarios for training
    n_steps:     int = 12    # time steps per scenario (each step = 6 h)
    vmax_ms:     float = 65.0  # maximum storm wind speed in m/s
    rmax_norm:   float = 0.25  # radius of maximum wind (normalised domain)
    school_disruption_threshold: float = 0.15

    # ── Training ───────────────────────────────────────────────────────────────
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    n_epochs:     int   = 40
    seed:         int   = 42
    demo:         bool  = False

    # ── Demo overrides ─────────────────────────────────────────────────────────
    def apply_demo_overrides(self) -> "DisasterGraphConfig":
        self.demo        = True
        self.n_atm       = 9      # 3×3 grid
        self.n_regions   = 4
        self.n_schools   = 6
        self.n_hospitals = 2
        self.n_shelters  = 3
        self.n_pop       = 5
        self.d_hidden    = 32
        self.d_type_emb  = 8
        self.d_edge_emb  = 8
        self.n_gnn_layers = 2
        self.n_scenarios = 40
        self.n_steps     = 8
        self.n_epochs    = 20
        return self

    @property
    def n_total_nodes(self) -> int:
        return (self.n_atm + self.n_regions + self.n_schools
                + self.n_hospitals + self.n_shelters + self.n_pop)

    def __str__(self) -> str:
        tag = "[DEMO] " if self.demo else ""
        return (f"{tag}DisasterGraphConfig | "
                f"nodes={self.n_total_nodes} ({self.n_atm} atm + "
                f"{self.n_regions} reg + {self.n_schools} sch + "
                f"{self.n_hospitals} hosp + {self.n_shelters} shlt + "
                f"{self.n_pop} pop) | "
                f"d_hidden={self.d_hidden} | layers={self.n_gnn_layers}")
