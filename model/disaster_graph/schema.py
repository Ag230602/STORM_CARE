"""
DisasterGraphSchema — node/edge type definitions and synthetic scenario generator.

Scenario physics
----------------
A synthetic tropical cyclone is generated as a Rankine vortex that moves
across the domain at ~5 m/s westward + 3 m/s northward (simplified steering).
At each time step the following update rules are applied:

  Wind speed at node i:
    V_i(t) = V_max · exp(−dist(i, storm_t) / R_max)

  Infrastructure damage (cumulative):
    D_i(t) = D_i(t−1) + α_i · V_i(t) · Δt,   α depends on node type

  Shelter occupancy (people seek shelter as storm approaches):
    O_j(t) = O_j(t−1) + Σ_k mobility_k · P(evac | V_k(t)) · pop_k

  Hospital load:
    H_j(t) = H_j(t−1) + Σ_i (damage_rate_i · pop_near_i · V_i(t))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from .config import DisasterGraphConfig

# ── Node type indices ──────────────────────────────────────────────────────────
ATM_TYPE      = 0
REGION_TYPE   = 1
SCHOOL_TYPE   = 2
HOSPITAL_TYPE = 3
SHELTER_TYPE  = 4
POP_TYPE      = 5

# ── Edge type indices ──────────────────────────────────────────────────────────
EDGE_PROPAGATION  = 0   # atm → atm
EDGE_EXPOSURE     = 1   # atm → infra
EDGE_TRANSPORT    = 2   # region → region
EDGE_INFRA_DEP    = 3   # region → school/hospital/shelter/pop


@dataclass
class DisasterScenario:
    """One synthetic disaster scenario at a single time step."""
    node_features: Tensor   # (N_total, d_feat_max)  zero-padded
    node_types:    Tensor   # (N_total,)              int64 type index
    edge_index:    Tensor   # (2, E)
    edge_types:    Tensor   # (E,)                   int64 edge type
    # Regression targets: per-node damage / stress score in [0, 1]
    targets:       Tensor   # (N_total,)
    # Global disaster state for WorldModel (produced by DisasterGNN at inference)
    storm_pos:     np.ndarray  # (2,) [lat, lon] of storm centre


def _build_graph(cfg: DisasterGraphConfig, rng: np.random.Generator) -> Tuple:
    """
    Build a fixed heterogeneous graph topology for the scenario.
    Returns (edge_index, edge_types, node_types, node_coords).
    node_coords: (N_total, 2) normalised [−1,1]² positions.
    """
    N = cfg.n_total_nodes
    n_atm = cfg.n_atm
    n_reg = cfg.n_regions
    n_sch = cfg.n_schools
    n_hos = cfg.n_hospitals
    n_sht = cfg.n_shelters
    n_pop = cfg.n_pop

    # Node indices
    atm_idx  = np.arange(0, n_atm)
    reg_idx  = np.arange(n_atm, n_atm + n_reg)
    sch_idx  = np.arange(n_atm + n_reg, n_atm + n_reg + n_sch)
    hos_idx  = np.arange(n_atm + n_reg + n_sch, n_atm + n_reg + n_sch + n_hos)
    sht_idx  = np.arange(n_atm + n_reg + n_sch + n_hos,
                          n_atm + n_reg + n_sch + n_hos + n_sht)
    pop_idx  = np.arange(n_atm + n_reg + n_sch + n_hos + n_sht, N)

    # Node types
    node_types = np.empty(N, dtype=np.int64)
    node_types[atm_idx] = ATM_TYPE
    node_types[reg_idx] = REGION_TYPE
    node_types[sch_idx] = SCHOOL_TYPE
    node_types[hos_idx] = HOSPITAL_TYPE
    node_types[sht_idx] = SHELTER_TYPE
    node_types[pop_idx] = POP_TYPE

    # Spatial coordinates for all nodes
    coords = rng.uniform(-1, 1, (N, 2))
    # Atmospheric grid: regular spacing
    grid_n = int(np.round(np.sqrt(n_atm)))
    lin = np.linspace(-0.9, 0.9, grid_n)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    coords[atm_idx, 0] = xx.ravel()[:n_atm]
    coords[atm_idx, 1] = yy.ravel()[:n_atm]

    src_list, dst_list, etype_list = [], [], []

    # ── Edge type 0: atm → atm (4-connected grid) ─────────────────────────────
    grid_n2 = grid_n
    for r in range(grid_n2):
        for c in range(grid_n2):
            i = r * grid_n2 + c
            if i >= n_atm:
                break
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_n2 and 0 <= nc < grid_n2:
                    j = nr * grid_n2 + nc
                    if j < n_atm:
                        src_list.append(atm_idx[i])
                        dst_list.append(atm_idx[j])
                        etype_list.append(EDGE_PROPAGATION)

    # ── Edge type 1: nearest atm → each infra node (exposure) ────────────────
    infra_all = np.concatenate([sch_idx, hos_idx, sht_idx, pop_idx, reg_idx])
    for inf_i in infra_all:
        dists = np.linalg.norm(coords[atm_idx] - coords[inf_i], axis=1)
        nearest = atm_idx[np.argmin(dists)]
        src_list.append(nearest)
        dst_list.append(inf_i)
        etype_list.append(EDGE_EXPOSURE)

    # ── Edge type 2: region → region (transport, random sparse) ───────────────
    for i in reg_idx:
        dists = np.linalg.norm(coords[reg_idx] - coords[i], axis=1)
        # Connect to 2 nearest regions
        order = np.argsort(dists)
        for j_idx in order[1:3]:
            j = reg_idx[j_idx]
            src_list.append(i); dst_list.append(j); etype_list.append(EDGE_TRANSPORT)
            src_list.append(j); dst_list.append(i); etype_list.append(EDGE_TRANSPORT)

    # ── Edge type 3: each region → its nearest infra nodes ────────────────────
    for inf_i in np.concatenate([sch_idx, hos_idx, sht_idx, pop_idx]):
        dists = np.linalg.norm(coords[reg_idx] - coords[inf_i], axis=1)
        nearest_reg = reg_idx[np.argmin(dists)]
        src_list.append(nearest_reg)
        dst_list.append(inf_i)
        etype_list.append(EDGE_INFRA_DEP)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_types = np.array(etype_list, dtype=np.int64)
    return edge_index, edge_types, node_types, coords


def generate_scenario(
    cfg: DisasterGraphConfig,
    rng: np.random.Generator,
    t: int,
    edge_index: np.ndarray,
    edge_types: np.ndarray,
    node_types: np.ndarray,
    coords: np.ndarray,
    storm_track: np.ndarray,
    vmax_ms: float,
    base_features: np.ndarray,
    damage: np.ndarray,
) -> DisasterScenario:
    """
    Generate the graph snapshot at time step t of a storm scenario.

    Args:
        t            : current time step index
        edge_index   : (2, E) pre-built graph topology
        edge_types   : (E,) pre-built edge type labels
        node_types   : (N,) node type indices
        coords       : (N, 2) node spatial coordinates
        storm_track  : (n_steps, 2) storm centre positions
        base_features: (N, d_feat_max) initial static node features
        damage       : (N,) accumulated damage state (modified in-place)
    """
    N = cfg.n_total_nodes
    storm_pos = storm_track[t]
    dist = np.linalg.norm(coords - storm_pos, axis=1)      # (N,)

    # Wind speed at each node (Rankine outer profile)
    V = vmax_ms * np.exp(-dist / (cfg.rmax_norm + 1e-6))  # (N,)

    # Damage accumulation (only non-atmospheric nodes).
    # The previous alpha * V * seconds formula saturated targets to one after
    # a single step.  This bounded hazard^2 update preserves variation across
    # node types and time while remaining a synthetic proxy target.
    hazard = np.clip(V / max(cfg.vmax_ms, 1e-6), 0.0, 1.0)
    alpha = np.zeros_like(hazard, dtype=np.float32)
    region_mask = node_types == REGION_TYPE
    school_mask = node_types == SCHOOL_TYPE
    hospital_mask = node_types == HOSPITAL_TYPE
    shelter_mask = node_types == SHELTER_TYPE
    pop_mask = node_types == POP_TYPE

    alpha[region_mask] = 0.200 * (0.5 + base_features[region_mask, 1])
    alpha[school_mask] = 0.300 * (1.2 - base_features[school_mask, 1])
    alpha[hospital_mask] = 0.280 * (1.25 - base_features[hospital_mask, 1]) * (
        1.0 - 0.25 * base_features[hospital_mask, 3]
    )
    alpha[shelter_mask] = 0.220 * (1.2 - base_features[shelter_mask, 2])
    alpha[pop_mask] = 0.160 * (0.7 + base_features[pop_mask, 1])
    damage += alpha * (hazard ** 2)
    damage = np.clip(damage, 0, 1)

    # Build node feature matrix at this time step
    feat = base_features.copy()
    # Update atm_cell wind_speed channel (index 4)
    atm_mask = node_types == ATM_TYPE
    feat[atm_mask, 4] = V[atm_mask] / cfg.vmax_ms   # normalise to [0,1]

    # Update shelter occupancy (index 1 of shelter feature = curr_occupancy)
    sht_mask = node_types == SHELTER_TYPE
    feat[sht_mask, 1] = np.clip(
        feat[sht_mask, 1] + 0.1 * V[sht_mask] / cfg.vmax_ms, 0, 1
    )

    # Regression target: damage score per node
    target = damage.copy()
    # atm cells: target is normalised wind speed (proxy for hazard level)
    target[atm_mask] = V[atm_mask] / cfg.vmax_ms

    node_feat_t = torch.from_numpy(feat).float()
    return DisasterScenario(
        node_features = node_feat_t,
        node_types    = torch.from_numpy(node_types),
        edge_index    = torch.from_numpy(edge_index),
        edge_types    = torch.from_numpy(edge_types),
        targets       = torch.from_numpy(target).float(),
        storm_pos     = storm_pos,
    )


def build_dataset(
    cfg: DisasterGraphConfig,
    seed: int = 42,
) -> List[List[DisasterScenario]]:
    """
    Generate cfg.n_scenarios full temporal scenarios, each with cfg.n_steps
    graph snapshots.  Returns a list of lists: [scenario][step] → DisasterScenario.
    """
    rng = np.random.default_rng(seed)
    all_scenarios = []

    # Pre-build one graph topology (shared across all scenarios)
    edge_index, edge_types, node_types, coords = _build_graph(cfg, rng)

    for sc_i in range(cfg.n_scenarios):
        sc_rng = np.random.default_rng(seed + sc_i + 1)

        # Random storm track: starts offshore, makes landfall
        vmax  = sc_rng.uniform(0.5 * cfg.vmax_ms, cfg.vmax_ms)
        start = np.array([sc_rng.uniform(-0.9, -0.4), sc_rng.uniform(-0.9, 0.9)])
        vel   = np.array([sc_rng.uniform(0.05, 0.12), sc_rng.uniform(-0.04, 0.04)])
        storm_track = np.stack([start + t * vel for t in range(cfg.n_steps)])

        # Initial node features (static properties)
        base = np.zeros((cfg.n_total_nodes, cfg.d_feat_max), dtype=np.float32)
        na = cfg.n_atm
        nr = cfg.n_regions
        ns = cfg.n_schools
        nh = cfg.n_hospitals
        nsh = cfg.n_shelters
        np_ = cfg.n_pop
        # Compute explicit slice starts for each node type
        o_atm  = 0
        o_reg  = na
        o_sch  = na + nr
        o_hos  = na + nr + ns
        o_sht  = na + nr + ns + nh
        o_pop  = na + nr + ns + nh + nsh
        base[o_atm:o_atm+na,   :] = sc_rng.uniform(0, 0.3, (na, cfg.d_feat_max))
        base[o_reg:o_reg+nr,  :5] = sc_rng.uniform(0.1, 0.9, (nr, 5))
        base[o_sch:o_sch+ns,  :4] = sc_rng.uniform(0.2, 0.8, (ns, 4))
        base[o_hos:o_hos+nh,  :4] = sc_rng.uniform(0.2, 0.8, (nh, 4))
        base[o_sht:o_sht+nsh, :4] = sc_rng.uniform(0.2, 0.8, (nsh, 4))
        base[o_pop:o_pop+np_, :3] = sc_rng.uniform(0.2, 0.8, (np_, 3))
        # Feature index 3 of pop_cluster = child_fraction (proportion aged <18)
        # Typical range 0.18–0.35 for US counties (CDC/Census data ranges)
        base[o_pop:o_pop+np_, 3]  = sc_rng.uniform(0.18, 0.35, np_)

        damage = np.zeros(cfg.n_total_nodes, dtype=np.float32)

        steps = []
        for t in range(cfg.n_steps):
            sc = generate_scenario(
                cfg, sc_rng, t,
                edge_index, edge_types, node_types, coords,
                storm_track, vmax, base, damage,
            )
            steps.append(sc)

        all_scenarios.append(steps)

    return all_scenarios


def node_offsets(cfg: DisasterGraphConfig) -> dict[str, int]:
    na = cfg.n_atm
    nr = cfg.n_regions
    ns = cfg.n_schools
    nh = cfg.n_hospitals
    nsh = cfg.n_shelters
    return {
        "atm": 0,
        "region": na,
        "school": na + nr,
        "hospital": na + nr + ns,
        "shelter": na + nr + ns + nh,
        "pop": na + nr + ns + nh + nsh,
    }


def humanitarian_targets(cfg: DisasterGraphConfig, sc: DisasterScenario) -> dict[str, Tensor]:
    """Simulator-derived proxy labels for the humanitarian heads.

    These are synthetic targets generated from the scenario state, not observed
    disaster labels.  They are separated from model inputs and used only as
    supervised labels/evaluation targets.
    """
    off = node_offsets(cfg)
    dmg = sc.targets.float()
    nf = sc.node_features.float()
    ns = cfg.n_schools
    nh = cfg.n_hospitals
    nsh = cfg.n_shelters
    np_ = cfg.n_pop
    o_reg = off["region"]
    o_sch = off["school"]
    o_hos = off["hospital"]
    o_sht = off["shelter"]
    o_pop = off["pop"]

    pop_damage = dmg[o_pop:o_pop + np_]
    child_exposure_frac = pop_damage.clamp(0.0, 1.0)
    pop_feat = nf[o_pop:o_pop + np_]
    exposed_children_count = (
        pop_feat[:, 0] * pop_feat[:, 3] * child_exposure_frac * 20_000.0
    )
    school_damage = dmg[o_sch:o_sch + ns].clamp(0.0, 1.0)
    hospital_access = (1.0 - dmg[o_hos:o_hos + nh]).clamp(0.0, 1.0)
    shelter_demand = (
        0.55 * dmg[o_sht:o_sht + nsh] + 0.45 * nf[o_sht:o_sht + nsh, 1]
    ).clamp(0.0, 1.0)
    recovery_priority = dmg[o_reg:o_pop].clamp(0.0, 1.0)
    return {
        "child_exposure_frac": child_exposure_frac,
        "exposed_children_count": exposed_children_count,
        "school_damage": school_damage,
        "school_disrupted": (school_damage > cfg.school_disruption_threshold).float(),
        "hospital_access": hospital_access,
        "shelter_demand": shelter_demand,
        "recovery_priority": recovery_priority,
    }


# ── Humanitarian output utilities ─────────────────────────────────────────────

def generate_hazard_map(
    cfg: DisasterGraphConfig,
    damage_scores: np.ndarray,      # (N_total,)
    node_types: np.ndarray,         # (N_total,)
) -> np.ndarray:
    """
    Reshape atmospheric-cell damage scores into a 2-D hazard map.

    Returns (grid_n, grid_n) float32 array where each cell contains
    the wind-hazard level [0, 1] at that grid point.
    This is the Meteorological output: Hazard Map.
    """
    atm_mask = node_types == ATM_TYPE
    atm_scores = damage_scores[atm_mask]
    grid_n = int(round(len(atm_scores) ** 0.5))
    n_use  = grid_n * grid_n
    return atm_scores[:n_use].reshape(grid_n, grid_n)


def generate_humanitarian_report(
    cfg: DisasterGraphConfig,
    model_out: dict,
    node_features: np.ndarray,      # (N_total, d_feat_max) — original features
) -> dict:
    """
    Compute all humanitarian output metrics from a DisasterGNN forward pass.

    Outputs
    -------
    Meteorological:
      hazard_map              (grid_n, grid_n)  2-D wind hazard level
      wind_field_max_ms       float             peak predicted wind speed

    Humanitarian:
      exposed_children_est    int               estimated children directly exposed
      school_disruption_pct   float             % schools with disruption score > 0.5
      hospital_access_avg     float             mean hospital accessibility [0,1]
      shelter_demand_avg      float             mean shelter demand pressure [0,1]
      shelter_at_capacity     int               # shelters with demand > 0.8

    Recovery:
      recovery_priority_zones list[int]         node indices ranked by priority (descending)
      top3_priority_labels    list[str]         human-readable labels for top-3 zones
    """
    import torch

    na  = cfg.n_atm
    nr  = cfg.n_regions
    ns  = cfg.n_schools
    nh  = cfg.n_hospitals
    nsh = cfg.n_shelters
    np_ = cfg.n_pop

    o_atm = 0
    o_reg = na
    o_sch = na + nr
    o_hos = na + nr + ns
    o_sht = na + nr + ns + nh
    o_pop = na + nr + ns + nh + nsh

    dmg = model_out["damage_scores"]
    if isinstance(dmg, torch.Tensor):
        dmg = dmg.detach().cpu().numpy()

    def _np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    # Hazard map
    atm_dmg   = dmg[o_atm:o_atm+na]
    grid_n    = int(round(na ** 0.5))
    hazard_map = atm_dmg[:grid_n*grid_n].reshape(grid_n, grid_n)

    # Wind field: atm feature index 4 = normalised wind speed
    wind_raw  = node_features[o_atm:o_atm+na, 4]
    wind_max  = float(wind_raw.max()) * cfg.vmax_ms    # convert back to m/s

    # Exposed children: pop_count (feat 0) × child_fraction (feat 3) × exposure
    pop_feat  = node_features[o_pop:o_pop+np_]
    pop_count = pop_feat[:, 0]                          # normalised count
    child_frac = pop_feat[:, 3]                         # fraction < 18
    exposure  = _np(model_out.get("child_exposure", dmg[o_pop:o_pop+np_]))
    # Assume pop_count feature ~0.5 ≈ 10,000 people (scale factor 20,000)
    exposed_children = int((pop_count * child_frac * exposure * 20_000).sum())

    # School disruption
    sch_disrupt = _np(model_out.get("school_disruption", dmg[o_sch:o_sch+ns]))
    school_disruption_pct = float((sch_disrupt > 0.5).mean() * 100)

    # Hospital accessibility
    hosp_access = _np(model_out.get("hospital_access", 1.0 - dmg[o_hos:o_hos+nh]))
    hospital_access_avg = float(hosp_access.mean())

    # Shelter demand
    shlt_demand = _np(model_out.get("shelter_demand", dmg[o_sht:o_sht+nsh]))
    shelter_demand_avg  = float(shlt_demand.mean())
    shelter_at_capacity = int((shlt_demand > 0.8).sum())

    # Recovery priority zones — rank all infra nodes by priority score
    rec_scores  = _np(model_out.get("recovery_priority", dmg[o_reg:o_pop]))
    infra_global_idx = np.arange(o_reg, o_pop)           # global node indices
    priority_order   = infra_global_idx[np.argsort(-rec_scores)]   # descending

    node_type_names = {
        o_reg:  "Region",
        o_sch:  "School",
        o_hos:  "Hospital",
        o_sht:  "Shelter",
    }

    def _label(global_i: int) -> str:
        if global_i < o_sch:
            return f"Region-{global_i - o_reg}"
        elif global_i < o_hos:
            return f"School-{global_i - o_sch}"
        elif global_i < o_sht:
            return f"Hospital-{global_i - o_hos}"
        else:
            return f"Shelter-{global_i - o_sht}"

    top3_labels = [_label(int(i)) for i in priority_order[:3]]

    return dict(
        # Meteorological
        hazard_map              = hazard_map,
        wind_field_max_ms       = round(wind_max, 1),
        # Humanitarian
        exposed_children_est    = exposed_children,
        school_disruption_pct   = round(school_disruption_pct, 1),
        hospital_access_avg     = round(hospital_access_avg, 3),
        shelter_demand_avg      = round(shelter_demand_avg, 3),
        shelter_at_capacity     = shelter_at_capacity,
        # Recovery
        recovery_priority_zones = priority_order.tolist(),
        top3_priority_labels    = top3_labels,
    )
