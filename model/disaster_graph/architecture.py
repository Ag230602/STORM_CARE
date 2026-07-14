"""
DisasterGNN — Heterogeneous Graph Neural Network for the Dynamic Disaster Graph.

Architecture
------------
1. NodeEncoder   : type-specific projection + learnable type embedding → d_hidden
2. GNNLayer ×n   : edge-type-conditioned message passing (scatter-add)
3. DamageHead    : per-node damage/stress score in [0, 1]
4. StateHead     : global mean-pool → MLP → d_disaster_state (for WorldModel)

Humanitarian output heads
--------------------------
5. RecoveryHead      : recovery priority score per infrastructure node (higher = more urgent)
6. ChildExposureHead : estimated exposed-children count per population cluster node
7. SchoolDisruptHead : school disruption level (0 = operational, 1 = fully disrupted)
8. HospitalAccessHead: hospital accessibility index (0 = inaccessible, 1 = fully accessible)
9. ShelterDemandHead : shelter demand pressure (0 = no demand, 1 = full / over capacity)

Message computation (one layer)
---------------------------------
  m_ij = MLP( [h_i ‖ h_j ‖ edge_type_emb_ij] )      ← kernel message
  H_i  = Σ_{j∈N(i)} m_ij                              ← scatter-add aggregate
  h_i' = GELU( LN( W_skip h_i  +  H_i  +  b ) )      ← residual update
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import DisasterGraphConfig
from .schema import DisasterScenario, ATM_TYPE, REGION_TYPE, SCHOOL_TYPE, HOSPITAL_TYPE, SHELTER_TYPE, POP_TYPE


class DisasterGNNLayer(nn.Module):
    """One message-passing layer over the heterogeneous disaster graph."""

    def __init__(self, d_hidden: int, d_edge_emb: int, n_edge_types: int):
        super().__init__()
        self.edge_emb = nn.Embedding(n_edge_types, d_edge_emb)

        msg_in = 2 * d_hidden + d_edge_emb
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.skip = nn.Linear(d_hidden, d_hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_hidden))
        self.norm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        h:          Tensor,   # (N, d_hidden)
        edge_index: Tensor,   # (2, E)   [src, dst]
        edge_types: Tensor,   # (E,)     int64
    ) -> Tensor:
        N, d = h.shape
        src, dst = edge_index[0], edge_index[1]

        et_emb = self.edge_emb(edge_types)              # (E, d_edge_emb)
        h_src  = h[src]                                  # (E, d_hidden)
        h_dst  = h[dst]                                  # (E, d_hidden)
        msg    = self.msg_mlp(
            torch.cat([h_src, h_dst, et_emb], dim=-1)
        )                                                # (E, d_hidden)

        # Scatter-add messages to destination nodes
        agg = torch.zeros(N, d, device=h.device, dtype=h.dtype)
        idx = dst.unsqueeze(-1).expand_as(msg)
        agg.scatter_add_(0, idx, msg)                    # (N, d_hidden)

        out = self.skip(h) + agg + self.bias
        return F.gelu(self.norm(out))                    # (N, d_hidden)


class DisasterGNN(nn.Module):
    """
    Full Dynamic Disaster GNN.

    Forward returns a dict with:
      node_emb           : (N, d_hidden)          per-node latent embeddings
      damage_scores      : (N,)                   damage/stress in [0, 1]
      disaster_state     : (d_disaster_state,)    global state for WorldModel

    Humanitarian outputs (derived from node-type-masked sub-sets):
      recovery_priority  : (N_infra,)             recovery urgency score [0,1]
      child_exposure     : (N_pop,)               exposed-children proportion [0,1]
      school_disruption  : (N_sch,)               school operational disruption [0,1]
      hospital_access    : (N_hosp,)              hospital accessibility index [0,1]
      shelter_demand     : (N_shlt,)              shelter demand pressure [0,1]

    Spatial outputs:
      hazard_grid        : (grid_n, grid_n)       2-D wind hazard map (atm nodes reshaped)
    """

    def __init__(self, cfg: DisasterGraphConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_hidden
        dt = cfg.d_type_emb

        # Project raw (padded) features + type embedding → d_hidden
        self.feat_proj   = nn.Linear(cfg.d_feat_max + dt, d)
        self.type_emb    = nn.Embedding(cfg.n_node_types, dt)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            DisasterGNNLayer(d, cfg.d_edge_emb, cfg.n_edge_types)
            for _ in range(cfg.n_gnn_layers)
        ])

        def _head(out_act=nn.Sigmoid):
            return nn.Sequential(
                nn.Linear(d, d // 2), nn.GELU(),
                nn.Linear(d // 2, 1), out_act(),
            )

        # Core outputs
        self.damage_head = _head(nn.Sigmoid)       # damage score per node
        self.state_head  = nn.Sequential(          # global disaster state → WorldModel
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, cfg.d_disaster_state),
        )

        # Humanitarian heads
        # recovery_priority: high damage × high vulnerability = urgent
        self.recovery_head     = _head(nn.Sigmoid)
        # child_exposure: what fraction of children in each pop cluster are exposed
        self.child_exposure_head = _head(nn.Sigmoid)
        # school_disruption: 0 = open, 1 = fully closed
        self.school_disruption_head = _head(nn.Sigmoid)
        # hospital_access: 0 = inaccessible, 1 = fully accessible (inverted damage)
        self.hospital_access_head   = _head(nn.Sigmoid)
        # shelter_demand: occupancy pressure on each shelter
        self.shelter_demand_head    = _head(nn.Sigmoid)

    def forward(self, scenario: DisasterScenario) -> dict:
        x   = scenario.node_features   # (N, d_feat_max)
        nt  = scenario.node_types       # (N,)
        ei  = scenario.edge_index       # (2, E)
        et  = scenario.edge_types       # (E,)

        # Embed node type and concatenate with features
        te  = self.type_emb(nt)         # (N, d_type_emb)
        h   = F.gelu(self.feat_proj(torch.cat([x, te], dim=-1)))   # (N, d_hidden)

        # Message passing
        for layer in self.gnn_layers:
            h = layer(h, ei, et)        # (N, d_hidden)

        # ── Core outputs ──────────────────────────────────────────────────────
        damage = self.damage_head(h).squeeze(-1)   # (N,)
        state  = self.state_head(h.mean(dim=0))    # (d_disaster_state,)

        # ── Node-type masks ───────────────────────────────────────────────────
        cfg = self.cfg
        na  = cfg.n_atm
        nr  = cfg.n_regions
        ns  = cfg.n_schools
        nh  = cfg.n_hospitals
        nsh = cfg.n_shelters
        np_ = cfg.n_pop

        o_atm  = 0
        o_reg  = na
        o_sch  = na + nr
        o_hos  = na + nr + ns
        o_sht  = na + nr + ns + nh
        o_pop  = na + nr + ns + nh + nsh

        h_atm  = h[o_atm:o_atm+na]
        h_infra = h[o_reg:o_reg+nr+ns+nh+nsh]  # regions + schools + hospitals + shelters
        h_sch  = h[o_sch:o_sch+ns]
        h_hos  = h[o_hos:o_hos+nh]
        h_sht  = h[o_sht:o_sht+nsh]
        h_pop  = h[o_pop:o_pop+np_]

        # ── Humanitarian outputs ──────────────────────────────────────────────
        # Recovery priority: all infra nodes ranked by urgency
        recovery_priority  = self.recovery_head(h_infra).squeeze(-1)         # (nr+ns+nh+nsh,)
        # Child exposure: proportion of children in each cluster exposed to storm
        child_exposure     = self.child_exposure_head(h_pop).squeeze(-1)     # (np_,)
        # School disruption: 0 = normal, 1 = fully disrupted
        school_disruption  = self.school_disruption_head(h_sch).squeeze(-1)  # (ns,)
        # Hospital accessibility: 1 = fully accessible (inverted → higher = better)
        hospital_access    = 1.0 - self.hospital_access_head(h_hos).squeeze(-1)  # (nh,)
        # Shelter demand pressure: 0 = empty, 1 = at/over capacity
        shelter_demand     = self.shelter_demand_head(h_sht).squeeze(-1)     # (nsh,)

        # ── 2-D hazard map from atm_cell damage scores ───────────────────────
        grid_n   = int(na ** 0.5)
        atm_dmg  = damage[o_atm:o_atm+na]          # (n_atm,)
        hazard_grid = atm_dmg[:grid_n * grid_n].reshape(grid_n, grid_n)

        return dict(
            node_emb           = h,
            damage_scores      = damage,
            disaster_state     = state,
            # Humanitarian
            recovery_priority  = recovery_priority,
            child_exposure     = child_exposure,
            school_disruption  = school_disruption,
            hospital_access    = hospital_access,
            shelter_demand     = shelter_demand,
            # Spatial
            hazard_grid        = hazard_grid,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
