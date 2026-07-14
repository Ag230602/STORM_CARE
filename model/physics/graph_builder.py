"""
GraphBuilder — k-NN spatial graph over the regular N×N domain grid.

Also provides GraphDifferentialOps: discrete differential operators
(gradient, Laplacian, divergence) needed by the physics residuals.

Graph topology
--------------
Nodes  : N² grid points, coordinates normalised to [-1, 1]² for stability.
Edges  : k-NN by Euclidean distance in the normalised space.
         edge_index[0] = destination (i), edge_index[1] = source (j).
         Message convention: information flows j → i.

Discrete gradient (inverse-distance weighted finite differences)
-----------------------------------------------------------------
For a scalar field f at node i:

  (∂f/∂x)_i ≈ Σ_{j∈N(i)} w_ij · (f_j − f_i) · ê_ij,x

where
  ê_ij  = (x_j − x_i) / ‖x_j − x_i‖   (unit vector from i → j)
  w_ij  = (1/‖x_j − x_i‖²) / Σ_k (1/‖x_k − x_i‖²)  (IDW weights, row-normalised)

Physical units: multiply the normalised-coordinate gradient by
  scale = (grid_size − 1) / (2 · grid_spacing_m)
to convert from [f / x_norm] to [f / m].

Graph Laplacian
---------------
  (∇²f)_i ≈ Σ_{j∈N(i)} w_ij · (f_j − f_i) · scale²

Divergence
----------
  (∇·u)_i = (∂u/∂x)_i + (∂v/∂y)_i
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .config import PIGNOConfig


@dataclass
class GraphData:
    """Holds graph topology and pre-computed geometric quantities."""
    x_coords:   Tensor   # (N², 2)  normalised coords ∈ [−1, 1]²
    edge_index: Tensor   # (2, E)   [dst; src], message j → i
    edge_vec:   Tensor   # (E, 2)   x_i − x_j  (vector from j to i)
    edge_dist:  Tensor   # (E,)     ‖x_i − x_j‖ in normalised coords
    edge_angle: Tensor   # (E,)     atan2(Δy, Δx) for x_i − x_j direction
    n_nodes:    int
    n_edges:    int


def build_grid_graph(cfg: PIGNOConfig, device: torch.device) -> GraphData:
    """
    Build a static k-NN graph on the N×N regular grid.

    Coordinates are normalised to [−1, 1]² and shared across all batch
    elements (the graph topology is identical for every hurricane sample).
    """
    N = cfg.grid_size
    k = cfg.k_neighbors

    # ── Node coordinates in [−1, 1]² ──────────────────────────────────────
    lin  = torch.linspace(-1.0, 1.0, N, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")    # (N, N)
    coords  = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N², 2)
    n_nodes = N * N

    # ── Pairwise distances and k-NN selection ─────────────────────────────
    # diff[i, j] = coords[i] − coords[j]
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)          # (N², N², 2)
    dist = diff.norm(dim=-1)                                   # (N², N²)
    dist.fill_diagonal_(1e9)                                   # exclude self-loops

    # k smallest distances per node  → indices of k nearest neighbours
    _, nn_idx = dist.topk(k, dim=-1, largest=False)           # (N², k)

    # Build edge_index: dst = i (receiving), src = j (sending), message j→i
    dst = (
        torch.arange(n_nodes, device=device)
        .unsqueeze(-1).expand(-1, k).reshape(-1)
    )                                                          # (E,)
    src = nn_idx.reshape(-1)                                   # (E,)

    edge_index = torch.stack([dst, src], dim=0)               # (2, E)
    edge_vec   = coords[dst] - coords[src]                    # (E, 2) = x_i − x_j
    edge_dist  = edge_vec.norm(dim=-1)                        # (E,)
    edge_angle = torch.atan2(edge_vec[:, 1], edge_vec[:, 0]) # (E,) angle of i−j vec

    return GraphData(
        x_coords   = coords,
        edge_index = edge_index,
        edge_vec   = edge_vec,
        edge_dist  = edge_dist,
        edge_angle = edge_angle,
        n_nodes    = n_nodes,
        n_edges    = edge_index.shape[1],
    )


class GraphDifferentialOps:
    """
    Discrete differential operators on the k-NN graph.

    All operators accept tensors of shape (B, N, C) and return the same shape.

    Pre-computed quantities (stored as tensors on the graph device):
      src, dst     : edge endpoint indices
      w            : IDW row-normalised weights  (E,)
      ex, ey       : x, y components of the unit vector  ê_{ij} = (x_j−x_i)/‖‖
                     i.e. pointing FROM i TO j
      scale        : converts normalised-coord gradient to physical (1/m)
    """

    def __init__(self, graph: GraphData, grid_spacing_m: float, grid_size: int):
        """
        Args:
            graph          : GraphData from build_grid_graph()
            grid_spacing_m : physical spacing in metres (for unit conversion)
            grid_size      : N (number of points along one axis)
        """
        self.graph = graph
        src, dst = graph.edge_index[0], graph.edge_index[1]
        self.src = src
        self.dst = dst
        N = graph.n_nodes

        # ── IDW weights: w_ij = 1/‖x_j−x_i‖² (row-normalised) ──────────
        eps   = 1e-8
        inv_d2 = 1.0 / (graph.edge_dist ** 2 + eps)           # (E,)
        w_sum  = torch.zeros(N, device=src.device)
        w_sum.scatter_add_(0, dst, inv_d2)
        self.w  = inv_d2 / (w_sum[dst] + eps)                  # (E,) normalised

        # ── Unit vectors ê_{ij} pointing FROM i TO j ─────────────────────
        # edge_vec = x_i − x_j  →  unit vec i→j = −edge_vec / ‖edge_vec‖
        self.ex = -graph.edge_vec[:, 0] / (graph.edge_dist + eps)  # (E,)
        self.ey = -graph.edge_vec[:, 1] / (graph.edge_dist + eps)  # (E,)

        # ── Physical scale factor ─────────────────────────────────────────
        # Normalised spacing: Δx_norm = 2/(N−1)
        # Physical spacing:   Δx_phys = grid_spacing_m
        # Scale: normalised_gradient → physical = (N−1)/(2·grid_spacing_m)
        dx_norm = 2.0 / max(grid_size - 1, 1)
        self.scale = dx_norm / grid_spacing_m                  # (1/m in physical)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _scatter_to_dst(self, msg: Tensor) -> Tensor:
        """
        Sum messages arriving at each destination node.
        msg  : (B, E, C)
        Returns (B, N, C).
        """
        B, E, C = msg.shape
        N = self.graph.n_nodes
        out = torch.zeros(B, N, C, device=msg.device, dtype=msg.dtype)
        idx = self.dst.view(1, -1, 1).expand(B, -1, C)
        out.scatter_add_(1, idx, msg)
        return out

    # ── Public operators ───────────────────────────────────────────────────

    def gradient(self, f: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Discrete spatial gradient:
          (∂f/∂x)_i ≈ Σ_j w_ij (f_j − f_i) ê_ij,x  ·  scale
          (∂f/∂y)_i ≈ Σ_j w_ij (f_j − f_i) ê_ij,y  ·  scale

        f      : (B, N, C)
        Returns: grad_x (B, N, C),  grad_y (B, N, C)  in units of [f / m]
        """
        B, N, C = f.shape
        f_i   = f[:, self.dst, :]                # (B, E, C)
        f_j   = f[:, self.src, :]                # (B, E, C)
        diff  = f_j - f_i                        # (B, E, C)

        w  = self.w .view(1, -1, 1)              # (1, E, 1)
        ex = self.ex.view(1, -1, 1)
        ey = self.ey.view(1, -1, 1)

        gx = self._scatter_to_dst(w * ex * diff) * self.scale  # (B, N, C)
        gy = self._scatter_to_dst(w * ey * diff) * self.scale
        return gx, gy

    def laplacian(self, f: Tensor) -> Tensor:
        """
        Discrete Laplacian (graph Laplacian):
          (∇²f)_i ≈ Σ_j w_ij (f_j − f_i) · scale²

        f      : (B, N, C)
        Returns: ∇²f (B, N, C)  in units of [f / m²]
        """
        B, N, C = f.shape
        f_i  = f[:, self.dst, :]
        f_j  = f[:, self.src, :]
        diff = f_j - f_i
        w    = self.w.view(1, -1, 1)
        lap  = self._scatter_to_dst(w * diff) * (self.scale ** 2)
        return lap

    def divergence(self, u: Tensor, v: Tensor) -> Tensor:
        """
        Discrete divergence of the 2-D vector field (u, v):
          (∇·F)_i = (∂u/∂x)_i + (∂v/∂y)_i

        u, v   : (B, N, 1)   zonal and meridional components
        Returns: (B, N, 1)   in units of [1/m · field_units]
        """
        du_dx, _    = self.gradient(u)
        _,     dv_dy = self.gradient(v)
        return du_dx + dv_dy

    def radial_gradient(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Radial pressure gradient ∂p/∂r in the direction of the wind vector.
          ∂p/∂r = (∂p/∂x) cos θ + (∂p/∂y) sin θ
        where θ = atan2(v, u) is the wind direction.

        p, u, v : (B, N, 1)
        Returns : (B, N, 1)
        """
        dp_dx, dp_dy = self.gradient(p)
        theta = torch.atan2(v, u)
        return dp_dx * torch.cos(theta) + dp_dy * torch.sin(theta)
