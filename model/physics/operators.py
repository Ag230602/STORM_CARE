"""
Neural operator layers for the PI-GNO.

Two complementary operator families are implemented and then composed:

┌──────────────────────────────────────────────────────────────────────────┐
│  1. Graph Neural Operator (GNO)  — local, physics-space                 │
│     Kernel integral on an unstructured k-NN graph:                       │
│       (K(a;φ)v)(x_i) = Σ_{j∈N(i)} κ(h_i, h_j, r_ij, θ_ij; φ) h_j      │
│     where κ: ℝ^{2d_v+3} → ℝ^{d_v} is a learned MLP (KernelNetwork).    │
│     This approximates ∫_D κ(x,y)v(y)dy on the graph support.            │
│                                                                          │
│  2. Fourier Neural Operator (FNO)  — global, spectral-space             │
│     (K(φ)v)(x) = F⁻¹[ R_φ(k) · F[v](k) ](x)                           │
│     Truncated Fourier expansion retaining the lowest n_modes modes.      │
│     SpectralConv2d implements this via rfft2 / irfft2.                   │
│     Captures global wave modes (Rossby, planetary waves) efficiently.    │
└──────────────────────────────────────────────────────────────────────────┘

GNO is applied first (physical-space local interactions), then FNO
(spectral global interactions).  The two operators are theoretically
complementary:
  - GNO: approximates L² integral operator with local support
  - FNO: exact for periodic domains, captures all-scale correlations

References
----------
  Li et al. (2020). "Neural Operator: Graph Kernel Network for PDEs."
    arXiv:2003.03485
  Li et al. (2021). "Fourier Neural Operator for Parametric PDEs."
    arXiv:2010.08895
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_builder import GraphData


# ── 1. Graph Neural Operator (GNO) ────────────────────────────────────────────

class KernelNetwork(nn.Module):
    """
    Edge-conditioned kernel approximation.

    Computes the message from node j to node i as a learned function of the
    concatenated features and edge geometry:

      m_ij = κ( [h_i ‖ h_j ‖ ‖r_ij‖ ‖ cos θ_ij ‖ sin θ_ij] ; φ )

    where r_ij = x_i − x_j (normalised), θ_ij = atan2(Δy, Δx).

    Input dim : 2·d_v + 3  (two node embeddings + 3 edge scalars)
    Output dim: d_v        (message added to destination node i)

    Memory: O(E · d_v) — much more efficient than the full kernel-matrix
    formulation O(E · d_v²) used in the original GKN paper.
    """

    def __init__(self, d_v: int, d_hidden: int):
        super().__init__()
        in_dim = 2 * d_v + 3   # h_i ‖ h_j ‖ dist ‖ cos θ ‖ sin θ
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_v),
        )
        self.d_v = d_v

    def forward(
        self,
        h_i:   Tensor,   # (E, d_v)  destination node features
        h_j:   Tensor,   # (E, d_v)  source node features
        dist:  Tensor,   # (E,)      ‖x_i − x_j‖ in normalised coords
        angle: Tensor,   # (E,)      atan2(Δy, Δx) for x_i − x_j
    ) -> Tensor:
        """Returns message tensor of shape (E, d_v)."""
        edge_feat = torch.cat(
            [
                h_i,
                h_j,
                dist.unsqueeze(-1),
                torch.cos(angle).unsqueeze(-1),
                torch.sin(angle).unsqueeze(-1),
            ],
            dim=-1,
        )                            # (E, 2·d_v + 3)
        return self.net(edge_feat)   # (E, d_v)


class GNOLayer(nn.Module):
    """
    One Graph Neural Operator layer.

    Update rule:
      h_i^{l+1} = σ( LayerNorm( W^l h_i^l  +  Σ_{j∈N(i)} m_ij^l + b^l ) )

    where  m_ij^l = κ^l(h_i^l, h_j^l, r_ij, θ_ij)  is the kernel message.

    The skip connection  W h_i  corresponds to the "pointwise" (W) term in
    the FNO literature.
    """

    def __init__(self, d_v: int, d_hidden: int):
        super().__init__()
        self.kernel = KernelNetwork(d_v, d_hidden)
        self.W      = nn.Linear(d_v, d_v, bias=False)
        self.bias   = nn.Parameter(torch.zeros(d_v))
        self.norm   = nn.LayerNorm(d_v)
        self.d_v    = d_v

    def forward(self, h: Tensor, graph: GraphData) -> Tensor:
        """
        h     : (B, N, d_v)
        graph : GraphData (same for every batch element)
        Returns (B, N, d_v)
        """
        B, N, d_v = h.shape
        src, dst = graph.edge_index[0], graph.edge_index[1]   # (E,)
        E = src.shape[0]

        # Gather features at source (j) and destination (i) for all edges
        # Flatten batch dimension for the kernel MLP
        h_i = h[:, dst, :].reshape(B * E, d_v)               # (B·E, d_v)
        h_j = h[:, src, :].reshape(B * E, d_v)               # (B·E, d_v)

        dist  = graph.edge_dist .unsqueeze(0).expand(B, -1).reshape(B * E)
        angle = graph.edge_angle.unsqueeze(0).expand(B, -1).reshape(B * E)

        # Compute messages
        msg = self.kernel(h_i, h_j, dist, angle)              # (B·E, d_v)
        msg = msg.reshape(B, E, d_v)                          # (B, E, d_v)

        # Aggregate messages at destination nodes (sum over in-edges)
        agg = torch.zeros(B, N, d_v, device=h.device, dtype=h.dtype)
        idx = dst.view(1, -1, 1).expand(B, -1, d_v)
        agg.scatter_add_(1, idx, msg)                         # (B, N, d_v)

        # Linear skip + bias + normalisation + activation
        out = self.W(h) + agg + self.bias
        return F.gelu(self.norm(out))                         # (B, N, d_v)


# ── 2. Fourier Neural Operator (FNO) ──────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """
    2-D Fourier integral operator (truncated).

    (K(φ) v)(x) = F⁻¹[ R_φ(k) · F[v](k) ](x)

    Only the lowest (n_modes_x × n_modes_y) Fourier modes are kept;
    higher-frequency content passes through the skip connection unchanged.

    Learnable parameters:
      W_re, W_im ∈ ℝ^{C_in × C_out × n_modes_x × n_modes_y}
      (stored as separate real/imag tensors for compatibility with older
      PyTorch versions that may not have full complex autograd support)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_modes_x:    int,
        n_modes_y:    int,
    ):
        super().__init__()
        self.out_ch = out_channels
        self.mx     = n_modes_x
        self.my     = n_modes_y

        scale   = 1.0 / (in_channels * out_channels)
        shape   = (in_channels, out_channels, n_modes_x, n_modes_y)
        self.W_re = nn.Parameter(scale * torch.randn(*shape))
        self.W_im = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x: Tensor) -> Tensor:
        """
        x       : (B, C_in,  H, W)  real-valued feature map
        Returns   (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        mx = self.mx
        my = min(self.my, W // 2 + 1)

        # 2-D real FFT → (B, C, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Learnable complex weights
        W_cmplx = torch.complex(self.W_re, self.W_im)         # (C, Co, mx, my)
        W_cmplx = W_cmplx[:, :, :mx, :my]

        # Truncated spectral multiplication (only low modes):
        # out_ft[b, d, kx, ky] = Σ_c W_cmplx[c, d, kx, ky] · x_ft[b, c, kx, ky]
        out_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :mx, :my] = torch.einsum(
            "bcxy,cdxy->bdxy",
            x_ft[:, :, :mx, :my].to(torch.cfloat),
            W_cmplx,
        )

        # Inverse 2-D real FFT → (B, C_out, H, W) real
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOLayer(nn.Module):
    """
    One FNO block.

    v^{l+1}(x) = σ( K(φ^l) v^l(x)  +  W^l v^l(x) )

    where  K(φ) = F⁻¹ R_φ F  is the spectral integral operator and
    W is a pointwise (1×1 convolution) skip connection.
    The two branches are added before the nonlinearity.
    """

    def __init__(self, width: int, n_modes_x: int, n_modes_y: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, n_modes_x, n_modes_y)
        self.skip     = nn.Conv2d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(width, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, width, H, W) → (B, width, H, W)"""
        return F.gelu(self.norm(self.spectral(x) + self.skip(x)))
