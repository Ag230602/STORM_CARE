"""
PIGNOConfig — Central configuration for the Physics-Informed Graph Neural Operator.

Mathematical background
-----------------------
The PI-GNO solves the hurricane-state evolution PDE system:

  Momentum (advection):
    ∂u/∂t + (u·∇)u + (1/ρ)∂p/∂x − fv − ν∇²u = 0          (1a)
    ∂v/∂t + (u·∇)v + (1/ρ)∂p/∂y + fu − ν∇²v = 0          (1b)

  Temperature diffusion:
    ∂T/∂t + u·∇T − κ∇²T = Q                                  (2)

  Mass conservation (incompressible approx.):
    ∇·u = ∂u/∂x + ∂v/∂y = 0                                  (3)

  Gradient wind balance (wind-pressure consistency):
    V²/r + fV + (1/ρ)(∂p/∂r) = 0                             (4)

  Kinetic-energy tendency:
    ∂E/∂t + ∇·(E·u) = 0,  E = ½(u² + v²)                    (5)

All five constraints are encoded as soft loss terms with learnable warm-up.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIGNOConfig:
    # ── Spatial domain ─────────────────────────────────────────────────────────
    # Regular N×N grid centred on the storm — nodes are graph vertices.
    # Physical domain: ±domain_radius_deg degrees, grid_spacing_deg resolution.
    grid_size: int            = 33      # N  → N² nodes per sample
    grid_spacing_deg: float   = 0.25    # degrees between grid points (~27.75 km)
    domain_radius_deg: float  = 8.0     # half-width of the domain in degrees

    # Input channels per node: [u850, v850, u500, v500, z500, T2m, MSLP]
    n_in_channels: int        = 7
    # Output channels (predicted field increments): same 7 channels
    n_out_channels: int       = 7

    # ── GNO architecture ───────────────────────────────────────────────────────
    # GNO = Graph Neural Operator (Li et al. 2020, arXiv:2003.03485).
    # Each layer computes:
    #   h_i^{l+1} = σ( W^l h_i^l  +  Σ_{j∈N(i)} κ^l(h_i, h_j, r_ij, θ_ij) )
    # where κ is a learned edge-conditioned kernel (MLP).
    d_v: int                  = 64      # lifting / latent dimension
    d_hidden: int             = 128     # hidden dim inside the kernel MLP
    n_gno_layers: int         = 4       # number of GNO message-passing layers
    k_neighbors: int          = 8       # k for k-NN graph construction

    # ── FNO architecture ───────────────────────────────────────────────────────
    # FNO = Fourier Neural Operator (Li et al. 2021, arXiv:2010.08895).
    # Each layer computes:
    #   v^{l+1} = σ( F⁻¹[ R_φ(k) · F[v^l](k) ] + W v^l )
    # where R_φ are learnable complex Fourier weights (truncated at n_modes).
    n_fno_layers: int         = 4       # number of FNO spectral layers
    n_modes_x: int            = 12      # retained Fourier modes (x-direction)
    n_modes_y: int            = 12      # retained Fourier modes (y-direction)
    fno_width: int            = 64      # must equal d_v (shared latent dim)

    # ── Physics constants ──────────────────────────────────────────────────────
    # Coriolis parameter at latitude φ: f = 2Ω sin(φ), Ω=7.29e-5 rad/s.
    # At ~20 °N: f ≈ 5.0e-5 s⁻¹.
    f_coriolis: float         = 5.0e-5  # s⁻¹

    # Horizontal eddy viscosity (atmospheric-scale diffusion).
    nu_viscosity: float       = 1.0e3   # m² s⁻¹

    # Thermodynamic diffusivity.
    kappa_diffusivity: float  = 1.0e3   # m² s⁻¹

    # Reference air density at 850 hPa.
    rho_air: float            = 1.225   # kg m⁻³

    # Physics time step for residual evaluation (= 1 hour in seconds).
    dt_physics: float         = 3600.0  # s

    # ── Physics loss weights ───────────────────────────────────────────────────
    # Total loss:
    #   L = λ_data · L_data  +  α(t) · [λ_adv R_adv + λ_diff R_diff
    #                                   + λ_mass R_mass + λ_wp R_wp
    #                                   + λ_cont R_cont + λ_nrg R_nrg]
    # where α(t) = min(t / T_warmup, 1) is the physics warm-up schedule.
    lambda_data: float        = 1.00
    lambda_adv: float         = 0.10
    lambda_diff: float        = 0.05
    lambda_mass: float        = 0.10
    lambda_wp: float          = 0.20
    lambda_cont: float        = 0.05
    lambda_energy: float      = 0.05

    # ── Training ───────────────────────────────────────────────────────────────
    lr: float                 = 1e-3
    weight_decay: float       = 1e-4
    n_epochs: int             = 50
    batch_size: int           = 8
    warmup_epochs: int        = 5       # learning-rate warm-up (cosine)
    # Physics loss warm-up: ramp λ_physics from 0 → full over this many epochs.
    physics_warmup_epochs: int = 10

    # ── Synthetic data ─────────────────────────────────────────────────────────
    n_synthetic_storms: int   = 200     # total storms (80 / 20 train/val split)
    n_steps_per_storm: int    = 12      # time steps per synthetic track

    # ── Reproducibility ────────────────────────────────────────────────────────
    seed: int                 = 42
    demo: bool                = False

    # ── Demo overrides ─────────────────────────────────────────────────────────
    def apply_demo_overrides(self) -> "PIGNOConfig":
        """Reduce model/data size for a fast CPU demonstration (~5 min)."""
        self.demo                  = True
        self.grid_size             = 17
        self.d_v                   = 32
        self.fno_width             = 32
        self.d_hidden              = 64
        self.n_gno_layers          = 2
        self.n_fno_layers          = 2
        self.n_modes_x             = 6
        self.n_modes_y             = 6
        self.n_epochs              = 20
        self.batch_size            = 4
        self.n_synthetic_storms    = 60
        self.n_steps_per_storm     = 8
        self.physics_warmup_epochs = 5
        return self

    # ── Derived properties ─────────────────────────────────────────────────────
    @property
    def n_grid_nodes(self) -> int:
        """Number of graph nodes = N²."""
        return self.grid_size * self.grid_size

    @property
    def grid_spacing_m(self) -> float:
        """Physical grid spacing in metres (1° lat ≈ 111 km)."""
        return self.grid_spacing_deg * 111_000.0

    def __str__(self) -> str:
        tag = "[DEMO] " if self.demo else ""
        return (
            f"{tag}PIGNOConfig | grid={self.grid_size}×{self.grid_size} "
            f"| d_v={self.d_v} | GNO×{self.n_gno_layers} "
            f"| FNO×{self.n_fno_layers} modes=({self.n_modes_x},{self.n_modes_y}) "
            f"| λ_data={self.lambda_data} λ_adv={self.lambda_adv} "
            f"λ_mass={self.lambda_mass} λ_wp={self.lambda_wp}"
        )
